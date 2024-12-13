import re
import os
import subprocess
import time
import shutil
import argparse
import sys

MAX_TRIALS = 3

SLURM_PARAMS_HIGH_MEM = {
    'nprocs': 24,
    'maxcore': 8024
}

SLURM_PARAMS_LOW_MEM = {
    'nprocs': 24,
    'maxcore': 2524
}

SUBMIT_COMMAND = ["ssubo", "-m", "No"]
CHECK_STATES = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']
RETRY_DELAY = 60
FREQ_THRESHOLD = -50


def run_subprocess(command, shell=False, capture_output=True, check=True,exit_on_error=True):
    try:
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=capture_output,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command '{' '.join(command)}' failed with error: {e.stderr}")
        sys.exit(1)


def check_job_status(job_id, interval=45, step=""):
    start_time = time.time()
    counter = 0
    while True:
        print(f'Checking job {job_id} status {step}')
        squeue = run_subprocess(['squeue', '-j', job_id, '-h'])
        if squeue.stdout.strip():
            if counter % 10 == 0:
                print(f'Job {job_id} is still in the queue.')
            counter += 1
        else:
            sacct = run_subprocess(['sacct', '-j', job_id, '--format=State', '--noheader'])
            if sacct.stderr.strip():
                print(f"Error from sacct: {sacct.stderr.strip()}")
                return "UNKNOWN"
            statuses = sacct.stdout.strip().split('\n')
            latest_status = statuses[-1].strip()
            print(f'Latest status for job {job_id}: {latest_status}')
            if latest_status in CHECK_STATES:
                end_time = time.time()
                print(f"Job {job_id} finished in {end_time - start_time} seconds")
                return latest_status
        time.sleep(interval)


def make_folder(dir_name):
    path = os.path.join(os.getcwd(), dir_name)
    if os.path.exists(path):
        if os.path.isdir(path):
            print(f"Removing existing folder {path}")
            shutil.rmtree(path)
        else:
            print(f"Removing existing file {path}")
            os.remove(path)
    os.makedirs(path)
    print(f"Created folder {path}")


def submit_job(input_file, output_file, walltime="24"):
    command = SUBMIT_COMMAND + ["-w", walltime, "-o", output_file, input_file]
    result = run_subprocess(command)
    for line in result.stdout.splitlines():
        if 'Submitted batch job' in line:
            job_id = line.split()[-1]
            print(f"Job submitted with ID: {job_id}")
            return job_id
    print(f"Failed to submit job {input_file}")
    sys.exit(1)


def grep_output(pattern, file):
    result = run_subprocess(f"grep '{pattern}' {file}", shell=True, capture_output=True, check=False,exit_on_error=False)
    return result.stdout.strip()


def optimise_reactants(charge=0, mult=1, trial=0, upper_limit=5, solvent="", xtb=True):
    trial += 1
    print(f"Trial {trial} for reactant optimisation")
    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent_formatted = f"ALPB({solvent})" if solvent and xtb else f"CPCM({solvent})" if solvent else ""
    method = "XTB2" if xtb else "r2scan-3c"

    if trial < 2:
        make_folder('OPT')
        run_subprocess("cp *.xyz OPT", shell=True)
        os.chdir('OPT')

    print('Starting reactant optimisation')

    job_inputs = {
        'educt_opt.inp': f"!{method} {solvent_formatted} opt\n%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n*xyzfile {charge} {mult} educt.xyz\n",
        'product_opt.inp': f"!{method} {solvent_formatted} opt\n%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n*xyzfile {charge} {mult} product.xyz\n"
    }

    for filename, content in job_inputs.items():
        with open(filename, 'w') as f:
            f.write(content)

    print('Submitting educt and product jobs to Slurm')
    job_ids = [submit_job(inp, f"{inp.split('.')[0]}_slurm.out") for inp in job_inputs.keys()]

    status_educt, status_product = [check_job_status(job_id, step=step) for job_id, step in zip(job_ids, ['educt', 'product'])]

    print(f"Job statuses - Educt: {status_educt}, Product: {status_product}")

    if status_educt == 'COMPLETED' and status_product == 'COMPLETED':
        educt_success = 'HURRAY' in grep_output('HURRAY', 'educt_opt.out')
        product_success = 'HURRAY' in grep_output('HURRAY', 'product_opt.out')
        if educt_success and product_success:
            os.chdir('..')
            return True

    print('One of the optimisation jobs failed, restarting both')
    for job_id in job_ids:
        run_subprocess(['scancel', job_id],exit_on_error=False) ## This step is for safety, 
    time.sleep(RETRY_DELAY)
    run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm* educt_opt.inp product_opt.inp", shell=True,exit_on_error=False)

    for xyz in ['educt_opt.xyz', 'product_opt.xyz']:
        if os.path.exists(xyz):
            os.rename(xyz, xyz.replace('_opt', ''))

    return optimise_reactants(charge, mult, trial, upper_limit, solvent, xtb)


def freq_job(struc_name="coord.xyz", charge=0, mult=1, trial=0, upper_limit=5, solvent="", xtb=False, ts=False):
    trial += 1
    print(f"Trial {trial} for frequency job")

    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    method = "XTB2 numfreq tightscf" if xtb else "r2scan-3c freq tightscf"
    solvent_formatted = f"CPCM({solvent})" if solvent and not xtb else f"ALPB({solvent})" if solvent else ""

    freq_input = f"! {method} {solvent_formatted}\n%pal nprocs {SLURM_PARAMS_HIGH_MEM['nprocs']} end\n%maxcore {SLURM_PARAMS_HIGH_MEM['maxcore']}\n*xyzfile {charge} {mult} {struc_name}\n"
    with open('freq.inp', 'w') as f:
        f.write(freq_input)

    print('Submitting frequency job to Slurm')
    job_id_freq = submit_job("freq.inp", "freq_slurm.out")
    status_freq = check_job_status(job_id_freq, step='Freq')

    if status_freq == 'COMPLETED':
        if grep_output('VIBRATIONAL FREQUENCIES', 'freq.out'):
            print('Frequency calculation completed successfully')
            if ts:
                output = grep_output('**imaginary mode***', 'freq.out')
                match = re.search(r'(-?\d+\.\d+)\s*cm\*\*-1', output)
                if match:
                    imag_freq = float(match.group(1))
                    if imag_freq < FREQ_THRESHOLD:
                        print('Negative frequency found, proceeding with TS optimisation')
                        return True
                    else:
                        print('Negative frequency above threshold, aborting TS optimisation')
                        return False
                print('No negative frequency found, aborting TS optimisation')
                return False
            return True

    print('Frequency job failed, restarting...')
    run_subprocess(['scancel', job_id_freq],exit_on_error=False)
    run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm*", shell=True,exit_on_error=False)
    return freq_job(struc_name, charge, mult, trial, upper_limit, solvent, ts)


def NEB_TS(charge=0, mult=1, trial=0, Nimages=16, upper_limit=5, xtb=True, fast=True, solvent="", switch=False):
    if switch:
        os.chdir('..')
        print("Switching to DFT run and optimizing reactants")
        if not optimise_reactants(charge, mult, trial=0, upper_limit=MAX_TRIALS, solvent=solvent, xtb=False):
            print("Reactant optimisation failed, aborting")
            return False
        switch = False
    job_step = 'NEB-TS'
    trial += 1
    print(f"Trial {trial} for NEB-TS")

    slurm_params=SLURM_PARAMS_HIGH_MEM

    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent and not xtb else f"ALPB({solvent})" if solvent else ""
    method = "FAST-NEB-TS XTB2 tightscf" if fast and xtb else \
             "FAST-NEB-TS r2scan-3c tightscf" if fast else \
             "NEB-TS XTB2 tightscf" if xtb else \
             "NEB-TS r2scan-3c tightscf"

    if trial < 2:
        make_folder("NEB")
        run_subprocess("cp OPT/educt_opt.xyz NEB/educt.xyz", shell=True)
        run_subprocess("cp OPT/product_opt.xyz NEB/product.xyz", shell=True)
        os.chdir('NEB')

    guess_block = ' TS "guess.xyz"\n' if os.path.exists("guess.xyz") else ""
    geom_block = ""
    if xtb:
        with open("product.xyz") as f:
            nAtoms = int(f.readline().strip())
        maxiter = nAtoms * 4
        geom_block = f"%geom\n Calc_Hess true\n Recalc_Hess 10\n MaxIter={maxiter} end \n"
        slurm_params=SLURM_PARAMS_LOW_MEM

    neb_input = (f"! {method} {solvent_formatted}\n"
                 f" {geom_block}"
                 f" %pal nprocs {slurm_params['nprocs']} end\n"
                 f"%maxcore {slurm_params['maxcore']}\n"
                 f"%neb \n Product \"product.xyz\" \n NImages {Nimages}  \n"
                 f" {guess_block} end\n"
                 f"*xyzfile {charge} {mult} educt.xyz\n")
    neb_input_name = "neb-fast-TS.inp" if fast else "neb-TS.inp"
    with open(neb_input_name, 'w') as f:
        f.write(neb_input)

    print(f'Submitting {neb_input_name} job to Slurm')
    job_id = submit_job(neb_input_name, f"{job_step}_slurm.out", walltime="48")
    status = check_job_status(job_id, step="NEB-TS")

    if status == 'COMPLETED' and grep_output('HURRAY', f'{neb_input_name.rsplit(".", 1)[0]}.out'):
        print('NEB-TS completed successfully')
        imag_freq = freq_job(
            struc_name=f'{neb_input_name.rsplit(".", 1)[0]}_NEB-TS_converged.xyz',
            charge=charge,
            mult=mult,
            trial=0,
            upper_limit=upper_limit,
            solvent=solvent,
            xtb=xtb,
            ts=True
        )

        if imag_freq:
            print("Negative mode below threshold found, continuing")
            run_subprocess(
                f"cp {neb_input_name.rsplit('.', 1)[0]}_NEB-TS_converged.xyz neb_completed.xyz",
                shell=True
            )
            os.chdir('..')
            return True
        else:
            if not handle_failed_imagfreq(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent):
                return False
            else:
                print("Negative mode found in guess. Continuing with TS optimisation.")
                run_subprocess("cp neb-TS_NEB-CI_converged.xyz neb_completed.xyz", shell=True)
                os.chdir('..')
                return True
                ### uper limit is technicaly not needed here
    elif grep_output('ORCA TERMINATED NORMALLY', f'{neb_input_name.rsplit(".", 1)[0]}.out'):
        if not handle_unconverged_neb(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent, job_id):
            return False
        else:
            print("R2SCAN-3c NEB-TS job did not converge, but freq job found negative frequency above guess. Fail might be due no Hess recalc. Continuing with TS optimisation.")
            run_subprocess("cp neb-TS_NEB-CI_converged.xyz neb_completed.xyz", shell=True)
            os.chdir('..')
            return True
            
    else:
        print(f"NEB-TS {neb_input_name} failed, restarting if there are trials left")
        print("Will be removed when segfaults are fixed")
        return NEB_TS(charge, mult, trial, Nimages, upper_limit, xtb, fast, solvent, switch)

   


def handle_failed_imagfreq(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent):
    print("No significant negative frequency found")
    if not xtb and fast:
        print("FAST-NEB-TS with r2scan-3c did not find TS. Retrying with regular NEB-TS.")
        print("Using FAST-NEB-TS guess as TS guess for NEB-TS")
        if os.path.exists("neb-fast-TS_NEB-HEI_converged.xyz"):

            run_subprocess("mv neb-fast-TS_NEB-HEI_converged.xyz guess.xyz", shell=True)
        else:
            print("Could not find NEB-HEI file, start new NEB without guess")
        run_subprocess("rm pmix* *densities* freq.inp slurm* neb*im* *neb*.inp", shell=True,exit_on_error=False)
        return NEB_TS(charge, mult, Nimages=8, trial=0, upper_limit=MAX_TRIALS, xtb=False, fast=False, solvent=solvent)
    elif not xtb and not fast:
        print("NEB-TS r2scan-3c did not find TS. Checking guess mode.")
        return freq_job(
            struc_name=f'neb-TS_NEB-CI_converged.xyz',
            charge=charge,
            mult=mult,
            trial=0,
            upper_limit=MAX_TRIALS,
            solvent=solvent,
            xtb=xtb,
            ts=True
        )

    elif xtb and not fast:
        print("NEB-TS XTB2 did not find TS. Retrying with FAST-NEB-TS with r2scan-3c.")
        print("Using NEB-TS guess as TS guess for FAST-NEB-TS")
        if os.path.exists("neb-TS_NEB-CI_converged.xyz"):
            run_subprocess("mv neb-TS_NEB-CI_converged.xyz guess.xyz", shell=True)
        else:
            print("Could not find NEB-CI file, NEB will be started without guess")
        run_subprocess("rm pmix* *densities* freq.inp slurm* neb*im* *neb*.inp", shell=True,exit_on_error=False)
        
        return NEB_TS(charge, mult, Nimages=8, trial=1, upper_limit=MAX_TRIALS+1, xtb=True, fast=True, solvent=solvent, switch=True)
    elif xtb and fast:
        print("Retrying with regular NEB-TS and XTB.")
        print("Using FAST-NEB-TS guess as TS guess for NEB-TS")
        if os.path.exists("neb-fast-TS_NEB-HEI_converged.xyz"):

            run_subprocess("mv neb-fast-TS_NEB-HEI_converged.xyz guess.xyz", shell=True)
        else:
            print("NEB-HEI file is not present, start new NEB withouth guess")
        run_subprocess("rm pmix* *densities* freq.inp slurm* neb*im* *neb*.inp", shell=True,exit_on_error=False)
        return NEB_TS(charge, mult, Nimages=24, trial=1, upper_limit=MAX_TRIALS+1, xtb=True, fast=False, solvent=solvent)
    return False


def handle_unconverged_neb(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent, job_id):
    print(f'NEB-TS job did not converge')
    run_subprocess(['scancel', job_id],exit_on_error=False)
    time.sleep(60)
    
    run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm* neb*im* *neb*.inp", shell=True,exit_on_error=False)
    if fast and not xtb:
        print("Restarting as R2SCAN-3c NEB-TS run using FAST-NEB-TS guess")
        if os.path.exists("neb-fast-TS_NEB-HEI_converged.xyz"):
            run_subprocess("cp neb-fast-TS_NEB-HEI_converged.xyz guess.xyz", shell=True)
        else:
            print("Could not find NEB-HEI file, NEB will be started without guess")
        return NEB_TS(charge, mult, Nimages=8, trial=1, upper_limit=MAX_TRIALS+1, xtb=False, fast=False, solvent=solvent, switch=False)
    if fast and xtb:
        print("Restarting as XTB2 NEB-TS")
        return NEB_TS(charge, mult, Nimages=24, trial=1, upper_limit=MAX_TRIALS+1, xtb=xtb, fast=False, solvent=solvent, switch=False)
    if not fast and  xtb:
        print("Restarting as R2SCAN-3c FAST-NEB-TS")
        return NEB_TS(charge, mult, Nimages=12, trial=0, upper_limit=MAX_TRIALS, xtb=xtb, fast=True, solvent=solvent, switch=True)
    if not fast and not xtb:
        print("R2SCAN-3c NEB-TS did not converge.")
        print("This might be due to challenging PES near the TS. Checking guess mode.")
        imag_freq = freq_job(
            struc_name=f'neb-TS_NEB-CI_converged.xyz',
            charge=charge,
            mult=mult,
            trial=0,
            upper_limit=MAX_TRIALS,
            solvent=solvent,
            xtb=xtb,
            ts=True
        )
        return imag_freq
    return False

def NEB_CI(charge=0, mult=1, trial=0, Nimages=8, upper_limit=5, xtb=True, solvent=""):
    trial += 1
    print(f"Trial {trial} for NEB-CI")
    
    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent and not xtb else f"ALPB({solvent})" if solvent else ""
    method = "XTB2" if xtb else "r2scan-3c"
    images = Nimages + 2 if xtb else Nimages
    job_step = 'NEB-CI'

    if trial < 2:
        make_folder(job_step)
        run_subprocess("cp OPT/educt_opt.xyz NEB/educt.xyz", shell=True)
        run_subprocess("cp OPT/product_opt.xyz NEB/product.xyz", shell=True)
        os.chdir('NEB')

    neb_input = (f"!NEB-CI {solvent_formatted} {method}\n"
                 f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
                 f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
                 f"%neb \n Product \"product.xyz\" \n NImages {images} {'Free_end true' if xtb else ''} \n end\n"
                 f"*xyzfile {charge} {mult} educt.xyz\n")
    with open('neb-CI.inp', 'w') as f:
        f.write(neb_input)

    print('Submitting NEB-CI job to Slurm')
    job_id = submit_job("neb-CI.inp", "neb-ci_slurm.out", walltime="48")
    status = check_job_status(job_id, step=job_step)

    if status == 'COMPLETED' and 'H U R R A Y' in grep_output('H U R R A Y', 'neb-CI.out'):
        print('NEB-CI completed successfully')
        os.chdir('..')
        return True

    print('NEB-CI job failed, restarting...')
    run_subprocess(['scancel', job_id])
    time.sleep(RETRY_DELAY)
    if grep_output('ORCA TERMINATED NORMALLY', 'neb-ci.out'):
        images = images * 2 if xtb else images + 2
    run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm* neb*im* *neb*.inp", shell=True)
    return NEB_CI(charge, mult, trial, images, upper_limit, xtb, solvent)


def TS_opt(charge=0, mult=1, trial=0, upper_limit=5, solvent=""):
    trial += 1
    print(f"Trial {trial} for TS optimisation")

    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent else ""
    job_step = 'TS'



    if trial < 2:
        make_folder(job_step)
        run_subprocess("cp NEB/neb_completed.xyz TS/ts_guess.xyz", shell=True)
        os.chdir('TS')

    print('Starting TS optimisation')

    if not freq_job(struc_name="ts_guess.xyz", charge=charge, mult=mult, trial=0, upper_limit=upper_limit, solvent=solvent, ts=True):
        print('Frequency job failed, aborting TS optimisation')
        return False

    TS_input = (f"!r2scan-3c OptTS tightscf {solvent_formatted}\n"
                f"%geom\ninhess read \ninhessname \"freq.hess\"\nCalc_Hess true\nrecalc_hess 15\nend\n"
                f"%pal nprocs {SLURM_PARAMS_HIGH_MEM['nprocs']} end\n"
                f"%maxcore {SLURM_PARAMS_HIGH_MEM['maxcore']}\n"
                f"*xyzfile {charge} {mult} ts_guess.xyz\n")
    with open('TS_opt.inp', 'w') as f:
        f.write(TS_input)

    print('Submitting TS optimisation job to Slurm')
    job_id_TS = submit_job("TS_opt.inp", "TS_opt_slurm.out", walltime="48")
    status_TS = check_job_status(job_id_TS, step=job_step)

    if status_TS == 'COMPLETED' and 'HURRAY' in grep_output('HURRAY', 'TS_opt.out'):
        print('TS optimisation completed successfully')
        os.chdir('..')
        return True

    print('TS optimisation job failed, restarting...')
    run_subprocess(['scancel', job_id_TS],exit_on_error=False)
    time.sleep(RETRY_DELAY)
    run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess", shell=True,exit_on_error=False)
    if os.path.exists('TS_opt.xyz'):
        os.rename('TS_opt.xyz', 'ts_guess.xyz')

    return TS_opt(charge, mult, trial, upper_limit, solvent)


def IRC_job(charge=0, mult=1, trial=0, upper_limit=5, solvent="", maxiter=70):
    trial += 1
    print(f"Trial {trial} for IRC")

    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent else ""
    job_step = 'IRC'

    if trial < 2:
        make_folder(job_step)
        run_subprocess("cp TS/TS_opt.xyz IRC/", shell=True)
        os.chdir('IRC')

        if not freq_job(struc_name="TS_opt.xyz", charge=charge, mult=mult, trial=0, upper_limit=upper_limit, solvent=solvent, ts=True):
            print('Frequency job failed or lacks negative frequency. Aborting IRC.')
            return False

    IRC_input = (f"!r2scan-3c IRC tightscf {solvent_formatted}\n"
                 f"%irc\n  maxiter {maxiter} \nInitHess read \nHess_Filename \"freq.hess\"\nend \n"
                 f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
                 f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
                 f"*xyzfile {charge} {mult} TS_opt.xyz\n")
    with open('IRC.inp', 'w') as f:
        f.write(IRC_input)

    print('Submitting IRC job to Slurm')
    job_id_IRC = submit_job("IRC.inp", "IRC_slurm.out")
    status_IRC = check_job_status(job_id_IRC, step=job_step)

    if status_IRC == 'COMPLETED' and 'HURRAY' in grep_output('HURRAY', 'IRC.out'):
        print('IRC completed successfully')
        os.chdir('..')
        return True

    print('IRC job failed, restarting...')
    run_subprocess(['scancel', job_id_IRC],exit_on_error=False)
    if 'ORCA TERMINATED NORMALLY' in grep_output('ORCA TERMINATED NORMALLY', 'IRC.out'):
        print("ORCA terminated normally, possibly due to non-convergence.")
        if maxiter < 100:
            maxiter += 30
        else:
            print("Max iterations reached, aborting IRC.")
            return False

    time.sleep(RETRY_DELAY)
    run_subprocess("rm -rf *.gbw pmix* *densities* IRC.inp slurm*", shell=True,exit_on_error=False)
    return IRC_job(charge, mult, trial, upper_limit, solvent, maxiter)


def restart_pipeline():
    print("Restart detected, loading previous settings.")
    if not os.path.exists("settings_neb_pipeline.txt"):
        print("Settings file not found, aborting.")
        sys.exit(1)

    with open("settings_neb_pipeline.txt", 'r') as f:
        settings = dict(line.strip().split(':') for line in f if ':' in line)

    charge = int(settings.get('charge', 0))
    mult = int(settings.get('mult', 1))
    solvent = settings.get('solvent', '')
    Nimages = int(settings.get('Nimages', 16))
    xtb = settings.get('xtb', 'False') == 'True'
    step_to_restart = settings.get('step', '')

    return step_to_restart, charge, mult, solvent, Nimages, xtb


def pipeline(step, charge=0, mult=1, solvent="", Nimages=16):
    steps_mapping = {
        "OPT_XTB": lambda: optimise_reactants(charge, mult, trial=0, upper_limit=MAX_TRIALS, solvent=solvent),
        "NEB_CI_XTB": lambda: NEB_CI(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS, xtb=True, solvent=solvent),
        "NEB_CI_DFT": lambda: NEB_CI(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS, xtb=False, solvent=solvent),
        "FAST_NEB-TS_XTB": lambda: NEB_TS(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS, xtb=True, fast=True, solvent=solvent),
        "FAST_NEB-TS_DFT": lambda: NEB_TS(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS, xtb=False, fast=True, solvent=solvent),
        "NEB-TS_XTB": lambda: NEB_TS(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS, xtb=True, fast=False, solvent=solvent),
        "NEB-TS_DFT": lambda: NEB_TS(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS, xtb=False, fast=False, solvent=solvent),
        "TS": lambda: TS_opt(charge, mult, trial=0, upper_limit=MAX_TRIALS, solvent=solvent),
        "IRC": lambda: IRC_job(charge, mult, trial=0, upper_limit=MAX_TRIALS, solvent=solvent)
    }

    step_function = steps_mapping.get(step)
    if not step_function:
        print("Invalid step")
        print("Valid steps: OPT_{XTB/DFT}, NEB_CI_{XTB/DFT}, (FAST)-NEB_TS_{XTB/DFT}, TS, IRC")
        sys.exit(1)

    return step_function()


def usage():
    print("Usage: neb_pipeline.py [options]")
    print("Options:")
    print("-c, --charge: Charge of the system")
    print("-m, --mult: Multiplicity of the system")
    print("-s, --solvent: Solvent model to use")
    print("-i, --Nimages: Number of images for NEB")
    print("--steps: Steps to run in the pipeline, comma separated")
    print("--restart: Restart from previous calculations")


def main(charge=0, mult=1, solvent="", Nimages=32, restart=False, steps=["OPT_XTB", "NEB-TS_XTB", "TS", "IRC"]):
    setting_file_name = "settings_neb_pipeline.txt"

    if restart:
        step_to_restart, charge, mult, solvent, Nimages, xtb = restart_pipeline()
        steps = steps[steps.index(step_to_restart):]
        print(f"Restarting from step: {step_to_restart}")

    for step in steps:
        print(f"Running step: {step}")
        if not pipeline(step=step, charge=charge, mult=mult, solvent=solvent, Nimages=Nimages):
            print(f"{step} failed. Saving state for restart.")
            with open(setting_file_name, 'w') as f:
                f.write(f"step: {step}\n")
                f.write(f"charge: {charge}\n")
                f.write(f"mult: {mult}\n")
                f.write(f"solvent: {solvent}\n")
                f.write(f"Nimages: {Nimages}\n")
            sys.exit(1)
        print(f"{step} completed successfully\n")

    print('All pipeline steps completed successfully.')
    with open(setting_file_name, 'w') as f:
        f.write(f"step: Completed\n")
        f.write(f"charge: {charge}\n")
        f.write(f"mult: {mult}\n")
        f.write(f"solvent: {solvent}\n")
        f.write(f"Nimages: {Nimages}\n")
        f.write(f"steps: {','.join(steps)}\n")
    sys.exit(0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_steps(steps_str):
    return steps_str.split(',')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NEB optimization pipeline.")
    parser.add_argument("--charge", "-c", type=int, default=0, help="Charge of the system")
    parser.add_argument("--mult", "-m", type=int, default=1, help="Multiplicity of the system")
    parser.add_argument("--solvent", "-s", type=str, default="", help="Solvent model to use")
    parser.add_argument("--Nimages", "-i", type=int, default=8, help="Number of images for NEB")
    parser.add_argument("--restart", type=str2bool, default=False, help="Restart from previous calculations")
    parser.add_argument("--steps", type=parse_steps, default=["OPT_XTB", "NEB-TS_XTB", "TS", "IRC"], help="Steps to run in the pipeline, comma separated")

    args = parser.parse_args()

    settings = (
        f'Charge: {args.charge}\n'
        f'Multiplicity: {args.mult}\n'
        f'Solvent: {args.solvent}\n'
        f'Nimages: {args.Nimages}\n'
        f'Steps: {",".join(args.steps)}\n'
        f'Restart: {args.restart}'
    )
    print("Starting Pipeline with parameters:")
    print(settings)

    main(
        charge=args.charge,
        mult=args.mult,
        solvent=args.solvent,
        Nimages=args.Nimages,
        restart=args.restart,
        steps=args.steps
    )

