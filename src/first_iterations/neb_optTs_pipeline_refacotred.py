import os
import subprocess
import time
import shutil
import argparse
import sys

MAX_TRIALS = 5

SLURM_PARAMS_FREQ = {
    'nprocs': 24,
    'maxcore': 8024
}

SLURM_PARAMS_OPT = {
    'nprocs': 24,
    'maxcore': 2524
}

def check_job_status(job_id, interval=20, step=""):
    counter = 0
    while True:
        print(f'Checking job {job_id} status {step}')
        squeue = subprocess.run(['squeue', '-j', job_id, '-h'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if squeue.stdout.strip():
            if counter%10 == 0:
                print(f'Job {job_id} is still in the queue.')
            print(f'Job {job_id} is still in the queue.')
            counter += 1
        else:
            sacct = subprocess.run(['sacct', '-j', job_id, '--format=State', '--noheader'],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if sacct.stderr.strip():
                print(f"Error from sacct: {sacct.stderr.strip()}")
                return "UNKNOWN"
            statuses = sacct.stdout.strip().split('\n')
            latest_status = statuses[-1].strip()
            print(f'Latest status for job {job_id}: {latest_status}')
            if latest_status in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                return latest_status
        time.sleep(interval)

def make_folders(folder_names):
    for dir_name in folder_names:
        path = os.path.join(os.getcwd(), dir_name)
        if os.path.exists(path):
            print(f"Deleting existing folder {path}")
            shutil.rmtree(path)
        os.makedirs(path)
        print(f"Created folder {path}")

def submit_job(input_file, output_file):
    try:
        result = subprocess.run(
            ["ssubo", "-o", output_file, input_file],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.splitlines():
            if 'Submitted batch job' in line:
                return line.split()[-1]
        print(f"Failed to submit job {input_file}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error submitting job {input_file}: {e.stderr}")
        sys.exit(1)

def grep_output(pattern, file):
    result = subprocess.run(f"grep '{pattern}' {file}", shell=True, stdout=subprocess.PIPE)
    return result.stdout.decode().strip()

def optimise_reactants(charge=0, mult=1, trial=0, upper_limit=5, solvent=""):
    trial += 1
    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent else ""
    job_step = 'OPT'

    if trial < 2:
        subprocess.run("cp *.xyz OPT", shell=True)
        os.chdir('OPT')

    print('Starting reactant optimisation\n\n')

    educt_opt_input = f"!r2scan-3c {solvent_formatted} opt\n%pal nprocs {SLURM_PARAMS_OPT['nprocs']} end\n%maxcore {SLURM_PARAMS_OPT['maxcore']}\n*xyzfile {charge} {mult} educt.xyz\n"
    product_opt_input = f"!r2scan-3c {solvent_formatted} opt\n%pal nprocs {SLURM_PARAMS_OPT['nprocs']} end\n%maxcore {SLURM_PARAMS_OPT['maxcore']}\n*xyzfile {charge} {mult} product.xyz\n"

    with open('educt_opt.inp', 'w') as f:
        f.write(educt_opt_input)

    with open('product_opt.inp', 'w') as f:
        f.write(product_opt_input)

    print('Submitting educt job to slurm')
    job_id_educt = submit_job("educt_opt.inp", "educt_slurm.out")

    print('Submitting product job to slurm')
    job_id_product = submit_job("product_opt.inp", "product_slurm.out")

    print("Checking job status")
    status_educt = check_job_status(job_id_educt, step='educt')
    status_product = check_job_status(job_id_product, step='product')

    print(f"Status educt: {status_educt}, Status product: {status_product}")
    print('Checking if optimisation completed successfully\n')

    if status_educt == 'COMPLETED' and status_product == 'COMPLETED':
        if grep_output('HURRAY', 'educt_opt.out'):
            print('Educt optimisation completed successfully')
            educt_success = True
        else:
            educt_success = False

        if grep_output('HURRAY', 'product_opt.out'):
            print('Product optimisation completed successfully')
            product_success = True
        else:
            product_success = False

        if educt_success and product_success:
            os.chdir('..')
            return True

    print('One of the jobs failed, restarting both from last optimisation')
    subprocess.run(f"scancel {job_id_educt}", shell=True)
    subprocess.run(f"scancel {job_id_product}", shell=True)
    print('Waiting for 60 seconds before restarting')
    time.sleep(60)
    subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess educt_opt.inp product_opt.inp", shell=True)

    if os.path.exists('educt_opt.xyz'):
        subprocess.run(["mv", "educt_opt.xyz", "educt.xyz"])
    if os.path.exists('product_opt.xyz'):
        subprocess.run(["mv", "product_opt.xyz", "product.xyz"])

    return optimise_reactants(charge, mult, trial, upper_limit, solvent)

def freq_job(struc_name="coord.xyz", charge=0, mult=1, trial=0, upper_limit=5, solvent="", ts=False):
    trial += 1
    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent = f"CPCM({solvent})" if solvent else ""

    freq_inp = f"!r2scan-3c {solvent} freq tightscf \n%pal nprocs {SLURM_PARAMS_FREQ['nprocs']} end\n%maxcore {SLURM_PARAMS_FREQ['maxcore']}\n*xyzfile {charge} {mult} {struc_name}\n"
    with open('freq.inp', 'w') as f:
        f.write(freq_inp)

    print('Submitting Freq job to slurm')
    job_id_freq = submit_job("freq.inp", "freq_slurm.out")
    status_freq = check_job_status(job_id_freq, step='Freq')

    print('Checking if Freq completed successfully')
    if status_freq == 'COMPLETED':
        if grep_output('VIBRATIONAL FREQUENCIES', 'freq.out'):
            print('Freq calculation completed successfully')
            if ts:
                if grep_output('**imaginary mode***', 'freq.out'):
                    print('Negative frequency found, continuing with TS optimisation')
                    return True
                else:
                    print('No negative frequency found, aborting TS optimisation')
                    return False
            return True
    print('Freq job failed, restarting...')
    subprocess.run(f"scancel {job_id_freq}", shell=True)
    subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess", shell=True)
    return freq_job(struc_name, charge, mult, trial, upper_limit, solvent, ts)

def NEB_CI(charge=0, mult=1, trial=0, Nimages=8, upper_limit=5, xtb=True, solvent=""):
    trial += 1
    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent and not xtb else f"ALPB({solvent})" if solvent else ""
    method = "XTB2" if xtb else "r2scan-3c"
    images = Nimages + 2 if xtb else Nimages
    job_step = 'NEB'

    if trial < 2:
        subprocess.run("cp OPT/educt_opt.xyz NEB/educt.xyz", shell=True)
        subprocess.run("cp OPT/product_opt.xyz NEB/product.xyz", shell=True)
        os.chdir('NEB')

    neb_input = f"!NEB-CI {solvent_formatted} {method} \n%pal nprocs {SLURM_PARAMS_OPT['nprocs']} end\n%maxcore {SLURM_PARAMS_OPT['maxcore']}\n%neb \n Product \"product.xyz\" \n NImages {images} {'Free_end true' if xtb else ''} \n end\n*xyzfile {charge} {mult} educt.xyz\n"
    with open('neb-CI.inp', 'w') as f:
        f.write(neb_input)

    print('Submitting NEB job to slurm')
    job_id = submit_job("neb-CI.inp", "neb-ci_slurm.out")
    status = check_job_status(job_id, step=job_step)

    if status == 'COMPLETED' and grep_output('H U R R A Y', 'neb-CI.out'):
        print('NEB optimization completed successfully')
        os.chdir('..')
        return True
    else:
        print('NEB job failed, restarting...')
        subprocess.run(f"scancel {job_id}", shell=True)
        time.sleep(30)
        if grep_output('ORCA TERMINATED NORMALLY', 'neb-ci.out'):
            images = images * 2 if xtb else images + 2
        subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess neb*im* *neb*.inp", shell=True)
        return NEB_CI(charge, mult, trial, images, upper_limit, xtb, solvent)

def TS_opt(charge=0, mult=1, trial=0, upper_limit=5, solvent=""):
    trial += 1
    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent else ""
    job_step = 'TS'

    if trial < 2:
        subprocess.run("cp NEB/*NEB-CI_converged.xyz TS/ts_guess.xyz", shell=True)
        os.chdir('TS')

    print('Starting TS optimisation\n\n')

    if not freq_job(struc_name="ts_guess.xyz", charge=charge, mult=mult, trial=0, upper_limit=upper_limit, solvent=solvent, ts=True):
        print('Freq job failed aborting TS optimisation')
        return False

    TS_opt_input = f"!r2scan-3c OptTS tightscf {solvent_formatted}\n%geom\ninhess read \ninhessname \"freq.hess\"\nCalc_Hess true \nrecalc_hess 15 \nend \n%pal nprocs {SLURM_PARAMS_OPT['nprocs']} end\n%maxcore {SLURM_PARAMS_FREQ['maxcore']}\n*xyzfile {charge} {mult} ts_guess.xyz\n"
    with open('TS_opt.inp', 'w') as f:
        f.write(TS_opt_input)

    print('Submitting TS job to slurm')
    job_id_TS = submit_job("TS_opt.inp", "TS_opt_slurm.out")
    status_TS = check_job_status(job_id_TS, step=job_step)

    print('Checking if optimisation completed successfully')
    if status_TS == 'COMPLETED' and grep_output('HURRAY', 'TS_opt.out'):
        print('TS optimisation completed successfully')
        os.chdir('..')
        return True

    print('TS job failed, restarting...')
    subprocess.run(f"scancel {job_id_TS}", shell=True)
    time.sleep(60)
    subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess", shell=True)
    if os.path.exists('TS_opt.xyz'):
        subprocess.run(["mv", "TS_opt.xyz", "ts_guess.xyz"], shell=True)

    return TS_opt(charge, mult, trial, upper_limit, solvent)

def IRC_job(charge=0, mult=1, trial=0, upper_limit=5, solvent="", maxiter=70):
    trial += 1
    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent else ""
    job_step = 'IRC'

    if trial < 2:
        subprocess.run("cp TS/TS_opt.xyz IRC/", shell=True)
        os.chdir('IRC')

        if not freq_job(struc_name="TS_opt.xyz", charge=charge, mult=mult, trial=0, upper_limit=upper_limit, solvent=solvent, ts=True):
            print('Freq job failed or did not have negative frequency. Aborting IRC calculation')
            return False

    IRC_input = f"!r2scan-3c IRC tightscf {solvent_formatted}\n%irc\n  maxiter {maxiter} \nInitHess read \nHess_Filename \"freq.hess\"\nend \n%pal nprocs {SLURM_PARAMS_OPT['nprocs']} end\n%maxcore {SLURM_PARAMS_OPT['maxcore']}\n*xyzfile {charge} {mult} TS_opt.xyz\n"
    with open('IRC.inp', 'w') as f:
        f.write(IRC_input)

    print('Submitting IRC job to slurm')
    job_id_IRC = submit_job("IRC.inp", "IRC_slurm.out")
    status_IRC = check_job_status(job_id_IRC, step=job_step)

    print('Checking if optimisation completed successfully')
    if status_IRC == 'COMPLETED' and grep_output('HURRAY', 'IRC.out'):
        print('IRC optimisation completed successfully')
        os.chdir('..')
        return True

    print('IRC job failed, restarting...')
    subprocess.run(f"scancel {job_id_IRC}", shell=True)

    if grep_output('ORCA TERMINATED NORMALLY', 'IRC.out'):
        print("ORCA terminated normally")
        print("This most likely means that the IRC failed to converge, will be refined in the future")
        if maxiter < 100:
            maxiter += 30
        else:
            print("Maxiter reached 100, aborting")
            return False

    print('Waiting for 60 seconds before restarting')
    time.sleep(60)
    subprocess.run("rm -rf *.gbw pmix* *densities* IRC.inp slurm* ", shell=True)

    return IRC_job(charge, mult, trial, upper_limit, solvent, maxiter)

def main(charge=0, mult=1, solvent="", Nimages=8, xtb=True):
    steps = ["OPT", "NEB", "TS", "IRC"]
    make_folders(steps)

    if not optimise_reactants(charge, mult, trial=0, upper_limit=5, solvent=solvent):
        print("OPT step failed")
        sys.exit(1)
    print("OPT step completed successfully")

    if not NEB_CI(charge, mult, trial=0, Nimages=Nimages, upper_limit=5, xtb=xtb, solvent=solvent):
        print("NEB step failed")
        sys.exit(1)
    print("NEB step completed successfully")

    if not TS_opt(charge, mult, trial=0, upper_limit=5, solvent=solvent):
        print("TS step failed")
        sys.exit(1)
    print("TS step completed successfully")

    if not IRC_job(charge, mult, trial=0, upper_limit=5, solvent=solvent):
        print("IRC step failed")
        sys.exit(1)
    print('All steps completed successfully.\nCheck IRC trajectory for the reaction path.\nExiting')
    sys.exit(0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NEB optimization pipeline.")
    parser.add_argument("--charge", type=int, default=0, help="Charge of the system")
    parser.add_argument("--mult", type=int, default=1, help="Multiplicity of the system")
    parser.add_argument("--solvent", type=str, default="", help="Solvent model to use")
    parser.add_argument("--Nimages", type=int, default=8, help="Number of images for NEB")
    parser.add_argument("--xtb", type=str2bool, nargs='?',const=True, default=False, help="Use xtb method")

    args = parser.parse_args()


    print("Starting Pipeline with parameters:")
    print(f"Charge: {args.charge}")
    print(f"Multiplicity: {args.mult}")
    print(f"Solvent: {args.solvent}")
    print(f"Nimages: {args.Nimages}")
    print(f"XTB: {args.xtb}")

    

    main(charge=args.charge, mult=args.mult, solvent=args.solvent, Nimages=args.Nimages, xtb=args.xtb)