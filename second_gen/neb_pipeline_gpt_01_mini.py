import argparse
import concurrent.futures
import os
import re
import shutil
import subprocess
import sys
import time
import logging
from typing import Tuple, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Constants
MAX_TRIALS = 5

SLURM_PARAMS_HIGH_MEM = {
    'nprocs': 16,
    'maxcore': 12000
}

SLURM_PARAMS_LOW_MEM = {
    'nprocs': 24,
    'maxcore': 2524
}

SUBMIT_COMMAND = ["ssubo", "-m", "No"]
CHECK_STATES = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']
RETRY_DELAY = 60
FREQ_THRESHOLD = -50

CONFIG_FILE = "settings_neb_pipeline.txt"

# Utility Functions
def run_subprocess(command: List[str], shell: bool = False, capture_output: bool = True,
                  check: bool = True, exit_on_error: bool = True, timeout: int = 60) -> Optional[subprocess.CompletedProcess]:
    """
    Runs a subprocess command with the given parameters.
    """
    try:
        result = subprocess.run(
            command,
            shell=shell,
            capture_output=capture_output,
            text=True,
            check=check,
            timeout=timeout
        )
        return result
    except subprocess.TimeoutExpired:
        logging.error(f"Command '{' '.join(command)}' timed out after {timeout} seconds.")
        return None
    except subprocess.CalledProcessError as e:
        if exit_on_error:
            logging.error(f"Command '{' '.join(command)}' failed with error: {e.stderr}")
            sys.exit(1)
        else:
            return None

def check_job_status(job_id: str, interval: int = 45, step: str = "", timeout: int = 60) -> str:
    """
    Checks the status of a Slurm job until it completes.
    """
    counter = 0
    while True:
        time.sleep(interval)
        logging.info(f'Checking job {job_id} status [{step}]')
        squeue = run_subprocess(['squeue', '-j', job_id, '-h'], timeout=timeout)
        if squeue and squeue.stdout.strip():
            if counter % 10 == 0:
                logging.info(f'Job {job_id} is still in the queue.')
            counter += 1
        else:
            sacct = run_subprocess(['sacct', '-j', job_id, '--format=State', '--noheader'], timeout=timeout)
            if sacct and sacct.stdout.strip():
                statuses = sacct.stdout.strip().split('\n')
                latest_status = statuses[-1].strip()
                logging.info(f'Latest status for job {job_id}: {latest_status}')
                if latest_status in CHECK_STATES:
                    return latest_status
                else:
                    logging.warning(f'Job {job_id} ended with status: {latest_status}')
                    return latest_status
            else:
                if sacct and sacct.stderr.strip():
                    logging.error(f"Error from sacct: {sacct.stderr.strip()}")
                logging.error(f'Job {job_id} status could not be determined.')
                return 'UNKNOWN'

def make_folder(dir_name: str) -> None:
    """
    Creates a directory after removing it if it already exists.
    """
    path = os.path.join(os.getcwd(), dir_name)
    if os.path.exists(path):
        if os.path.isdir(path):
            logging.info(f"Removing existing folder {path}")
            shutil.rmtree(path)
        else:
            logging.info(f"Removing existing file {path}")
            os.remove(path)
    os.makedirs(path)
    logging.info(f"Created folder {path}")

def submit_job(input_file: str, output_file: str, walltime: str = "24") -> str:
    """
    Submits a job to Slurm and returns the job ID.
    """
    command = SUBMIT_COMMAND + ["-w", walltime, "-o", output_file, input_file]
    result = run_subprocess(command)
    if result:
        for line in result.stdout.splitlines():
            if 'Submitted batch job' in line:
                job_id = line.split()[-1]
                logging.info(f"Job submitted with ID: {job_id}")
                return job_id
    logging.error(f"Failed to submit job {input_file}")
    sys.exit(1)

def grep_output(pattern: str, file: str) -> str:
    """
    Greps for a pattern in a file.
    """
    result = run_subprocess(f"grep '{pattern}' {file}", shell=True, check=False)
    return result.stdout.strip() if result else ""

# Pipeline Functions
def optimise_reactants(charge: int = 0, mult: int = 1, trial: int = 0, upper_limit: int = 5,
                      solvent: str = "", xtb: bool = True) -> bool:
    trial += 1
    logging.info(f"Trial {trial} for reactant optimisation")
    if trial > upper_limit:
        logging.error('Too many trials aborting reactant optimisation.')
        return False

    solvent_formatted = f"ALPB({solvent})" if solvent and xtb else f"CPCM({solvent})" if solvent else ""
    method = "XTB2" if xtb else "r2scan-3c"

    if trial < 2:
        make_folder('OPT')
        run_subprocess("cp *.xyz OPT", shell=True)
        os.chdir('OPT')

    logging.info('Starting reactant optimisation')

    job_inputs = {
        'educt_opt.inp': (
            f"!{method} {solvent_formatted} opt\n"
            f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
            f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
            f"*xyzfile {charge} {mult} educt.xyz\n"
        ),
        'product_opt.inp': (
            f"!{method} {solvent_formatted} opt\n"
            f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
            f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
            f"*xyzfile {charge} {mult} product.xyz\n"
        )
    }

    for filename, content in job_inputs.items():
        with open(filename, 'w') as f:
            f.write(content)

    logging.info('Submitting educt and product jobs to Slurm')
    job_ids = [submit_job(inp, f"{inp.split('.')[0]}_slurm.out") for inp in job_inputs.keys()]

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_educt = executor.submit(check_job_status, job_ids[0], step='educt')
        future_product = executor.submit(check_job_status, job_ids[1], step='product')

        status_educt = future_educt.result()
        status_product = future_product.result()

    logging.info(f'Educt Status: {status_educt}')
    logging.info(f'Product Status: {status_product}')

    if status_educt == 'COMPLETED' and status_product == 'COMPLETED':
        educt_success = 'HURRAY' in grep_output('HURRAY', 'educt_opt.out')
        product_success = 'HURRAY' in grep_output('HURRAY', 'product_opt.out')
        if educt_success and product_success:
            os.chdir('..')
            return True

    logging.warning('One of the optimisation jobs failed, restarting both')
    for job_id in job_ids:
        run_subprocess(['scancel', job_id], check=False)
    time.sleep(RETRY_DELAY)
    run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm* educt_opt.inp product_opt.inp", shell=True, check=False)

    for xyz in ['educt_opt.xyz', 'product_opt.xyz']:
        if os.path.exists(xyz):
            os.rename(xyz, xyz.replace('_opt', ''))

    return optimise_reactants(charge, mult, trial, upper_limit, solvent, xtb)

def freq_job(struc_name: str = "coord.xyz", charge: int = 0, mult: int = 1, trial: int = 0,
             upper_limit: int = 5, solvent: str = "", xtb: bool = False, ts: bool = False) -> bool:
    trial += 1
    logging.info(f"Trial {trial} for frequency job")

    if trial > upper_limit:
        logging.error('Too many trials aborting frequency job.')
        return False

    method = "r2scan-3c freq tightscf"
    solvent_formatted = f"CPCM({solvent})" if solvent else ""

    freq_input = (
        f"! {method} {solvent_formatted}\n"
        f"%pal nprocs {SLURM_PARAMS_HIGH_MEM['nprocs']} end\n"
        f"%maxcore {SLURM_PARAMS_HIGH_MEM['maxcore']}\n"
        f"*xyzfile {charge} {mult} {struc_name}\n"
    )
    with open('freq.inp', 'w') as f:
        f.write(freq_input)

    logging.info('Submitting frequency job to Slurm')
    job_id_freq = submit_job("freq.inp", "freq_slurm.out")
    status_freq = check_job_status(job_id_freq, step='Freq')

    if status_freq == 'COMPLETED':
        if grep_output('VIBRATIONAL FREQUENCIES', 'freq.out'):
            logging.info('Frequency calculation completed successfully')
            if ts:
                output = grep_output('**imaginary mode***', 'freq.out')
                match = re.search(r'(-?\d+\.\d+)\s*cm\*\*-1', output)
                if match:
                    imag_freq = float(match.group(1))
                    if imag_freq < FREQ_THRESHOLD:
                        logging.info('Negative frequency found, proceeding with TS optimisation')
                        return True
                    else:
                        logging.warning('Negative frequency above threshold, aborting TS optimisation')
                        return False
                logging.warning('No negative frequency found, aborting TS optimisation')
                return False
            return True

    logging.warning('Frequency job failed, restarting...')
    run_subprocess(['scancel', job_id_freq], check=False)
    run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm*", shell=True, check=False)
    return freq_job(struc_name, charge, mult, trial, upper_limit, solvent, xtb, ts)

def NEB_TS(charge: int = 0, mult: int = 1, trial: int = 0, Nimages: int = 16,
           upper_limit: int = 5, xtb: bool = True, fast: bool = True,
           solvent: str = "", switch: bool = False) -> bool:
    if switch:
        trial = 0  # Reset trial counter if switching from XTB to DFT
        os.chdir('..')
        logging.info("Switching to DFT run and optimizing reactants")
        if not optimise_reactants(charge, mult, trial=0, upper_limit=MAX_TRIALS, solvent=solvent, xtb=False):
            logging.error("Reactant optimisation failed, aborting NEB-TS.")
            return False
        switch = False

    job_step = 'NEB-TS'
    trial += 1
    logging.info(f"Trial {trial} for NEB-TS")

    if trial > upper_limit:
        logging.error('Too many trials aborting NEB-TS.')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent and not xtb else f"ALPB({solvent})" if solvent else ""
    method = ("FAST-NEB-TS XTB2 tightscf" if fast and xtb else
              "FAST-NEB-TS r2scan-3c tightscf" if fast and not xtb else
              "NEB-TS XTB2 tightscf" if xtb and not fast else
              "NEB-TS r2scan-3c tightscf")

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
        geom_block = f"%geom\n Calc_Hess true\n Recalc_Hess 1\n MaxIter={maxiter} end \n"

    neb_input = (
        f"! {method} {solvent_formatted}\n"
        f"{geom_block}"
        f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
        f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
        f"%neb \nProduct \"product.xyz\" \nNImages {Nimages}  \n"
        f"{guess_block} end\n"
        f"*xyzfile {charge} {mult} educt.xyz\n"
    )
    neb_input_name = "neb-fast-TS.inp" if fast else "neb-TS.inp"
    with open(neb_input_name, 'w') as f:
        f.write(neb_input)

    logging.info(f'Submitting {neb_input_name} job to Slurm')
    job_id = submit_job(neb_input_name, f"{job_step}_slurm.out", walltime="48")
    status = check_job_status(job_id, step="NEB-TS")

    out_file = f'{neb_input_name.rsplit(".", 1)[0]}.out'
    if status == 'COMPLETED' and grep_output('HURRAY', out_file):
        logging.info('NEB-TS completed successfully')

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
            logging.info("Negative mode below threshold found, continuing")
            time.sleep(60)
            run_subprocess(
                f"cp {neb_input_name.rsplit('.', 1)[0]}_NEB-TS_converged.xyz neb_completed.xyz", shell=True)
            os.chdir('..')
            return True
        else:
            if not handle_failed_imagfreq(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent):
                os.chdir('..')
                return False
            else:
                logging.info("Negative mode found in guess. Continuing with TS optimisation.")
                time.sleep(60)
                run_subprocess("cp neb-TS_NEB-CI_converged.xyz neb_completed.xyz", shell=True)
                os.chdir('..')
                return True
    elif grep_output('ORCA TERMINATED NORMALLY', out_file):
        if not handle_unconverged_neb(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent, job_id):
            os.chdir('..')
            return False
        else:
            logging.info("R2SCAN-3c NEB-TS job did not converge, but freq job found negative frequency above guess.")
            logging.info("Continuing with TS optimisation.")
            time.sleep(60)
            run_subprocess("cp neb-TS_NEB-CI_converged.xyz neb_completed.xyz", shell=True)
            os.chdir('..')
            return True
    else:
        if grep_output("TS OPTIMIZATION", out_file):
            logging.warning("NEB-TS started to optimized a TS guess but aborted during the run")
            logging.info("Checking whether last struct has significant negative frequency")
            run_subprocess(["scancel", job_id], check=False)
            logging.info("Waiting for a minute to retrieve files from the node")
            time.sleep(60)
            struc_name = f'{neb_input_name.rsplit(".", 1)[0]}.xyz'

            imag_freq = freq_job(
                struc_name=struc_name,
                charge=charge,
                mult=mult,
                trial=0,
                upper_limit=upper_limit,
                solvent=solvent,
                xtb=xtb,
                ts=True
            )
            if imag_freq:
                logging.info("Negative mode below threshold found, continuing with TS optimisation")
                time.sleep(60)
                run_subprocess(["cp", struc_name, "neb_completed.xyz"], shell=True)
                os.chdir('..')
                return True

        logging.error(f"NEB-TS {neb_input_name} failed, restarting if there are trials left")
        return NEB_TS(charge, mult, trial, Nimages, upper_limit, xtb, fast, solvent, switch=True)

def handle_failed_imagfreq(charge: int, mult: int, Nimages: int, trial: int,
                           upper_limit: int, xtb: bool, fast: bool, solvent: str) -> bool:
    logging.warning("No significant negative frequency found")
    if not xtb and fast:
        logging.info("FAST-NEB-TS with r2scan-3c did not find TS. Retrying with regular NEB-TS.")
        logging.info("Using FAST-NEB-TS guess as TS guess for NEB-TS")
        if os.path.exists("neb-fast-TS_NEB-HEI_converged.xyz"):
            run_subprocess("mv neb-fast-TS_NEB-HEI_converged.xyz guess.xyz", shell=True)
        else:
            logging.warning("Could not find NEB-HEI file, starting new NEB without guess")
        run_subprocess("rm pmix* *densities* freq.inp slurm* neb*im* *neb*.inp", shell=True, check=False)
        return NEB_TS(charge, mult, Nimages=8, trial=1, upper_limit=MAX_TRIALS + 1, xtb=False, fast=False, solvent=solvent)
    elif not xtb and not fast:
        logging.info("NEB-TS r2scan-3c did not find TS. Checking guess mode.")
        return freq_job(
            struc_name='neb-TS_NEB-CI_converged.xyz',
            charge=charge,
            mult=mult,
            trial=0,
            upper_limit=MAX_TRIALS,
            solvent=solvent,
            xtb=xtb,
            ts=True
        )
    elif xtb and not fast:
        logging.info("NEB-TS XTB2 did not find TS. Retrying with FAST-NEB-TS with r2scan-3c.")
        logging.info("Using NEB-TS guess as TS guess for FAST-NEB-TS")
        if os.path.exists("neb-TS_NEB-CI_converged.xyz"):
            run_subprocess("mv neb-TS_NEB-CI_converged.xyz guess.xyz", shell=True)
        else:
            logging.warning("Could not find NEB-CI file, NEB will be started without guess")
        run_subprocess("rm pmix* *densities* freq.inp slurm* neb*im* *neb*.inp", shell=True, check=False)
        return NEB_TS(charge, mult, Nimages=12, trial=0, upper_limit=MAX_TRIALS, xtb=False, fast=True, solvent=solvent, switch=True)
    elif xtb and fast:
        logging.info("Retrying with regular NEB-TS and XTB.")
        logging.info("Using FAST-NEB-TS guess as TS guess for NEB-TS")
        if os.path.exists("neb-fast-TS_NEB-HEI_converged.xyz"):
            run_subprocess("mv neb-fast-TS_NEB-HEI_converged.xyz guess.xyz", shell=True)
        else:
            logging.warning("NEB-HEI file is not present, starting new NEB without guess")
        run_subprocess("rm pmix* *densities* freq.inp slurm* neb*im* *neb*.inp", shell=True, check=False)
        return NEB_TS(charge, mult, Nimages=24, trial=1, upper_limit=MAX_TRIALS + 1, xtb=True, fast=False, solvent=solvent)
    return False

def handle_unconverged_neb(charge: int, mult: int, Nimages: int, trial: int, upper_limit: int,
                           xtb: bool, fast: bool, solvent: str, job_id: str) -> bool:
    logging.error('NEB-TS job did not converge')
    run_subprocess(['scancel', job_id], check=False)
    time.sleep(60)

    run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm* neb*im* *neb*.inp", shell=True, check=False)
    if fast and not xtb:
        logging.info("Restarting as R2SCAN-3c NEB-TS run using FAST-NEB-TS guess")
        if os.path.exists("neb-fast-TS_NEB-HEI_converged.xyz"):
            run_subprocess("cp neb-fast-TS_NEB-HEI_converged.xyz guess.xyz", shell=True)
        else:
            logging.warning("Could not find NEB-HEI file, NEB will be started without guess")
        return NEB_TS(charge, mult, Nimages=8, trial=1, upper_limit=MAX_TRIALS + 1,
                     xtb=False, fast=False, solvent=solvent, switch=False)
    if fast and xtb:
        logging.info("Restarting as XTB2 NEB-TS")
        return NEB_TS(charge, mult, Nimages=24, trial=1, upper_limit=MAX_TRIALS + 1,
                     xtb=True, fast=False, solvent=solvent, switch=False)
    if not fast and xtb:
        logging.info("Restarting as R2SCAN-3c FAST-NEB-TS")
        return NEB_TS(charge, mult, Nimages=12, trial=0, upper_limit=MAX_TRIALS,
                     xtb=False, fast=True, solvent=solvent, switch=True)
    if not fast and not xtb:
        logging.info("R2SCAN-3c NEB-TS did not converge.")
        logging.info("This might be due to challenging PES near the TS. Checking guess mode.")
        imag_freq = freq_job(
            struc_name='neb-TS_NEB-CI_converged.xyz',
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

def NEB_CI(charge: int = 0, mult: int = 1, trial: int = 0, Nimages: int = 8,
           upper_limit: int = 5, xtb: bool = True, solvent: str = "") -> bool:
    trial += 1
    logging.info(f"Trial {trial} for NEB-CI")

    if trial > upper_limit:
        logging.error('Too many trials aborting NEB-CI.')
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

    neb_input = (
        f"!NEB-CI {solvent_formatted} {method}\n"
        f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
        f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
        f"%neb \nProduct \"product.xyz\" \nNImages {images} {'Free_end true' if xtb else ''} \nend\n"
        f"*xyzfile {charge} {mult} educt.xyz\n"
    )
    with open('neb-CI.inp', 'w') as f:
        f.write(neb_input)

    logging.info('Submitting NEB-CI job to Slurm')
    job_id = submit_job("neb-CI.inp", "neb-ci_slurm.out", walltime="48")
    status = check_job_status(job_id, step=job_step)

    out_file = 'neb-CI.out'
    if status == 'COMPLETED' and 'H U R R A Y' in grep_output('H U R R A Y', out_file):
        logging.info('NEB-CI completed successfully')
        os.chdir('..')
        return True

    logging.warning('NEB-CI job failed, restarting...')
    run_subprocess(['scancel', job_id], check=False)
    time.sleep(RETRY_DELAY)
    if grep_output('ORCA TERMINATED NORMALLY', 'neb-ci.out'):
        images = images * 2 if xtb else images + 2
    run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm* neb*im* *neb*.inp", shell=True, check=False)
    return NEB_CI(charge, mult, trial, images, upper_limit, xtb, solvent)

def TS_opt(charge: int = 0, mult: int = 1, trial: int = 0, upper_limit: int = 5,
           solvent: str = "") -> bool:
    trial += 1
    logging.info(f"Trial {trial} for TS optimisation")

    if trial > upper_limit:
        logging.error('Too many trials aborting TS optimisation.')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent else ""
    job_step = 'TS'

    if trial < 2:
        make_folder(job_step)
        run_subprocess("cp NEB/neb_completed.xyz TS/ts_guess.xyz", shell=True)
        os.chdir('TS')

    logging.info('Starting TS optimisation')

    if not freq_job(struc_name="ts_guess.xyz", charge=charge, mult=mult, trial=0,
                   upper_limit=upper_limit, solvent=solvent, xtb=False, ts=True):
        logging.error('Frequency job failed, aborting TS optimisation')
        return False

    TS_input = (
        f"!r2scan-3c OptTS tightscf {solvent_formatted}\n"
        f"%geom\ninhess read \ninhessname \"freq.hess\"\nCalc_Hess true\nrecalc_hess 15\nend\n"
        f"%pal nprocs {SLURM_PARAMS_HIGH_MEM['nprocs']} end\n"
        f"%maxcore {SLURM_PARAMS_HIGH_MEM['maxcore']}\n"
        f"*xyzfile {charge} {mult} ts_guess.xyz\n"
    )
    with open('TS_opt.inp', 'w') as f:
        f.write(TS_input)

    logging.info('Submitting TS optimisation job to Slurm')
    job_id_TS = submit_job("TS_opt.inp", "TS_opt_slurm.out", walltime="48")
    status_TS = check_job_status(job_id_TS, step=job_step)

    out_file = 'TS_opt.out'
    if status_TS == 'COMPLETED' and 'HURRAY' in grep_output('HURRAY', out_file):
        logging.info('TS optimisation completed successfully')
        os.chdir('..')
        return True

    logging.warning('TS optimisation job failed, restarting...')
    run_subprocess(['scancel', job_id_TS], check=False)
    time.sleep(RETRY_DELAY)
    run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess", shell=True, check=False)
    if os.path.exists('TS_opt.xyz'):
        os.rename('TS_opt.xyz', 'ts_guess.xyz')

    return TS_opt(charge, mult, trial, upper_limit, solvent)

def IRC_job(charge: int = 0, mult: int = 1, trial: int = 0, upper_limit: int = 5,
            solvent: str = "", maxiter: int = 70) -> bool:
    trial += 1
    logging.info(f"Trial {trial} for IRC")

    if trial > upper_limit:
        logging.error('Too many trials aborting IRC.')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent else ""
    job_step = 'IRC'

    if trial < 2:
        make_folder(job_step)
        run_subprocess("cp TS/TS_opt.xyz IRC/", shell=True)
        os.chdir('IRC')

        if not freq_job(struc_name="TS_opt.xyz", charge=charge, mult=mult, trial=0,
                       upper_limit=upper_limit, solvent=solvent, xtb=False, ts=True):
            logging.error('Frequency job failed or lacks negative frequency. Aborting IRC.')
            return False

    IRC_input = (
        f"!r2scan-3c IRC tightscf {solvent_formatted}\n"
        f"%irc\n  maxiter {maxiter} \nInitHess read \nHess_Filename \"freq.hess\"\nend \n"
        f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
        f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
        f"*xyzfile {charge} {mult} TS_opt.xyz\n"
    )
    with open('IRC.inp', 'w') as f:
        f.write(IRC_input)

    logging.info('Submitting IRC job to Slurm')
    job_id_IRC = submit_job("IRC.inp", "IRC_slurm.out")
    status_IRC = check_job_status(job_id_IRC, step=job_step)

    out_file = 'IRC.out'
    if status_IRC == 'COMPLETED' and 'HURRAY' in grep_output('HURRAY', out_file):
        logging.info('IRC completed successfully')
        os.chdir('..')
        return True

    logging.warning('IRC job failed, restarting...')
    run_subprocess(['scancel', job_id_IRC], check=False)
    if 'ORCA TERMINATED NORMALLY' in grep_output('ORCA TERMINATED NORMALLY', out_file):
        logging.info("ORCA terminated normally, possibly due to non-convergence.")
        if maxiter < 100:
            maxiter += 30
        else:
            logging.error("Max iterations reached, aborting IRC.")
            return False

    time.sleep(RETRY_DELAY)
    run_subprocess("rm -rf *.gbw pmix* *densities* IRC.inp slurm*", shell=True, check=False)
    return IRC_job(charge, mult, trial, upper_limit, solvent, maxiter)

def SP(charge: int = 0, mult: int = 1, trial: int = 0, upper_limit: int = 5,
       solvent: str = "", method: str = "r2scanh", basis: str = "def2-QZVPP") -> bool:
    trial += 1
    logging.info(f"Trial {trial} for single point calculation")

    if trial > upper_limit:
        logging.error('Too many trials aborting single point calculation.')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent else ""
    job_step = "SP"

    if trial < 2:
        make_folder(job_step)
        run_subprocess("cp TS/TS_opt.xyz SP/", shell=True)
        os.chdir('SP')

    SP_input = (
        f"!{method} {basis} {solvent_formatted} verytightscf defgrid3 d4\n"
        f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
        f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
        f"*xyzfile {charge} {mult} TS_opt.xyz\n"
    )
    with open('SP.inp', 'w') as f:
        f.write(SP_input)

    logging.info(f"Performing single point calculation with {method} {basis} {solvent_formatted}")
    logging.info(f'Submitting {job_step} job to Slurm')
    job_id = submit_job("SP.inp", f"{job_step}_slurm.out")
    status = check_job_status(job_id, step=job_step)

    out_file = f'{job_step}.out'
    if status == 'COMPLETED' and 'FINAL SINGLE POINT ENERGY' in grep_output('HURRAY', out_file):
        logging.info(f'{job_step} completed successfully')
        os.chdir('..')
        return True

    logging.warning(f'{job_step} job failed, restarting...')
    run_subprocess(['scancel', job_id], check=False)
    time.sleep(RETRY_DELAY)
    run_subprocess("rm -rf *.gbw pmix* *densities* SP.inp slurm*", shell=True, check=False)
    return SP(charge, mult, trial, upper_limit, solvent, method=method, basis=basis)

def FOD(charge: int = 0, mult: int = 1, coord: str = "coord.xyz") -> Optional[None]:
    """
    FOD analysis runs in the current directory as ORCA plot searches for GBW in the given path.
    Default parameters for FOD are used: method: tpssh
    """
    # Implementation can be added here if needed
    return None

def restart_pipeline() -> Tuple[str, int, int, str, int, bool]:
    logging.info("Restart detected, loading previous settings.")
    if not os.path.exists(CONFIG_FILE):
        logging.error("Settings file not found, aborting restart.")
        sys.exit(1)

    with open(CONFIG_FILE, 'r') as f:
        settings = dict(line.strip().split(':') for line in f if ':' in line)

    charge = int(settings.get('Charge', 0))
    mult = int(settings.get('Multiplicity', 1))
    solvent = settings.get('Solvent', '')
    Nimages = int(settings.get('Nimages', 16))
    steps = settings.get('Steps', "")
    xtb = settings.get('xtb', 'False') == 'True'
    step_to_restart = settings.get('step', 'Completed')

    return step_to_restart, charge, mult, solvent, Nimages, xtb

def pipeline(step: str, charge: int = 0, mult: int = 1, solvent: str = "", Nimages: int = 16) -> bool:
    steps_mapping: Dict[str, callable] = {
        "OPT_XTB": lambda: optimise_reactants(charge, mult, trial=0, upper_limit=MAX_TRIALS, solvent=solvent),
        "OPT_DFT": lambda: optimise_reactants(charge, mult, trial=0, upper_limit=MAX_TRIALS, solvent=solvent, xtb=False),

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
        logging.error("Invalid step")
        logging.info("Valid steps: OPT_{XTB/DFT}, NEB_CI_{XTB/DFT}, (FAST)-NEB_TS_{XTB/DFT}, TS, IRC")
        sys.exit(1)

    return step_function()

def usage() -> None:
    logging.info("Usage: neb_pipeline.py [options]")
    logging.info("Options:")
    logging.info("-c, --charge: Charge of the system")
    logging.info("-m, --mult: Multiplicity of the system")
    logging.info("-s, --solvent: Solvent model to use")
    logging.info("-i, --Nimages: Number of images for NEB")
    logging.info("--steps: Steps to run in the pipeline, comma separated")
    logging.info("--restart: Restart from previous calculations")

def main(charge: int = 0, mult: int = 1, solvent: str = "", Nimages: int = 32,
         restart: bool = False, steps: List[str] = ["OPT_XTB", "NEB-TS_XTB", "TS", "IRC"]) -> None:
    home_dir = os.getcwd()

    if restart:
        step_to_restart, charge, mult, solvent, Nimages, xtb = restart_pipeline()
        try:
            start_index = steps.index(step_to_restart)
            steps = steps[start_index:]
            logging.info(f"Restarting from step: {step_to_restart}")
        except ValueError:
            logging.error(f"Step to restart '{step_to_restart}' not found in the current steps.")
            sys.exit(1)

    for step in steps:
        logging.info(f"Running step: {step}")
        if not pipeline(step=step, charge=charge, mult=mult, solvent=solvent, Nimages=Nimages):
            logging.error(f"{step} failed. Saving state for restart.")
            os.chdir(home_dir)
            with open("FAILED", "w") as f:
                f.write("")

            with open(CONFIG_FILE, 'w') as f:
                f.write(f"step: {step}\n")
                f.write(f"Charge: {charge}\n")
                f.write(f"Multiplicity: {mult}\n")
                f.write(f"Solvent: {solvent}\n")
                f.write(f"Nimages: {Nimages}\n")
            sys.exit(1)
        logging.info(f"{step} completed successfully\n")

    logging.info('All pipeline steps completed successfully.')

    with open("COMPLETED", "w") as f:
        f.write("")

    with open(CONFIG_FILE, 'w') as f:
        f.write("step: Completed\n")
        f.write(f"Charge: {charge}\n")
        f.write(f"Multiplicity: {mult}\n")
        f.write(f"Solvent: {solvent}\n")
        f.write(f"Nimages: {Nimages}\n")
        f.write(f"Steps: {','.join(steps)}\n")
    sys.exit(0)

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_steps(steps_str: str) -> List[str]:
    return steps_str.split(',')

# Entry Point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NEB optimization pipeline.")
    parser.add_argument("--charge", "-c", type=int, default=0, help="Charge of the system")
    parser.add_argument("--mult", "-m", type=int, default=1, help="Multiplicity of the system")
    parser.add_argument("--solvent", "-s", type=str, default="", help="Solvent model to use")
    parser.add_argument("--Nimages", "-i", type=int, default=8, help="Number of images for NEB")
    parser.add_argument("--restart", type=str2bool, default=False, help="Restart from previous calculations")
    parser.add_argument("--steps", type=parse_steps, default=["OPT_XTB", "NEB-TS_XTB", "TS", "IRC"],
                        help="Steps to run in the pipeline, comma separated")

    args = parser.parse_args()

    settings = (
        f'Charge: {args.charge}\n'
        f'Multiplicity: {args.mult}\n'
        f'Solvent: {args.solvent}\n'
        f'Nimages: {args.Nimages}\n'
        f'Steps: {",".join(args.steps)}\n'
        f'Restart: {args.restart}'
    )
    logging.info("Starting Pipeline with parameters:")
    logging.info(settings)

    main(
        charge=args.charge,
        mult=args.mult,
        solvent=args.solvent,
        Nimages=args.Nimages,
        restart=args.restart,
        steps=args.steps
    )