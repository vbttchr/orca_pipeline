
import glob

import re
import os
import subprocess
import time
import shutil
import argparse
import sys
import concurrent.futures

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


 # Added helper functions for handling file operations without shell=True
def remove_temp_files(*patterns):
    """Remove files/directories matching the given glob patterns."""
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
            except IsADirectoryError:
                shutil.rmtree(path, ignore_errors=True)
            except FileNotFoundError:
                pass

def copy_files(src_pattern, dest_folder):
    """Copy files matching src_pattern to dest_folder."""
    for src in glob.glob(src_pattern):
        shutil.copy(src, dest_folder)

    


def run_subprocess(command, shell=False, capture_output=True, check=True, exit_on_error=True, timeout=60):
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
        print(f"Command '{' '.join(command)}' timed out after {timeout} seconds.")
        return False
    except subprocess.CalledProcessError as e:
        if exit_on_error:
            print(f"Command '{' '.join(command)}' failed with error: {e.stderr}")
            sys.exit(1)
        else:
            return False
   

def check_job_status(job_id, interval=45, step="", timeout=60):
    
    counter = 0
    while True:
        time.sleep(interval)
        print(f'Checking job {job_id} status {step} ')
        squeue = run_subprocess(['squeue', '-j', job_id, '-h'], shell=False, exit_on_error=False, timeout=timeout)
        if squeue and squeue.stdout.strip():
            if counter % 10 == 0:
                print(f'Job {job_id} is still in the queue.')
            counter += 1
        else:
            sacct = run_subprocess(['sacct', '-j', job_id, '--format=State', '--noheader'], shell=False, exit_on_error=False, timeout=timeout)
            if sacct and sacct.stdout.strip():
                statuses = sacct.stdout.strip().split('\n')
                latest_status = statuses[-1].strip()
                print(f'Latest status for job {job_id}: {latest_status}')
                if latest_status in CHECK_STATES:
                    return latest_status
                else:
                    print(f'Job {job_id} ended with status: {latest_status}')
                    return latest_status
            else:
                if sacct and sacct.stderr.strip():
                    print(f"Error from sacct: {sacct.stderr.strip()}")
                    print(f'Job {job_id} status could not be determined.')
                    return 'UNKNOWN'
                



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
    
    
    # Replace the shell command string with a list of arguments
    command = ["grep", pattern, file]
    result = run_subprocess(command, shell=False, capture_output=True, check=False, exit_on_error=False)
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
        
        # Replace “cp *.xyz OPT” shell=True with copy_files
        copy_files("*.xyz", "OPT")
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

    

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        future_educt = executor.submit(check_job_status, job_ids[0], step='educt')
        future_product = executor.submit(check_job_status, job_ids[1], step='product')

        status_educt = future_educt.result()
        status_product = future_product.result()

    print(f'Educt Status: {status_educt}')
    print(f'Product Status: {status_product}')


    

    if status_educt == 'COMPLETED' and status_product == 'COMPLETED':
        educt_success = 'HURRAY' in grep_output('HURRAY', 'educt_opt.out')
        product_success = 'HURRAY' in grep_output('HURRAY', 'product_opt.out')
        if educt_success and product_success:
            os.chdir('..')
            return True


    print('One of the optimisation jobs failed, restarting both')
    for job_id in job_ids:
        run_subprocess(['scancel', job_id], exit_on_error=False)
    time.sleep(RETRY_DELAY)

    remove_temp_files("*.gbw", "pmix*", "*densities*", "freq.inp", "slurm*", "educt_opt.inp", "product_opt.inp")

    for xyz in ['educt_opt.xyz', 'product_opt.xyz']:
        if os.path.exists(xyz):
            os.rename(xyz, xyz.replace('_opt', ''))    

    return optimise_reactants(charge, mult, trial, upper_limit, solvent, xtb)

def freq_job(struc_name="coord.xyz", charge=0, mult=1, trial=0, upper_limit=5, solvent="", xtb=False, ts=False):
    // ...existing code...
    print('Frequency job failed, restarting...')
    run_subprocess(['scancel', job_id_freq], exit_on_error=False)
    {
    # Replace “rm -rf *.gbw pmix* ...” shell=True with remove_temp_files
    remove_temp_files("*.gbw", "pmix*", "*densities*", "freq.inp", "slurm*")
    }
    return freq_job(struc_name, charge, mult, trial, upper_limit, solvent, ts)

def NEB_TS(charge=0, mult=1, trial=0, Nimages=16, upper_limit=5, xtb=True, fast=True, solvent="", switch=False):
    // ...existing code...
    if trial < 2:
        make_folder("NEB")
        {
        # Replace shell=True with shell=False or helper function
        run_subprocess(["cp", "OPT/educt_opt.xyz", "NEB/educt.xyz"], shell=False)
        run_subprocess(["cp", "OPT/product_opt.xyz", "NEB/product.xyz"], shell=False)
        os.chdir('NEB')
        }
    // ...existing code...
    if status == 'COMPLETED' and grep_output('HURRAY', f'{neb_input_name.rsplit(".", 1)[0]}.out'):
        // ...existing code...
        {
        run_subprocess(["cp", f"{neb_input_name.rsplit('.', 1)[0]}_NEB-TS_converged.xyz", "neb_completed.xyz"], shell=False)
        }
        // ...existing code...
    // ...existing code...
    else:
        // ...existing code...
        if grep_output("TS OPTIMIZATION", f'{neb_input_name.rsplit(".", 1)[0]}.out'):
            // ...existing code...
            run_subprocess(["scancel", job_id], exit_on_error=False)
            // ...existing code...
            {
            run_subprocess(["cp", struc_name, "neb_completed.xyz"], shell=False)
            }
            // ...existing code...
        return NEB_TS(charge, mult, trial, Nimages, upper_limit, xtb, fast, solvent, switch)

def handle_failed_imagfreq(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent):
    // ...existing code...
    if not xtb and fast:
        // ...existing code...
        {
        run_subprocess(["mv", "neb-fast-TS_NEB-HEI_converged.xyz", "guess.xyz"], shell=False)
        }
        {
        remove_temp_files("pmix*", "*densities*", "freq.inp", "slurm*", "neb*im*", "*neb*.inp")
        }
        return NEB_TS(charge, mult, Nimages=8, trial=1, upper_limit=MAX_TRIALS+1, xtb=False, fast=False, solvent=solvent)
    // ...existing code...
    elif xtb and fast:
        // ...existing code...
        {
        run_subprocess(["mv", "neb-fast-TS_NEB-HEI_converged.xyz", "guess.xyz"], shell=False)
        }
        {
        remove_temp_files("pmix*", "*densities*", "freq.inp", "slurm*", "neb*im*", "*neb*.inp")
        }
        return NEB_TS(charge, mult, Nimages=24, trial=1, upper_limit=MAX_TRIALS+1, xtb=True, fast=False, solvent=solvent)
    return False

def handle_unconverged_neb(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent, job_id):
    // ...existing code...
    run_subprocess(['scancel', job_id], exit_on_error=False)
    time.sleep(60)
    {
    remove_temp_files("*.gbw", "pmix*", "*densities*", "freq.inp", "slurm*", "neb*im*", "*neb*.inp")
    }
    // ...existing code...
    return NEB_TS(charge, mult, Nimages=8, trial=1, upper_limit=MAX_TRIALS+1, xtb=False, fast=False, solvent=solvent, switch=False)

// ...existing code...

def NEB_CI(charge=0, mult=1, trial=0, Nimages=8, upper_limit=5, xtb=True, solvent=""):
    // ...existing code...
    if trial < 2:
        make_folder(job_step)
        {
        run_subprocess(["cp", "OPT/educt_opt.xyz", "NEB/educt.xyz"], shell=False)
        run_subprocess(["cp", "OPT/product_opt.xyz", "NEB/product.xyz"], shell=False)
        os.chdir('NEB')
        }
    // ...existing code...
    print('NEB-CI job failed, restarting...')
    run_subprocess(['scancel', job_id])
    time.sleep(RETRY_DELAY)
    {
    remove_temp_files("*.gbw", "pmix*", "*densities*", "freq.inp", "slurm*", "neb*im*", "*neb*.inp")
    }
    return NEB_CI(charge, mult, trial, images, upper_limit, xtb, solvent)

// ...existing code...

def TS_opt(charge=0, mult=1, trial=0, upper_limit=5, solvent=""):
    // ...existing code...
    if trial < 2:
        make_folder(job_step)
        {
        run_subprocess(["cp", "NEB/neb_completed.xyz", "TS/ts_guess.xyz"], shell=False)
        os.chdir('TS')
        }
    // ...existing code...
    run_subprocess(['scancel', job_id_TS], exit_on_error=False)
    time.sleep(RETRY_DELAY)
    {
    remove_temp_files("*.gbw", "pmix*", "*densities*", "freq.inp", "slurm*", "*.hess")
    }
    // ...existing code...
    return TS_opt(charge, mult, trial, upper_limit, solvent)

// ...existing code...

def IRC_job(charge=0, mult=1, trial=0, upper_limit=5, solvent="", maxiter=70):
    // ...existing code...
    if trial < 2:
        make_folder(job_step)
        {
        run_subprocess(["cp", "TS/TS_opt.xyz", "IRC/"], shell=False)
        os.chdir('IRC')
        }
        if not freq_job(struc_name="TS_opt.xyz", charge=charge, mult=mult, trial=0, upper_limit=upper_limit, solvent=solvent, ts=True):
            // ...existing code...
    // ...existing code...
    run_subprocess(['scancel', job_id_IRC], exit_on_error=False)
    time.sleep(RETRY_DELAY)
    {
    remove_temp_files("*.gbw", "pmix*", "*densities*", "IRC.inp", "slurm*")
    }
    return IRC_job(charge, mult, trial, upper_limit, solvent, maxiter)

// ...existing code...

def SP(charge=0, mult=1, trial=0, upper_limit=5, solvent="",method="r2scanh",basis="def2-QZVPP"):
    // ...existing code...
    if trial < 2:
        make_folder(job_step)
        {
        run_subprocess(["cp", "TS/TS_opt.xyz", "SP/"], shell=False)
        os.chdir('SP')
        }
    // ...existing code...
    print(f'Performing single point calculation with {method} {basis} {solvent_formatted}')
    // ...existing code...
    run_subprocess(['scancel', job_id], exit_on_error=False)
    time.sleep(RETRY_DELAY)
    {
    remove_temp_files("*.gbw", "pmix*", "*densities*", "SP.inp", "slurm*")
    }
    return SP(charge, mult, trial, upper_limit, solvent, method=method, basis=basis)

// ...existing code...