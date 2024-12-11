import re
import os
import subprocess
import time
import shutil
import argparse
import sys
import time



### TODO  do the opt restart when switching from XTB to DFT in the NEB-TS step. Check paths

MAX_TRIALS = 5

SLURM_PARAMS_FREQ = {
    'nprocs': 24,
    'maxcore': 8024
}

SLURM_PARAMS_OPT = {
    'nprocs': 24,
    'maxcore': 2524
}

def check_job_status(job_id, interval=45, step=""):

    start_time = time.time()

    counter = 0
    while True:
        print(f'Checking job {job_id} status {step}')
        squeue = subprocess.run(['squeue', '-j', job_id, '-h'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if squeue.stdout.strip():
            if counter%10 == 0:
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
                end_time = time.time()
                print(f"Job {job_id} finished in {end_time - start_time} seconds")
                return latest_status
        time.sleep(interval)

def make_folder(dir_name):
    path = os.path.join(os.getcwd(), dir_name)
    if os.path.exists(path):
        if os.path.isdir(path):
            print(f"Removing existing folder {path}")
            shutil.rmtree(path)  # Remove the existing directory
        else:
            print(f"Removing existing file {path}")
            os.remove(path)  # Remove the existing file
    
    os.makedirs(path)
    print(f"Created folder {path}")
             

def submit_job(input_file, output_file, walltime="24"):
    try:
        result = subprocess.run(
            ["ssubo", "-m ","No","-w", walltime, "-o", output_file, input_file],
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

def optimise_reactants(charge=0, mult=1, trial=0, upper_limit=5, solvent="",xtb=True):
    
    trial += 1
    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    if solvent and xtb:

        solvent_formatted = f"ALPB({solvent})" 
    elif solvent:
        solvent_formatted = f"CPCM({solvent})"
    else:
        solvent_formatted = ""
    
    method = "XTB2" if xtb else "r2scan-3c"

    if trial < 2:
        make_folder('OPT')
        subprocess.run("cp *.xyz OPT", shell=True)
        os.chdir('OPT')

    print('Starting reactant optimisation\n\n')

    educt_opt_input = f"!{method} {solvent_formatted} opt\n%pal nprocs {SLURM_PARAMS_OPT['nprocs']} end\n%maxcore {SLURM_PARAMS_OPT['maxcore']}\n*xyzfile {charge} {mult} educt.xyz\n"
    product_opt_input = f"!{method} {solvent_formatted} opt\n%pal nprocs {SLURM_PARAMS_OPT['nprocs']} end\n%maxcore {SLURM_PARAMS_OPT['maxcore']}\n*xyzfile {charge} {mult} product.xyz\n"

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
    subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm*  educt_opt.inp product_opt.inp", shell=True)

    if os.path.exists('educt_opt.xyz'):
        subprocess.run(["mv", "educt_opt.xyz", "educt.xyz"])
    if os.path.exists('product_opt.xyz'):
        subprocess.run(["mv", "product_opt.xyz", "product.xyz"])

    return optimise_reactants(charge, mult, trial, upper_limit, solvent)

def freq_job(struc_name="coord.xyz", charge=0, mult=1, trial=0, upper_limit=5, solvent="", xtb=False, ts=False):
    trial += 1
    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    method = "XTB2 numfreq tightscf" if xtb else "r2scan-3c freq tightscf"
    solvent_formatted = f"CPCM({solvent})" if solvent and not xtb else f"ALPB({solvent})" if solvent else ""
   

    freq_inp = f"! {method} {solvent_formatted}  \n%pal nprocs {SLURM_PARAMS_FREQ['nprocs']} end\n%maxcore {SLURM_PARAMS_FREQ['maxcore']}\n*xyzfile {charge} {mult} {struc_name}\n"
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
                output= grep_output('**imaginary mode***', 'freq.out')
                match = re.search(r'(-?\d+\.\d+)\s*cm\*\*-1', output)
                if match:
                    imag_freq = float(output.gropu(1))

                    if imag_freq < -50:


                        print('Negative frequency found, continuing with TS optimisation')
                        return True
                    else: 
                        print('Negative frequency above -50 cm**-1, aborting TS optimisation')
                        return False
                else:
                    print('No negative frequency found, aborting TS optimisation')
                    return False
            return True
    print('Freq job failed, restarting...')
    subprocess.run(f"scancel {job_id_freq}", shell=True)
    subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm* ", shell=True)
    return freq_job(struc_name, charge, mult, trial, upper_limit, solvent, ts)





def NEB_TS(charge=0, mult=1,  trial=0, Nimages=16, upper_limit=5, xtb=True,fast=True,solvent="",switch=False):

    """
    Performs NEB-TS runs with restarts if euler fails. 
    IF XTB is chosen and the number of images is greater than 32 (means that the NEB-TS run has twice not achieved convergence ), the run will restarted as a DFT run with 2 more trials
    """
    if switch:
        os.chdir('..')
        print("Switching to DFT run")
        print("Need to optimize reactants again")
        success = optimise_reactants(charge, mult, trial=0, upper_limit=MAX_TRIALS, solvent=solvent,xtb=False)
        if not success:
            print("Optimisation of reactants failed, aborting")
            return False
        switch=False
    

    job_step = "NEB"
    
    trial += 1


    images=Nimages
    if trial > upper_limit:
        print('Too many trials aborting')
        return False
    solvent_formatted = f"CPCM({solvent})" if solvent and not xtb else f"ALPB({solvent})" if solvent else ""
    method=""
    if fast and xtb:
        method = "FAST-NEB-TS XTB2 tightscf"
    elif fast and not xtb:
        method = "FAST-NEB-TS r2scan-3c tightscf"
    elif xtb:
        method = "NEB-TS XTB2 tightscf"
    else:
        method = "NEB-TS r2scan-3c tightscf"
    
    
    if trial < 2:
        make_folder(job_step)
        subprocess.run("cp OPT/educt_opt.xyz NEB/educt.xyz", shell=True)
        subprocess.run("cp OPT/product_opt.xyz NEB/product.xyz", shell=True)
        os.chdir('NEB')
    guess_block=""
    if os.path.exists("guess.xyz"):
        guess_block=f" TS \"guess.xyz\" \n"
    geom_block=""
    if xtb:
        nAtoms = int(subprocess.run("cat product.xyz | head -1 ",shell=True,stdout=subprocess.PIPE).stdout.decode().strip())
        maxiter = nAtoms * 4        

        geom_block =f"%geom\n Calc_Hess true\n Recalc_Hess 1\n MaxIter={maxiter} end \n"
        

    neb_input = f"! {method} {solvent_formatted} \n {geom_block}   %pal nprocs {SLURM_PARAMS_OPT['nprocs']} end\n%maxcore {SLURM_PARAMS_OPT['maxcore']}\n%neb \n Product \"product.xyz\" \n NImages {images}  \n {guess_block} end\n*xyzfile {charge} {mult} educt.xyz\n"
    neb_input_name = "neb-fast-TS.inp" if fast else "neb-TS.inp"
    with open(neb_input_name, 'w') as f:
        f.write(neb_input)
    print(f'Submitting {job_step} job to slurm')
    job_id = submit_job(neb_input_name, f"{job_step}_slurm.out",walltime="48")
    status = check_job_status(job_id, step=job_step)
    if status == 'COMPLETED' and grep_output('HURRAY', f'{neb_input_name.rsplit(".",1)[0]}.out'):
        print(f'{job_step}  completed successfully')
        print('Start freq job to check for significant imaginary frequency')
        imag_freq=freq_job(struc_name=f'{neb_input_name.rsplit(".",1)[0]}_NEB-TS_converged.xyz', charge=charge, mult=mult, trial=0, upper_limit=upper_limit, solvent=solvent, xtb=xtb, ts=True)

        if imag_freq:
            print("Negative mode below -50 cm**-1 found, continuing with next step")
            subprocess.run(f"cp {neb_input_name.rsplit('.',1)[0]}_NEB-TS_converged.xyz neb_completed.xyz", shell=True)
            os.chdir('..')
            return True
        else:
            print("No negative frequency found or are above -50 cm**-1, aborting")
            return False
    else:
        print(f'{job_step} job failed, restarting...')
        
        print(f'Status of failed Job {status}, if COMPLETED the jobs has not converged')
        subprocess.run(f"scancel {job_id}", shell=True)
        time.sleep(30)
        if grep_output('ORCA TERMINATED NORMALLY', f'{neb_input_name.rsplit(".",1)[0]}.out'):
            
            if fast and not xtb:
                print("Restarting as DFT NEB-TS run using DFT FAST-NEB-TS as guess")
                if os.path.exists("neb-fast-TS_NEB-HEI_converged.xyz"):
                    subprocess.run("cp neb-fast-TS_NEB-HEI_converged.xyz guess.xyz", shell=True)
                fast=False
                xtb=False
                Nimages=8
                trial =1
                upper_limit=MAX_TRIALS-2


            if fast and xtb :
                print("Restarting as NEB-TS run")
                fast=False

            if images >= 32 and xtb:
                print("Restarting as FAST-NEB-TS DFT run")
                Nimages=12
                fast=True
                xtb=False
                trial=1
                upper_limit=MAX_TRIALS
                switch=True
            if trial > 1:
                images = images * 2 if xtb else images + 2
            
        subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm*  neb*im* *neb*.inp", shell=True)
        
        return NEB_TS(charge=charge, mult=mult, trial=trial, upper_limit=upper_limit, xtb=xtb, fast=fast, solvent=solvent,switch=switch)






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
        make_folder(job_step)
        subprocess.run("cp OPT/educt_opt.xyz NEB/educt.xyz", shell=True)
        subprocess.run("cp OPT/product_opt.xyz NEB/product.xyz", shell=True)
        os.chdir('NEB')

    neb_input = f"!NEB-CI {solvent_formatted} {method} \n%pal nprocs {SLURM_PARAMS_OPT['nprocs']} end\n%maxcore {SLURM_PARAMS_OPT['maxcore']}\n%neb \n Product \"product.xyz\" \n NImages {images} {'Free_end true' if xtb else ''} \n end\n*xyzfile {charge} {mult} educt.xyz\n"
    with open('neb-CI.inp', 'w') as f:
        f.write(neb_input)

    print('Submitting NEB job to slurm')
    job_id = submit_job("neb-CI.inp", "neb-ci_slurm.out",walltime="48")
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
        subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm*  neb*im* *neb*.inp", shell=True)
        
        return NEB_CI(charge, mult, trial, images, upper_limit, xtb, solvent)

def TS_opt(charge=0, mult=1, trial=0, upper_limit=5, solvent=""):
    trial += 1
    if trial > upper_limit:
        print('Too many trials aborting')
        return False

    solvent_formatted = f"CPCM({solvent})" if solvent else ""
    job_step = 'TS'

    if trial < 2:
        make_folder(job_step)
        subprocess.run("cp NEB/neb_completed.xyz TS/ts_guess.xyz", shell=True)

        os.chdir('TS')

    print('Starting TS optimisation\n\n')

    if not freq_job(struc_name="ts_guess.xyz", charge=charge, mult=mult, trial=0, upper_limit=upper_limit, solvent=solvent, ts=True):
        print('Freq job failed aborting TS optimisation')
        return False

    TS_opt_input = f"!r2scan-3c OptTS tightscf {solvent_formatted}\n%geom\ninhess read \ninhessname \"freq.hess\"\nCalc_Hess true \nrecalc_hess 15 \nend \n%pal nprocs {SLURM_PARAMS_OPT['nprocs']} end\n%maxcore {SLURM_PARAMS_FREQ['maxcore']}\n*xyzfile {charge} {mult} ts_guess.xyz\n"
    with open('TS_opt.inp', 'w') as f:
        f.write(TS_opt_input)

    print('Submitting TS job to slurm')
    job_id_TS = submit_job("TS_opt.inp", "TS_opt_slurm.out",walltime="48")
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
        make_folder(job_step)
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





def restart_pipeline():
    print("Restart detected, checking for previous calculations")
    if not os.path.exists("settings_neb_pipeline.txt"):
        print("No settings file found, aborting")
        sys.exit(1)
    
    with open("settings_neb_pipeline.txt", 'r') as f:
        settings = f.read()
        print(f"Settings: {settings}")
        settings = settings.split('\n')
        charge = int(settings[0].split(':')[-1].strip())
        mult = int(settings[1].split(':')[-1].strip())
        solvent = settings[2].split(':')[-1].strip()
        Nimages = int(settings[3].split(':')[-1].strip())
        xtb = settings[4].split(':')[-1].strip()
        xtb = True if xtb == 'True' else False
        step_to_restart = settings[5].split(':')[-1].strip()

        return step_to_restart, charge, mult, solvent, Nimages, xtb



def pipeline(step,charge=0, mult=1, solvent="", Nimages=16):

    
    match step:

        case "OPT_XTB":
            return optimise_reactants(charge, mult, trial=0, upper_limit=MAX_TRIALS, solvent=solvent)
        case "NEB_CI_XTB":
            return NEB_CI(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS, xtb=True, solvent=solvent)
        case "NEB_CI_DFT":
            return NEB_CI(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS-2, xtb=False, solvent=solvent)
        case "FAST_NEB-TS_XTB":
            return NEB_TS(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS, xtb=True, fast=True, solvent=solvent)
        case "FAST_NEB-TS_DFT":
            return NEB_TS(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS-2, xtb=False, fast=True, solvent=solvent)
        case "NEB-TS_XTB":
            return NEB_TS(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS-2, xtb=True, fast=False, solvent=solvent)
        case "NEB-TS_DFT":
            return NEB_TS(charge, mult, trial=0, Nimages=Nimages, upper_limit=MAX_TRIALS-2, xtb=False, fast=False, solvent=solvent)
        case "TS":
            return TS_opt(charge, mult, trial=0, upper_limit=MAX_TRIALS-2, solvent=solvent)
        case "IRC":
            return IRC_job(charge, mult, trial=0, upper_limit=MAX_TRIALS, solvent=solvent)
        case _:
            print("Invalid step")
            print("Valid steps: OPT_{XTB/DFT}, NEB_CI_{XTB/DFT}, (FAST)-NEB_TS_{XTB/DFT}, TS, IRC")
            
            sys.exit(1)




   

def ussage():

    print("Usage: neb_pipeline.py [options]")
    print("Options:")
    print("-c, --charge: Charge of the system")
    print("-m, --mult: Multiplicity of the system")
    print("-s, --solvent: Solvent model to use")
    print("-i, --Nimages: Number of images for NEB")
    print("--steps: Steps to run in the pipeline, comma separated")
    print("--restart: Restart from previous calculations")
        
    

######MAIN FUNCTION######

#########################

def main(charge=0, mult=1, solvent="", Nimages=16,  restart=False,steps=["OPT_XTB", "FAST_NEB-TS_XTB", "TS", "IRC"]):
    
    setting_file_name="settings_neb_pipeline.txt"
    
    """
    
    Does not work currently, will be fixed in the future

    if restart:
        step_to_restart,charge,mult,solvent,Nimages,xtb= restart_pipeline()
        steps = steps[steps.index(step_to_restart):]
        print(f"Restarting from {step_to_restart} step")

    """
    for step in steps:
        print(f"Running {step} step")
        if not pipeline(step=step, charge=charge, mult=mult, solvent=solvent, Nimages=Nimages):
            print(f"{step} step failed")
            with open(setting_file_name, 'w') as f:
                f.write("step: " + step + "\n")
                f.write(f"charge: {charge}\n")
                f.write(f"mult: {mult}\n")
                f.write(f"solvent: {solvent}\n")
                f.write(f"Nimages: {Nimages}\n")
                

            sys.exit(1)
        print(f"{step} step completed successfully\n")
    
    
    
    print('All steps completed successfully.')

    print(f"Writing pipeline to {setting_file_name}")
    with open(setting_file_name, 'w') as f:
        f.write("Step: " + steps + "\n")
        f.write(f"charge: {charge}\n")
        f.write(f"mult: {mult}\n")
        f.write(f"solvent: {solvent}\n")
        f.write(f"Nimages: {Nimages}\n")
        f.write(f"steps: {steps}\n")
        
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
    
def parse_steps(steps_str):
    return steps_str.split(',')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NEB optimization pipeline.")
    parser.add_argument("--charge","-c", type=int, default=0, help="Charge of the system")
    parser.add_argument("--mult","-m", type=int, default=1, help="Multiplicity of the system")
    parser.add_argument("--solvent","-s", type=str, default="", help="Solvent model to use")
    parser.add_argument("--Nimages","-i", type=int, default=8, help="Number of images for NEB")
    parser.add_argument("--restart", type=str2bool,  default=False, help="Restart from previous calculations")
    parser.add_argument("--steps",  type=parse_steps, default=["OPT_XTB", "FAST_NEB-TS", "TS", "IRC"], help="Steps to run in the pipeline")

    args = parser.parse_args()




    settings=f'Charge: {args.charge}\nMultiplicity: {args.mult}\nSolvent: {args.solvent}\nNimages: {args.Nimages}\nSteps: {args.steps}\nRestart: {args.restart}'
    print("Starting Pipeline with parameters:")
    print(settings)




    main(charge=args.charge, mult=args.mult, solvent=args.solvent, Nimages=args.Nimages,  restart=args.restart,steps=args.steps)


