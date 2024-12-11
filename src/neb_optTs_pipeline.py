import os
import subprocess
import time
import glob
import sys
import shutil
import argparse

 ## coutn trials for each function after 3 abort run. 

### the script autmates the start of a reactant optimisisation, followed by an NEB-Ci run  with xtb 
## Followed by Freq followed by optTS  followed by freq and IRC

def check_job_status(job_id, interval=3,step=""):
    """
    Check the status of a SLURM job using squeue and sacct.

    Args:
        job_id (str): The SLURM job ID.
        max_checks (int): Maximum number of status checks before timing out.
        interval (int): Time in seconds between each check.

    Returns:
        str: Final status of the job.
    """
    
    while True:
        print(f'Checking job {job_id} status {step}')
        
        # Check if the job is still in the queue
        squeue_check = subprocess.run(
            ['squeue', '-j', job_id, '-h'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if squeue_check.stdout.strip():
            print(f'Job {job_id} is still in the queue.')
        else:
            # Job is no longer in the queue; get the final status using sacct
            sacct_result = subprocess.run(
                ['sacct', '-j', job_id, '--format=State', '--noheader'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            sacct_output = sacct_result.stdout.strip()
            sacct_error = sacct_result.stderr.strip()

            if sacct_error:
                print(f"Error from sacct: {sacct_error}")
                return "UNKNOWN"

            if sacct_output:
                # sacct can return multiple lines if there are job steps; take the primary job's status
                statuses = sacct_output.split('\n')
                latest_status = statuses[-1].strip()
                print(f'Latest status for job {job_id}: {latest_status}')

                if latest_status in ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                    return latest_status
            else:
                print(f"No sacct information found for job ID {job_id}.")
                return "UNKNOWN"
        
        
        time.sleep(interval)


def make_folders(folder_names):
     
    for dir_name in folder_names:
        
       dir_name_path = os.path.join(os.getcwd(),dir_name)
       if not os.path.exists(dir_name_path):
           os.makedirs(dir_name_path)
       else:
           print(f"Delete {dir_name_path} in the future a restart option will be added")
           shutil.rmtree(dir_name_path)
           os.makedirs(dir_name_path)
           
           


def NEB_CI(charge=0,mult=1,trial=0,Nimages=8,upper_limit=3,xtb=True,solvent=""):

    trial +=1

    if trial > upper_limit:
        print('Too many trials aborting')
        return False
    
    neb_sucess = False
    neb_input = ""
    if solvent and xtb==False:
        solvent = f"CPCM({solvent}"
    elif solvent and xtb:
        solvent = f"ALPB({solvent})"
    else:
        solvent = ""

    job_name_neb = ""
    if(trial < 2):
        subprocess.run("cp OPT/educt_opt.xyz NEB/educt.xyz", shell=True)
        subprocess.run("cp OPT/product_opt.xyz NEB/product.xyz", shell=True)


        os.chdir('NEB')
    
    if xtb:
        Nimages = Nimages +2 
        print(f'Starting NEB-TS calculation using {Nimages} images and gfn2-xtb as method  \n\n')
        print('Doing free end NEB-TS since educt and product are optimized on r2scanc-3 surface\n')
        neb_input = f"!NEB-CI {solvent} XTB2 \n%pal nprocs 24 end\n%maxcore 2024\n %neb \n Product \"product.xyz\" \n NImages {Nimages} Free_end true \n end\n *xyzfile {charge} {mult} educt.xyz\n"
    else:
        print(f'Starting NEB-CI calculation using {Nimages} images and r2scan-3c as method  \n\n')  ### maybe b97-3c as alternative. 
        neb_input = f"!NEB-CI r2scan-3c {solvent} \n%pal nprocs 24 end\n%maxcore 2024\n %neb \n Product \"product.xyz\" \n NImages {Nimages}  end\n *xyzfile {charge} {mult} educt.xyz\n"

    with open('neb-CI.inp','w') as f:
       f.write(neb_input)
    
   
    print('Submitting NEB job to slurm\n')
        
    ssubo_output = subprocess.run(["ssubo","-o neb-ci_slurm.out","neb-CI.inp"],capture_output=True)

    # Extract the job ID from the ssubo output
    ssubo_output = ssubo_output.stdout.decode().strip()
    job_id_line = [line for line in ssubo_output.split('\n') if 'Submitted batch job' in line]
    if job_id_line:
        job_name_neb = job_id_line[0].split()[-1]
    else:
        print("Failed to extract job ID from ssubo output aborting")
        exit(1)


    print("Checking NEB-CI job status\n")
    #time.sleep(10)
    status_NEB = check_job_status(job_name_neb,step='NEB')


    print('Checking if optimisation completed successfully\n')

    if status_NEB == 'COMPLETED':

        ## check if HURRAY is present if yes continue if no restart job with last opt. 

        hurray_educt= subprocess.run("grep 'H U R R A Y' neb-CI.out", shell=True, stdout=subprocess.PIPE)
        hurray_educt = hurray_educt.stdout.decode().strip()



        if bool(hurray_educt):
                
                print('Educt optimisation completed successfully\n')
                neb_sucess = True

       
    








    if neb_sucess:
        os.chdir('..')
        return True
    else:
        print('NEB-CI job failed restarting')


        subprocess.run(f"scancel  {job_name_neb}", shell=True, capture_output=True, text=True)
       
        print('Waiting for 30 seconds before restarting to wait for the files to be copied\n')  
        time.sleep(30)

        normal_termination=subprocess.run("grep 'ORCA TERMINATED NORMALLY' neb-ci.out", shell=True, capture_output=True, text=True )
        subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess slurm* neb*im* *neb*.inp ", shell=True)
        


        if normal_termination:
            print("ORCA terminated normally")
            print("This most likely means that the CI failed to converge, will be refined in the future")
            print("Restarting with more images double for xtb, and +2 for r2scan-3c")

            if xtb:
                Nimages = Nimages*2
            else:
                Nimages = Nimages + 2



        

        
        
            ## start from lats opt


        

        success = NEB_CI(charge=charge,mult=mult,trial=trial,solvent=solvent,xtb=xtb,Nimages=Nimages,upper_limit=upper_limit)
        return success


def optimis_reactants(charge=0,mult=1,trial=0,upper_limit=3,solvent=""):

    
    trial +=1

    if trial > 3:
        print('Too many trials aborting')
        return False

    educt_sucess = False
    product_sucess = False

    if solvent:
        solvent = f"CPCM({solvent})"

    job_name_educt = ""
    job_name_product = ""
    if(trial < 2):
        subprocess.run("cp *.xyz OPT", shell=True)

        os.chdir('OPT')

    print('Starting reactant optimisation\n\n') 

    educt_opt_input = f"!r2scan-3c  {solvent} opt\n%pal nprocs 24 end\n%maxcore 2024\n*xyzfile {charge} {mult} educt.xyz\n"
    product_opt_input = f"!r2scan-3c {solvent} opt\n%pal nprocs 24 end\n%maxcore 2024\n*xyzfile {charge} {mult} product.xyz\n"
    
    with open('educt_opt.inp','w') as f:
        f.write(educt_opt_input)
    
    with open('product_opt.inp','w') as f:
        f.write(product_opt_input)


    print('Submitting educt job to slurm\n')
        
    ssubo_output = subprocess.run(["ssubo","-o educt_slurm.out","educt_opt.inp"],capture_output=True)

    # Extract the job ID from the ssubo output
    ssubo_output = ssubo_output.stdout.decode().strip()
    job_id_line = [line for line in ssubo_output.split('\n') if 'Submitted batch job' in line]
    if job_id_line:
        job_name_educt = job_id_line[0].split()[-1]
    else:
        print("Failed to extract job ID from ssubo output aborting")
        exit(1)
        
    
    #job_name_educt = subprocess.run("grep 'Batch' educt_slurm.out | awk '{print $NF}'", shell=True, stdout=subprocess.PIPE)
    #job_name_educt = job_name_educt.stdout.decode().strip()



    print('Submitting product job to slurm\n')

    ssubo_output=subprocess.run(["ssubo","-o product_slurm.out","product_opt.inp"],capture_output=True)

    ssubo_output = ssubo_output.stdout.decode().strip()
    job_id_line = [line for line in ssubo_output.split('\n') if 'Submitted batch job' in line]
    if job_id_line:
        job_name_product = job_id_line[0].split()[-1]
    else:
        print("Failed to extract job ID from ssubo output aborting")
        exit(1)
      # Wait for the job to be submitted and slurm file to be created
    #job_name_product = subprocess.run("grep 'Batch' product_slurm.out | awk '{print $NF}'", shell=True, stdout=subprocess.PIPE)
    #job_name_product = job_name_product.stdout.decode().strip()


    ## check if educt is completed
    print("Checking job status\n")
    #time.sleep(10)
    status_educt = check_job_status(job_name_educt,step='educt')
    print("Status educt: ",status_educt)
    status_product = check_job_status(job_name_product,step='product')

    print(f"Status educt: {status_educt}, Status product: {status_product}")
    print('Checking if optimisation completed successfully\n')

    if status_educt == 'COMPLETED' and status_product == 'COMPLETED':

        ## check if HURRAY is present if yes continue if no restart job with last opt. 

        hurray_educt= subprocess.run("grep 'HURRAY' educt_opt.out", shell=True, stdout=subprocess.PIPE)
        hurray_educt = hurray_educt.stdout.decode().strip()



        if bool(hurray_educt):
                
                print('Educt optimisation completed successfully\n')
                educt_sucess = True

        hurray_product= subprocess.run("grep 'HURRAY' product_opt.out", shell=True, stdout=subprocess.PIPE)
        hurray_product = hurray_product.stdout.decode().strip()

        if bool(hurray_product):
                    
                    print('Product optimisation completed successfully\n')
                    product_sucess = True
    








    if educt_sucess and product_sucess:
        os.chdir('..')
        return True
    else:
        print('One of the jobs failed restarting both, from last opt')

        subprocess.run(f"scancel  {job_name_product}", shell=True, capture_output=True, text=True)
        subprocess.run(f"scancel  {job_name_educt}", shell=True, capture_output=True, text=True)

        print('Waiting for 30 seconds before restarting to wait for the files to be copied\n')  
        time.sleep(60)
        
        subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess educt_opt.inp product_opt.inp", shell=True)
        
            ## start from lats opt
        if os.path.exists('educt_opt.xyz'):
            subprocess.run(["mv","educt_opt.xyz","educt.xyz"])
        if os.path.exists('product_opt.xyz'):    
            subprocess.run(["mv","product_opt.xyz","product.xyz"])

        

        success = optimis_reactants(charge=charge,mult=mult,trial=trial,solvent=solvent)
        return success



def freq_job(struc_name="coord.xyz",charge=0,mult=1,trial=0,upper_limit=3,solvent="",ts=False):

    ## ts flag checks for imag mode if not present TS opt will  not be continu

    trial +=1


    freq_inp = f"!r2scan-3c {solvent} freq tightscf \n%pal nprocs 24 end\n%maxcore 8024\n*xyzfile {charge} {mult} {struc_name}\n"

    with open('freq.inp','w') as f:
        f.write(freq_inp)

    job_name_freq = ""

    ssubo_output = subprocess.run(["ssubo","-o freq_slurm.out","freq.inp"],capture_output=True)

    
        # Extract the job ID from the ssubo output
    ssubo_output = ssubo_output.stdout.decode().strip()
    job_id_line = [line for line in ssubo_output.split('\n') if 'Submitted batch job' in line]
    if job_id_line:
        job_name_freq = job_id_line[0].split()[-1]
    else:
            print("Failed to extract job ID from ssubo output aborting")
            exit(1)
            
        

    
    
    
    print("Checking Freq job status\n")
        #time.sleep(10)
    status_freq = check_job_status(job_name_freq,step='Freq')
    
    print('Checking if freq completed successfully\n')
    
    if status_freq == 'COMPLETED':
    
        # check if HURRAY is present if yes continue if no restart job with last opt. 
    
        vib= subprocess.run("grep 'VIBRATIONAL FREQUENCIES' freq.out", shell=True, stdout=subprocess.PIPE)
        vib = vib.stdout.decode().strip()



        if bool(vib):
                
                print('Freq calculation completed successfully\n')
                if ts==True:
                    neg_vib = subprocess.run("grep '**imaginary mode***' freq.out", shell=True, stdout=subprocess.PIPE)
                    if bool(neg_vib):
                        print('Negative frequency found, continuing with TS optimisation\n')
                        return True
                    else:
                        print('No negative frequency found, aborting TS optimisation\n')
                        return False
                else:
                    return True
        else:
            print('No vibrational frequencies found, restarting job\n')
            subprocess.run(f"scancel  {job_name_freq}", shell=True, capture_output=True, text=True)
            subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess", shell=True)
            
            success = freq_job(charge=charge,mult=mult,trial=trial,upper_limit=upper_limit,solvent=solvent,ts=ts)
            return success


        
def TS_opt(charge=0,mult=1,trial=0,upper_limit=3,solvent=""):
        

        #### Initial frequency calculation of the TS guess

        
        
        trial +=1
    
        if trial > upper_limit:
            print('Too many trials aborting')
            return False
    
        TS_sucess = False
    
        if solvent:
            solvent = f"CPCM({solvent})"
    
        job_name_TS = ""
        if(trial < 2):
            subprocess.run("cp NEB/*NEB-CI_converged.xyz TS/ts_guess.xyz", shell=True)
    
            os.chdir('TS')
    
        print('Starting TS optimisation\n\n') 

        freq_succes=freq_job(struc_name="ts_guess.xyz",charge=charge,mult=mult,trial=0,upper_limit=upper_limit,solvent=solvent,ts=True)

        if not freq_succes:
            print('Freq job failed aborting TS optimisation')
            return False

        TS_opt_input = f"!r2scan-3c OptTS tightscf  {solvent}\n %geom\n inhess read \n inhessname \"freq.hess\"\n Calc_Hess true \n recalc_hess 15 \n end \n  %pal nprocs 24 end\n%maxcore 8024\n*xyzfile {charge} {mult} ts_guess.xyz\n"
        



        with open('TS_opt.inp','w') as f:
            f.write(TS_opt_input)
        
        print('Submitting TS job to slurm\n')
            
        ssubo_output = subprocess.run(["ssubo","-o TS_opt_slurm.out","TS_opt.inp"],capture_output=True)

    
        # Extract the job ID from the ssubo output
        ssubo_output = ssubo_output.stdout.decode().strip()
        job_id_line = [line for line in ssubo_output.split('\n') if 'Submitted batch job' in line]
        if job_id_line:
            job_name_TS = job_id_line[0].split()[-1]
        else:
            print("Failed to extract job ID from ssubo output aborting")
            exit(1)
            
        
        #job_name_educt = subprocess.run("grep 'Batch' educt_slurm.out | awk '{print $NF}'", shell=True, stdout=subprocess.PIPE)
        #job_name_educt = job_name_educt.stdout.decode().strip()
    
    
    
        print("Checking Freq job status\n")
        #time.sleep(10)
        status_ts = check_job_status(job_name_TS,step='TS')
    
        print('Checking if optimisation completed successfully\n')
    
        if status_ts == 'COMPLETED':
    
            ## check if HURRAY is present if yes continue if no restart job with last opt. 
    
            hurray_TS= subprocess.run("grep 'HURRAY' TS_opt.out", shell=True, stdout=subprocess.PIPE)
            hurray_TS = hurray_TS.stdout.decode().strip()

            if bool(hurray_TS   ):
                
                print('TS optimisation completed successfully\n')
                TS_sucess = True
        if TS_sucess:
            os.chdir('..')
            return True
        else:
            print('TS job failed restarting')
    
            subprocess.run(f"scancel  {job_name_TS}", shell=True, capture_output=True, text=True)
            
    
            print('Waiting for 30 seconds before restarting to wait for the files to be copied\n')  
            time.sleep(60)
            
            subprocess.run("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess", shell=True)
                ## start from lats opt
            if os.path.exists('TS_opt.xyz'):
                subprocess.run("mv","TS_opt.xyz","ts_guess.xyz", shell=True)
            
    
            
    
            success = TS_opt(charge=charge,mult=mult,trial=trial,solvent=solvent)
            return success


    
    
def IRC_job(charge=0,mult=1,trial=0,upper_limit=3,solvent="",maxiter=70):

    print('Starting IRC calculation\n\n')
    trial +=1
    
    if trial > upper_limit:
        print('Too many trials aborting')
        return False
    
    IRC_sucess = False
    
    if solvent:
        solvent = f"CPCM({solvent})"
    
    job_name_IRC = ""
    if(trial < 2):
        subprocess.run("cp TS/TS_opt.xyz IRC ", shell=True)
    
        os.chdir('IRC')
    
        freq_ts = freq_job(struc_name="TS_opt.xyz",charge=charge,mult=mult,trial=0,upper_limit=upper_limit,solvent=solvent,ts=True)

        if not freq_ts:
            print('Freq job failed or did not have negative frequency. Aborting IRC calculation')
            return False

    IRC_input = f"!r2scan-3c IRC tightscf {solvent}\n %irc\n  maxiter {maxiter} \n InitHess read \n Hess_Filename \"freq.hess\"\n end \n  %pal nprocs 24 end\n%maxcore 2024\n*xyzfile {charge} {mult} TS_opt.xyz\n"
    
    with open('IRC.inp','w') as f:
            f.write(IRC_input)
        
    print('Submitting IRC job to slurm\n')
            
    ssubo_output = subprocess.run(["ssubo","-o IRC_slurm.out","IRC.inp"],capture_output=True)

    
        # Extract the job ID from the ssubo output
    ssubo_output = ssubo_output.stdout.decode().strip()
    job_id_line = [line for line in ssubo_output.split('\n') if 'Submitted batch job' in line]
    if job_id_line:
        job_name_IRC = job_id_line[0].split()[-1]
    else:
        print("Failed to extract job ID from ssubo output aborting")
        exit(1)
            
        
        #job_name_educt = subprocess.run("grep 'Batch' educt_slurm.out | awk '{print $NF}'", shell=True, stdout=subprocess.PIPE)
        #job_name_educt = job_name_educt.stdout.decode().strip()
    
    
    
        print("Checking IRC job status\n")
        #time.sleep(10)
        status_irc = check_job_status(job_name_IRC,step='IRC')
    
        print('Checking if optimisation completed successfully\n')
    
        if status_irc == 'COMPLETED':
    
            ## check if HURRAY is present if yes continue if no restart job with last opt. 
    
            hurray_irc= subprocess.run("grep 'HURRAY' IRC.out", shell=True, stdout=subprocess.PIPE)
            hurray_irc = hurray_irc.stdout.decode().strip()

            if bool(hurray_irc):
                
                print('IRC optimisation completed successfully\n')
                IRC_sucess = True
        if IRC_sucess:
            os.chdir('..')
            return True
        else:
            print('IRC job failed restarting')
            subprocess.run(f"scancel  {job_name_IRC}", shell=True, capture_output=True, text=True)

            if subprocess.run("grep 'ORCA TERMINATED NORMALLY' IRC.out", shell=True, capture_output=True, text=True):
                print("ORCA terminated normally")
                print("This most likely means that the IRC failed to converge, will be refined in the future")
                
                if maxiter < 100:
                    maxiter = maxiter + 30
                else:
                    print("Maxiter reached 100, aborting")
                    return False
            
            
    
            print('Waiting for 30 seconds before restarting to wait for the files to be copied\n')  
            time.sleep(60)
            
            subprocess.run("rm -rf *.gbw pmix* *densities* IRC.inp slurm* ", shell=True)
                ## start from lats opt

            
    
            
    
            success = IRC_input(charge=charge,mult=mult,trial=trial,solvent=solvent,maxiter=maxiter)
            return success




def main(charge=0,mult=1,solvent="",Nimages=8,xtb=True):
     
    step = 0

    steps = ["OPT","NEB","TS","IRC"]

    make_folders(steps)

    
    success =optimis_reactants(charge=charge,mult=mult,trial=0,upper_limit=5)

    if not success:
         print(f"Step {steps[step]} failed")
         print('Exiting')
         exit(1)         
    print(f" {steps[step]} step completed successfully")

    step+=1

    print(f"Starting {steps[step]} step")

    success = NEB_CI(charge=charge,mult=mult,trial=0,Nimages=Nimages,upper_limit=5,solvent=solvent,xtb=xtb)
    
    if not success:
        print(f"Step {steps[step]} failed")
        print('Exiting')
        exit(1)
    print(f" {steps[step]} step completed successfully")
    
    step+=1

    print(f"Starting {steps[step]} step")

    success = TS_opt(charge=charge,mult=mult,trial=0,upper_limit=5,solvent=solvent)

    if not success:
        print(f"Step {steps[step]} failed")
        print('Exiting')
        exit(1)
    print(f" {steps[step]} step completed successfully")

    step+=1

    print(f"Starting {steps[step]} step")

    success = IRC_job(charge=charge,mult=mult,trial=0,upper_limit=5,solvent=solvent)

    if not success:
        print(f"Step {steps[step]} failed")
        print('Exiting')
        exit(1)
    
    print(f'All steps completed successfully. \n Chekc IRC trajectory for the reaction path\n Exiting')

    exit(0)







if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run NEB optimization pipeline.")
    parser.add_argument("--charge", type=int, default=0, help="Charge of the system")
    parser.add_argument("--mult", type=int, default=1, help="Multiplicity of the system")
    parser.add_argument("--solvent", type=str, default="", help="Solvent model to use")
    parser.add_argument("--Nimages", type=int, default=8, help="Number of images for NEB")
    parser.add_argument("--xtb", type=bool, default=True, help="Use xtb method")

    args = parser.parse_args()

    main(charge=args.charge, mult=args.mult, solvent=args.solvent, Nimages=args.Nimages, xtb=args.xtb)