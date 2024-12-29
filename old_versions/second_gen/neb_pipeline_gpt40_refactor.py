import re
import os
import subprocess
import time
import shutil
import argparse
import sys
import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# Configuration
@dataclass
class PipelineConfig:
    max_trials: int = 5
    retry_delay: int = 60
    freq_threshold: float = -50.0
    slurm_params_high: Dict[str, int] = field(default_factory=lambda: {'nprocs': 16, 'maxcore': 12000})
    slurm_params_low: Dict[str, int] = field(default_factory=lambda: {'nprocs': 24, 'maxcore': 2524})
    submit_command: List[str] = field(default_factory=lambda: ["ssubo", "-m", "No"])
    check_states: List[str] = field(default_factory=lambda: ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'])

class NEBPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("NEBPipeline")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler("neb_pipeline.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def run_subprocess(self, command: Union[str, List[str]], shell: bool = False, capture_output: bool = True, check: bool = True, exit_on_error: bool = True, timeout: int = 60) -> Optional[subprocess.CompletedProcess]:
        try:
            result = subprocess.run(command, shell=shell, capture_output=capture_output, text=True, check=check, timeout=timeout)
            return result
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command '{' '.join(command)}' timed out after {timeout} seconds.")
            return None
        except subprocess.CalledProcessError as e:
            if exit_on_error:
                self.logger.error(f"Command '{' '.join(command)}' failed with error: {e.stderr}")
                sys.exit(1)
            else:
                return None

    def check_job_status(self, job_id: str, interval: int = 45, step: str = "", timeout: int = 60) -> str:
        counter = 0
        while True:
            time.sleep(interval)
            self.logger.info(f'Checking job {job_id} status {step}')
            squeue = self.run_subprocess(['squeue', '-j', job_id, '-h'], shell=False, exit_on_error=False, timeout=timeout)
            if squeue and squeue.stdout.strip():
                if counter % 10 == 0:
                    self.logger.info(f'Job {job_id} is still in the queue.')
                counter += 1
            else:
                sacct = self.run_subprocess(['sacct', '-j', job_id, '--format=State', '--noheader'], shell=False, exit_on_error=False, timeout=timeout)
                if sacct and sacct.stdout.strip():
                    statuses = sacct.stdout.strip().split('\n')
                    latest_status = statuses[-1].strip()
                    self.logger.info(f'Latest status for job {job_id}: {latest_status}')
                    if latest_status in self.config.check_states:
                        return latest_status
                    else:
                        self.logger.info(f'Job {job_id} ended with status: {latest_status}')
                        return latest_status
                else:
                    if sacct and sacct.stderr.strip():
                        self.logger.error(f"Error from sacct: {sacct.stderr.strip()}")
                        self.logger.error(f'Job {job_id} status could not be determined.')
                        return 'UNKNOWN'

    def make_folder(self, dir_name: str):
        path = os.path.join(os.getcwd(), dir_name)
        if os.path.exists(path):
            if os.path.isdir(path):
                self.logger.info(f"Removing existing folder {path}")
                shutil.rmtree(path)
            else:
                self.logger.info(f"Removing existing file {path}")
                os.remove(path)
        os.makedirs(path)
        self.logger.info(f"Created folder {path}")

    def submit_job(self, input_file: str, output_file: str, walltime: str = "24") -> Optional[str]:
        command = self.config.submit_command + ["-w", walltime, "-o", output_file, input_file]
        result = self.run_subprocess(command)
        if result:
            for line in result.stdout.splitlines():
                if 'Submitted batch job' in line:
                    job_id = line.split()[-1]
                    self.logger.info(f"Job submitted with ID: {job_id}")
                    return job_id
        self.logger.error(f"Failed to submit job {input_file}")
        sys.exit(1)

    def grep_output(self, pattern: str, file: str) -> str:
        result = self.run_subprocess(f"grep '{pattern}' {file}", shell=True, capture_output=True, check=False, exit_on_error=False)
        return result.stdout.strip() if result else ''

    def optimise_reactants(self, charge: int = 0, mult: int = 1, trial: int = 0, upper_limit: int = 5, solvent: str = "", xtb: bool = True) -> bool:
        trial += 1
        self.logger.info(f"Trial {trial} for reactant optimisation")
        if trial > upper_limit:
            self.logger.error('Too many trials aborting')
            return False

        solvent_formatted = f"ALPB({solvent})" if solvent and xtb else f"CPCM({solvent})" if solvent else ""
        method = "XTB2" if xtb else "r2scan-3c"

        if trial < 2:
            self.make_folder('OPT')
            self.run_subprocess("cp *.xyz OPT", shell=True)
            os.chdir('OPT')

        self.logger.info('Starting reactant optimisation')

        job_inputs = {
            'educt_opt.inp': f"!{method} {solvent_formatted} opt\n%pal nprocs {self.config.slurm_params_low['nprocs']} end\n%maxcore {self.config.slurm_params_low['maxcore']}\n*xyzfile {charge} {mult} educt.xyz\n",
            'product_opt.inp': f"!{method} {solvent_formatted} opt\n%pal nprocs {self.config.slurm_params_low['nprocs']} end\n%maxcore {self.config.slurm_params_low['maxcore']}\n*xyzfile {charge} {mult} product.xyz\n"
        }

        for filename, content in job_inputs.items():
            with open(filename, 'w') as f:
                f.write(content)

        self.logger.info('Submitting educt and product jobs to Slurm')
        job_ids = [self.submit_job(inp, f"{inp.split('.')[0]}_slurm.out") for inp in job_inputs.keys()]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_educt = executor.submit(self.check_job_status, job_ids[0], step='educt')
            future_product = executor.submit(self.check_job_status, job_ids[1], step='product')

            status_educt = future_educt.result()
            status_product = future_product.result()

        self.logger.info(f'Educt Status: {status_educt}')
        self.logger.info(f'Product Status: {status_product}')

        if status_educt == 'COMPLETED' and status_product == 'COMPLETED':
            educt_success = 'HURRAY' in self.grep_output('HURRAY', 'educt_opt.out')
            product_success = 'HURRAY' in self.grep_output('HURRAY', 'product_opt.out')
            if educt_success and product_success:
                os.chdir('..')
                return True

        self.logger.warning('One of the optimisation jobs failed, restarting both')
        for job_id in job_ids:
            self.run_subprocess(['scancel', job_id], exit_on_error=False)
        time.sleep(self.config.retry_delay)
        self.run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm* educt_opt.inp product_opt.inp", shell=True, exit_on_error=False)

        for xyz in ['educt_opt.xyz', 'product_opt.xyz']:
            if os.path.exists(xyz):
                os.rename(xyz, xyz.replace('_opt', ''))

        return self.optimise_reactants(charge, mult, trial, upper_limit, solvent, xtb)

    def freq_job(self, struc_name: str = "coord.xyz", charge: int = 0, mult: int = 1, trial: int = 0, upper_limit: int = 5, solvent: str = "", xtb: bool = False, ts: bool = False) -> bool:
        trial += 1
        self.logger.info(f"Trial {trial} for frequency job")

        if trial > upper_limit:
            self.logger.error('Too many trials aborting')
            return False

        method = "r2scan-3c freq tightscf"
        solvent_formatted = f"CPCM({solvent})" if solvent else ""

        freq_input = f"! {method} {solvent_formatted}\n%pal nprocs {self.config.slurm_params_high['nprocs']} end\n%maxcore {self.config.slurm_params_high['maxcore']}\n*xyzfile {charge} {mult} {struc_name}\n"
        with open('freq.inp', 'w') as f:
            f.write(freq_input)

        self.logger.info('Submitting frequency job to Slurm')
        job_id_freq = self.submit_job("freq.inp", "freq_slurm.out")
        status_freq = self.check_job_status(job_id_freq, step='Freq')

        if status_freq == 'COMPLETED':
            if self.grep_output('VIBRATIONAL FREQUENCIES', 'freq.out'):
                self.logger.info('Frequency calculation completed successfully')
                if ts:
                    output = self.grep_output('**imaginary mode***', 'freq.out')
                    match = re.search(r'(-?\d+\.\d+)\s*cm\*\*-1', output)
                    if match:
                        imag_freq = float(match.group(1))
                        if imag_freq < self.config.freq_threshold:
                            self.logger.info('Negative frequency found, proceeding with TS optimisation')
                            return True
                        else:
                            self.logger.error('Negative frequency above threshold, aborting TS optimisation')
                            return False
                    self.logger.error('No negative frequency found, aborting TS optimisation')
                    return False
                return True

        self.logger.warning('Frequency job failed, restarting...')
        self.run_subprocess(['scancel', job_id_freq], exit_on_error=False)
        self.run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm*", shell=True, exit_on_error=False)
        return self.freq_job(struc_name, charge, mult, trial, upper_limit, solvent, ts)

    def NEB_TS(self, charge: int = 0, mult: int = 1, trial: int = 0, Nimages: int = 16, upper_limit: int = 5, xtb: bool = True, fast: bool = True, solvent: str = "", switch: bool = False) -> bool:
        if switch:
            trial = 0
            os.chdir('..')
            self.logger.info("Switching to DFT run and optimizing reactants")
            if not self.optimise_reactants(charge, mult, trial=0, upper_limit=self.config.max_trials, solvent=solvent, xtb=False):
                self.logger.error("Reactant optimisation failed, aborting")
                return False
            switch = False
        job_step = 'NEB-TS'
        trial += 1
        self.logger.info(f"Trial {trial} for NEB-TS")

        slurm_params = self.config.slurm_params_low

        if trial > upper_limit:
            self.logger.error('Too many trials aborting')
            return False

        solvent_formatted = f"CPCM({solvent})" if solvent and not xtb else f"ALPB({solvent})" if solvent else ""
        method = "FAST-NEB-TS XTB2 tightscf" if fast and xtb else \
                 "FAST-NEB-TS r2scan-3c tightscf" if fast and not xtb else \
                 "NEB-TS XTB2 tightscf" if xtb and not fast else \
                 "NEB-TS r2scan-3c tightscf"

        if trial < 2:
            self.make_folder("NEB")
            self.run_subprocess("cp OPT/educt_opt.xyz NEB/educt.xyz", shell=True)
            self.run_subprocess("cp OPT/product_opt.xyz NEB/product.xyz", shell=True)
            os.chdir('NEB')

        guess_block = ' TS "guess.xyz"\n' if os.path.exists("guess.xyz") else ""
        geom_block = ""
        if xtb:
            with open("product.xyz") as f:
                nAtoms = int(f.readline().strip())
            maxiter = nAtoms * 4
            geom_block = f"%geom\n Calc_Hess true\n Recalc_Hess 1\n MaxIter={maxiter} end \n"
            slurm_params = self.config.slurm_params_low

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

        self.logger.info(f'Submitting {neb_input_name} job to Slurm')
        job_id = self.submit_job(neb_input_name, f"{job_step}_slurm.out", walltime="48")
        status = self.check_job_status(job_id, step="NEB-TS")

        if status == 'COMPLETED' and self.grep_output('HURRAY', f'{neb_input_name.rsplit(".", 1)[0]}.out'):
            self.logger.info('NEB-TS completed successfully')

            imag_freq = self.freq_job(
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
                self.logger.info("Negative mode below threshold found, continuing")
                time.sleep(60)
                self.run_subprocess(
                    f"cp {neb_input_name.rsplit('.', 1)[0]}_NEB-TS_converged.xyz neb_completed.xyz", shell=True)
                os.chdir('..')
                return True
            else:
                if not self.handle_failed_imagfreq(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent):
                    os.chdir('..')
                    return False
                else:
                    self.logger.info("Negative mode found in guess. Continuing with TS optimisation.")
                    time.sleep(60)
                    self.run_subprocess("cp neb-TS_NEB-CI_converged.xyz neb_completed.xyz", shell=True)
                    os.chdir('..')
                    return True
        elif self.grep_output('ORCA TERMINATED NORMALLY', f'{neb_input_name.rsplit(".", 1)[0]}.out'):
            if not self.handle_unconverged_neb(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent, job_id):
                os.chdir('..')
                return False
            else:
                self.logger.info("R2SCAN-3c NEB-TS job did not converge, but freq job found negative frequency above guess. Fail might be due no Hess recalc. Continuing with TS optimisation.")
                time.sleep(60)
                self.run_subprocess("cp neb-TS_NEB-CI_converged.xyz neb_completed.xyz", shell=True)
                os.chdir('..')
                return True
        else:
            if self.grep_output("TS OPTIMIZATION", f'{neb_input_name.rsplit(".", 1)[0]}.out'):
                self.logger.info("NEB-TS started to optimized a TS guess but aborted during the run")
                self.logger.info("Check whether last struct has significant negative frequency")
                self.run_subprocess(["scancel", job_id], exit_on_error=False)
                self.logger.info("Wait for a minute to wait for the files from the node")
                time.sleep(60)
                struc_name = f'{neb_input_name.rsplit(".", 1)[0]}.xyz'

                imag_freq = self.freq_job(
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
                    self.logger.info("Negative mode below threshold found, Continuing with TS opt")
                    time.sleep(60)
                    self.run_subprocess(["cp", struc_name, "neb_completed.xyz"], shell=True)
                    os.chdir('..')
                    return True

            self.logger.warning(f"NEB-TS {neb_input_name} failed, restarting if there are trials left")
            return self.NEB_TS(charge, mult, trial, Nimages, upper_limit, xtb, fast, solvent, switch)

    def handle_failed_imagfreq(self, charge: int, mult: int, Nimages: int, trial: int, upper_limit: int, xtb: bool, fast: bool, solvent: str) -> bool:
        self.logger.info("No significant negative frequency found")
        if not xtb and fast:
            self.logger.info("FAST-NEB-TS with r2scan-3c did not find TS. Retrying with regular NEB-TS.")
            self.logger.info("Using FAST-NEB-TS guess as TS guess for NEB-TS")
            if os.path.exists("neb-fast-TS_NEB-HEI_converged.xyz"):
                self.run_subprocess("mv neb-fast-TS_NEB-HEI_converged.xyz guess.xyzimport re")
import os
import subprocess
import time
import shutil
import argparse
import sys
import concurrent.futures
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# Configuration
@dataclass
class PipelineConfig:
    max_trials: int = 5
    retry_delay: int = 60
    freq_threshold: float = -50.0
    slurm_params_high: Dict[str, int] = field(default_factory=lambda: {'nprocs': 16, 'maxcore': 12000})
    slurm_params_low: Dict[str, int] = field(default_factory=lambda: {'nprocs': 24, 'maxcore': 2524})
    submit_command: List[str] = field(default_factory=lambda: ["ssubo", "-m", "No"])
    check_states: List[str] = field(default_factory=lambda: ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'])

class NEBPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger("NEBPipeline")
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        fh = logging.FileHandler("neb_pipeline.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        return logger

    def run_subprocess(self, command: Union[str, List[str]], shell: bool = False, capture_output: bool = True, check: bool = True, exit_on_error: bool = True, timeout: int = 60) -> Optional[subprocess.CompletedProcess]:
        try:
            result = subprocess.run(command, shell=shell, capture_output=capture_output, text=True, check=check, timeout=timeout)
            return result
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command '{' '.join(command)}' timed out after {timeout} seconds.")
            return None
        except subprocess.CalledProcessError as e:
            if exit_on_error:
                self.logger.error(f"Command '{' '.join(command)}' failed with error: {e.stderr}")
                sys.exit(1)
            else:
                return None

    def check_job_status(self, job_id: str, interval: int = 45, step: str = "", timeout: int = 60) -> str:
        counter = 0
        while True:
            time.sleep(interval)
            self.logger.info(f'Checking job {job_id} status {step}')
            squeue = self.run_subprocess(['squeue', '-j', job_id, '-h'], shell=False, exit_on_error=False, timeout=timeout)
            if squeue and squeue.stdout.strip():
                if counter % 10 == 0:
                    self.logger.info(f'Job {job_id} is still in the queue.')
                counter += 1
            else:
                sacct = self.run_subprocess(['sacct', '-j', job_id, '--format=State', '--noheader'], shell=False, exit_on_error=False, timeout=timeout)
                if sacct and sacct.stdout.strip():
                    statuses = sacct.stdout.strip().split('\n')
                    latest_status = statuses[-1].strip()
                    self.logger.info(f'Latest status for job {job_id}: {latest_status}')
                    if latest_status in self.config.check_states:
                        return latest_status
                    else:
                        self.logger.info(f'Job {job_id} ended with status: {latest_status}')
                        return latest_status
                else:
                    if sacct and sacct.stderr.strip():
                        self.logger.error(f"Error from sacct: {sacct.stderr.strip()}")
                        self.logger.error(f'Job {job_id} status could not be determined.')
                        return 'UNKNOWN'

    def make_folder(self, dir_name: str):
        path = os.path.join(os.getcwd(), dir_name)
        if os.path.exists(path):
            if os.path.isdir(path):
                self.logger.info(f"Removing existing folder {path}")
                shutil.rmtree(path)
            else:
                self.logger.info(f"Removing existing file {path}")
                os.remove(path)
        os.makedirs(path)
        self.logger.info(f"Created folder {path}")

    def submit_job(self, input_file: str, output_file: str, walltime: str = "24") -> Optional[str]:
        command = self.config.submit_command + ["-w", walltime, "-o", output_file, input_file]
        result = self.run_subprocess(command)
        if result:
            for line in result.stdout.splitlines():
                if 'Submitted batch job' in line:
                    job_id = line.split()[-1]
                    self.logger.info(f"Job submitted with ID: {job_id}")
                    return job_id
        self.logger.error(f"Failed to submit job {input_file}")
        sys.exit(1)

    def grep_output(self, pattern: str, file: str) -> str:
        result = self.run_subprocess(f"grep '{pattern}' {file}", shell=True, capture_output=True, check=False, exit_on_error=False)
        return result.stdout.strip() if result else ''

    def optimise_reactants(self, charge: int = 0, mult: int = 1, trial: int = 0, upper_limit: int = 5, solvent: str = "", xtb: bool = True) -> bool:
        trial += 1
        self.logger.info(f"Trial {trial} for reactant optimisation")
        if trial > upper_limit:
            self.logger.error('Too many trials aborting')
            return False

        solvent_formatted = f"ALPB({solvent})" if solvent and xtb else f"CPCM({solvent})" if solvent else ""
        method = "XTB2" if xtb else "r2scan-3c"

        if trial < 2:
            self.make_folder('OPT')
            self.run_subprocess("cp *.xyz OPT", shell=True)
            os.chdir('OPT')

        self.logger.info('Starting reactant optimisation')

        job_inputs = {
            'educt_opt.inp': f"!{method} {solvent_formatted} opt\n%pal nprocs {self.config.slurm_params_low['nprocs']} end\n%maxcore {self.config.slurm_params_low['maxcore']}\n*xyzfile {charge} {mult} educt.xyz\n",
            'product_opt.inp': f"!{method} {solvent_formatted} opt\n%pal nprocs {self.config.slurm_params_low['nprocs']} end\n%maxcore {self.config.slurm_params_low['maxcore']}\n*xyzfile {charge} {mult} product.xyz\n"
        }

        for filename, content in job_inputs.items():
            with open(filename, 'w') as f:
                f.write(content)

        self.logger.info('Submitting educt and product jobs to Slurm')
        job_ids = [self.submit_job(inp, f"{inp.split('.')[0]}_slurm.out") for inp in job_inputs.keys()]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_educt = executor.submit(self.check_job_status, job_ids[0], step='educt')
            future_product = executor.submit(self.check_job_status, job_ids[1], step='product')

            status_educt = future_educt.result()
            status_product = future_product.result()

        self.logger.info(f'Educt Status: {status_educt}')
        self.logger.info(f'Product Status: {status_product}')

        if status_educt == 'COMPLETED' and status_product == 'COMPLETED':
            educt_success = 'HURRAY' in self.grep_output('HURRAY', 'educt_opt.out')
            product_success = 'HURRAY' in self.grep_output('HURRAY', 'product_opt.out')
            if educt_success and product_success:
                os.chdir('..')
                return True

        self.logger.warning('One of the optimisation jobs failed, restarting both')
        for job_id in job_ids:
            self.run_subprocess(['scancel', job_id], exit_on_error=False)
        time.sleep(self.config.retry_delay)
        self.run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm* educt_opt.inp product_opt.inp", shell=True, exit_on_error=False)

        for xyz in ['educt_opt.xyz', 'product_opt.xyz']:
            if os.path.exists(xyz):
                os.rename(xyz, xyz.replace('_opt', ''))

        return self.optimise_reactants(charge, mult, trial, upper_limit, solvent, xtb)

    def freq_job(self, struc_name: str = "coord.xyz", charge: int = 0, mult: int = 1, trial: int = 0, upper_limit: int = 5, solvent: str = "", xtb: bool = False, ts: bool = False) -> bool:
        trial += 1
        self.logger.info(f"Trial {trial} for frequency job")

        if trial > upper_limit:
            self.logger.error('Too many trials aborting')
            return False

        method = "r2scan-3c freq tightscf"
        solvent_formatted = f"CPCM({solvent})" if solvent else ""

        freq_input = f"! {method} {solvent_formatted}\n%pal nprocs {self.config.slurm_params_high['nprocs']} end\n%maxcore {self.config.slurm_params_high['maxcore']}\n*xyzfile {charge} {mult} {struc_name}\n"
        with open('freq.inp', 'w') as f:
            f.write(freq_input)

        self.logger.info('Submitting frequency job to Slurm')
        job_id_freq = self.submit_job("freq.inp", "freq_slurm.out")
        status_freq = self.check_job_status(job_id_freq, step='Freq')

        if status_freq == 'COMPLETED':
            if self.grep_output('VIBRATIONAL FREQUENCIES', 'freq.out'):
                self.logger.info('Frequency calculation completed successfully')
                if ts:
                    output = self.grep_output('**imaginary mode***', 'freq.out')
                    match = re.search(r'(-?\d+\.\d+)\s*cm\*\*-1', output)
                    if match:
                        imag_freq = float(match.group(1))
                        if imag_freq < self.config.freq_threshold:
                            self.logger.info('Negative frequency found, proceeding with TS optimisation')
                            return True
                        else:
                            self.logger.error('Negative frequency above threshold, aborting TS optimisation')
                            return False
                    self.logger.error('No negative frequency found, aborting TS optimisation')
                    return False
                return True

        self.logger.warning('Frequency job failed, restarting...')
        self.run_subprocess(['scancel', job_id_freq], exit_on_error=False)
        self.run_subprocess("rm -rf *.gbw pmix* *densities* freq.inp slurm*", shell=True, exit_on_error=False)
        return self.freq_job(struc_name, charge, mult, trial, upper_limit, solvent, ts)

    def NEB_TS(self, charge: int = 0, mult: int = 1, trial: int = 0, Nimages: int = 16, upper_limit: int = 5, xtb: bool = True, fast: bool = True, solvent: str = "", switch: bool = False) -> bool:
        if switch:
            trial = 0
            os.chdir('..')
            self.logger.info("Switching to DFT run and optimizing reactants")
            if not self.optimise_reactants(charge, mult, trial=0, upper_limit=self.config.max_trials, solvent=solvent, xtb=False):
                self.logger.error("Reactant optimisation failed, aborting")
                return False
            switch = False
        job_step = 'NEB-TS'
        trial += 1
        self.logger.info(f"Trial {trial} for NEB-TS")

        slurm_params = self.config.slurm_params_low

        if trial > upper_limit:
            self.logger.error('Too many trials aborting')
            return False

        solvent_formatted = f"CPCM({solvent})" if solvent and not xtb else f"ALPB({solvent})" if solvent else ""
        method = "FAST-NEB-TS XTB2 tightscf" if fast and xtb else \
                 "FAST-NEB-TS r2scan-3c tightscf" if fast and not xtb else \
                 "NEB-TS XTB2 tightscf" if xtb and not fast else \
                 "NEB-TS r2scan-3c tightscf"

        if trial < 2:
            self.make_folder("NEB")
            self.run_subprocess("cp OPT/educt_opt.xyz NEB/educt.xyz", shell=True)
            self.run_subprocess("cp OPT/product_opt.xyz NEB/product.xyz", shell=True)
            os.chdir('NEB')

        guess_block = ' TS "guess.xyz"\n' if os.path.exists("guess.xyz") else ""
        geom_block = ""
        if xtb:
            with open("product.xyz") as f:
                nAtoms = int(f.readline().strip())
            maxiter = nAtoms * 4
            geom_block = f"%geom\n Calc_Hess true\n Recalc_Hess 1\n MaxIter={maxiter} end \n"
            slurm_params = self.config.slurm_params_low

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

        self.logger.info(f'Submitting {neb_input_name} job to Slurm')
        job_id = self.submit_job(neb_input_name, f"{job_step}_slurm.out", walltime="48")
        status = self.check_job_status(job_id, step="NEB-TS")

        if status == 'COMPLETED' and self.grep_output('HURRAY', f'{neb_input_name.rsplit(".", 1)[0]}.out'):
            self.logger.info('NEB-TS completed successfully')

            imag_freq = self.freq_job(
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
                self.logger.info("Negative mode below threshold found, continuing")
                time.sleep(60)
                self.run_subprocess(
                    f"cp {neb_input_name.rsplit('.', 1)[0]}_NEB-TS_converged.xyz neb_completed.xyz", shell=True)
                os.chdir('..')
                return True
            else:
                if not self.handle_failed_imagfreq(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent):
                    os.chdir('..')
                    return False
                else:
                    self.logger.info("Negative mode found in guess. Continuing with TS optimisation.")
                    time.sleep(60)
                    self.run_subprocess("cp neb-TS_NEB-CI_converged.xyz neb_completed.xyz", shell=True)
                    os.chdir('..')
                    return True
        elif self.grep_output('ORCA TERMINATED NORMALLY', f'{neb_input_name.rsplit(".", 1)[0]}.out'):
            if not self.handle_unconverged_neb(charge, mult, Nimages, trial, upper_limit, xtb, fast, solvent, job_id):
                os.chdir('..')
                return False
            else:
                self.logger.info("R2SCAN-3c NEB-TS job did not converge, but freq job found negative frequency above guess. Fail might be due no Hess recalc. Continuing with TS optimisation.")
                time.sleep(60)
                self.run_subprocess("cp neb-TS_NEB-CI_converged.xyz neb_completed.xyz", shell=True)
                os.chdir('..')
                return True
        else:
            if self.grep_output("TS OPTIMIZATION", f'{neb_input_name.rsplit(".", 1)[0]}.out'):
                self.logger.info("NEB-TS started to optimized a TS guess but aborted during the run")
                self.logger.info("Check whether last struct has significant negative frequency")
                self.run_subprocess(["scancel", job_id], exit_on_error=False)
                self.logger.info("Wait for a minute to wait for the files from the node")
                time.sleep(60)
                struc_name = f'{neb_input_name.rsplit(".", 1)[0]}.xyz'

                imag_freq = self.freq_job(
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
                    self.logger.info("Negative mode below threshold found, Continuing with TS opt")
                    time.sleep(60)
                    self.run_subprocess(["cp", struc_name, "neb_completed.xyz"], shell=True)
                    os.chdir('..')
                    return True

            self.logger.warning(f"NEB-TS {neb_input_name} failed, restarting if there are trials left")
            return self.NEB_TS(charge, mult, trial, Nimages, upper_limit, xtb, fast, solvent, switch)

    def handle_failed_imagfreq(self, charge: int, mult: int, Nimages: int, trial: int, upper_limit: int, xtb: bool, fast: bool, solvent: str) -> bool:
        self.logger.info("No significant negative frequency found")
        if not xtb and fast:
            self.logger.info("FAST-NEB-TS with r2scan-3c did not find TS. Retrying with regular NEB-TS.")
            self.logger.info("Using FAST-NEB-TS guess as TS guess for NEB-TS")
            if os.path.exists("neb-fast-TS_NEB-HEI_converged.xyz"):
                self.run_subprocess("mv neb-fast-TS_NEB-HEI_converged.xyz guess.xyz", shell=True)