#!/usr/bin/env python3
"""
step_runner.py

Contains the StepRunner class with all major pipeline steps 
(e.g., reactant optimization, frequency, NEB, TS, IRC, single point).
"""

import os
import re
import time
import shutil
import concurrent.futures
from typing import Optional


from hpc_driver import HPCDriver  # Import your HPC driver

# Optionally move these to a separate constants file
MAX_TRIALS = 5
FREQ_THRESHOLD = -50
RETRY_DELAY = 60

SLURM_PARAMS_HIGH_MEM = {
    'nprocs': 16,
    'maxcore': 12000
}

SLURM_PARAMS_LOW_MEM = {
    'nprocs': 24,
    'maxcore': 2524
}


class StepRunner:
    """
    Contains the methods for each major pipeline step 
    (e.g., optimization, frequency, NEB, TS, IRC, single point).
    Uses HPCDriver for all HPC interactions.
    """

    def __init__(self, hpc_driver: HPCDriver) -> None:
        self.hpc_driver = hpc_driver

    @staticmethod
    def make_folder(dir_name: str) -> None:
        """
        Creates a new folder, removing existing one if present.
        """
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

    def grep_output(self, pattern: str, file_path: str) -> str:
        """
        Wrapper around grep to return matched lines as a string.
        """
        command = f"grep '{pattern}' {file_path}"
        result = self.hpc_driver.shell_command(command)
        if result and result.stdout:
            return result.stdout.strip()
        return ""

    # ---------- OPTIMIZATION STEP ---------- #
    def optimise_reactants(self,
                           charge: int = 0,
                           mult: int = 1,
                           trial: int = 0,
                           upper_limit: int = 5,
                           solvent: str = "",
                           xtb: bool = True) -> bool:
        """
        Optimize educt and product with XTB or r2scan-3c. Retries if fails.
        """
        trial += 1
        print(f"[OPT] Trial {trial} for reactant optimisation")

        if trial > upper_limit:
            print("Too many trials aborting.")
            return False

        method = "XTB2" if xtb else "r2scan-3c"
        solvent_formatted = ""
        if solvent:
            solvent_formatted = f"ALPB({solvent})" if xtb else f"CPCM({solvent})"

        if trial == 1:
            self.make_folder('OPT')
            self.hpc_driver.shell_command("cp *.xyz OPT")
            os.chdir('OPT')

        print("Starting reactant optimisation")
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

        # Write input files
        for fname, content in job_inputs.items():
            with open(fname, 'w') as f:
                f.write(content)

        # Submit jobs
        job_ids = []
        for inp_file in job_inputs.keys():
            out_file = f"{inp_file.split('.')[0]}_slurm.out"
            job_id = self.hpc_driver.submit_job(inp_file, out_file)
            job_ids.append(job_id)

        # Wait for completion
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(self.hpc_driver.check_job_status, jid) for jid in job_ids]
            statuses = [f.result() for f in futures]

        educt_status, product_status = statuses
        print(f'[OPT] Educt Status: {educt_status}')
        print(f'[OPT] Product Status: {product_status}')

        if educt_status == 'COMPLETED' and product_status == 'COMPLETED':
            educt_success = 'HURRAY' in self.grep_output('HURRAY', 'educt_opt.out')
            product_success = 'HURRAY' in self.grep_output('HURRAY', 'product_opt.out')
            if educt_success and product_success:
                os.chdir('..')
                return True

        # If failed
        print("[OPT] One of the optimisation jobs failed, restarting both.")
        for jid in job_ids:
            self.hpc_driver.scancel_job(jid)

        time.sleep(RETRY_DELAY)
        self.hpc_driver.shell_command("rm -rf *.gbw pmix* *densities* freq.inp slurm* educt_opt.inp product_opt.inp")
        for xyz in ["educt_opt.xyz", "product_opt.xyz"]:
            if os.path.exists(xyz):
                os.rename(xyz, xyz.replace("_opt", ""))

        return self.optimise_reactants(charge, mult, trial, upper_limit, solvent, xtb)

    # ---------- FREQUENCY STEP ---------- #
    def freq_job(self,
                 struc_name: str = "coord.xyz",
                 charge: int = 0,
                 mult: int = 1,
                 trial: int = 0,
                 upper_limit: int = 5,
                 solvent: str = "",
                 xtb: bool = False,
                 ts: bool = False) -> bool:
        """
        Runs a frequency calculation with r2scan-3c freq tightscf.
        If ts=True, check for imaginary freq below FREQ_THRESHOLD.
        """
        trial += 1
        print(f"[FREQ] Trial {trial} on {struc_name}")
        if trial > upper_limit:
            print("[FREQ] Too many trials, aborting.")
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

        job_id_freq = self.hpc_driver.submit_job("freq.inp", "freq_slurm.out")
        status_freq = self.hpc_driver.check_job_status(job_id_freq, step='Freq')

        if status_freq == 'COMPLETED':
            if self.grep_output('VIBRATIONAL FREQUENCIES', 'freq.out'):
                print('[FREQ] Calculation completed successfully.')
                if ts:
                    output = self.grep_output('**imaginary mode***', 'freq.out')
                    match = re.search(r'(-?\d+\.\d+)\s*cm\*\*-1', output)
                    if match:
                        imag_freq = float(match.group(1))
                        if imag_freq < FREQ_THRESHOLD:
                            print('[FREQ] Negative frequency found (TS).')
                            return True
                        else:
                            print('[FREQ] Negative frequency above threshold, not a TS.')
                            return False
                    print('[FREQ] No negative frequency found, not a TS.')
                    return False
                return True

        print('[FREQ] Job failed or no frequencies found. Retrying...')
        self.hpc_driver.scancel_job(job_id_freq)
        self.hpc_driver.shell_command("rm -rf *.gbw pmix* *densities* freq.inp slurm*")
        return self.freq_job(struc_name, charge, mult, trial, upper_limit, solvent, xtb, ts)

    # ---------- NEB-TS STEP ---------- #
    def neb_ts(self,
               charge: int = 0,
               mult: int = 1,
               trial: int = 0,
               Nimages: int = 16,
               upper_limit: int = 5,
               xtb: bool = True,
               fast: bool = True,
               solvent: str = "") -> bool:
        """
        Performs a NEB-TS calculation (optionally FAST) with either XTB or r2scan-3c.
        Checks for convergence with 'HURRAY' and runs a freq job on the converged TS.
        """



        trial += 1
        print(f"[NEB_TS] Trial {trial}, Nimages={Nimages}, xtb={xtb}, fast={fast}")
        if trial > upper_limit:
            print('[NEB_TS] Too many trials aborting.')
            return False

        if not os.path.isdir("OPT"):
            print("[NEB_TS] No 'OPT' directory found. You must run an optimization step first.")
            return False

        # On first NEB trial, create NEB folder if not existing
        if trial == 1:
            self.make_folder("NEB")
            # Copy converged reactant/product from OPT
            self.hpc_driver.shell_command("cp OPT/educt_opt.xyz NEB/educt.xyz")
            self.hpc_driver.shell_command("cp OPT/product_opt.xyz NEB/product.xyz")
            os.chdir("NEB")

        # Decide method
        if solvent and xtb:
            solvent_formatted = f"ALPB({solvent})"
        elif solvent and not xtb:
            solvent_formatted = f"CPCM({solvent})"
        else:
            solvent_formatted = ""

        if fast and xtb:
            method = "FAST-NEB-TS XTB2 tightscf"
        elif fast and not xtb:
            method = "FAST-NEB-TS r2scan-3c tightscf"
        elif not fast and xtb:
            method = "NEB-TS XTB2 tightscf"
        else:
            method = "NEB-TS r2scan-3c tightscf"

        geom_block = ""
        if xtb:
            with open("product.xyz") as f:
                n_atoms = int(f.readline().strip())
            maxiter = n_atoms * 4
            geom_block = f"%geom\n Calc_Hess true\n Recalc_Hess 1\n MaxIter={maxiter} end\n"

        guess_block = ""
        if os.path.exists("guess.xyz"):
            guess_block = ' TS "guess.xyz"\n'

        neb_input = (
            f"! {method} {solvent_formatted}\n"
            f"{geom_block}"
            f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
            f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
            f"%neb\n   Product \"product.xyz\"\n   NImages {Nimages}\n   {guess_block}end\n"
            f"*xyzfile {charge} {mult} educt.xyz\n"
        )

        neb_input_name = "neb-fast-TS.inp" if fast else "neb-TS.inp"
        with open(neb_input_name, 'w') as f:
            f.write(neb_input)

        job_id = self.hpc_driver.submit_job(neb_input_name, "NEB_TS_slurm.out", walltime="48")
        status = self.hpc_driver.check_job_status(job_id, step="NEB_TS")

        out_name = neb_input_name.rsplit(".", 1)[0] + ".out"
        if status == 'COMPLETED' and 'HURRAY' in self.grep_output('HURRAY', out_name):
            print('[NEB_TS] NEB converged successfully. Checking frequency of TS structure...')
            # Typically the NEB TS is in something like: neb-TS_NEB-TS_converged.xyz
            ts_xyz = neb_input_name.rsplit('.', 1)[0] + "_NEB-TS_converged.xyz"
            if os.path.exists(ts_xyz):
                freq_success = self.freq_job(struc_name=ts_xyz, charge=charge, mult=mult, ts=True)
                if freq_success:
                    # Copy the final TS structure to a known file
                    self.hpc_driver.shell_command(f"cp {ts_xyz} neb_completed.xyz")
                    os.chdir('..')
                    return True
                else:
                    print('[NEB_TS] Frequency job indicates no valid TS. Retry or fallback needed.')
            else:
                print(f"[NEB_TS] Could not find converged TS XYZ file: {ts_xyz}")

        else:
            print("[NEB_TS] NEB did not converge or job failed. Retrying if trials remain.")
            self.hpc_driver.scancel_job(job_id)
            # If needed, remove partial files or rename them
            self.hpc_driver.shell_command("rm -rf *.gbw pmix* *densities* freq.inp slurm* *neb*.inp")

        # Retry
        return self.neb_ts(charge, mult, trial, Nimages, upper_limit, xtb, fast, solvent)



    # ---------- NEB-CI STEP ---------- #
    def neb_ci(self,
                  charge: int = 0,
                  mult: int = 1,
                  trial: int = 0,
                  Nimages: int = 8,
                  upper_limit: int = 5,
                  xtb: bool = True,
                  solvent: str = "") -> bool:
           """
           Performs NEB-CI (Climbing Image) calculation. 
           Similar logic to NEB_TS but with !NEB-CI <method>.
           """
           trial += 1
           print(f"[NEB_CI] Trial {trial}, Nimages={Nimages}, xtb={xtb}")
           if trial > upper_limit:
               print('[NEB_CI] Too many trials aborting.')
               return False

           if trial == 1:
               self.make_folder("NEB_CI")
               # Copy from OPT or existing NEB
               self.hpc_driver.shell_command("cp OPT/educt_opt.xyz NEB_CI/educt.xyz")
               self.hpc_driver.shell_command("cp OPT/product_opt.xyz NEB_CI/product.xyz")
               os.chdir("NEB_CI")

           if xtb and solvent:
               solvent_formatted = f"ALPB({solvent})"
           elif solvent and not xtb:
               solvent_formatted = f"CPCM({solvent})"
           else:
               solvent_formatted = ""

           method = "XTB2" if xtb else "r2scan-3c"
           # Possibly free_end if xtb, depends on your usage
           free_end = "Free_end true" if xtb else ""

           neb_input = (
               f"!NEB-CI {solvent_formatted} {method}\n"
               f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
               f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
               f"%neb\n  Product \"product.xyz\"\n  NImages {Nimages} {free_end}\nend\n"
               f"*xyzfile {charge} {mult} educt.xyz\n"
           )

           with open('neb-CI.inp', 'w') as f:
               f.write(neb_input)

           job_id = self.hpc_driver.submit_job("neb-CI.inp", "neb-ci_slurm.out", walltime="48")
           status = self.hpc_driver.check_job_status(job_id, step="NEB-CI")

           if status == 'COMPLETED' and 'H U R R A Y' in self.grep_output('H U R R A Y', 'neb-CI.out'):
               print('[NEB_CI] Completed successfully.')
               os.chdir('..')
               return True
           else:
               print('[NEB_CI] Job failed or did not converge. Retrying...')
               self.hpc_driver.scancel_job(job_id)
               time.sleep(RETRY_DELAY)
               self.hpc_driver.shell_command("rm -rf *.gbw pmix* *densities* freq.inp slurm* neb*im* *neb*.inp")
               return self.neb_ci(charge, mult, trial, Nimages, upper_limit, xtb, solvent)

    # ---------- TS Optimization STEP ---------- #
    def ts_opt(self,
               charge: int = 0,
               mult: int = 1,
               trial: int = 0,
               upper_limit: int = 5,
               solvent: str = "") -> bool:
        """
        Takes a guessed TS (e.g., from NEB) and optimizes it with r2scan-3c OptTS.
        Checks for 'HURRAY' in output. Retries if fails.
        """
        trial += 1
        print(f"[TS_OPT] Trial {trial} for TS optimization")
        if trial > upper_limit:
            print("[TS_OPT] Too many trials, aborting.")
            return False

        if trial == 1:
            self.make_folder("TS")
            if not os.path.exists("NEB/neb_completed.xyz"):
                print("[TS_OPT] No 'neb_completed.xyz' found from NEB. Exiting.")
                return False
            self.hpc_driver.shell_command("cp NEB/neb_completed.xyz TS/ts_guess.xyz")
            os.chdir("TS")

        # Always run freq job on the guess to ensure negative frequency
        if not self.freq_job(struc_name="ts_guess.xyz", charge=charge, mult=mult, ts=True):
            print("[TS_OPT] Frequency job indicates no negative frequency. Aborting.")
            return False

        solvent_formatted = f"CPCM({solvent})" if solvent else ""

        ts_input = (
            f"!r2scan-3c OptTS tightscf {solvent_formatted}\n"
            f"%geom\ninhess read\ninhessname \"freq.hess\"\nCalc_Hess true\nrecalc_hess 15\nend\n"
            f"%pal nprocs {SLURM_PARAMS_HIGH_MEM['nprocs']} end\n"
            f"%maxcore {SLURM_PARAMS_HIGH_MEM['maxcore']}\n"
            f"*xyzfile {charge} {mult} ts_guess.xyz\n"
        )

        with open("TS_opt.inp", "w") as f:
            f.write(ts_input)

        job_id = self.hpc_driver.submit_job("TS_opt.inp", "TS_opt_slurm.out", walltime="48")
        status = self.hpc_driver.check_job_status(job_id, step="TS_OPT")

        if status == 'COMPLETED' and 'HURRAY' in self.grep_output('HURRAY', 'TS_opt.out'):
            print("[TS_OPT] TS optimization succeeded.")
            os.chdir("..")
            return True

        print("[TS_OPT] TS optimization failed. Retrying...")
        self.hpc_driver.scancel_job(job_id)
        time.sleep(RETRY_DELAY)
        self.hpc_driver.shell_command("rm -rf *.gbw pmix* *densities* freq.inp slurm* *.hess")
        if os.path.exists('TS_opt.xyz'):
            os.rename('TS_opt.xyz', 'ts_guess.xyz')

        return self.ts_opt(charge, mult, trial, upper_limit, solvent)

    def irc_job(self,
                charge: int = 0,
                mult: int = 1,
                trial: int = 0,
                upper_limit: int = 5,
                solvent: str = "",
                maxiter: int = 70) -> bool:
        """
        Runs an Intrinsic Reaction Coordinate (IRC) calculation from an optimized TS.
        Checks for 'HURRAY' in 'IRC.out'. Retries with more steps if needed.
        """
        trial += 1
        print(f"[IRC] Trial {trial} with maxiter={maxiter}")
        if trial > upper_limit:
            print("[IRC] Too many trials, aborting.")
            return False

        if trial == 1:
            self.make_folder("IRC")
            if not os.path.exists("TS/TS_opt.xyz"):
                print("[IRC] TS_opt.xyz not found in TS folder. Exiting.")
                return False
            self.hpc_driver.shell_command("cp TS/TS_opt.xyz IRC/")
            os.chdir("IRC")

            # Run freq job to ensure negative frequency for TS
            if not self.freq_job(struc_name="TS_opt.xyz", charge=charge, mult=mult, ts=True):
                print("[IRC] TS frequency job invalid. Aborting IRC.")
                return False

        solvent_formatted = f"CPCM({solvent})" if solvent else ""

        irc_input = (
            f"!r2scan-3c IRC tightscf {solvent_formatted}\n"
            f"%irc\n  maxiter {maxiter}\n  InitHess read\n  Hess_Filename \"freq.hess\"\nend\n"
            f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
            f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
            f"*xyzfile {charge} {mult} TS_opt.xyz\n"
        )

        with open("IRC.inp", "w") as f:
            f.write(irc_input)

        job_id = self.hpc_driver.submit_job("IRC.inp", "IRC_slurm.out")
        status = self.hpc_driver.check_job_status(job_id, step="IRC")

        if status == 'COMPLETED' and 'HURRAY' in self.grep_output('HURRAY', 'IRC.out'):
            print("[IRC] IRC completed successfully.")
            os.chdir('..')
            return True

        print("[IRC] IRC job did not converge or failed. Attempting restart with more steps.")
        self.hpc_driver.scancel_job(job_id)
        if 'ORCA TERMINATED NORMALLY' in self.grep_output('ORCA TERMINATED NORMALLY', 'IRC.out'):
            if maxiter < 100:
                maxiter += 30
            else:
                print("[IRC] Max iteration limit reached. Aborting.")
                return False

        time.sleep(RETRY_DELAY)
        self.hpc_driver.shell_command("rm -rf *.gbw pmix* *densities* IRC.inp slurm*")
        return self.irc_job(charge, mult, trial, upper_limit, solvent, maxiter)

    # ---------- SINGLE POINT (SP) STEP ---------- #
    def sp_calc(self,
                charge: int = 0,
                mult: int = 1,
                trial: int = 0,
                upper_limit: int = 5,
                solvent: str = "",
                method: str = "r2scanh",
                basis: str = "def2-QZVPP") -> bool:
        """
        Runs a high-level single-point calculation on a TS or other structure (e.g., TS_opt.xyz).
        """
        trial += 1
        print(f"[SP] Trial {trial} with {method}/{basis}")
        if trial > upper_limit:
            print("[SP] Too many trials, aborting.")
            return False

        if trial == 1:
            self.make_folder("SP")
            if not os.path.exists("TS/TS_opt.xyz"):
                print("[SP] TS_opt.xyz not found. Exiting.")
                return False
            self.hpc_driver.shell_command("cp TS/TS_opt.xyz SP/")
            os.chdir("SP")

        solvent_formatted = f"CPCM({solvent})" if solvent else ""

        sp_input = (
            f"!{method} {basis} {solvent_formatted} verytightscf defgrid3 d4\n"
            f"%pal nprocs {SLURM_PARAMS_LOW_MEM['nprocs']} end\n"
            f"%maxcore {SLURM_PARAMS_LOW_MEM['maxcore']}\n"
            f"*xyzfile {charge} {mult} TS_opt.xyz\n"
        )

        with open("SP.inp", "w") as f:
            f.write(sp_input)

        job_id = self.hpc_driver.submit_job("SP.inp", "SP_slurm.out")
        status = self.hpc_driver.check_job_status(job_id, step="SP")

        # Some users look for "FINAL SINGLE POINT ENERGY" in SP.out 
        # or a "HURRAY" in ORCA's final lines. Adjust accordingly.
        if status == 'COMPLETED' and "HURRAY" in self.grep_output("HURRAY", "SP.out"):
            print("[SP] Single point completed successfully.")
            os.chdir('..')
            return True

        print("[SP] Failed or not converged. Retrying...")
        self.hpc_driver.scancel_job(job_id)
        time.sleep(RETRY_DELAY)
        self.hpc_driver.shell_command("rm -rf *.gbw pmix* *densities* SP.inp slurm*")
        return self.sp_calc(charge, mult, trial, upper_limit, solvent, method, basis)



   