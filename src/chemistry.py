# Standard library imports
from typing import List, Union, Tuple
import numpy as np
import os
import time
import re
import shutil
import concurrent.futures

# Own imports
from constants import MAX_TRIALS, RETRY_DELAY, FREQ_THRESHOLD

from hpc_driver import HPCDriver

# TODO put functions from step_runner here. Step Runer is just and orchestrator of the steps.


def read_xyz(filepath: str) -> Tuple[List[str], np.ndarray]:
    """
    Reads an XYZ file and returns a list of atoms and a 3xN NumPy array of coordinates.
    """
    atoms = []
    coords = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"XYZ file not found: {filepath}")

    with open(filepath, 'r') as file:
        lines = file.readlines()
        if len(lines) < 3:
            raise ValueError(
                f"XYZ file {filepath} is too short to contain atomic data.")

        # Skip the first two lines (header)
        for idx, line in enumerate(lines[2:], start=3):
            parts = line.strip().split()
            if len(parts) < 4:
                print(
                    f"Warning: Skipping invalid line {idx} in {filepath}: '{line.strip()}'")
                continue
            atom, x, y, z = parts[:4]
            try:
                coords.append([float(x), float(y), float(z)])
            except ValueError as e:
                print(
                    f"Warning: Invalid coordinates on line {idx} in {filepath}: {e}")
                continue
            atoms.append(atom)

    if not atoms or not coords:
        raise ValueError(f"No valid atomic data found in {filepath}.")

    coords = np.array(coords)  # Transpose to get a Nx3 array
    return atoms, coords


class Molecule:
    """
    Represents a molecule.


    """

    def __init__(self, name: str, atoms: List[str], coords: np.ndarray, mult: int, charge: int, solvent: str = None, method: str = "r2scan-3c", sp_method: str = "r2scanh def2-qzvpp d4") -> None:
        self.name = name
        self.atoms = atoms
        self.coords = coords
        self.mult = mult
        self.charge = charge
        self.solvent = solvent

        # method functional, basis set, [opt] Dispersion correction, composite methods are also supported (e.g r2scan-3c)
        self.method = method
        self.sp_method = sp_method

        if len(atoms) != len(coords):
            print(len(atoms), len(coords))
            raise ValueError("Number of atoms and coordinates must match.")

        if mult < 1:
            raise ValueError("Multiplicity must be at least 1.")

    @classmethod
    def from_xyz(cls, filepath: str, charge: int, mult: int, solvent: str = None, name: str = "mol", method: str = "r2scan-3c", sp_method="r2scanh def2-qzvpp d4") -> 'Molecule':
        """
        Creates a Molecule instance from an XYZ file.
        """
        if name == None:
            name = os.path.basename(filepath).split('.')[0]
        atoms, coords = read_xyz(filepath)
        return cls(name, atoms, coords, charge=charge, mult=mult, solvent=solvent, method=method, sp_method=sp_method)

    def __str__(self) -> str:
        return f"Molecule(name={self.name},atoms={self.atoms}, coordinates={self.coords}, Multiplicity={self.mult}, Charge={self.charge})"

    def __repr__(self) -> str:
        return f"Molecule(name={self.name},atoms={self.atoms}, coordinates={self.coords}, Multiplicity={self.mult}, Charge={self.charge})"

    def translate(self, vector: List[float]) -> None:
        """
        Translates the molecule by the given vector.
        """
        self.coords += np.array(vector).reshape(3, 1)

    def rotate(self, rotation_matrix: np.ndarray) -> None:
        """
        Rotates the molecule using the given 3x3 rotation matrix.
        """
        self.coords = np.dot(rotation_matrix, self.coords)

    def to_xyz(self, filepath: str = None) -> None:
        """
        Writes the molecule to an XYZ file.
        """
        if filepath is None:
            current_dir = os.getcwd()
            filepath = os.path.join(current_dir, f"{self.name}.xyz")

        with open(filepath, 'w') as file:
            file.write(f"{len(self.atoms)}\n")
            file.write("\n")
            for atom, coord in zip(self.atoms, self.coords):
                file.write(
                    f"{atom} {coord[0]:.9f} {coord[1]:.9f} {coord[2]:.9f}\n")

    def update_coords_from_xyz(self, filepath: str) -> None:
        """
        Updates the coordinates of the molecule from an XYZ file.
        """
        atoms, coords = read_xyz(filepath)
        self.atoms = atoms
        self.coords = coords

    def get_xyz_block(self) -> str:
        xyz_block = ""
        for atom, coord in zip(self.atoms, self.coords):
            xyz_block += f"{atom}  {coord[0]:.9f} {coord[1]:.9f} {coord[2]:.9f} \n"
        return xyz_block

    def geometry_opt(self, driver: HPCDriver, slurm_params: dict, trial: int = 1, upper_limit: int = MAX_TRIALS) -> bool:
        """
        Optimises the geometry of the molecule.
        returns True if the optimisation was successful, False otherwise.
        If successful, the molecule's coordinates are updated.



        """

        trial += 1
        print(f"[OPT] Trial {trial} for reactant optimisation")

        if trial > upper_limit:
            print("Too many trials aborting.")
            return False

        solvent_formatted = ""

        if self.solvent:
            if "xtb" in self.method.lower():
                solvent_formatted = f"ALPB({self.solvent})"
            else:
                solvent_formatted = f"CPCM({self.solvent})"

        if trial == 1:
            self.to_xyz(f"{self.name}_start.xyz")

        print(f"Starting optimisation of {self.name}")

        input_name = f"{self.name}_opt.inp"

        input = (
            f"!{self.method} {solvent_formatted} opt\n"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyz {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )

        # Write input files

        with open(input_name, 'w') as f:
            f.write(input)

        # Submit jobs

        out_file = f"{self.name}_slurm.out"
        job_id = driver.submit_job(f"{input_name}", out_file)
        status = driver.check_job_status(job_id)

        # Wait for completion

        if status == 'COMPLETED':
            if 'HURRAY' in driver.grep_output('HURRAY', input_name.split('.')[0] + '.out'):
                print("[OPT] Optimisation jobs completed.")

                self.update_coords_from_xyz(f"{input_name.split('.')[0]}.xyz")

                # directories are handle by step_runner or custom script.
                return True

        print("[OPT] Optimisation jobs failed. Retrying...")

        driver.scancel_job(job_id)
        time.sleep(RETRY_DELAY)
        driver.shell_command(
            "rm -rf *.gbw pmix* *densities*  slurm* ")
        self.update_coords_from_xyz(f"{input_name.split('.')[0]}.xyz")
        return self.geometry_opt(driver=driver, slurm_params=slurm_params, trial=trial, upper_limit=upper_limit)

    def freq_job(self, driver: HPCDriver, slurm_params: dict, trial: int = 0, upper_limit: int = MAX_TRIALS, ts: bool = False) -> bool:
        """
        Runs a frequency calculation with the given method with tightscf.
        If ts=True, check for imaginary freq below FREQ_THRESHOLD.
        """
        trial += 1
        print(f"[FREQ] Trial {trial} on {self.name}")
        if trial > upper_limit:
            print("[FREQ] Too many trials, aborting.")
            return False

        solvent_formatted = ""

        if self.solvent:

            if "xtb" in self.method.lower():
                solvent_formatted = f"ALPB({self.solvent})" if self.solvent else ""
            else:
                solvent_formatted = f"CPCM({self.solvent})" if self.solvent else ""

          # iterate over the atmos and coords to get the xyz block
        freq_input = (
            f"! {self.method} freq tightscf {solvent_formatted}\n"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyz {self.charge} {self.mult}\n"
            f"{self.get_xyz_block()}  *"
        )
        with open(f'{self.name}_freq.inp', 'w') as f:
            f.write(freq_input)

        job_id_freq = driver.submit_job(
            f"{self.name}_freq.inp", f"{self.name}_freq_slurm.out")
        status_freq = driver.check_job_status(
            job_id_freq, step='Freq')

        if status_freq == 'COMPLETED':
            if driver.grep_output('VIBRATIONAL FREQUENCIES', f'{self.name}_freq.out'):
                print('[FREQ] Calculation completed successfully.')
                if ts:

                    output = driver.grep_output(
                        '**imaginary mode***', f'{self.name}_freq.out')
                    match = re.search(r'(-?\d+\.\d+)\s*cm\*\*-1', output)
                    if match:
                        imag_freq = float(match.group(1))

                        if imag_freq < FREQ_THRESHOLD:
                            print('[FREQ] Negative frequency found (TS).')
                            return True
                        else:
                            print(
                                '[FREQ] Negative frequency above threshold, not a TS.')
                            return False
                    print('[FREQ] No negative frequency found, not a TS.')
                    return False
                return True

        print('[FREQ] Job failed or no frequencies found. Retrying...')
        driver.scancel_job(job_id_freq)
        driver.shell_command(
            "rm -rf *.gbw pmix* *densities*  slurm*")
        return self.freq_job(driver=driver, slurm_params=slurm_params, trial=trial, upper_limit=upper_limit, ts=ts)

    def ts_opt(self, driver: HPCDriver, slurm_params: dict, trial: int = 0, upper_limit: int = MAX_TRIALS) -> bool:
        """
        Takes a guessed TS (e.g., from NEB) and optimizes it with selected method.
        Checks for 'HURRAY' in output. Retries if fails.
        Looks for guess.hess before starting the run, if not present new hess is calculated.
        """

        trial += 1
        print(f"[TS_OPT] Trial {trial} for TS optimization")
        if trial > upper_limit:
            print("[TS_OPT] Too many trials, aborting.")
            return False

        if "xtb" in self.method.lower():
            print(
                "TS optimization will not be conducted with semiemporical methods switching to r2scan-3c .")
            self.method = "r2scan-3c"
            print("Calculate Hess")
            if not self.freq_job(driver=driver, slurm_params=slurm_params, ts=True):
                print("Guess has no significant imaginary frequency. Aborting.")
                return False
            else:
                shutil.move(f'{self.name}_freq.hess',
                            f'{self.name}_guess.hess')

            # step_runner handles hess copying. If run from other script, copy hess file to current directory.
        if not os.path.exists(f'{self.name}_guess.hess'):
            print("Hessian file not found. Doing freq job on guess.")
            if not self.freq_job(driver=driver, slurm_params=slurm_params, ts=True):
                print("Guess has no significant imaginary frequency. Aborting.")
                return False
            else:
                shutil.move(f'{self.name}_freq.hess',
                            f'{self.name}_guess.hess')

        if trial == 1:
            self.to_xyz("ts_guess.xyz")

        # TODO use tightopt settings for TS optimization
        solvent_formatted = f"CPCM({self.solvent})" if self.solvent else ""

        ts_input = (
            f"!{self.method} OptTS tightscf {solvent_formatted}\n"
            f"%geom\ninhess read\ninhessname \"{self.name}_guess.hess\"\nCalc_Hess true\n recalc_hess 15\n end\n"
            f"%pal nprocs {slurm_params['nprocs']} end \n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyz {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )

        with open(f"{self.name}_TS_opt.inp", "w") as f:
            f.write(ts_input)

        job_id = driver.submit_job(
            f"{self.name}_TS_opt.inp", f"{self.name}_TS_opt_slurm.out", walltime="48")
        status = driver.check_job_status(job_id, step="TS_OPT")

        if status == 'COMPLETED' and 'HURRAY' in driver.grep_output('HURRAY', f'{self.name}_TS_opt.out'):
            print("[TS_OPT] TS optimization succeeded.")
            self.update_coords_from_xyz(
                f"{self.name}_TS_opt.xyz")
            self.name = "ts"

            if self.freq_job(driver, slurm_params, ts=True):

                return True
            else:
                print(
                    "[TS_OPT] TS has no significant imaginary frequency. Check negative mode of guess.")
                self.update_coords_from_xyz("ts_guess.xyz")
                self.name = "ts_guess"

                return False

        print("[TS_OPT] TS optimization failed. Retrying...")
        driver.scancel_job(job_id)
        time.sleep(RETRY_DELAY)
        driver.shell_command(
            "rm -rf *.gbw pmix* *densities*  slurm* *.hess")
        if os.path.exists(f'{self.name}_TS_opt.xyz'):
            self.update_coords_from_xyz(f'{self.name}_TS_opt.xyz')

        return self.ts_opt(trial=trial, upper_limit=upper_limit)

    def irc_job(self, driver: HPCDriver, slurm_params: dict, trial: int = 0, upper_limit: int = 5, maxiter: int = 70) -> bool:
        """
        Runs an Intrinsic Reaction Coordinate (IRC) calculation from an optimized TS.
        Checks for 'HURRAY' in 'IRC.out'. Retries with more steps if needed.
        """

        if self.method.lower() == "xtb":
            print(
                "IRC calculation will not be conducted with semiemporical methods. Switching to r2scan-3c. If TS was optimised with other method indicate in the input.")

            self.method = "r2scan-3c"
        trial += 1
        print(f"[IRC] Trial {trial} with maxiter={maxiter}")
        if trial > upper_limit:
            print("[IRC] Too many trials, aborting.")
            return False

        if trial == 1:

            if not os.path.exists(f'{self.name}_freq.hess'):
                print("Hessian file not found. Doing freq job on guess.")
                if not self.freq_job(driver=driver, slurm_params=slurm_params, ts=True):
                    print("Guess has no significant imaginary frequency. Aborting.")

            self.to_xyz(f"{self.name}_TS_opt.xyz")

            # Run freq job to ensure negative frequency for TS
            if not self.freq_job(driver=driver, slurm_params=slurm_params, ts=True):
                print("[IRC] TS frequency job invalid. Aborting IRC.")
                return False

        solvent_formatted = f"CPCM({self.educt.solvent})" if self.educt.solvent else ""

        irc_input = (
            f"!{self.method} IRC tightscf {solvent_formatted}\n"
            f"%irc\n  maxiter {maxiter}\n  InitHess read\n  Hess_Filename \"{self.name}_freq.hess\"\nend\n"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyzfile {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )

        with open(f"{self.name}_IRC.inp", "w") as f:
            f.write(irc_input)

        job_id = driver.submit_job(
            f"{self.name}_IRC.inp", f"{self.name}_IRC_slurm.out")
        status = driver.check_job_status(job_id, step="IRC")

        if status == 'COMPLETED' and 'HURRAY' in driver.grep_output('HURRAY', f'{self.name}_IRC.out'):
            print("[IRC] IRC completed successfully.")

            return True

        print(
            "[IRC] IRC job did not converge or failed. Attempting restart with more steps.")
        driver.scancel_job(job_id)
        if 'ORCA TERMINATED NORMALLY' in driver.grep_output('ORCA TERMINATED NORMALLY', f'{self.name}_IRC.out'):
            if maxiter < 100:
                maxiter += 30
            else:
                print("[IRC] Max iteration limit reached. Aborting.")
                return False

        time.sleep(RETRY_DELAY)
        driver.shell_command(
            "rm -rf *.gbw pmix* *densities*  slurm*")
        return self.irc_job(driver=driver, slurm_params=slurm_params, trial=trial, upper_limit=upper_limit, maxiter=maxiter)

    # ---------- SINGLE POINT (SP) STEP ----------

    def sp_calc(self,
                driver: HPCDriver,
                slurm_params: dict,
                trial: int = 0,
                upper_limit: int = 5,
                ) -> bool:
        """
        Runs a single point calculation on the molecule iwth the sp_method given.
        Default is r2scanh def2-qzvpp d4.

        """

        solvent_formatted = f"CPCM({self.solvent})" if self.solvent else ""

        trial += 1
        print(f"[SP] Trial {trial} ")
        if trial > upper_limit:
            print("[SP] Too many trials, aborting.")
            return False

        if trial == 1:
            self.to_xyz(f"{self.name}_sp.xyz")

        sp_input = (
            f"!{self.method} {solvent_formatted} verytightscf defgrid3 \n"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyzfile {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )
        with open(f"{self.name}_SP.inp", "w") as f:
            f.write(sp_input)
        job_id = driver.submit_job(
            f"{self.name}_SP.inp", f"{self.name}SP_slurm.out")

        status = driver.check_job_status(job_id, step="SP")

        if status == 'COMPLETED' and 'FINAL SINGLE POINT ENERGY' in driver.grep_output('FINAL SINGLE POINT ENERGY', f'{self.name}SP.out'):
            print("[SP] Single point calculation completed successfully.")
            return True

        print("[SP] Failed or not converged. Retrying...")
        driver.scancel_job(job_id)
        time.sleep(RETRY_DELAY)
        driver.shell_command(
            "rm -rf *.gbw pmix* *densities* SP.inp slurm*")
        return self.sp_calc(driver=driver, slurm_params=slurm_params, trial=trial, upper_limit=upper_limit)

    def get_lowest_confomers(self, driver: HPCDriver, slurm_params: dict, trial: int = 0, upper_limit: int = 5) -> bool:

        trial += 1
        print(f"[CREST] Trial {trial} ")
        if trial > upper_limit:
            print("[CREST] Too many trials, aborting.")
            return False

        print(f"[CREST] Generating conformers for {self.name}")

        print("CREST needs xtb2 optimized structures.")

        self.to_xyz(f"{self.name}_before_crest.xyz")

        # for now do everything from commnand since orca can have some problems which xtb does not have
        solvent = f"--alpb {self.solvent}" if self.solvent else ""

        driver.shell_command(
            f"xtb {self.name}_before_crest.xyz --opt --charge {self.charge} --uhf {self.mult -1} {solvent}")

        if not os.path.exists("xtbopt.xyz"):
            print("Optimization failed. Aborting.")
            return False

        """
        generates crest ensembel
        """


class Reaction:
    """
    Represents a elementary step in a reaction.
    """

    def __init__(self, educt: Molecule, product: Molecule, transition_state: Molecule = None, nimages: int = 16, method: str = "r2scan-3c", sp_method="r2scanh def2-qzvpp d4", solvent="", name="reaction", fast: bool = False, zoom: bool = False) -> None:
        self.educt = educt
        self.product = product
        self.transition_state = transition_state
        self.nimages = nimages
        self.method = method
        self.sp_method = sp_method
        self.solvent = solvent
        self.name = name
        self.fast = fast
        self.zoom = zoom
        self.charge = educt.charge
        self.mult = educt.mult

        if educt.charge != product.charge:
            raise ValueError("Charge of educt and product must match.")

        if educt.mult != product.mult:
            raise ValueError("Exicited state reactions are not supported yet.")

        # TODO validation of object

    def __str__(self) -> str:
        reactant_strs = f"{self.educt} \n"
        product_strs = f"{self.product}\n"
        if self.transition_state:
            ts_str = f"TS: {self.transition_state}\n"
            return f"{reactant_strs} => {ts_str} => {product_strs}"
        return f"{reactant_strs} => {product_strs}"

    def __repr__(self) -> str:
        reactant_strs = f"{self.educt}\n"
        product_strs = f"{self.product}\n"
        return f"{reactant_strs} => {product_strs}"

    def __repr__(self) -> str:
        reactant_strs = " + ".join(f"{self.educt}")
        product_strs = " + ".join(self.product)
        return f"{reactant_strs} => {product_strs}"

    @ classmethod
    def from_xyz(cls, educt_filepath: str, product_filepath: str, transition_state_filepath: str = None, nimages: int = 16, method: str = "r2scan-3c", charge: int = 0, mult: int = 1, solvent: str = None, sp_method="r2scanh def2-qzvpp d4", name: str = "reaction", fast: bool = False, zoom: bool = False) -> 'Reaction':
        """
        Creates an ElementaryStep instance from XYZ files.
        """
        educt = Molecule.from_xyz(educt_filepath, charge=charge,
                                  mult=mult, solvent=solvent, method=method, name="educt")
        product = Molecule.from_xyz(product_filepath, charge=charge,
                                    mult=mult, solvent=solvent, method=method, name="product")
        transition_state = Molecule.from_xyz(transition_state_filepath, charge=charge, mult=mult,
                                             solvent=solvent, name="ts", method=method) if transition_state_filepath else None
        return cls(educt=educt, product=product, transition_state=transition_state, nimages=nimages, method=method, solvent=solvent, sp_method=sp_method, name=name, fast=fast, zoom=zoom)

    def optimise_reactants(self, driver: HPCDriver, slurm_params: dict, trial: int = 0, upper_limit: int = MAX_TRIALS) -> bool:
        """
        Optimises the reactant geometry.
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = [
                executor.submit(self.educt.geometry_opt, driver,
                                slurm_params, trial, upper_limit),
                executor.submit(self.product.geometry_opt, driver,
                                slurm_params, trial, upper_limit)
            ]

        return all([result.result() for result in results])

    def neb_ci(self, driver: HPCDriver, slurm_params: dict,  trial=0, upper_limit: int = MAX_TRIALS,) -> bool:
        """
        Performs NEB-CI (Climbing Image) calculation.
        Assums that the educt and products are already optimized with the same method
        """
        trial += 1
        print(
            f"[NEB_CI] Trial {trial}, Nimages={self.nimages}, Method={self.method}")
        if trial > upper_limit:
            print('[NEB_CI] Too many trials aborting.')
            return False

        if trial == 1:

            self.educt.to_xyz("educt.xyz")
            self.product.to_xyz("product.xyz")
        solvent_formatted = ""
        if self.solvent:

            solvent_formatted = f"ALPB({self.solvent})" if "xtb" in self.method.lower(
            ) else f"CPCM({self.solvent})"

        neb_method = "NEB-CI" if not self.zoom else "ZOOM-NEB-CI"

        neb_input = (
            f"! {neb_method} {self.method} {solvent_formatted}  tightscf\n"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"%neb\n  Product \"product.xyz\"\n  NImages {self.nimages} \nend\n"
            f"*xyzfile {self.charge} {self.mult} educt.xyz\n"
        )

        with open(f'{self.name}_neb-CI.inp', 'w') as f:
            f.write(neb_input)

        job_id = driver.submit_job(
            f"{self.name}_neb-CI.inp", f"{self.name}_neb-ci_slurm.out", walltime="72")
        status = driver.check_job_status(job_id, step="NEB-CI")

        if status == 'COMPLETED' and 'H U R R A Y' in driver.grep_output('H U R R A Y', 'neb-CI.out'):
            print('[NEB_CI] Completed successfully.')
            time.sleep(20)
            pot_ts = Molecule.from_xyz(
                f"{self.name}_neb-CI_NEB-CI_converged.xyz", charge=self.charge, mult=self.mult, solvent=self.solvent, method="r2scan-3c", name="ts_guess")

            slurm_params_freq = slurm_params.copy()
            # a bit conservative maybe double is enough
            slurm_params_freq['maxcore'] = slurm_params['maxcore']*4
            freq_success = pot_ts.freq_job(
                driver, slurm_params=slurm_params_freq, ts=True)
            if freq_success:
                self.transition_state = pot_ts
                return True
            else:
                print("TS has no significant imag freq Retry with refined method")
                if self.nimages == 24:
                    print("Max number of images reached ")

                    return False
                driver.scancel_job(job_id)
                time.sleep(RETRY_DELAY)
                driver.shell_command(
                    "rm -rf *.gbw pmix* *densities* freq.inp slurm* *neb*.inp")
                self.nimages += 4
                return self.neb_ci(trial, upper_limit)
        else:
            print('[NEB_CI] Job failed or did not converge. Retrying...')
            driver.scancel_job(job_id)
            time.sleep(RETRY_DELAY)
            driver.shell_command(
                "rm -rf *.gbw pmix* *densities* freq.inp slurm* neb*im* *neb*.inp")
            return self.neb_ci(driver=driver, slurm_params=slurm_params, trial=trial, upper_limit=upper_limit)

    def neb_ts(self,
               driver: HPCDriver,
               slurm_params: dict,
               trial=0,
               upper_limit: int = MAX_TRIALS
               ) -> Tuple[str, bool]:
        """
        Performs a NEB-TS calculation (optionally FAST) with either XTB or r2scan-3c.
        Checks for convergence with 'HURRAY' and runs a freq job on the converged TS.


        FAST method with xtb is discouraged as xtb2 is already fast enough
        """

        trial += 1
        print(
            f"[NEB_TS] Trial {trial}, Nimages={self.nimages}, method={self.method}, fast={self.fast}")
        if trial > upper_limit:
            print('[NEB_TS] Too many trials aborting.')
            return False

        # On first NEB trial, create NEB folder if not existing
        if trial == 1:

            self.educt.to_xyz("educt.xyz")
            self.product.to_xyz("product.xyz")

        geom_block = ""

        if "xtb" in self.method.lower():
            maxiter = len(self.educt.atoms) * 4
            geom_block = f"%geom\n Calc_Hess true\n Recalc_Hess 1\n MaxIter={maxiter} end\n"
        neb_block = "Fast-NEB-TS" if self.fast else "NEB-TS"
        neb_block = "ZOOM-NEB-TS" if self.zoom else neb_block

        solvent_formatted = ""
        if self.solvent:
            solvent_formatted = f"ALPB({self.solvent})" if "xtb" in self.method.lower(
            ) else f"CPCM({self.solvent})"
        neb_input = (
            f"! {self.method} {neb_block} {solvent_formatted} tightscf\n"
            f"{geom_block}"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"%neb\n   Product \"product.xyz\"\n   NImages {self.nimages}\n  end\n"
            f"*xyzfile {self.charge} {self.mult} educt.xyz\n"
        )

        neb_input_name = "neb-fast-TS.inp" if self.fast else "neb-TS.inp"
        with open(neb_input_name, 'w') as f:
            f.write(neb_input)

        job_id = driver.submit_job(
            neb_input_name, "NEB_TS_slurm.out", walltime="72")
        status = driver.check_job_status(job_id, step="NEB_TS")

        out_name = neb_input_name.rsplit(".", 1)[0] + ".out"
        if status == 'COMPLETED' and 'HURRAY' in driver.grep_output('HURRAY', out_name):
            print(
                '[NEB_TS] NEB converged successfully. Checking frequency of TS structure...')
            # Typically the NEB TS is in something like: neb-TS_NEB-TS_converged.xyz
            ts_xyz = neb_input_name.rsplit('.', 1)[0] + "_NEB-TS_converged.xyz"
            time.sleep(RETRY_DELAY)

            # TODO we need a retry mechanism here if ssubo is to slow.
            if os.path.exists(ts_xyz):
                potential_ts = Molecule.from_xyz(
                    ts_xyz, charge=self.charge, mult=self.mult, solvent=self.solvent, name="ts_guess", method=self.method, sp_method=self.sp_method)
                slurm_params_freq = slurm_params
                slurm_params_freq["maxcore"] = slurm_params_freq["maxcore"]*4
                freq_success = potential_ts.freq_job(
                    driver=driver, slurm_params=slurm_params_freq, ts=True)
                if freq_success:
                    # Copy the final TS structure to a known file
                    self.transition_state = potential_ts

                    return "", True
                else:
                    print("TS has no significant imag freq Retry with refined method")
                    driver.scancel_job(job_id)
                    time.sleep(RETRY_DELAY)
                    driver.shell_command(
                        "rm -rf *.gbw pmix* *densities* freq.inp slurm* *neb*.inp")

                    return "failed", False

        elif driver.grep_output('ORCA TERMINATED NORMALLY', f'{neb_input_name.rsplit(".", 1)[0]}.out'):
            print("ORCA has terminated normally, optimisation did not converge.")
            print("Restart neb with more precise settings, use ")
            driver.scancel_job(job_id)
            # If needed, remove partial files or rename them
            driver.shell_command(
                "rm -rf *.gbw pmix* *densities* freq.inp slurm* *neb*.inp")

            return "failed", False
  # already changed dir
        print("There was an error during the run, Restart with the same settings")
        driver.scancel_job(job_id)
        time.sleep(RETRY_DELAY)
        driver.shell_command(
            "rm -rf *.gbw pmix* *densities* freq.inp slurm* *neb*.inp")

        return self.neb_ts(driver=driver, slurm_params=slurm_params, trial=trial, upper_limit=upper_limit)
