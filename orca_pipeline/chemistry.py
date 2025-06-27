# Standard library imports
from typing import List, Union, Tuple
import numpy as np
import os
import time
import re
import shutil
import concurrent.futures
import pandas as pd
import matplotlib.pyplot as plt


# Own imports
from orca_pipeline.constants import MAX_TRIALS, RETRY_DELAY, FREQ_THRESHOLD

from orca_pipeline.hpc_driver import HPCDriver

# TODO ad gather results function  for SP.
# TODO add FOD functio
# TODO add plotting function
# TODO add functionality for plotting function to several Steps.
# TODO add censo option to confomers
# TODO For the Future maybe usew tight opt Settings for TS opt and add this also to the final optimisation of the sampled conformers
# TODO Add option to caluclate solvation energy with cosmors
# TODO add something to reaction to do stuff with seperated reactants.
# TODO  add correction to G if we go from 2 to 1 species etc.
# TODO Do we need to do micokinetic modeling to get concentraitions?
# TODO Add NBO stuff
# TODO make function which handles the copying of failed calulc

### ---UTILS----###


def read_xyz(filepath: str) -> Tuple[List[str], np.ndarray]:
    """
    Reads an XYZ file and returns a list of atoms and a 3xN NumPy array of coordinates.
    """
    atoms = []
    coords = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"XYZ file not found: {filepath}")

    with open(filepath, "r") as file:
        lines = file.readlines()
        if len(lines) < 3:
            raise ValueError(
                f"XYZ file {filepath} is too short to contain atomic data."
            )

        # Skip the first two lines (header)
        for idx, line in enumerate(lines[2:], start=3):
            parts = line.strip().split()
            if len(parts) < 4:
                print(
                    f"Warning: Skipping invalid line {idx} in {filepath}: '{line.strip()}'"
                )
                continue
            atom, x, y, z = parts[:4]
            try:
                coords.append([float(x), float(y), float(z)])
            except ValueError as e:
                print(f"Warning: Invalid coordinates on line {idx} in {filepath}: {e}")
                continue
            atoms.append(atom)

    if not atoms or not coords:
        raise ValueError(f"No valid atomic data found in {filepath}.")

    coords = np.array(coords)  # Transpose to get a Nx3 array
    return atoms, coords


def rmsd(mol1, mol2) -> float:
    """
    Calculates the RMSD between two molecules.

    """
    if type(mol1) is not type(mol2):
        raise ValueError("Both arguments must be of the same type.")

    if isinstance(mol1, Molecule):
        path_mol1 = f"{mol1.name}.xyz"
        mol1.to_xyz(path_mol1)

        path_mol2 = f"{mol2.name}.xyz"
        mol2.to_xyz(path_mol2)
    else:
        path_mol1 = mol1
        path_mol2 = mol2

    driver = HPCDriver()
    result = driver.shell_command(
        f"crest --rmsd {path_mol1} {path_mol2} | grep 'RMSD' | awk '{{print $NF}}'"
    )

    return float(result.stdout.strip())


### ---MOLECULE----###


class Molecule:
    """
    Represents a molecule.


    """

    def __init__(
        self,
        name: str,
        atoms: List[str],
        coords: np.ndarray,
        mult: int,
        charge: int,
        solvent: str = None,
        cosmo: bool = False,
        method: str = "r2scan-3c",
        sp_method: str = "r2scanh def2-qzvpp d4",
        conf_method="CREST",
        dif_scf: bool = False,
    ) -> None:
        self.name = name
        self.atoms = atoms
        self.coords = coords
        self.mult = mult
        self.charge = charge
        self.solvent = solvent
        self.cosmo = cosmo
        self.conf_method = conf_method
        # If true use  settings which should work foo all systems, if wft is convergabel, could tak long
        self.dif_scf = dif_scf
        # method functional, basis set, [opt] Dispersion correction, composite methods are also supported (e.g r2scan-3c)
        self.method = method
        self.sp_method = sp_method

        if len(atoms) != len(coords):
            print(len(atoms), len(coords))
            raise ValueError("Number of atoms and coordinates must match.")

        if mult < 1:
            raise ValueError("Multiplicity must be at least 1.")

    @classmethod
    def from_xyz(
        cls,
        filepath: str,
        charge: int,
        mult: int,
        solvent: str = None,
        cosmo: bool = False,
        name: str = None,
        method: str = "r2scan-3c",
        sp_method="r2scanh def2-qzvpp d4",
        conf_method="CREST",
        dif_scf: bool = False,
    ) -> "Molecule":
        """
        Creates a Molecule instance from an XYZ file.
        """
        if name is None:
            name = os.path.basename(filepath).split(".")[0]
        atoms, coords = read_xyz(filepath)
        return cls(
            name,
            atoms,
            coords,
            charge=charge,
            mult=mult,
            solvent=solvent,
            cosmo=cosmo,
            method=method,
            sp_method=sp_method,
            conf_method=conf_method,
            dif_scf=dif_scf,
        )

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

    def to_xyz(self, filepath: str = "") -> None:
        """
        Writes the molecule to an XYZ file.
        """
        if filepath == "":
            current_dir = os.getcwd()
            filepath = os.path.join(current_dir, f"{self.name}.xyz")

        with open(filepath, "w") as file:
            file.write(f"{len(self.atoms)}\n")
            file.write("\n")
            for atom, coord in zip(self.atoms, self.coords):
                file.write(f"{atom} {coord[0]:.9f} {coord[1]:.9f} {coord[2]:.9f}\n")

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

    ### ---OPT----###

    def geometry_opt(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        upper_limit: int = MAX_TRIALS,
        tight: bool = False,
        dif_scf: bool = False,
    ) -> bool:
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
            self.to_xyz(f"{self.name}.xyz")

        print(f"Starting optimisation of {self.name}")

        input_name = f"{self.name}_opt.inp"
        opt_block = "tightscf opt" if tight else "opt"
        scf_block = ""
        if dif_scf:
            scf_block = (
                f"%scf\n maxiter 1000\n DIISMaxeq 20\n directresetfreq 10\n end \n"
            )

        input = (
            f"!{self.method} {solvent_formatted} {opt_block}\n"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"{scf_block}"
            f"*xyz {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )

        # Write input files

        with open(input_name, "w") as f:
            f.write(input)

        # Submit jobs
        out_file = f"{input_name.split('.')[0]}_slurm.out"
        job_id = driver.submit_job(f"{input_name}", out_file)
        status = driver.check_job_status(job_id)

        # Wait for completion

        if status == "COMPLETED":
            time.sleep(45)
            if "HURRAY" in driver.grep_output(
                "HURRAY", input_name.split(".")[0] + ".out"
            ):
                print("[OPT] Optimisation jobs completed.")

                self.update_coords_from_xyz(f"{input_name.split('.')[0]}.xyz")
                self.name = f"{self.name}"

                # directories are handle by step_runner or custom script.
                return True

        print("[OPT] Optimisation jobs failed. Retrying...")

        driver.scancel_job(job_id)
        time.sleep(RETRY_DELAY)
        if not os.path.exists("Failed_calculations"):
            os.mkdir("Failed_calculations")

        shutil.move(
            input_name.split(".")[0] + ".out",
            f"Failed_calculations/{input_name.split('.')[0]}_failed_on_trial_{trial}.out",
        )

        driver.shell_command("rm -rf *.gbw pmix* *densities*  slurm* ")
        self.update_coords_from_xyz(f"{input_name.split('.')[0]}.xyz")
        return self.geometry_opt(
            driver=driver,
            slurm_params=slurm_params,
            trial=trial,
            upper_limit=upper_limit,
            tight=tight,
        )

    ### ---FREQ----###

    def freq_job(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        upper_limit: int = MAX_TRIALS,
        ts: bool = False,
        version: int = 601,
    ) -> bool:
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

        input_name = f"{self.name}_freq.inp"
        with open(input_name, "w") as f:
            f.write(freq_input)

        job_id_freq = driver.submit_job(
            input_name, input_name.split(".")[0] + "_slurm.out", version=version
        )
        status_freq = driver.check_job_status(job_id_freq, step="Freq")

        if status_freq == "COMPLETED":
            if driver.grep_output(
                "VIBRATIONAL FREQUENCIES", input_name.split(".")[0] + ".out"
            ):
                print("[FREQ] Calculation completed successfully.")
                if ts:
                    output = driver.grep_output(
                        "**imaginary mode***", input_name.split(".")[0] + ".out"
                    )
                    match = re.search(r"(-?\d+\.\d+)\s*cm\*\*-1", output)
                    if match:
                        imag_freq = float(match.group(1))

                        if imag_freq < FREQ_THRESHOLD:
                            print("[FREQ] Negative frequency found (TS).")
                            return True
                        else:
                            print(
                                "[FREQ] Negative frequency above threshold, not a TS."
                            )
                            return False
                    print("[FREQ] No negative frequency found, not a TS.")
                    return False
                return True

        print("[FREQ] Job failed or no frequencies found. Retrying...")
        if not os.path.exists("Failed_calculations"):
            os.mkdir("Failed_calculations")

        driver.scancel_job(job_id_freq)
        shutil.move(
            input_name.split(".")[0] + ".out",
            f"Failed_calculations/{input_name.split('.')[0]}_failed_on_trial_{trial}.out",
        )
        driver.shell_command("rm -rf *.gbw pmix* *densities*  slurm*")
        return self.freq_job(
            driver=driver,
            slurm_params=slurm_params,
            trial=trial,
            upper_limit=upper_limit,
            ts=ts,
        )

    ### ---TS-OPT----###

    def ts_opt(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        upper_limit: int = MAX_TRIALS,
    ) -> bool:
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
                "TS optimization will not be conducted with semiemporical methods switching to r2scan-3c ."
            )
            self.method = "r2scan-3c"
            print("Calculate Hess")
            if not self.freq_job(driver=driver, slurm_params=slurm_params, ts=True):
                print("Guess has no significant imaginary frequency. Aborting.")
                return False
            else:
                shutil.move(f"{self.name}_freq.hess", f"{self.name}_guess.hess")

            # step_runner checks if a hess is present in NEB copyies it. If one wants to use antohter hess, one can copy it to the guess.hess file
        if not os.path.exists(f"{self.name}_guess.hess"):
            print("Hessian file not found. Doing freq job on guess.")
            if not self.freq_job(driver=driver, slurm_params=slurm_params, ts=True):
                print("Guess has no significant imaginary frequency. Aborting.")
                return False
            else:
                shutil.move(f"{self.name}_freq.hess", f"{self.name}_guess.hess")

        if trial == 1:
            self.to_xyz("ts_guess.xyz")

        # TODO use tightopt settings for TS optimization
        solvent_formatted = f"CPCM({self.solvent})" if self.solvent else ""

        input_name = f"{self.name}_TS_opt.inp"

        ts_input = (
            f"!{self.method} OptTS tightscf {solvent_formatted}\n"
            f'%geom\ninhess read\ninhessname "{self.name}_guess.hess"\nCalc_Hess true\n recalc_hess 15\n end\n'
            f"%pal nprocs {slurm_params['nprocs']} end \n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyz {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )

        with open(input_name, "w") as f:
            f.write(ts_input)

        job_id = driver.submit_job(
            input_name, input_name.split(".")[0] + "_slurm.out", walltime="48"
        )
        status = driver.check_job_status(job_id, step="TS_OPT")

        if status == "COMPLETED" and "HURRAY" in driver.grep_output(
            "HURRAY", input_name.split(".")[0] + ".out"
        ):
            print("[TS_OPT] TS optimization succeeded.")
            self.update_coords_from_xyz(
                input_name.split(".")[0] + ".xyz"
            )  # Update coords from output")
            self.name = "ts"

            if self.freq_job(driver, slurm_params, ts=True):
                return True
            else:
                print(
                    "[TS_OPT] TS has no significant imaginary frequency. Check negative mode of guess."
                )

                self.update_coords_from_xyz("ts_guess.xyz")
                self.name = "ts_guess"

                return False

        print("[TS_OPT] TS optimization failed. Retrying...")
        driver.scancel_job(job_id)
        time.sleep(RETRY_DELAY)
        if not os.path.exists("Failed_calculations"):
            os.mkdir("Failed_calculations")
        shutil.move(
            input_name.split(".")[0] + ".out",
            f"Failed_calculations/{input_name.split('.')[0]}_failed_on_trial_{trial}.out",
        )
        driver.shell_command("rm -rf *.gbw pmix* *densities*  slurm* ")
        if os.path.exists(f"{input_name}.split('.')[0].xyz"):
            self.update_coords_from_xyz(f"{input_name.split('.')[0]}.xyz")

        return self.ts_opt(
            driver, slurm_params=slurm_params, trial=trial, upper_limit=upper_limit
        )

    ### ---QRC---###

    def qrc_job(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        upper_limit: int = 5,
    ) -> bool:
        print(f"[QRC] Trial {trial}")
        print(
            "The module uses pyQRC to make the displacement checkout there git (https://github.com/patonlab/pyQRC) for correct citation and usage."
        )

        if "xtb" in self.method.lower():
            print(
                "QRC calculation will not be conducted with semiemporical methods. Switching to r2scan-3c."
            )
            self.method = "r2scan-3c"

        print(
            "QRC does currently not suport ORCA 6.0.1. Need to do the freq job with ORCA 5.0.4"
        )

        trial += 1

        if trial > upper_limit:
            print("[QRC] Too many trials, aborting.")
            return False

        if trial == 1:
            slurm_params_freq = slurm_params.copy()
            slurm_params_freq["maxcore"] = slurm_params_freq["maxcore"] * 4

            if not os.path.exists("QRC"):
                with open("QRC", "w") as f:
                    f.write("")

                print("Doing freq job on guess.")
                print("Deleting old hess and gbw files")
                driver.shell_command("rm -rf *.gbw *.hess")
                if not self.freq_job(
                    driver=driver, slurm_params=slurm_params_freq, ts=True, version=504
                ):
                    print("Guess has no significant imaginary frequency. Aborting.")
                    return False
            elif not os.path.exists(f"{self.name}_freq.hess"):
                print("Hessian file not found. Doing freq job on guess.")
                if not self.freq_job(
                    driver=driver, slurm_params=slurm_params_freq, ts=True, version=504
                ):
                    print("Guess has no significant imaginary frequency. Aborting.")
                    return False

            # input is bassically the same as example 2 from the pyQRC git
        solvent_formatted = f"CPCM({self.solvent})" if self.solvent else ""

        driver.shell_command(
            f"python -m pyqrc --nproc {slurm_params['nprocs']} --mem {slurm_params['maxcore']} --amp 0.3 --name QRC_Forwards --route '{self.method} opt {solvent_formatted} ' {self.name}_freq.out "
        )
        driver.shell_command(
            f"python -m pyqrc --nproc {slurm_params['nprocs']} --mem {slurm_params['maxcore']} --amp -0.3 --name QRC_Backwards --route '{self.method} opt {solvent_formatted}' {self.name}_freq.out "
        )

        input_name_front = f"{self.name}_freq_QRC_Forward.inp"
        # pyqrc takes basename of input file and appends what is passed in the --name flag
        input_name_back = f"{self.name}_freq_QRC_Backwards.inp"

        # submission fails sometimes, yet when I test it manually it works.  wait 30 sek to ensure all files are present
        time.sleep(30)

        job_id_front = driver.submit_job(
            input_file=input_name_front,
            output_file=input_name_front.split(".")[0] + "_slurm.out",
            job_type="orca",
        )
        job_id_back = driver.submit_job(
            input_file=input_name_back,
            output_file=input_name_back.split(".")[0] + "_slurm.out",
            job_type="orca",
        )

        statuses = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_front = executor.submit(
                driver.check_job_status, job_id_front, step="QRC_FRONT"
            )
            future_back = executor.submit(
                driver.check_job_status, job_id_back, step="QRC_BACK"
            )

            statuses.append(future_front.result())
            statuses.append(future_back.result())

        if (
            all(status == "COMPLETED" for status in statuses)
            and "HURRAY"
            in driver.grep_output("HURRAY", input_name_front.split(".")[0] + ".out")
            and "HURRAY"
            in driver.grep_output("HURRAY", input_name_back.split(".")[0] + ".out")
        ):
            print("[QRC] QRC calculations completed successfully.")
            return True

        print("[QRC] QRC calculations failed. Retrying...")

        if "ORCA TERMINATED NORMALLY" in driver.grep_output(
            "ORCA TERMINATED NORMALLY", input_name_front.split(".")[0] + ".out"
        ) and "ORCA TERMINATED NORMALLY" in driver.grep_output(
            "ORCA TERMINATED NORMALLY", input_name_back.split(".")[0] + ".out"
        ):
            print("ORCA terminated normally.")
            print("One or both  of the optimization did not converege.")

        print("ORCA did not terminate normally. Retrying.")
        driver.scancel_job(job_id_front)
        driver.scancel_job(job_id_back)
        if not os.path.exists("Failed_calculations"):
            os.mkdir("Failed_calculations")
        shutil.move(
            input_name_front.split(".")[0] + ".out",
            f"Failed_calculations/{input_name_front.split('.')[0]}_failed_on_trial_{trial}.out",
        )

        shutil.move(
            input_name_back.split(".")[0] + ".out",
            f"Failed_calculations/{input_name_back.split('.')[0]}_failed_on_trial_{trial}.out",
        )

        time.sleep(RETRY_DELAY)
        driver.shell_command("rm -rf *.gbw pmix* *densities*  slurm*")
        return self.qrc_job(
            driver=driver,
            slurm_params=slurm_params,
            trial=trial,
            upper_limit=upper_limit,
        )

    ### ---IRC----###

    def irc_job(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        upper_limit: int = 5,
        maxiter: int = 70,
    ) -> bool:
        """
        Runs an Intrinsic Reaction Coordinate (IRC) calculation from an optimized TS.
        Checks for 'HURRAY' in 'IRC.out'. Retries with more steps if needed.
        """

        # TODO add after n trials thqat it switches to pyqrc. We need to recaluclate the hess with orca 5.04 also wee need to cahnge the method to opt after that orca 6 should be fine  This can be changed whencclib 2 is released

        if "xtb" in self.method.lower():
            print(
                "IRC calculation will not be conducted with semiemporical methods. Switching to r2scan-3c. If TS was optimised with other method indicate in the input."
            )

            self.method = "r2scan-3c"
        trial += 1
        # if trial > upper_limit/2:
        """
        if trial == 1:
            print("[IRC] seems to have an issue try with qrc. QRC has 3 trials")
            return self.qrc_job(driver=driver, slurm_params=slurm_params, trial=trial-1, upper_limit=upper_limit/2)
        print(f"[IRC] Trial {trial} with maxiter={maxiter}")
        if trial > upper_limit:
            print("[IRC] Too many trials, aborting.")
            return False
        """

        if trial == 1:
            slurm_params_freq = slurm_params.copy()
            slurm_params_freq["maxcore"] = slurm_params_freq["maxcore"] * 4
            if not os.path.exists(f"{self.name}_freq.hess"):
                print("Hessian file not found. Doing freq job on guess.")
                if not self.freq_job(
                    driver=driver, slurm_params=slurm_params_freq, ts=True
                ):
                    print("Guess has no significant imaginary frequency. Aborting.")
                    return False

        solvent_formatted = f"CPCM({self.solvent})" if self.solvent else ""
        input_name = f"{self.name}_IRC.inp"

        irc_input = (
            f"!{self.method} IRC tightscf {solvent_formatted}\n"
            f'%irc\n  maxiter {maxiter}\n  InitHess read\n  Hess_Filename "{self.name}_freq.hess"\nend\n'
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyz {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )

        with open(input_name, "w") as f:
            f.write(irc_input)

        job_id = driver.submit_job(input_name, input_name.split(".")[0] + "_slurm.out")
        status = driver.check_job_status(job_id, step="IRC")

        if status == "COMPLETED" and (
            "HURRAY" in driver.grep_output("HURRAY", f"{self.name}_IRC.out")
            and driver.grep_output("ORCA TERMINATED NORMALLY", f"{self.name}_IRC.out")
        ):
            print("[IRC] IRC completed successfully.")

            return True

        print(
            "[IRC] IRC job did not converge or failed. Attempting restart with more steps."
        )
        driver.scancel_job(job_id)
        if "ORCA TERMINATED NORMALLY" in driver.grep_output(
            "ORCA TERMINATED NORMALLY", f"{self.name}_IRC.out"
        ):
            if maxiter < 100:
                maxiter += 30
            else:
                print("[IRC] Max iteration limit reached. Aborting.")
                return False

        time.sleep(RETRY_DELAY)
        if not os.path.exists("Failed_calculations"):
            os.mkdir("Failed_calculations")
        shutil.move(
            input_name.split(".")[0] + ".out",
            f"Failed_calculations/{input_name.split('.')[0]}_failed_on_trial_{trial}.out",
        )
        driver.shell_command("rm -rf *.gbw pmix* *densities*  slurm*")
        return self.irc_job(
            driver=driver,
            slurm_params=slurm_params,
            trial=trial,
            upper_limit=upper_limit,
            maxiter=maxiter,
        )

    # ---------- SINGLE POINT (SP) STEP ----------

    def sp_calc(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        upper_limit: int = 5,
        with_freq=True,
    ) -> bool:
        """
        Runs a single point calculation on the molecule iwth the sp_method given.
        Default is r2scanh def2-qzvpp d4.

        """

        solvent_formatted = f"CPCM({self.solvent})" if self.solvent else ""

        if self.cosmo:
            solvent_formatted = ""

        if "xtb" in self.sp_method.lower():
            print(
                "SP calculation will not be conducted with semiemporical methods. Switching to r2scanh def2-qzvpp d4 defgrid 3 verytightscf."
            )
            self.sp_method = "r2scanh def2-qzvpp d4 "

        if with_freq:
            print("Doing freq job to get THERMO data")
            if os.path.exists(f"{self.name}_freq.out"):
                print("Freq job already done")
            else:
                slurm_params_freq = slurm_params.copy()
                slurm_params_freq["maxcore"] = slurm_params_freq["maxcore"] * 4
                if slurm_params_freq["maxcore"] * slurm_params_freq["nprocs"] > 500000:
                    slurm_params_freq["maxcore"] = slurm_params_freq["maxcore"] * (
                        3 / 4
                    )

                self.freq_job(driver=driver, slurm_params=slurm_params_freq, ts=False)

        trial += 1
        print(f"[SP] Trial {trial} ")
        if trial > upper_limit:
            print("[SP] Too many trials, aborting.")
            return False

        if trial == 1:
            self.to_xyz()

        sp_input = (
            f"!{self.sp_method} {solvent_formatted} verytightscf defgrid3 \n"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyz {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )
        input_name = f"{self.name}_SP.inp"
        with open(input_name, "w") as f:
            f.write(sp_input)
        job_id = driver.submit_job(input_name, input_name.split(".")[0] + "_slurm.out")

        status = driver.check_job_status(job_id, step="SP")

        if status == "COMPLETED" and "FINAL SINGLE POINT ENERGY" in driver.grep_output(
            "FINAL SINGLE POINT ENERGY", input_name.split(".")[0] + ".out"
        ):
            print("[SP] Single point calculation completed successfully.")
            if self.cosmo:
                print(f"COSMO was selected, start COSMO-RS job with {self.solvent}")

                succes = self.cosmo_rs(driver=driver, slurm_params=slurm_params)

                if succes:
                    return True
                else:
                    print(f" COSMO-RS job failed for {self.name} .")
                    return False

            return True

        print("[SP] Failed or not converged. Retrying...")
        driver.scancel_job(job_id)
        time.sleep(RETRY_DELAY)
        driver.shell_command("rm -rf *.gbw pmix* *densities* SP.inp slurm*")
        return self.sp_calc(
            driver=driver,
            slurm_params=slurm_params,
            trial=trial,
            upper_limit=upper_limit,
        )

    def cosmo_rs(self, driver: HPCDriver, slurm_params: dict):
        """
        Runs a COSMO-RS calculation on the molecule.
        """

        print(f"Starting COSMO-RS calculation for {self.name}.")
        cosmors_input = (
            f"! COSMORS({self.solvent})\n"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyz {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )
        input_name = f"{self.name}_cosmors.inp"

        with open(input_name, "w") as f:
            f.write(cosmors_input)

        job_id = driver.submit_job(input_name, input_name.split(".")[0] + "_slurm.out")
        status = driver.check_job_status(job_id, step="COSMO-RS")

        if status == "COMPLETED" and (
            "ORCA TERMINATED NORMALLY"
            in driver.grep_output(
                "ORCA TERMINATED NORMALLY", input_name.split(".")[0] + ".out"
            )
            and "Free energy of solvation"
            in driver.grep_output(
                "Free energy of solvation", input_name.split(".")[0] + ".out"
            )
        ):
            print("[COSMO-RS] COSMO-RS calculation completed successfully.")
            return True
        else:
            print("[COSMO-RS] COSMO-RS calculation failed or not converged.")
            print(status)
            return False

    def get_lowest_confomer_goat(
        self, driver: HPCDriver, slurm_params: dict, cwd=None, censo: bool = False
    ) -> bool:
        print(
            f"Finding confomers for {self.name} with GOAT. XTB2 is used as method as dft metods are to exepnsive "
        )
        solvent_formatted = f"ALPB({self.solvent})" if self.solvent else ""

        self.to_xyz(f"{self.name}_before_goat.xyz")

        goat_input = (
            f"!XTB2 {solvent_formatted} GOAT \n"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyz {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )

        input_name = f"{self.name}_goat.inp"
        with open(input_name, "w") as f:
            f.write(goat_input)

        job_id = driver.submit_job(input_name, input_name.split(".")[0] + "_slurm.out")
        status = driver.check_job_status(job_id, step="GOAT")

        if status == "COMPLETED" and "ORCA TERMINATED NORMALLY" in driver.grep_output(
            "ORCA TERMINATED NORMALLY", input_name.split(".")[0] + ".out"
        ):
            print("[GOAT] GOAT calculation completed successfully.")

            self.update_coords_from_xyz(input_name.split(".")[0] + ".globalminimum.xyz")

            return True
        else:
            print("GOAT failed chekc output")
            print(status)
            return False

    def get_lowest_confomer_crest(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        upper_limit: int = 5,
        cwd=None,
    ) -> bool:
        trial += 1
        print(f"[CREST] Trial {trial} ")
        if trial > upper_limit:
            print("[CREST] Too many trials, aborting.")
            return False
        print(f"[CREST] Generating conformers for {self.name}")

        print("CREST needs xtb2 optimized structures.")

        self.to_xyz(f"{self.name}_before_crest.xyz")

        # for now do everything from commnand since orca can have some problems which xtb does not have

        driver.shell_command(
            f"xtb {self.name}_before_crest.xyz --opt vtight --charge {self.charge} --uhf {self.mult - 1} --alpb {self.solvent} --namespace {self.name} ",
            timeout=1200,
        )

        if not os.path.exists(f"{self.name}.xtbopt.xyz"):
            print("Optimization failed. Aborting.")
            return False

        print("Optimization done. Starting CREST")
        print("Copying optimized structure to CREST directory")
        if cwd:
            shutil.copy(f"{self.name}.xtbopt.xyz", cwd)
        else:
            cwd = os.getcwd()

        job_id = driver.submit_job(
            input_file=f"{self.name}.xtbopt.xyz",
            walltime="120",
            output_file=f"{self.name}_slurm.out",
            charge=self.charge,
            mult=self.mult,
            solvent=self.solvent,
            job_type="crest",
            cwd=cwd,
        )  # submit_job handles conversion from mult to uhf

        status = driver.check_job_status(job_id, step="CREST")
        if status == "COMPLETED":
            time.sleep(30)
            output = os.path.join(cwd, f"{self.name}.xtbopt_out.log")
            crest_best = os.path.join(cwd, "crest_best.xyz")

            if "CREST terminated normally" in driver.grep_output(
                "CREST terminated normally", output, flags="-a"
            ) and os.path.exists(crest_best):
                print("[CREST] Conformers generated successfully.")
                print("Updating coordinates from CREST output.")

                self.update_coords_from_xyz(crest_best)
                return True
            else:
                print("[CREST] CREST failed")
                print("grep output")
                print(
                    driver.grep_output("CREST terminated normally", output, flags="-a")
                )
                print(f"{os.getcwd()} current directory")
                return False
        return False

    def fod_calc(self, driver, slurm_params, trial=0, upper_limit=5):
        trial += 1
        print(f"[FOD] Trial {trial} ")
        if trial > upper_limit:
            print("[FOD] Too many trials, aborting.")
            return False
        print(f"[FOD] Generating FOD for {self.name}")
        print(
            "ORCA needs to be excuted in same dir, else density plots are not possible"
        )
        print(
            "FOD is currently conducted with default settings -> TPSS def2-TZVP at 5000K"
        )

        input_name = f"{self.name}_FOD.inp"
        fod_input = (
            f"!FOD\n"
            f"%pal nprocs {slurm_params['nprocs']} end\n"
            f"%maxcore {slurm_params['maxcore']}\n"
            f"*xyz {self.charge} {self.mult} \n"
            f"{self.get_xyz_block()}*"
        )

        with open(input_name, "w") as f:
            f.write(fod_input)

        job_id = driver.submit_job(
            input_name, input_name.split(".")[0] + "_slurm.out", job_type="fod"
        )

        status = driver.check_job_status(job_id, step="FOD")

        if status == "COMPLETED" and "ORCA TERMINATED NORMALLY" in driver.grep_output(
            "ORCA TERMINATED NORMALLY", input_name.split(".")[0] + ".out"
        ):
            print("[FOD] FOD calculation completed successfully.")
            print("Plotting Cube files")

            with open(f"{self.name}_fod_plot.inp", "w") as f:
                f.write(
                    f" 1\n 2\n n\n {input_name.split('.')[0]}.scfp_fod\n 4\n 100\n 5\n 7\n 11\n 12\n"
                )
            result = driver.shell_command(
                f"orca_plot  {input_name.split('.')[0]}.gbw -i < {self.name}_fod_plot.inp",
                timeout=3600,
            )
            if not result:
                print("orca_plot failed")

            return True
        else:
            # TODO think about restart if we want it here
            print("[FOD] FOD calculation failed.")
            return False


def get_reaction_energies(self) -> bool:
    """
    self.energies = pd.DataFrame(
        columns=["step", "single_point_energy", "free_energy_correction" "inner_energy_correction","enthalpy_correction"  "entropy",  "temperature","method", "sp_method"])

    #TODO add reaction name to step naming, to easier concat the csv. Currently probably not possible since reaction name is not perfect
    """

    driver = HPCDriver()
    if os.path.exists(f"{self.name}_energies.csv"):
        print(f"{self.name}_energies.csv exist")
        return True

    if not os.path.exists("SP"):
        raise FileNotFoundError("SP folder not found. Run SP calculations first.")

    steps = [f"{self.name}"]
    sp_energies = []
    free_energy_corrections = []
    inner_energy_corrections = []
    enthalpy_corrections = []
    entropies = []
    temperatures = []
    methods = []
    sp_methods = []
    solvents = []

    for step in [self.educt.name, self.transition_state.name, self.product.name]:
        sp_file = f"SP/{step}_SP.out"
        freq_file = f"SP/{step}_freq.out"
        if not os.path.exists(sp_file) or not os.path.exists(freq_file):
            raise FileNotFoundError(
                f"SP file {sp_file} or FREQ file {freq_file} not found. Run SP calculations first."
            )
        print("Chekcking if reactants are true minima and TS true saddle point")

        imags = driver.grep_output("***imaginary mode***", freq_file).split("\n")
        imags = [line for line in imags if line.strip() != ""]

        if len(imags) > 0:
            if step == self.transition_state.name and len(imags) == 1:
                pass
            else:
                print(
                    f"Step {step} is not a true minimum or saddle point. Check the frequency calculation and take the results with caution."
                )

        sp_energies.append(
            float(
                driver.grep_output("FINAL SINGLE POINT ENERGY", sp_file).split(" ")[-1]
            )
        )
        # maybe it is super consistent and we can just count where it is. This is kinda fool proofed

        tmp = driver.grep_output("G-E(el)", freq_file).split(" ")
        index = tmp.index("Eh") - 1
        free_energy_corrections.append(float(tmp[index]))

        tmp = driver.grep_output("Total correction", freq_file).split(" ")
        index = tmp.index("Eh") - 1
        inner_energy_corrections.append(float(tmp[index]))

        tmp = driver.grep_output("Thermal Enthalpy correction", freq_file).split(" ")
        index = tmp.index("Eh") - 1
        enthalpy_corrections.append(float(tmp[index]))

        tmp = driver.grep_output("Final entropy term", freq_file).split(" ")
        index = tmp.index("Eh") - 1
        entropies.append(float(tmp[index]))
        temperatures.append(
            float(driver.grep_output("Temperature", freq_file).split(" ")[-2])
        )
        methods.append(self.educt.method)
        sp_methods.append(self.educt.sp_method)
        solvents.append(self.educt.solvent)

    self.energies = pd.DataFrame(
        {
            "step": steps,
            "single_point_energy": sp_energies,
            "free_energy_correction": free_energy_corrections,
            "inner_energy_correction": inner_energy_corrections,
            "enthalpy_correction": enthalpy_corrections,
            "entropy": entropies,
            "temperature": temperatures,
            "method": methods,
            "sp_method": sp_methods,
            "solvent": solvents,
        }
    )
    self.energies.to_csv(f"{self.name}_energies.csv", index=False)
    return True


class Reaction:
    """
    Represents a elementary step in a reaction.
    """

    # TODO function which calculates RMSD between two molecules. To be used to identify which of the endpoints is educt and which is product.

    def __init__(
        self,
        educt: Molecule,
        product: Molecule,
        transition_state: Molecule = None,
        nimages: int = 16,
        method: str = "r2scan-3c",
        sp_method="r2scanh def2-qzvpp d4",
        solvent="",
        name="reaction",
        fast: bool = False,
        zoom: bool = False,
        energies_path: str = None,
        conf_method="CREST",
    ) -> None:
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
        self.conf_method = conf_method
        self.energies = pd.DataFrame(
            columns=[
                "step",
                "single_point_energy",
                "free_energy_correctioninner_energy_correction",
                "enthalpy_correction",
                "entropy",
                "temperature",
                "method",
                "sp_method",
                "solvent",
                "cosmo",
            ],
        )
        if energies_path:
            self.energies = pd.read_csv(energies_path)

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
        reactant_strs = " + ".join(f"{self.educt}")
        product_strs = " + ".join(self.product)
        return f"{reactant_strs} => {product_strs}"

    @classmethod
    def from_xyz(
        cls,
        educt_filepath: str,
        product_filepath: str,
        transition_state_filepath: str = None,
        nimages: int = 16,
        method: str = "r2scan-3c",
        charge: int = 0,
        mult: int = 1,
        solvent: str = None,
        cosmo: bool = False,
        sp_method: str = "r2scanh def2-qzvpp d4",
        name: str = "reaction",
        fast: bool = False,
        zoom: bool = False,
        energy_file: str = None,
        conf_method: str = "CREST",
    ) -> "Reaction":
        """
        Creates an Reaction instance from XYZ files.
        """
        educt = Molecule.from_xyz(
            educt_filepath,
            charge=charge,
            mult=mult,
            solvent=solvent,
            cosmo=cosmo,
            method=method,
            sp_method=sp_method,
            name="educt",
            conf_method=conf_method,
        )
        product = Molecule.from_xyz(
            product_filepath,
            charge=charge,
            mult=mult,
            solvent=solvent,
            cosmo=cosmo,
            method=method,
            sp_method=sp_method,
            name="product",
            conf_method=conf_method,
        )
        transition_state = (
            Molecule.from_xyz(
                transition_state_filepath,
                charge=charge,
                mult=mult,
                solvent=solvent,
                cosmo=cosmo,
                name="ts",
                method=method,
                sp_method=sp_method,
            )
            if transition_state_filepath
            else None
        )
        return cls(
            educt=educt,
            product=product,
            transition_state=transition_state,
            nimages=nimages,
            method=method,
            solvent=solvent,
            sp_method=sp_method,
            name=name,
            fast=fast,
            zoom=zoom,
            energies_path=energy_file,
            conf_method=conf_method,
        )

    def optimise_reactants(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        upper_limit: int = MAX_TRIALS,
        tight: bool = False,
    ) -> bool:
        """
        Optimises the reactant geometry.
        """

        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            results = [
                executor.submit(
                    self.educt.geometry_opt,
                    driver,
                    slurm_params,
                    trial,
                    upper_limit,
                    tight,
                ),
                executor.submit(
                    self.product.geometry_opt,
                    driver,
                    slurm_params,
                    trial,
                    upper_limit,
                    tight,
                ),
            ]

        return all([result.result() for result in results])

    def neb_ci(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial=0,
        upper_limit: int = MAX_TRIALS,
    ) -> bool:
        """
        Performs NEB-CI (Climbing Image) calculation.
        Assums that the educt and products are already optimized with the same method
        """
        trial += 1
        print(f"[NEB_CI] Trial {trial}, Nimages={self.nimages}, Method={self.method}")
        if trial > upper_limit:
            print("[NEB_CI] Too many trials aborting.")
            return False

        if trial == 1:
            self.educt.to_xyz("educt.xyz")
            self.product.to_xyz("product.xyz")
        solvent_formatted = ""
        if self.solvent:
            solvent_formatted = (
                f"ALPB({self.solvent})"
                if "xtb" in self.method.lower()
                else f"CPCM({self.solvent})"
            )

        neb_method = "NEB-CI" if not self.zoom else "ZOOM-NEB-CI"
        nprocs = ""
        maxcore = slurm_params["maxcore"]
        if "xtb" not in self.method.lower():
            nprocs = (
                slurm_params["nprocs"]
                if 4 * self.nimages < slurm_params["nprocs"]
                else 4 * self.nimages
            )

        else:
            # xtb2 is quite fast and can handle a lot of images
            nprocs = slurm_params["nprocs"] if self.nimages < 64 else 64
            maxcore = 512

        neb_input = (
            f"! {neb_method} {self.method} {solvent_formatted} tightscf \n"
            f"%pal nprocs {nprocs} end\n"
            f"%maxcore {maxcore}\n"
            f'%neb\n  Product "product.xyz"\n  NImages {self.nimages} \nend\n'
            f"*xyzfile {self.charge} {self.mult} educt.xyz\n"
        )
        input_name = f"{self.name}_neb-CI.inp"

        with open(input_name, "w") as f:
            f.write(neb_input)

        job_id = driver.submit_job(
            input_name, input_name.split(".")[0] + "_slurm.out", walltime="72"
        )
        status = driver.check_job_status(job_id, step="NEB-CI")

        if (
            status == "COMPLETED"
            and "THE NEB OPTIMIZATION HAS CONVERGED"
            in driver.grep_output(
                "THE NEB OPTIMIZATION HAS CONVERGED", input_name.split(".")[0] + ".out"
            )
        ):
            print("[NEB_CI] Completed successfully.")
            time.sleep(20)
            pot_ts = Molecule.from_xyz(
                input_name.split(".")[0] + "_NEB-CI_converged.xyz",
                charge=self.charge,
                mult=self.mult,
                solvent=self.solvent,
                method=self.method,
                name="ts_guess",
                sp_method=self.sp_method,
            )

            slurm_params_freq = slurm_params.copy()
            # a bit conservative maybe double is enough

            slurm_params_freq["maxcore"] = maxcore * 4

            freq_success = pot_ts.freq_job(
                driver, slurm_params=slurm_params_freq, ts=True
            )
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
                if not os.path.exists("Failed_calculations"):
                    os.mkdir("Failed_calculations")
                shutil.move(
                    input_name.split(".")[0] + "_out",
                    f"Failed_calculations/{input_name.split('.')[0]}_failed_on_trial_{trial}.out",
                )

                driver.shell_command(
                    "rm -rf *.gbw pmix* *densities* freq.inp slurm* *neb*.inp"
                )
                self.nimages += 4
                return self.neb_ci(driver, slurm_params, trial, upper_limit)
        else:
            print("[NEB_CI] Job failed or did not converge. Retrying...")
            driver.scancel_job(job_id)
            if not os.path.exists("Failed_calculations"):
                os.mkdir("Failed_calculations")
            shutil.move(
                input_name.split(".")[0] + ".out",
                f"Failed_calculations/{input_name.split('.')[0]}_failed_on_trial_{trial}.out",
            )

            time.sleep(RETRY_DELAY)
            driver.shell_command(
                "rm -rf *.gbw pmix* *densities* freq.inp slurm* neb*im* *neb*.inp"
            )
            return self.neb_ci(
                driver=driver,
                slurm_params=slurm_params,
                trial=trial,
                upper_limit=upper_limit,
            )

    def neb_ts(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial=0,
        upper_limit: int = MAX_TRIALS,
    ) -> Tuple[str, bool]:
        """
        Performs a NEB-TS calculation (optionally FAST) with either XTB or r2scan-3c.
        Checks for convergence with 'HURRAY' and runs a freq job on the converged TS.



        """

        trial += 1
        print(
            f"[NEB_TS] Trial {trial}, Nimages={self.nimages}, method={self.method}, fast={self.fast}"
        )
        if trial > upper_limit:
            print("[NEB_TS] Too many trials aborting.")
            return "failed", False

        # On first NEB trial, create NEB folder if not existing
        if trial == 1:
            self.educt.to_xyz("educt.xyz")
            self.product.to_xyz("product.xyz")

        geom_block = "%geom\n Calc_Hess true\n Recalc_Hess 15 end\n"
        nprocs = slurm_params["nprocs"]
        maxcore = slurm_params["maxcore"]

        if "xtb" in self.method.lower():
            self.fast = False
            maxiter = len(self.educt.atoms) * 4
            geom_block = (
                f"%geom\n Calc_Hess true\n Recalc_Hess 1\n MaxIter={maxiter} end\n"
            )
            nprocs = (
                slurm_params["nprocs"]
                if self.nimages < slurm_params["nprocs"]
                else self.nimages
            )
            maxcore = 1500

        neb_block = "Fast-NEB-TS" if self.fast else "NEB-TS"
        neb_block = "ZOOM-NEB-TS" if self.zoom else neb_block
        solvent_formatted = ""
        if self.solvent:
            solvent_formatted = (
                f"ALPB({self.solvent})"
                if "xtb" in self.method.lower()
                else f"CPCM({self.solvent})"
            )
        neb_input = (
            f"! {self.method} {neb_block} {solvent_formatted} tightscf \n"
            f"{geom_block}"
            f"%pal nprocs {nprocs}  end\n"
            f"%maxcore {maxcore}\n"
            f'%neb\n   Product "product.xyz"\n   NImages {self.nimages}\n  end\n'
            f"*xyzfile {self.charge} {self.mult} educt.xyz\n"
        )

        neb_input_name = (
            f"{self.name}_neb-fast-TS.inp" if self.fast else f"{self.name}_neb-TS.inp"
        )
        with open(neb_input_name, "w") as f:
            f.write(neb_input)

        job_id = driver.submit_job(
            neb_input_name, neb_input_name.split(".")[0] + "_slurm.out", walltime="72"
        )
        status = driver.check_job_status(job_id, step="NEB_TS")

        out_name = neb_input_name.rsplit(".", 1)[0] + ".out"
        if status == "COMPLETED" and "HURRAY" in driver.grep_output("HURRAY", out_name):
            print(
                "[NEB_TS] NEB converged successfully. Checking frequency of TS structure..."
            )
            # Typically the NEB TS is in something like: neb-TS_NEB-TS_converged.xyz
            ts_xyz = neb_input_name.rsplit(".", 1)[0] + "_NEB-TS_converged.xyz"
            time.sleep(RETRY_DELAY)

            # TODO we need a retry mechanism here if ssubo is to slow.
            if os.path.exists(ts_xyz):
                potential_ts = Molecule.from_xyz(
                    ts_xyz,
                    charge=self.charge,
                    mult=self.mult,
                    solvent=self.solvent,
                    name="ts_guess",
                    method=self.method,
                    sp_method=self.sp_method,
                )
                freq_success = potential_ts.freq_job(
                    driver=driver, slurm_params=slurm_params, ts=True
                )
                if freq_success:
                    # Copy the final TS structure to a known file
                    self.transition_state = potential_ts

                    return "", True
                else:
                    print("TS has no significant imag freq Retry with refined method")
                    driver.scancel_job(job_id)
                    time.sleep(RETRY_DELAY)
                    if not os.path.exists("Failed_calculations"):
                        os.mkdir("Failed_calculations")
                    shutil.move(
                        neb_input_name.split(".")[0] + ".out",
                        f"Failed_calculations/{neb_input_name.split('.')[0]}_failed_on_trial_{trial}.out",
                    )
                    if os.path.exists(
                        neb_input_name.split(".")[0] + "_NEB-CI_converged.xyz"
                    ):
                        shutil.move(
                            neb_input_name.split(".")[0] + "_NEB-CI_converged.xyz",
                            "Failed_calculations",
                        )

                    driver.shell_command(
                        "rm -rf *.gbw pmix* *densities* freq.inp slurm* *neb*.inp"
                    )

                    return "failed", False

        elif driver.grep_output(
            "ORCA TERMINATED NORMALLY", f"{neb_input_name.rsplit('.', 1)[0]}.out"
        ) or driver.grep_output(
            "The optimization did not converge",
            f"{neb_input_name.rsplit('.', 1)[0]}.out",
        ):
            # TODO Maybe do something else than just restart with other settings
            print("ORCA has terminated normally, optimisation did not converge.")
            print("Restart neb with more precise settings, use ")
            driver.scancel_job(job_id)

            if not os.path.exists("Failed_calculations"):
                os.mkdir("Failed_calculations")
            shutil.move(neb_input_name.split(".")[0] + ".out", "Failed_calculations")
            if os.path.exists(neb_input_name.split(".")[0] + "_NEB-CI_converged.xyz"):
                shutil.move(
                    neb_input_name.split(".")[0] + "_NEB-CI_converged.xyz",
                    "Failed_calculations",
                )

            # If needed, remove partial files or rename them
            driver.shell_command(
                "rm -rf *.gbw pmix* *densities* *freq.inp slurm* *neb*.inp"
            )

            return "failed", False
        # already changed dir
        print(
            "There was an error during the run or it was aborted, Restart like it was not converged"
        )
        driver.scancel_job(job_id)
        time.sleep(RETRY_DELAY)
        if not os.path.exists("Failed_calculations"):
            os.mkdir("Failed_calculations")
        shutil.move(neb_input_name.split(".")[0] + ".out", "Failed_calculations")
        if os.path.exists(neb_input_name.split(".")[0] + "_NEB-CI_converged.xyz"):
            shutil.move(
                neb_input_name.split(".")[0] + "_NEB-CI_converged.xyz",
                "Failed_calculations",
            )
        driver.shell_command("rm -rf *.gbw pmix* *densities* freq.inp slurm* *neb*.inp")

        return "failed", False

    def sp_calc(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        cosmo: bool = False,
        upper_limit: int = MAX_TRIALS,
    ) -> bool:
        """
        Runs a single point calculation on educt transition state and product.
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = [
                executor.submit(
                    self.educt.sp_calc, driver, slurm_params, trial, upper_limit
                ),
                executor.submit(
                    self.product.sp_calc, driver, slurm_params, trial, upper_limit
                ),
                executor.submit(
                    self.transition_state.sp_calc,
                    driver,
                    slurm_params,
                    trial,
                    upper_limit,
                ),
            ]
        return all([result.result() for result in results])

    def fod_calc(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        upper_limit: int = MAX_TRIALS,
    ) -> bool:
        """
        Does FOD with all reactants
        """
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            results = [
                executor.submit(
                    self.educt.fod_calc, driver, slurm_params, trial, upper_limit
                ),
                executor.submit(
                    self.product.fod_calc, driver, slurm_params, trial, upper_limit
                ),
                executor.submit(
                    self.transition_state.fod_calc,
                    driver,
                    slurm_params,
                    trial,
                    upper_limit,
                ),
            ]
        return all([result.result() for result in results])

    def get_lowest_confomers(
        self,
        driver: HPCDriver,
        slurm_params: dict,
        trial: int = 0,
        upper_limit: int = MAX_TRIALS,
        crest: bool = True,
        conf_exclude: str = "",
    ) -> bool:
        """
        Generates conformers for the educt and product.
        Function expects that there are educt and product directories in the current working directory.

        exclude: educt, product. Sometimes it makes sense to omit the conf ensemble for a specific step.




        """
        # TODO implement option to do it with GOAT
        # TODO implement TS conf-generation
        results = []

        if crest:
            print("Searching for lowest conformers with CREST")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                if "educt" not in conf_exclude:
                    results.append(
                        executor.submit(
                            self.educt.get_lowest_confomer_crest,
                            driver,
                            slurm_params,
                            trial,
                            upper_limit,
                            cwd="educt_confs",
                        )
                    )
                if "product" not in conf_exclude:
                    results.append(
                        executor.submit(
                            self.product.get_lowest_confomer_crest,
                            driver,
                            slurm_params,
                            trial,
                            upper_limit,
                            cwd="product_confs",
                        )
                    )
        else:
            print("Searching for lowest confomers using GOAT")
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                if "educt" not in conf_exclude:
                    results.append(
                        executor.submit(
                            self.educt.get_lowest_confomer_goat, driver, slurm_params
                        )
                    )
                if "product" not in conf_exclude:
                    results.append(
                        executor.submit(
                            self.product.get_lowest_confomer_goat, driver, slurm_params
                        )
                    )

        return all([result.result() for result in results])

    def get_reaction_energies(self) -> bool:
        """
        self.energies = pd.DataFrame(
            columns=["step", "single_point_energy", "free_energy_correction" "inner_energy_correction","enthalpy_correction"  "entropy",  "temperature","method", "sp_method"])

        #TODO add reaction name to step naming, to easier concat the csv. Currently probably not possible since reaction name is not perfect
        #TODO add ZPE energy seperately to have E_el + ZPE
        """

        driver = HPCDriver()
        if os.path.exists(f"{self.name}_energies.csv"):
            print("Energies already calculated")
            return True

        if not os.path.exists("SP"):
            raise FileNotFoundError("SP folder not found. Run SP calculations first.")

        steps = ["educt", "transition_state", "product"]
        sp_energies = []
        free_energy_corrections = []
        inner_energy_corrections = []
        enthalpy_corrections = []
        entropies = []
        temperatures = []
        methods = []
        sp_methods = []
        solvents = []

        for step in [self.educt.name, self.transition_state.name, self.product.name]:
            sp_file = f"SP/{step}_SP.out"
            freq_file = f"SP/{step}_freq.out"
            if not os.path.exists(sp_file) or not os.path.exists(freq_file):
                raise FileNotFoundError(
                    f"SP file {sp_file} or FREQ file {freq_file} not found. Run SP calculations first."
                )
            print("Chekcking if reactants are true minima and TS true saddle point")

            imags = driver.grep_output("***imaginary mode***", freq_file).split("\n")
            imags = [line for line in imags if line.strip() != ""]

            if len(imags) > 0:
                if step == self.transition_state.name and len(imags) == 1:
                    pass
                else:
                    print(
                        f"Step {step} is not a true minimum or saddle point. Check the frequency calculation and take the results with caution."
                    )

            sp_energies.append(
                float(
                    driver.grep_output("FINAL SINGLE POINT ENERGY", sp_file).split(" ")[
                        -1
                    ]
                )
            )
            # maybe it is super consistent and we can just count where it is. This is kinda fool proofed

            tmp = driver.grep_output("G-E(el)", freq_file).split(" ")
            index = tmp.index("Eh") - 1
            free_energy_corrections.append(float(tmp[index]))

            tmp = driver.grep_output("Total correction", freq_file).split(" ")
            index = tmp.index("Eh") - 1
            inner_energy_corrections.append(float(tmp[index]))

            tmp = driver.grep_output("Thermal Enthalpy correction", freq_file).split(
                " "
            )
            index = tmp.index("Eh") - 1
            enthalpy_corrections.append(float(tmp[index]))

            tmp = driver.grep_output("Final entropy term", freq_file).split(" ")
            index = tmp.index("Eh") - 1
            entropies.append(float(tmp[index]))
            temperatures.append(
                float(driver.grep_output("Temperature", freq_file).split(" ")[-2])
            )
            methods.append(self.educt.method)
            sp_methods.append(self.educt.sp_method)
            solvents.append(self.educt.solvent)

        self.energies = pd.DataFrame(
            {
                "step": steps,
                "single_point_energy": sp_energies,
                "free_energy_correction": free_energy_corrections,
                "inner_energy_correction": inner_energy_corrections,
                "enthalpy_correction": enthalpy_corrections,
                "entropy": entropies,
                "temperature": temperatures,
                "method": methods,
                "sp_method": sp_methods,
                "solvent": solvents,
            }
        )
        self.energies.to_csv(f"{self.name}_energies.csv", index=False)
        return True
