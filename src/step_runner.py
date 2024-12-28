import concurrent.futures
import json
import os
import logging
import shutil
from typing import List, Callable, Dict, Union
from chemistry import Reaction, Molecule  # Ensure correct import paths
from hpc_driver import HPCDriver
from constants import MAX_TRIALS, RETRY_DELAY, SLURM_PARAMS_BIG_HIGH_MEM, SLURM_PARAMS_BIG_LOW_MEM, DEFAULT_STEPS
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

SETTINGS_FILE = "settings.json"
COMPLETED_MARKER = "COMPLETED"
FAILED_MARKER = "FAILED"


class StepRunner:
    def __init__(self,
                 hpc_driver: HPCDriver,
                 target: Union[Reaction, Molecule],
                 steps: List[str] = DEFAULT_STEPS,
                 slurm_params_high_mem: Dict = SLURM_PARAMS_BIG_HIGH_MEM,
                 slurm_params_low_mem: Dict = SLURM_PARAMS_BIG_LOW_MEM,
                 home_dir: str = "."):
        self.hpc_driver = hpc_driver
        self.target = target
        self.slurm_params_high_mem = slurm_params_high_mem
        self.slurm_params_low_mem = slurm_params_low_mem
        self.home_dir = home_dir
        self.state_file = os.path.join(self.home_dir, "pipeline_state.json")
        self.completed_steps = self.load_state()

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

    def load_state(self) -> set:
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return set(json.load(f))
        return set()

    def save_state(self, step_name: str):
        self.completed_steps.add(step_name)
        with open(self.state_file, 'w') as f:
            json.dump(list(self.completed_steps), f)

    def save_initial_state(self, step: str, charge: int, mult: int, solvent: str, Nimages: int) -> None:
        """
        Saves initial pipeline state.
        """
        os.chdir(self.home_dir)
        with open(FAILED_MARKER, "w") as f:
            f.write("")

        settings = {
            "step": step,
            "charge": charge,
            "mult": mult,
            "solvent": solvent,
            "Nimages": Nimages

        }
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        logging.info("Initial pipeline state saved.")

    def save_completion_state(self, steps: List[str], charge: int, mult: int, solvent: str, Nimages: int) -> None:
        """
        Saves final pipeline state with 'COMPLETED' marker.
        """
        os.chdir(self.home_dir)
        with open(COMPLETED_MARKER, "w") as f:
            f.write("")

        settings = {
            "step": "Completed",
            "charge": charge,
            "mult": mult,
            "solvent": solvent,
            "Nimages": Nimages,
            "steps": steps
        }
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        logging.info("Pipeline completed successfully.")

    def pipeline(self, step: str) -> bool:
        """
        Maps step strings to methods within StepRunner based on target type.
        """

        steps_mapping = {
            "OPT": self.geometry_optimisation,
            "NEB_TS": self.neb_ts,
            "NEB_CI": self.neb_ci,
            "TS": self.ts_opt,
            "IRC": self.irc_job,
            "SP": self.sp_calc,
        }

        step_function: Callable[[], bool] = steps_mapping.get(step.upper())
        if not step_function:
            logging.error(f"Step '{step}' is not recognized.")
            return False

        return step_function()

    def execute_step(self, step: str) -> bool:
        if step in self.completed_steps:
            logging.info(f"Step '{step}' already completed. Skipping.")
            return True

        success = self.pipeline(step)
        if success:
            self.save_state(step)
        else:
            logging.error(f"Step '{step}' failed.")
        return success

    def run_pipeline(self, steps: List[str]) -> bool:
        """
        Executes the entire pipeline sequence.
        """
        charge = self.target.charge if isinstance(
            self.target, Molecule) else self.target.educt.charge
        mult = self.target.mult if isinstance(
            self.target, Molecule) else self.target.educt.mult
        solvent = self.target.solvent if isinstance(
            self.target, Molecule) else self.target.educt.solvent
        Nimages = self.target.nimages if isinstance(
            self.target, Reaction) else 0
        self.save_initial_state(
            step=steps[0], charge=charge, mult=mult, solvent=solvent, Nimages=Nimages)

        for step in steps:

            success = self.execute_step(step)
            os.chdir(self.home_dir)
            if not success:
                logging.error(
                    f"Pipeline halted due to failure in step '{step}'.")
                with open(FAILED_MARKER, "w") as f:
                    f.write(f"Failed at step: {step}\n")
                return False

        self.save_completion_state(
            steps=steps, charge=charge, mult=mult, solvent=solvent, Nimages=Nimages)
        return True

    # Define all pipeline step methods
    def geometry_optimisation(self) -> bool:
        logging.info("Starting geometry optimization.")
        if isinstance(self.target, Reaction):
            self.make_folder("OPT")
            os.chdir("OPT")

            return self.target.optimise_reactants(self.hpc_driver, self.slurm_params_low_mem, trial=0, upper_limit=MAX_TRIALS)
        elif isinstance(self.target, Molecule):
            self.make_folder("OPT")
            os.chdir("OPT")
            return self.target.geometry_opt(self.hpc_driver, self.slurm_params_low_mem, trial=0, upper_limit=MAX_TRIALS)
        else:
            logging.error("Unsupported target type for geometry optimization.")
            return False

    def neb_ci(self) -> bool:
        logging.info("Starting NEB-CI calculation.")
        if isinstance(self.target, Reaction):
            self.make_folder("NEB")
            os.chdir("NEB")
            return self.target.neb_ci(self.hpc_driver, self.slurm_params_low_mem, trial=0, upper_limit=MAX_TRIALS)
        else:
            logging.error(
                "NEB-CI step is only applicable to Reaction objects.")
            return False

    def neb_ts(self, fast: bool, switch: bool) -> bool:
        logging.info("Starting NEB-TS calculation.")
        if isinstance(self.target, Reaction):
            self.make_folder("NEB")
            os.chdir("NEB")

            message, success = self.target.neb_ts(
                self.hpc_driver, self.slurm_params_low_mem, trial=0, upper_limit=MAX_TRIALS, fast=fast, switch=switch)
            if message == "UNCONVERGED":
                os.chdir("..")
                return self.handle_unconverged_neb(0, MAX_TRIALS, fast)
            elif message == "freq":
                os.chdir("..")
                return self.handle_failed_imagfreq(0, MAX_TRIALS, fast)
            else:
                return success
        else:
            logging.error(
                "NEB-TS step is only applicable to Reaction objects.")
            return False

    def ts_opt(self) -> bool:
        logging.info("Starting TS optimization.")
        if isinstance(self.target, Reaction):
            self.make_folder("TS")

            self.hpc_driver.shell_command("cp NEB/*.hess TS/guess.hess")
            os.chdir("TS")

            return self.target.transition_state.ts_opt(self.hpc_driver, self.slurm_params, trial=0, upper_limit=MAX_TRIALS)
        elif isinstance(self.target, Molecule):
            return self.target.ts_opt(self.hpc_driver, self.slurm_params, trial=0, upper_limit=MAX_TRIALS)
        else:

            logging.error(
                "TS optimization is only applicable to Reaction objects.")
            return False

    def irc_job(self) -> bool:
        logging.info("Starting IRC job.")
        if isinstance(self.target, Reaction):
            self.make_folder("IRC")
            self.hpc_driver.shell_command(
                f"cp TS/{self.name}_freq.hess IRC/TS.xyz")
            return self.target.transition_state.irc_job(self.hpc_driver, self.slurm_params_low_mem, trial=0, upper_limit=MAX_TRIALS)
        elif isinstance(self.target, Molecule):
            return self.target.irc_job(self.hpc_driver, self.slurm_params_low_mem, trial=0, upper_limit=MAX_TRIALS)
        else:
            logging.error("Unsupported target type for IRC job.")
            return False

    def sp_calc(self) -> bool:
        logging.info("Starting single point calculation.")
        if isinstance(self.target, Reaction):
            self.make_folder("SP")
            os.chdir("SP")
            return self.target.sp_calc(self.hpc_driver, self.slurm_params)
        elif isinstance(self.target, Molecule):
            self.make_folder("SP")
            os.chdir("SP")
            return self.target.sp_calc(self.hpc_driver, self.slurm_params)
        else:
            logging.error(
                "Unsupported target type for single point calculation.")
            return False

    def handle_unconverged_neb(self,  uper_limit, fast):
        if "xtb" in self.target.method.lower():
            print("Restating with FAST-NEB r2scan-3c")
            self.reaction.method = "r2scan-3c"
            self.reaction.educt.method = "r2scan-3c"
            self.reaction.product.method = "r2scan-3c"
            print("Need to reoptimize reactants")
            return self.neb_ts(trial=0, upper_limit=uper_limit, fast=True, switch=True)
        elif fast:
            print("Restarting with NEB-TS r2scan-3c")
            return self.neb_ts(trial=0, upper_limit=uper_limit, fast=False, switch=False)
        else:
            return self.neb_ci()

    def handle_failed_imagfreq(self, trial, upper_limit, fast):
        """
       Handles when NEB-TS finished but no significant imag freq is found.
        """

        # check what method was used

        if "xtb" in self.reaction.method.lower():
            print("No significant imaginary frequency found. Retrying with r2scan-3c.")
            self.reaction.method = "r2scan-3c"
            self.reaction.educt.method = "r2scan-3c"
            self.reaction.product.method = "r2scan-3c"
            return self.neb_ts(trial=0, upper_limit=upper_limit, fast=True, switch=True)
        elif fast:
            print("No significant imaginary frequency found. Retrying with fast=False.")
            return self.neb_ts(trial=0, upper_limit=upper_limit, fast=False)
        else:
            print("No significant imag freq found.")
            print("Do a neb_ci run to try to get TS guess.")
            return self.neb_ci()
