
import json
import os
import logging
import shutil
import glob
import sys
from typing import List, Callable, Dict, Union
from chemistry import Reaction, Molecule  # Ensure correct import paths
from hpc_driver import HPCDriver
from constants import MAX_TRIALS, RETRY_DELAY, SLURM_PARAMS_BIG_HIGH_MEM, SLURM_PARAMS_BIG_LOW_MEM, DEFAULT_STEPS
# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

SETTINGS_FILE = "settings.json"


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
        self.state = self.load_state()

        self.steps = steps
        if not self.state:
            logging.info("No previous state found.")
            initial_state = {
                "steps": self.steps,
                "last_completed_step": "",
                "charge": 0,
                "mult": 1,
                "solvent": "",
                "Nimages": 0
            }
            self.state = initial_state
        self.completed_steps = self.state.get("last_completed_steps", "")
        if self.completed_steps:
            index = self.steps.index(self.completed_steps)
            next_step = self.steps[index+1]
            self.steps = self.steps[self.steps.index(next_step):]
            logging.info(
                f"Resuming pipeline with steps: {self.steps}")
            match next_step:
                case "NEB_TS" | "NEB_CI":
                    if not os.path.exists("OPT"):
                        logging.error(
                            "Cannot resume NEB-TS without OPT step.")
                        sys.exit(1)
                    self.target.educt = Molecule.from_xyz(filepath="OPT/educt.xyz", charge=self.target.educt.charge, mult=self.target.educt.mult,
                                                          solvent=self.target.educt.solvent, method=self.target.educt.method, sp_method=self.target.educt.sp_method)
                    self.target.product = Molecule.from_xyz(filepath="OPT/product.xyz", charge=self.target.product.charge, mult=self.target.product.mult,
                                                            solvent=self.target.product.solvent, method=self.target.product.method, sp_method=self.target.product.sp_method)
                case "TS":
                    if self.target.transition_state is None:
                        if not os.path.exists("NEB"):
                            logging.error(
                                "Cannot resume TS without NEB step providing a guess.")
                            sys.exit(1)
                        self.target.educt = Molecule.from_xyz(filepath="NEB/educt.xyz", charge=self.target.educt.charge, mult=self.target.educt.mult,
                                                              solvent=self.target.educt.solvent, method=self.target.educt.method, sp_method=self.target.educt.sp_method)
                        self.target.product = Molecule.from_xyz(filepath="NEB/product.xyz", charge=self.target.product.charge, mult=self.target.product.mult,
                                                                solvent=self.target.product.solvent, method=self.target.product.method, sp_method=self.target.product.sp_method)
                        if os.path.exists("TS"):
                            self.target.transition_state = Molecule.from_xyz(filepath="TS/ts_guess.xyz", charge=self.target.educt.charge,
                                                                             mult=self.target.educt.mult, solvent=self.target.educt.solvent, method=self.target.educt.method, sp_method=self.target.educt.sp_method)
                        else:
                            file_path = ""
                            if os.path.exists("NEB/neb-TS_converged.xyz"):
                                file_path = "NEB/neb-TS_converged.xyz"
                            elif os.path.exists("NEB/neb-TS.xyz"):
                                pattern = "NEB/*_NEB-TS_converged.xyz"
                                matches = glob.glob(pattern)
                                if matches:
                                    file_path = matches[0]

                            self.target.transition_state = Molecule.from_xyz(filepath=file_path, charge=self.target.educt.charge,
                                                                             mult=self.target.educt.mult, solvent=self.target.educt.solvent, method=self.target.educt.method, sp_method=self.target.educt.sp_method)
                case "IRC":
                    if not os.path.exists("TS"):
                        logging.error(
                            "Cannot resume IRC without TS step.")
                        sys.exit(1)
                    self.target.educt = Molecule.from_xyz(filepath="NEB/educt.xyz", charge=self.target.educt.charge, mult=self.target.educt.mult,
                                                          solvent=self.target.educt.solvent, method=self.target.educt.method, sp_method=self.target.educt.sp_method)
                    self.target.product = Molecule.from_xyz(filepath="NEB/product.xyz", charge=self.target.product.charge, mult=self.target.product.mult,
                                                            solvent=self.target.product.solvent, method=self.target.product.method, sp_method=self.target.product.sp_method)

                    pattern = "TS/*_TS_opt.xyz"
                    matches = glob.glob(pattern)
                    if matches:
                        self.target.transition_state = Molecule.from_xyz(filepath=matches[0], charge=self.target.educt.charge,
                                                                         mult=self.target.educt.mult, solvent=self.target.educt.solvent, method=self.target.educt.method, sp_method=self.target.educt.sp_method)
                    else:
                        logging.erro(
                            "Cannot resume IRC step could not load TS_opt")
                        sys.exit(1)

    def make_folder(self, dir_name: str) -> None:
        """
        Creates a new folder, removing existing one if present.
        """
        path = os.path.join(os.getcwd(), dir_name)
        logging.info(f"Creating folder at path: {path}")
        if os.path.exists(path):
            if os.path.isdir(path):
                logging.info(f"Removing existing folder {path}")
                shutil.rmtree(path)
            else:
                logging.info(f"Removing existing file {path}")
                os.remove(path)
        os.makedirs(path)
        logging.info(f"Created folder {path}")

    def load_state(self) -> dict:
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                return json.load(f)
        return {}

    def save_state(self, steps: list, charge: int, mult: int, solvent: str, Nimages: int) -> None:
        """
        Saves initial pipeline state.
        """
        os.chdir(self.home_dir)

        with open("w") as f:
            f.write("")

        settings = {
            "steps": steps,
            "charge": charge,
            "mult": mult,
            "solvent": solvent,
            "Nimages": Nimages

        }
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=4)
        logging.info("Initial pipeline state saved.")

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

    def run_pipeline(self) -> bool:
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
            step=self.steps[0], charge=charge, mult=mult, solvent=solvent, Nimages=Nimages)

        for step in self.steps:

            success = self.execute_step(step)
            os.chdir(self.home_dir)
            if not success:
                logging.error(
                    f"Pipeline halted due to failure in step '{step}'.")
                with open(FAILED_MARKER, "w") as f:
                    f.write(f"Failed at step: {step}\n")
                return False

        self.save_completion_state(
            steps=self.steps, charge=charge, mult=mult, solvent=solvent, Nimages=Nimages)
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

    def neb_ts(self) -> bool:
        logging.info("Starting NEB-TS calculation.")
        if isinstance(self.target, Reaction):

            self.make_folder("NEB")
            os.chdir("NEB")

            message, success = self.target.neb_ts(
                self.hpc_driver, self.slurm_params_low_mem, trial=0, upper_limit=MAX_TRIALS)
            if message == "failed":
                os.chdir("..")
                return self.handle_failed_neb(MAX_TRIALS)
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

    def handle_failed_neb(self,  uper_limit):
        if "xtb" in self.target.method.lower():
            print("Restating with FAST-NEB r2scan-3c")
            self.target.method = "r2scan-3c"
            self.target.educt.method = "r2scan-3c"
            self.target.product.method = "r2scan-3c"
            self.target.fast = True
            self.target.nimages = 16

            print("Need to reoptimize reactants")
            if not self.target.optimise_reactants(
                    self.hpc_driver, self.slurm_params_low_mem, trial=0, upper_limit=uper_limit):
                print("Failed to reoptimize reactants.")
                return False
            print("Reactants reoptimized. Restarting FAST-NEB-TS with r2scan.")

            return self.neb_ts()
        elif self.target.fast:
            self.target.fast = False
            self.target.nimages = 12
            print("Restarting with NEB-TS r2scan-3c")
            return self.neb_ts()
        else:
            print("Trying to get a better initial guess with NEB-CI")
            self.target.nimages = 16
            return self.neb_ci()
