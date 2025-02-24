import json
import os
import logging
import shutil
import glob
import sys
from typing import List, Callable, Dict, Union

# Ensure correct import paths
from orca_pipeline.chemistry import Reaction, Molecule, rmsd
from orca_pipeline.hpc_driver import HPCDriver
from orca_pipeline.constants import (
    MAX_TRIALS,
    RETRY_DELAY,
    SLURM_PARAMS_BIG_HIGH_MEM,
    SLURM_PARAMS_BIG_LOW_MEM,
    SLURM_PARAMS_SMALL_HIGH_MEM,
    SLURM_PARAMS_SMALL_LOW_MEM,
    DEFAULT_STEPS,
    LOW_MEM_ELEMENTS,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# TODO Restart/ Methods should check if a folder exists, if some steps are full filled
# TODO Abortion due to other events than job failuer, slurm, my mistake etc.
# TODO if last completed step ist last step, program aborts with errro, should gracefully terminate and tell the user that pipeline is done


class StepRunner:
    # TODO make naming scheme of file names better.
    def __init__(
        self,
        hpc_driver: HPCDriver,
        target: Union[Reaction, Molecule],
        steps: List[str] = DEFAULT_STEPS,
        slurm_params_high_mem: Dict = SLURM_PARAMS_BIG_HIGH_MEM,
        slurm_params_low_mem: Dict = SLURM_PARAMS_BIG_LOW_MEM,
        home_dir: str = ".",
    ):
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
                "Nimages": 0,
            }
            self.state = initial_state
        self.completed_steps = self.state.get("last_completed_step", "")
        if self.completed_steps:
            index = self.steps.index(self.completed_steps)

            next_step = self.steps[index + 1]
            self.steps = self.steps[self.steps.index(next_step) :]
            logging.info(f"Resuming pipeline with steps: {self.steps}")
            match next_step:
                case "NEB_TS" | "NEB_CI":
                    if not os.path.exists("OPT"):
                        logging.error("Cannot resume NEB-TS without OPT step.")
                        sys.exit(1)
                    self.target.educt = Molecule.from_xyz(
                        filepath="OPT/educt_opt.xyz",
                        charge=self.target.educt.charge,
                        mult=self.target.educt.mult,
                        solvent=self.target.educt.solvent,
                        method=self.target.educt.method,
                        sp_method=self.target.educt.sp_method,
                        name="educt_opt",
                    )
                    self.target.product = Molecule.from_xyz(
                        filepath="OPT/product_opt.xyz",
                        charge=self.target.product.charge,
                        mult=self.target.product.mult,
                        solvent=self.target.product.solvent,
                        method=self.target.product.method,
                        sp_method=self.target.product.sp_method,
                        name="product_opt",
                    )
                case "TS":
                    if isinstance(self.target, Reaction):
                        if self.target.transition_state is None:
                            if not os.path.exists("NEB"):
                                logging.error(
                                    "Cannot resume TS without NEB step providing a guess."
                                )
                                sys.exit(1)
                            self.target.educt = Molecule.from_xyz(
                                filepath="NEB/educt.xyz",
                                charge=self.target.educt.charge,
                                mult=self.target.educt.mult,
                                solvent=self.target.educt.solvent,
                                method=self.target.educt.method,
                                sp_method=self.target.educt.sp_method,
                                name="educt_opt",
                            )
                            self.target.product = Molecule.from_xyz(
                                filepath="NEB/product.xyz",
                                charge=self.target.product.charge,
                                mult=self.target.product.mult,
                                solvent=self.target.product.solvent,
                                method=self.target.product.method,
                                sp_method=self.target.product.sp_method,
                                name="product_opt",
                            )
                            if os.path.exists("TS") and os.path.exists(
                                "TS/ts_guess.xyz"
                            ):
                                self.target.transition_state = Molecule.from_xyz(
                                    filepath="TS/ts_guess.xyz",
                                    charge=self.target.educt.charge,
                                    mult=self.target.educt.mult,
                                    solvent=self.target.educt.solvent,
                                    method=self.target.educt.method,
                                    sp_method=self.target.educt.sp_method,
                                    name="ts_guess",
                                )
                            else:
                                file_path = ""

                                pattern = "NEB/*TS_converged.xyz"
                                matches = glob.glob(pattern)
                                if matches:
                                    file_path = matches[0]
                                else:
                                    pattern = "NEB/*CI_converged.xyz"
                                    matches = glob.glob(pattern)
                                    if matches:
                                        file_path = matches[0]
                                print(f"Using {file_path} as TS guess")

                                self.target.transition_state = Molecule.from_xyz(
                                    filepath=file_path,
                                    charge=self.target.educt.charge,
                                    mult=self.target.educt.mult,
                                    solvent=self.target.educt.solvent,
                                    method=self.target.educt.method,
                                    sp_method=self.target.educt.sp_method,
                                    name="ts_guess",
                                )
                case "IRC":
                    if not os.path.exists("TS"):
                        logging.error("Cannot resume IRC without TS step.")
                        sys.exit(1)
                    self.target.educt = Molecule.from_xyz(
                        filepath="NEB/educt.xyz",
                        charge=self.target.educt.charge,
                        mult=self.target.educt.mult,
                        solvent=self.target.educt.solvent,
                        method=self.target.educt.method,
                        sp_method=self.target.educt.sp_method,
                        name="educt",
                    )
                    self.target.product = Molecule.from_xyz(
                        filepath="NEB/product.xyz",
                        charge=self.target.product.charge,
                        mult=self.target.product.mult,
                        solvent=self.target.product.solvent,
                        method=self.target.product.method,
                        sp_method=self.target.product.sp_method,
                        name="product",
                    )

                    pattern = "TS/*_TS_opt.xyz"
                    matches = glob.glob(pattern)
                    if matches:
                        self.target.transition_state = Molecule.from_xyz(
                            filepath=matches[0],
                            charge=self.target.educt.charge,
                            mult=self.target.educt.mult,
                            solvent=self.target.educt.solvent,
                            method=self.target.educt.method,
                            sp_method=self.target.educt.sp_method,
                            name="ts",
                        )
                    else:
                        logging.erro("Cannot resume IRC step could not load TS_opt")
                        sys.exit(1)
                case "SP":
                    if isinstance(self.target, Reaction):
                        if not os.path.exists("TS"):
                            logging.error("Cannot resume SP without TS step.")
                            sys.exit(1)
                        if os.path.exists("OPT") and not os.path.exists(
                            "CONF/best_confs_opt"
                        ):
                            self.target.educt = Molecule.from_xyz(
                                filepath="OPT/educt_opt.xyz",
                                charge=self.target.educt.charge,
                                mult=self.target.educt.mult,
                                solvent=self.target.educt.solvent,
                                method=self.target.educt.method,
                                sp_method=self.target.educt.sp_method,
                                name="educt",
                            )
                            self.target.product = Molecule.from_xyz(
                                filepath="OPT/product_opt.xyz",
                                charge=self.target.product.charge,
                                mult=self.target.product.mult,
                                solvent=self.target.product.solvent,
                                method=self.target.product.method,
                                sp_method=self.target.product.sp_method,
                                name="product",
                            )
                        elif os.path.exists("CONF/best_confs_opt"):
                            self.target.educt = Molecule.from_xyz(
                                filepath="CONF/best_confs_opt/educt_opt.xyz",
                                charge=self.target.educt.charge,
                                mult=self.target.educt.mult,
                                solvent=self.target.educt.solvent,
                                method=self.target.educt.method,
                                sp_method=self.target.educt.sp_method,
                                name="educt",
                            )
                            self.target.product = Molecule.from_xyz(
                                filepath="CONF/best_confs_opt/educt_opt.xyz",
                                charge=self.target.product.charge,
                                mult=self.target.product.mult,
                                solvent=self.target.product.solvent,
                                method=self.target.product.method,
                                sp_method=self.target.product.sp_method,
                                name="product",
                            )
                        self.target.transition_state = Molecule.from_xyz(
                            filepath="TS/ts_guess_TS_opt.xyz",
                            charge=self.target.educt.charge,
                            mult=self.target.educt.mult,
                            solvent=self.target.educt.solvent,
                            method=self.target.educt.method,
                            sp_method=self.target.educt.sp_method,
                            name="ts",
                        )
                    elif isinstance(self.target, Molecule):
                        if os.path.exists("OPT"):
                            self.target = Molecule.from_xyz(
                                filepath=f"OPT/{self.target.name}_opt.xyz",
                                charge=self.target.charge,
                                mult=self.target.mult,
                                solvent=self.target.solvent,
                                method=self.target.method,
                                sp_method=self.target.sp_method,
                                name=f"{self.target.name}",
                            )
                        if os.path.exists("CONF"):
                            self.target = Molecule.from_xyz(
                                filepath=f"CONF/{self.target.name}_opt.xyz",
                                charge=self.target.charge,
                                mult=self.target.mult,
                                solvent=self.target.solvent,
                                method=self.target.method,
                                sp_method=self.target.sp_method,
                                name=f"{self.target.name}",
                            )
                case "CONF":
                    if isinstance(self.target, Reaction):
                        if os.path.exists("TS"):
                            self.target.transition_state = Molecule.from_xyz(
                                filepath="TS/ts_guess_TS_opt.xyz",
                                charge=self.target.educt.charge,
                                mult=self.target.educt.mult,
                                solvent=self.target.educt.solvent,
                                method=self.target.educt.method,
                                sp_method=self.target.educt.sp_method,
                                name="ts",
                            )
                        if os.path.exists("OPT"):
                            self.target.educt = Molecule.from_xyz(
                                filepath="OPT/educt_opt.xyz",
                                charge=self.target.educt.charge,
                                mult=self.target.educt.mult,
                                solvent=self.target.educt.solvent,
                                method=self.target.educt.method,
                                sp_method=self.target.educt.sp_method,
                                name="educt_opt",
                            )
                            self.target.product = Molecule.from_xyz(
                                filepath="OPT/product_opt.xyz",
                                charge=self.target.product.charge,
                                mult=self.target.product.mult,
                                solvent=self.target.product.solvent,
                                method=self.target.product.method,
                                sp_method=self.target.product.sp_method,
                                name="product_opt",
                            )
                case "PLOT":
                    if isinstance(self.target, Molecule):
                        print("Not supported for Molecule objects")
                        sys.exit(1)

                    if not os.path.exists("SP"):
                        logging.error("Cannot resume PLOT without SP step.")
                        sys.exit(1)
                    method = (
                        self.target.educt.method
                        if not "xtb" in self.target.educt.method.lower()
                        else "r2scan-3c"
                    )
                    self.target.educt = Molecule.from_xyz(
                        filepath="SP/educt.xyz",
                        charge=self.target.educt.charge,
                        mult=self.target.educt.mult,
                        method=self.target.educt.method,
                        sp_method=self.target.educt.sp_method,
                        solvent=self.target.educt.solvent,
                        name="educt",
                    )
                    self.target.product = Molecule.from_xyz(
                        filepath="SP/product.xyz",
                        charge=self.target.product.charge,
                        mult=self.target.product.mult,
                        method=self.target.product.method,
                        sp_method=self.target.product.sp_method,
                        solvent=self.target.product.solvent,
                        name="product",
                    )
                    self.target.transition_state = Molecule.from_xyz(
                        filepath="SP/ts.xyz",
                        charge=self.target.educt.charge,
                        mult=self.target.educt.mult,
                        method=self.target.educt.method,
                        sp_method=self.target.educt.sp_method,
                        solvent=self.target.educt.solvent,
                        name="ts",
                    )

                    #

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
            with open(self.state_file, "r") as f:
                return json.load(f)
        return {}

    def save_state(self) -> None:
        """
        Saves initial pipeline state.
        """
        os.chdir(self.home_dir)

        settings = self.state
        with open("pipeline_state.json", "w") as f:
            json.dump(settings, f, indent=4)

    def update_state(self, step: str) -> None:
        self.state["last_completed_step"] = step
        self.state["charge"] = self.target.charge
        self.state["mult"] = self.target.mult
        self.state["solvent"] = self.target.solvent
        if isinstance(self.target, Reaction):
            self.state["Nimages"] = self.target.nimages

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
            "CONF": self.conf_calc,
            "PLOT": self.plot,
            "FOD": self.fod_job,
        }

        step_function: Callable[[], bool] = steps_mapping.get(step.upper())
        if not step_function:
            logging.error(f"Step '{step}' is not recognized.")
            return False

        return step_function()

    def execute_step(self, step: str) -> bool:
        success = self.pipeline(step)
        if success:
            self.update_state(step)
            self.save_state()
        else:
            logging.error(f"Step '{step}' failed.")
        return success

    def run_pipeline(self) -> bool:
        """
        Executes the entire pipeline sequence.
        """

        for step in self.steps:
            success = self.execute_step(step)
            os.chdir(self.home_dir)
            if not success:
                logging.error(f"Pipeline halted due to failure in step '{step}'.")
                with open("FAILED.out", "w") as f:
                    f.write(f"Failed at step: {step}\n")
                self.save_state()
                return False

        self.save_state()
        return True

    # Define all pipeline step methods
    def geometry_optimisation(self) -> bool:
        logging.info("Starting geometry optimization.")
        if isinstance(self.target, Reaction):
            self.make_folder("OPT")
            os.chdir("OPT")

            return self.target.optimise_reactants(
                self.hpc_driver,
                self.slurm_params_low_mem,
                trial=0,
                upper_limit=MAX_TRIALS,
            )
        elif isinstance(self.target, Molecule):
            self.make_folder("OPT")
            os.chdir("OPT")
            return self.target.geometry_opt(
                self.hpc_driver,
                self.slurm_params_low_mem,
                trial=0,
                upper_limit=MAX_TRIALS,
            )
        else:
            logging.error("Unsupported target type for geometry optimization.")
            return False

    def neb_ci(self) -> bool:
        logging.info("Starting NEB-CI calculation.")
        if isinstance(self.target, Reaction):
            self.make_folder("NEB")
            os.chdir("NEB")
            return self.target.neb_ci(
                self.hpc_driver,
                self.slurm_params_low_mem,
                trial=0,
                upper_limit=MAX_TRIALS,
            )
        else:
            logging.error("NEB-CI step is only applicable to Reaction objects.")
            return False

    def neb_ts(self) -> bool:
        logging.info("Starting NEB-TS calculation.")
        if isinstance(self.target, Reaction):
            self.make_folder("NEB")
            os.chdir("NEB")

            message, success = self.target.neb_ts(
                self.hpc_driver,
                self.slurm_params_low_mem,
                trial=0,
                upper_limit=MAX_TRIALS,
            )
            if message == "failed":
                os.chdir("..")
                return self.handle_failed_neb(MAX_TRIALS)
            else:
                return success
        else:
            logging.error("NEB-TS step is only applicable to Reaction objects.")
            return False

    def ts_opt(self) -> bool:
        logging.info("Starting TS optimization.")
        if isinstance(self.target, Reaction):
            self.make_folder("TS")
            if not "xtb" in self.target.educt.method.lower():
                if os.path.exists(f"NEB/{self.target.transition_state.name}_freq.hess"):
                    self.hpc_driver.shell_command(
                        f"cp NEB/{self.target.transition_state.name}_freq.hess TS/{self.target.transition_state.name}_guess.hess"
                    )
            else:
                print(
                    "If xtb was chosen as method r2scan-3c is used for all subsequent steps starting from TS-optimization"
                )
                self.target.educt.method = "r2scan-3c"
                self.target.product.method = "r2scan-3c"
                self.target.transition_state.method = "r2scan-3c"
                if "xtb" in self.target.educt.sp_method.lower():
                    print(
                        "If xtb was chosen as sp_method r2scan-3c is used for all subsequent steps starting from TS-optimization"
                    )
                    self.target.educt.sp_method = "r2scan-3c"
                    self.target.product.sp_method = "r2scan-3c"
                    self.target.transition_state.sp_method = "r2scan-3c"
                print("slurm_params will be adjusted accordingly")

                self.slurm_params_high_mem = SLURM_PARAMS_SMALL_HIGH_MEM
                self.slurm_params_low_mem = SLURM_PARAMS_SMALL_LOW_MEM

                for atom in self.target.educt.atoms:
                    if atom not in LOW_MEM_ELEMENTS:
                        self.slurm_params_high_mem = SLURM_PARAMS_BIG_HIGH_MEM
                        self.slurm_params_low_mem = SLURM_PARAMS_BIG_LOW_MEM
                        break

            os.chdir("TS")

            return self.target.transition_state.ts_opt(
                self.hpc_driver,
                self.slurm_params_high_mem,
                trial=0,
                upper_limit=MAX_TRIALS,
            )
        elif isinstance(self.target, Molecule):
            return self.target.ts_opt(
                self.hpc_driver,
                self.slurm_params_high_mem,
                trial=0,
                upper_limit=MAX_TRIALS,
            )
        else:
            logging.error("TS optimization is only applicable to Reaction objects.")
            return False

    def irc_job(self) -> bool:
        logging.info("Starting IRC job.")
        if isinstance(self.target, Reaction):
            self.make_folder("IRC")
            print()
            self.hpc_driver.shell_command(
                f"cp TS/{self.target.transition_state.name}_freq.hess IRC/"
            )
            os.chdir("IRC")
            return self.target.transition_state.irc_job(
                self.hpc_driver,
                self.slurm_params_low_mem,
                trial=0,
                upper_limit=MAX_TRIALS,
            )
        elif isinstance(self.target, Molecule):
            return self.target.irc_job(
                self.hpc_driver,
                self.slurm_params_low_mem,
                trial=0,
                upper_limit=MAX_TRIALS,
            )
        else:
            logging.error("Unsupported target type for IRC job.")
            return False

    def sp_calc(self) -> bool:
        logging.info("Starting single point calculation.")
        if isinstance(self.target, Reaction):
            self.make_folder("SP")
            os.chdir("SP")
            return self.target.sp_calc(self.hpc_driver, self.slurm_params_low_mem)
        elif isinstance(self.target, Molecule):
            self.make_folder("SP")
            os.chdir("SP")
            return self.target.sp_calc(self.hpc_driver, self.slurm_params_low_mem)
        else:
            logging.error("Unsupported target type for single point calculation.")
            return False

    def conf_calc(self) -> bool:
        logging.info("Starting conformer calculation.")
        if isinstance(self.target, Molecule):
            if "xtb" in self.target.method.lower():
                print(
                    "Changing method to r2scan-3c to optimize best confomer, crest_best is already xtb2 optimized"
                )
                self.target.method = "r2scan-3c"
            self.make_folder("CONF")
            os.chdir("CONF")
            success = False
            if self.target.conf_method == "CREST":
                success = self.target.get_lowest_confomer_crest(
                    self.hpc_driver, self.slurm_params_low_mem
                )
            else:
                success = self.target.get_lowest_confomer_goat(
                    self.hpc_driver, self.slurm_params_low_mem
                )

            if success:
                print("Conformer calculation successful. Optimizing best confomer")
                return self.target.geometry_opt(
                    self.hpc_driver,
                    self.slurm_params_low_mem,
                    trial=0,
                    upper_limit=MAX_TRIALS,
                    tight=True,
                )
            else:
                print("Conformer calculation failed.")
                return False

        elif isinstance(self.target, Reaction):
            self.make_folder("CONF")
            if "xtb" in self.target.method.lower():
                print(
                    "Changing method to r2scan-3c to optimize best confomer, crest_best is already xtb2 optimized"
                )
                self.target.method = "r2scan-3c"
                self.target.educt.method = "r2scan-3c"
                self.target.product.method = "r2scan-3c"
                self.target.transition_state.method = "r2scan-3c"
                print("SP method will be unchanged")
            if os.path.exists("IRC"):
                print("Starting conformer calculation from IRC or QRC end points")

                if os.path.exists("IRC/QRC"):
                    self.hpc_driver.shell_command("cp IRC/*Backwards.xyz CONF/back.xyz")
                    self.hpc_driver.shell_command("cp IRC/*Forwards.xyz CONF/front.xyz")
                else:
                    self.hpc_driver.shell_command("cp IRC/*IRC_B.xyz CONF/back.xyz")
                    self.hpc_driver.shell_command("cp IRC/*IRC_F.xyz CONF/front.xyz")
                print("Checking which IRC endpoint is closer to educt")
                os.chdir("CONF")
                self.target.educt.to_xyz("educt.xyz")
                # self.target.product.to_xyz("product.xyz")
                rmsd_educt_irc_b = rmsd("educt.xyz", "back.xyz")
                rmsd_educt_irc_f = rmsd("educt.xyz", "front.xyz")
                if rmsd_educt_irc_b < rmsd_educt_irc_f:
                    self.target.educt = Molecule.from_xyz(
                        f"back.xyz",
                        charge=self.target.educt.charge,
                        mult=self.target.educt.mult,
                        solvent=self.target.educt.solvent,
                        method=self.target.educt.method,
                        sp_method=self.target.educt.sp_method,
                        name="educt",
                    )
                    self.target.product = Molecule.from_xyz(
                        "front.xyz",
                        charge=self.target.product.charge,
                        mult=self.target.product.mult,
                        solvent=self.target.product.solvent,
                        method=self.target.product.method,
                        sp_method=self.target.product.sp_method,
                        name="product",
                    )
                else:
                    self.target.educt = Molecule.from_xyz(
                        "front.xyz",
                        charge=self.target.educt.charge,
                        mult=self.target.educt.mult,
                        solvent=self.target.educt.solvent,
                        method=self.target.educt.method,
                        sp_method=self.target.educt.sp_method,
                        name="educt",
                    )
                    self.target.product = Molecule.from_xyz(
                        "back.xyz",
                        charge=self.target.product.charge,
                        mult=self.target.product.mult,
                        solvent=self.target.product.solvent,
                        method=self.target.product.method,
                        sp_method=self.target.product.sp_method,
                        name="product",
                    )
            else:
                print("Starting conformer calculation form given reactants")
                os.chdir("CONF")

            self.make_folder("educt_confs")
            self.make_folder("product_confs")
            success = False
            if self.target.conf_method == "CREST":
                success = self.target.get_lowest_confomers(
                    self.hpc_driver, self.slurm_params_low_mem, crest=True
                )
            else:
                sucess = self.target.get_lowest_confomers(
                    self.hpc_driver, self.slurm_params_low_mem, crest=False
                )
            if success:
                print("Conformer calculation successful. Optimizing best confomers")
                self.make_folder("best_confs_opt")
                os.chdir("best_confs_opt")
                return self.target.optimise_reactants(
                    self.hpc_driver,
                    self.slurm_params_low_mem,
                    trial=0,
                    upper_limit=MAX_TRIALS,
                    tight=True,
                )
            else:
                print("Conformer calculation failed.")
                return False

        print("Unsupported target type for conformer calculation.")
        return False

    def plot(self) -> bool:
        if isinstance(self.target, Molecule):
            print("Not supported for Molecule objects")
            return False

        # TODO implement plotting for reaction objects
        print("Gathering energies for plot wil be saved to CSV in parent directory")
        return self.target.get_reaction_energies()

    def fod_job(self) -> bool:
        self.make_folder("FOD")
        os.chdir("FOD")
        return self.target.fod_calc(self.hpc_driver, self.slurm_params_low_mem)

    def handle_failed_neb(self, uper_limit):
        if "xtb" in self.target.method.lower():
            print("Restating with FAST-NEB r2scan-3c")
            self.target.method = "r2scan-3c"
            self.target.educt.method = "r2scan-3c"
            self.target.product.method = "r2scan-3c"
            self.target.fast = True
            self.target.nimages = 16

            print("Need to reoptimize reactants")
            if not self.geometry_optimisation():
                print("Failed to reoptimize reactants.")
                return False
            print("Reactants reoptimized. Restarting FAST-NEB-TS with r2scan-3c.")
            os.chdir("..")

            return self.neb_ts()
        elif self.target.fast:
            self.target.fast = False
            self.target.nimages = 12
            print("Restarting with NEB-TS r2scan-3c")
            return self.neb_ts()
        elif (
            "xtb" not in self.target.method and self.target.fast == False
        ) and self.target.zoom == False:
            print("Try with zoom-option")
            self.target.zoom = True
            self.target.nimages = 12
            return self.neb_ts()

        else:
            print("Trying to get a better initial guess with NEB-CI")
            self.target.zoom = False
            self.target.nimages = 12
            return self.neb_ci()
