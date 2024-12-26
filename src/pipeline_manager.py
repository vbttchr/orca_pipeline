#!/usr/bin/env python3
"""
pipeline_manager.py

Contains the PipelineManager class for coordinating pipeline steps, 
handling restarts, and saving/loading pipeline state.
"""

import sys
import os
from typing import List, Tuple

from step_runner import StepRunner

SETTINGS_FILE = "settings_neb_pipeline.txt"
MAX_TRIALS = 5


class PipelineManager:
    """
    Manages the overall pipeline: which steps to run, handling restarts, etc.
    """

    def __init__(self,
                 step_runner: StepRunner,
                 home_dir: str = None) -> None:
        self.step_runner = step_runner
        self.home_dir = home_dir or os.getcwd()

    def load_settings(self) -> Tuple[str, int, int, str, int, bool]:
        """
        Loads settings from file to resume from a previous state.
        Returns (step, charge, mult, solvent, Nimages, xtb).
        """
        print("Restart detected, loading previous settings.")
        if not os.path.exists(SETTINGS_FILE):
            print("Settings file not found, aborting.")
            sys.exit(1)

        with open(SETTINGS_FILE, 'r') as f:
            settings = dict(line.strip().split(':')
                            for line in f if ':' in line)

        step = settings.get('step', '')
        charge = int(settings.get('charge', 0))
        mult = int(settings.get('mult', 1))
        solvent = settings.get('solvent', '')
        Nimages = int(settings.get('Nimages', 16))
        xtb = settings.get('xtb', 'False') == 'True'
        return step, charge, mult, solvent, Nimages, xtb

    def save_failure_state(self,
                           step: str,
                           charge: int,
                           mult: int,
                           solvent: str,
                           Nimages: int) -> None:
        """
        Saves the current pipeline state to file and marks as 'FAILED'.
        """
        os.chdir(self.home_dir)
        with open("FAILED", "w") as f:
            f.write("")

        with open(SETTINGS_FILE, 'w') as f:
            f.write(f"step: {step}\n")
            f.write(f"charge: {charge}\n")
            f.write(f"mult: {mult}\n")
            f.write(f"solvent: {solvent}\n")
            f.write(f"Nimages: {Nimages}\n")

    def save_completion_state(self,
                              steps: List[str],
                              charge: int,
                              mult: int,
                              solvent: str,
                              Nimages: int) -> None:
        """
        Saves final pipeline state with 'COMPLETED' marker.
        """
        os.chdir(self.home_dir)
        with open("COMPLETED", "w") as f:
            f.write("")

        with open(SETTINGS_FILE, 'w') as f:
            f.write("step: Completed\n")
            f.write(f"charge: {charge}\n")
            f.write(f"mult: {mult}\n")
            f.write(f"solvent: {solvent}\n")
            f.write(f"Nimages: {Nimages}\n")
            f.write(f"steps: {','.join(steps)}\n")

    def pipeline(self,
                 step: str,
                 ) -> bool:
        """
        Maps step strings to methods in StepRunner.
        """
        fast = False
        if "NEB" in step:
            fast = False if "xtb" in self.step_runner.reaction.method.lower() else True
        steps_mapping = {
            "OPT": lambda: self.step_runner.geometry_optimisation(trial=0, upper_limit=MAX_TRIALS),

            "NEB_TS": lambda: self.step_runner.neb_ts(trial=0, upper_limit=MAX_TRIALS, fast=fast, switch=False),

            "NEB_CI": lambda: self.step_runner.neb_ci(trial=0, upper_limit=MAX_TRIALS),
            "TS": lambda: self.step_runner.ts_opt(trial=0, upper_limit=MAX_TRIALS),
            "IRC": lambda: self.step_runner.irc_job(trial=0, upper_limit=MAX_TRIALS),
            "SP": lambda: self.step_runner.sp_calc(),
        }

        step_function = steps_mapping.get(step)
        if not step_function:
            print(f"Invalid step '{step}'. Valid steps:")
            for valid_step in steps_mapping.keys():
                print(f"  - {valid_step}")
            sys.exit(1)

        return step_function()

    def run_pipeline(self,
                     steps: List[str]) -> None:
        """
        Coordinates the entire pipeline, including restarts if specified.
        """

        for stp in steps:
            print(f"--- Running step: {stp} ---")
            success = self.pipeline(stp)
            if not success:
                print(f"{stp} failed. Saving state for restart.")
                self.save_failure_state(stp)
                sys.exit(1)
            print(f"--- Step {stp} completed successfully ---\n")

        print("All pipeline steps completed successfully.")
        self.save_completion_state(steps, charge, mult, solvent, Nimages)
        sys.exit(0)
