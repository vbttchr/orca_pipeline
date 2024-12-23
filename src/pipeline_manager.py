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
            settings = dict(line.strip().split(':') for line in f if ':' in line)

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
                 charge: int,
                 mult: int,
                 solvent: str,
                 Nimages: int) -> bool:
        """
        Maps step strings to methods in StepRunner.
        """
        steps_mapping = {
            "OPT_XTB": lambda: self.step_runner.optimise_reactants(charge, mult, 0, MAX_TRIALS, solvent, xtb=True),
            "OPT_DFT": lambda: self.step_runner.optimise_reactants(charge, mult, 0, MAX_TRIALS, solvent, xtb=False),
            "NEB_TS_XTB": lambda: self.step_runner.neb_ts(charge, mult, 0, Nimages, MAX_TRIALS, xtb=True, fast=False, solvent=solvent),
            "FAST_NEB_TS_XTB": lambda: self.step_runner.neb_ts(charge, mult, 0, Nimages, MAX_TRIALS, xtb=True, fast=True, solvent=solvent),
            "NEB_TS_DFT": lambda: self.step_runner.neb_ts(charge, mult, 0, Nimages, MAX_TRIALS, xtb=False, fast=False, solvent=solvent),
            "FAST_NEB_TS_DFT": lambda: self.step_runner.neb_ts(charge, mult, 0, Nimages, MAX_TRIALS, xtb=False, fast=True, solvent=solvent),
            "NEB_CI_XTB": lambda: self.step_runner.neb_ci(charge, mult, 0, Nimages, MAX_TRIALS, xtb=True, solvent=solvent),
            "NEB_CI_DFT": lambda: self.step_runner.neb_ci(charge, mult, 0, Nimages, MAX_TRIALS, xtb=False, solvent=solvent),
            "TS": lambda: self.step_runner.ts_opt(charge, mult, 0, MAX_TRIALS, solvent),
            "IRC": lambda: self.step_runner.irc_job(charge, mult, 0, MAX_TRIALS, solvent),
            "SP": lambda: self.step_runner.sp_calc(charge, mult, 0, MAX_TRIALS, solvent),
        }

        step_function = steps_mapping.get(step)
        if not step_function:
            print(f"Invalid step '{step}'. Valid steps:")
            for valid_step in steps_mapping.keys():
                print(f"  - {valid_step}")
            sys.exit(1)

        return step_function()

    def run_pipeline(self,
                     charge: int,
                     mult: int,
                     solvent: str,
                     Nimages: int,
                     restart: bool,
                     steps: List[str]) -> None:
        """
        Coordinates the entire pipeline, including restarts if specified.
        """
        if restart:
            step_to_restart, charge, mult, solvent, Nimages, _ = self.load_settings()
            if step_to_restart in steps:
                steps = steps[steps.index(step_to_restart):]
            print(f"Restarting from step: {step_to_restart}")

        for stp in steps:
            print(f"--- Running step: {stp} ---")
            success = self.pipeline(stp, charge, mult, solvent, Nimages)
            if not success:
                print(f"{stp} failed. Saving state for restart.")
                self.save_failure_state(stp, charge, mult, solvent, Nimages)
                sys.exit(1)
            print(f"--- Step {stp} completed successfully ---\n")

        print("All pipeline steps completed successfully.")
        self.save_completion_state(steps, charge, mult, solvent, Nimages)
        sys.exit(0)
