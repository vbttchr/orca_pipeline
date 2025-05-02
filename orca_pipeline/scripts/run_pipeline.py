#!/usr/bin/env python3
"""
run_pipeline.py

The main script that parses command-line arguments and orchestrates the
pipeline using HPCDriver, StepRunner, and PipelineManager.

maybe associate the methods of step_runner to Reaction and Molecule classes.

"""
# TODO make parsing case insensitive
# TODO remove state file if input does not match steps in state file. probably better in steprunner

import argparse
import os
from typing import List
import sys
import time

from orca_pipeline.hpc_driver import HPCDriver
from orca_pipeline.step_runner import StepRunner
from orca_pipeline.constants import (
    DEFAULT_STEPS,
    SLURM_PARAMS_BIG_HIGH_MEM,
    SLURM_PARAMS_BIG_LOW_MEM,
)
from orca_pipeline.chemistry import Molecule, Reaction
import yaml

sys.stdout = open(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), "w", buffering=1)

"""
# config.yaml

method: "r2scan-3c"
solvent: "water"
mult: 1
charge: 0
steps: "opt, freq, neb,ts..."
coords: educt.xyz, product.xyz, ts.xyz # or just a filename
Nimages: 8


"""


def parse_yaml(file_path: str) -> dict:
    """
    Parses a YAML file and returns the contents as a dictionary.
    """
    if not file_path or not os.path.exists(file_path):
        print(
            "No configuration file provided or file does not exist. Using default values."
        )
        return {}
    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            return {}


def str2bool(v: str) -> bool:
    """
    Converts a string to a boolean for CLI arg parsing.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_steps(steps_str: str) -> List[str]:
    """
    Parses comma-separated steps into a list.

    """
    if isinstance(steps_str, list):
        return steps_str
    return steps_str.split(",")


def main() -> None:
    """
    Main function to run the ORCA pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the ORCA pipeline.")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default=None,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()
    if not args.config:
        print("Error: No configuration file provided.")
        parser.print_help()
        sys.exit(1)

    config = parse_yaml(args.config)

    charge = config.get("charge")
    mult = config.get("mult")
    method = config.get("method")
    coords = config.get("coords")
    solvent = config.get("solvent")
    cosmo = config.get("cosmo")
    Nimages = config.get("Nimages")
    fast = config.get("fast")
    zoom = config.get("zoom")
    sp_method = config.get("sp_method")
    name = config.get("name")
    steps = parse_steps(config.get("steps"))
    conf_method = config.get("conf_method")
    slurm_params_low_mem = config.get("slurm_params_low_mem")
    slurm_params_high_mem = config.get("slurm_params_high_mem")

    required_keys = [
        "charge",
        "mult",
        "method",
        "coords",
        "name",
        "steps",
        "slurm_params_low_mem",
        "slurm_params_high_mem",
    ]
    for key in required_keys:
        if config.get(key) is None:
            print(f"Error: Missing required key '{key}' in configuration file.")
            sys.exit(1)

    print("Starting Pipeline with parameters:")
    print(config)

    for filepath in coords:
        if not os.path.exists(filepath):
            print(f"Error: Coordinate file '{filepath}' does not exist.")
            print(
                "Please ensure all coordinate files are present or provide a valid configuration."
            )
            return
    # Initialize HPCDriver and StepRunn
    mol = None
    reaction = None
    if len(coords) == 1:
        name = os.path.basename(coords[0]).split(".")[0] if not name else name
        mol = Molecule.from_xyz(
            filepath=coords[0],
            charge=charge,
            mult=mult,
            solvent=solvent,
            cosmo=cosmo,
            name=name,
            method=method,
            sp_method=sp_method,
            conf_method=conf_method,
        )  # Molecule
    elif len(coords) == 2:
        name = "Reaction" if not name else name

        reaction = Reaction.from_xyz(
            educt_filepath=coords[0],
            product_filepath=coords[1],
            transition_state_filepath=None,
            nimages=Nimages,
            method=method,
            charge=charge,
            mult=mult,
            solvent=solvent,
            cosmo=cosmo,
            sp_method=sp_method,
            name=name,
            fast=fast,
            zoom=zoom,
            conf_method=conf_method,
        )  # Reaction

    elif len(coords) == 3:
        name = "Reaction" if not name else name
        reaction = Reaction.from_xyz(
            educt_filepath=coords[0],
            product_filepath=coords[1],
            transition_state_filepath=coords[2],
            nimages=Nimages,
            method=method,
            charge=charge,
            mult=mult,
            solvent=solvent,
            cosmo=cosmo,
            sp_method=sp_method,
            name=name,
            fast=fast,
            zoom=zoom,
            conf_method=conf_method,
        )  # Reaction  # Reaction
    else:
        print("Error: Invalid number of coordinate files provided.")
        return
    target = reaction if reaction else mol

    hpc_driver = HPCDriver()
    step_runner = StepRunner(
        hpc_driver=hpc_driver,
        target=target,
        steps=steps,
        home_dir=os.getcwd(),
        slurm_params_low_mem=slurm_params_low_mem,
        slurm_params_high_mem=slurm_params_high_mem,
    )
    # continue here
    start_time = time.time()
    success = step_runner.run_pipeline()
    end_time = time.time()

    total_time = end_time - start_time

    days = total_time // (24 * 3600)

    hours = (total_time - days * 24 * 3600) // 3600
    minutes = (total_time - days * 24 * 3600 - hours * 3600) // 60
    seconds = total_time - days * 24 * 3600 - hours * 3600 - minutes * 60

    if success:
        print("Pipeline completed successfully.")
        print(
            f"Total time: {days} days, {hours} hours, {minutes} minutes, {seconds:.0f} seconds"
        )
        open("COMPLETED", "w").close()
    else:
        print("Pipeline failed check logs for more information.")
        print(
            f"Total time: {days} days, {hours} hours, {minutes} minutes, {seconds:.0f} seconds"
        )


if __name__ == "__main__":
    main()
