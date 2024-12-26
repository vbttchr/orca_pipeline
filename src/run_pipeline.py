#!/usr/bin/env python3
"""
run_pipeline.py

The main script that parses command-line arguments and orchestrates the
pipeline using HPCDriver, StepRunner, and PipelineManager.

maybe associate the methods of step_runner to Reaction and Molecule classes.

"""

import argparse
import os
from typing import List

from hpc_driver import HPCDriver
from step_runner import StepRunner
from pipeline_manager import PipelineManager
from constants import DEFAULT_STEPS
from chemistry import Molecule, Reaction
import yaml


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
            "No configuration file provided or file does not exist. Using default values.")
        return {}
    with open(file_path, 'r') as file:
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
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_steps(steps_str: str) -> List[str]:
    """
    Parses comma-separated steps into a list.
    """
    if isinstance(steps_str, list):
        return steps_str
    return steps_str.split(',')


def main() -> None:
    """
    Main function to run the ORCA pipeline.
    """
    parser = argparse.ArgumentParser(description="Run the ORCA pipeline.")
    parser.add_argument('config', type=str, nargs='?', default=None,
                        help='Path to the YAML configuration file.')
    args = parser.parse_args()

    config = parse_yaml(args.config)

    charge = config.get('charge', 0)
    mult = config.get('mult', 1)
    method = config.get('method', 'r2scan-3c')
    coords = config.get('coords', ["educt.xyz", "product.xyz"])
    solvent = config.get('solvent', None)
    Nimages = config.get('Nimages', 16)
    steps = parse_steps(config.get('steps', DEFAULT_STEPS))

    print("Starting Pipeline with parameters:")
    print(config)

    for filepath in coords:
        if not os.path.exists(filepath):
            print(f"Error: Coordinate file '{filepath}' does not exist.")
            print(
                "Please ensure all coordinate files are present or provide a valid configuration.")
            return
    # Initialize HPCDriver and StepRunner
    mol = None
    reaction = None
    if len(coords) == 1:
        name = os.path.basename(coords[0]).split('.')[0]
        mol = Molecule.from_xyz(
            filepath=coords[0], charge=charge, mult=mult, solvent=solvent, name=name, method=method)
    elif len(coords) == 2:

        reaction = Reaction.from_xyz(
            educt_filepath=coords[0], product_filepath=coords[1], transition_state_filepath=None, nimages=Nimages, method=method, charge=charge, mult=mult, solvent=solvent)  # Reaction

    elif len(coords) == 3:
        reaction = Reaction.from_xyz(
            educt_filepath=coords[0], product_filepath=coords[1], transition_state_filepath=coords[2], nimages=Nimages, method=method, charge=charge, mult=mult, solvent=solvent)  # Reaction  # Reaction

    hpc_driver = HPCDriver()
    step_runner = StepRunner(hpc_driver, reaction=reaction, molecule=mol)
    # continue here
    pipeline_manager = PipelineManager(step_runner)

    # Run the pipeline
    pipeline_manager.run_pipeline(steps=steps)


if __name__ == "__main__":
    main()
