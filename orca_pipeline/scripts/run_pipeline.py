#!/usr/bin/env python3
"""
run_pipeline.py

The main script that parses command-line arguments and orchestrates the
pipeline using HPCDriver, StepRunner, and PipelineManager.

maybe associate the methods of step_runner to Reaction and Molecule classes.

"""
# TODO make parsing case insensitive
# TODO remove state file if input does not match steps in state file. probably better in steprunner

import logging
logger = logging.getLogger(__name__)
# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

import argparse
import os
from typing import cast
import sys
import time
from collections import defaultdict
from pathlib import Path
from orca_pipeline.hpc_driver import HPCDriver
from orca_pipeline.step_runner import StepRunner
from orca_pipeline.constants import (
    DEFAULT_STEPS,
    SLURM_PARAMS_BIG_HIGH_MEM,
    SLURM_PARAMS_BIG_LOW_MEM,
)
from orca_pipeline.chemistry import Molecule, Reaction
import yaml
import pprint

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
        raise FileNotFoundError(
            "No configuration file provided or file does not exist. Using default values."
        )
        # return {}
    with open(file_path, "r") as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as exc:
            raise RuntimeError(f"Error parsing YAML file: {exc}")
            # return {}


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


def parse_steps(steps_str: str | list | None) -> list[str]:
    """
    Parses comma-separated steps into a list.

    """
    if steps_str is None:
        raise TypeError(f'Steps: {steps_str} are empty')
    elif isinstance(steps_str, list):
        assert all(isinstance(i, str) for i in steps_str)
        return steps_str
    a = steps_str.split(",")
    return a

def main(argv=None) -> None:
    """
    Main function to run the ORCA pipeline.

    Accept an arg vector s.t. script can be entered from another python script
    to enable debugging without copying this file out of the package repo.

    If this files main() is called with main(["lol.yaml"]) then this will be parsed
    into the config attribute of args. Otherwise its none and it will go to CLI.
    """
    parser = argparse.ArgumentParser(description="Run the ORCA pipeline.")
    parser.add_argument(
        "config",
        type=str,
        nargs="?",
        default="",
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args(argv)
    
    if not args.config:
        print("Error: No configuration file provided.")
        parser.print_help()
        sys.exit(1)
    
    # For debugging, __file__ always goes to the repo -> do not use
    logger.info(f" __file__ = {__file__}")
    logger.info(f"cwd      = {os.getcwd()}")
    logger.info(f"args.config      = {args.config}")
    logger.info(f"config   = {os.path.abspath(os.path.join(os.getcwd(), args.config))}")

    # Check path validity; can be absolute already or relative (cwd + config)
    # I keep to cases to see if there are cases where os.path cannot manage rel paths
    if os.path.exists(args.config):
        config_path = os.path.abspath(args.config)
    else:
        config_path = os.path.abspath(os.path.join(os.getcwd(), args.config))
    logger.info(f"config_path   = {config_path}")
    assert os.path.exists(config_path)==True

    config = defaultdict(lambda: "")
    config.update(parse_yaml(config_path))
    config_dir = Path(os.path.dirname(config_path))
    charge = cast(int, config["charge"])
    # charge = config.get("charge")
    mult = cast(int,config["mult"])
    # mult = config.get("mult")
    method = config.get("method","r2scan-3c") # take default from molecule
    coords = config["coords"]
    coords = [Path(os.path.join(config_dir,cur_coord)) for cur_coord in coords]
    # coords = config.get("coords")
    solvent = config.get("solvent")
    cosmo = str2bool(config.get("cosmo","false"))
    Nimages = config.get("Nimages")
    fast = str2bool(config.get("fast","false"))
    zoom = str2bool(config.get("zoom","false"))
    dif_scf = str2bool(config.get("dif_scf","false"))
    sp_method = config.get("sp_method","r2scanh def2-qzvpp d4") # take default from molecule
    name = config["name"]
    # name = config.get("name")
    steps = parse_steps(config.get("steps"))
    conf_method = config.get("conf_method") # take default from molecule
    # conf_method = config.get("conf_method","CREST") # take default from molecule
    conf_exclude = config.get("conf_exlude", "")
    slurm_params_low_mem = config["slurm_params_low_mem"]
    assert isinstance(slurm_params_low_mem,dict)
    # slurm_params_low_mem = config.get("slurm_params_low_mem")
    slurm_params_high_mem = config["slurm_params_high_mem"]
    assert isinstance(slurm_params_high_mem,dict)
    # slurm_params_high_mem = config.get("slurm_params_high_mem")

    # Latest addition to the yaml: scans / constraints
    scan = config.get("scan")

    required_vals = [
        charge, mult, method, coords, name,
        steps, slurm_params_low_mem, slurm_params_high_mem
        ]
    for val in required_vals:
        if val is None:
            print(
                f"Error: Missing required key '{val}' in configuration file.")
            sys.exit(1)
            raise ValueError

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
            print(
                f"Error: Missing required key '{key}' in configuration file.")
            sys.exit(1)

    print("Starting Pipeline with parameters:")
    pprint.pprint(config)

    # Create absolute path to all coord files (assume that in the config yaml only the file names are listed)
    # and that the yaml is placed in same directory as the coord files
    
    if "CONF" in steps:
        assert conf_method is not None
    
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
            dif_scf=dif_scf,
            scan=scan,
            home_dir=config_dir
        )  # Molecule
    # ! HAVE NOT TOUCHED REACTIONS YET
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
            dif_scf=dif_scf,
        )  # Reaction
    # ! HAVE NOT TOUCHED REACTIONS YET
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
            dif_scf=dif_scf,
        )  # Reaction  # Reaction
    else:
        print("Error: Invalid number of coordinate files provided.")
        return
    target = reaction if reaction else mol

    hpc_driver = HPCDriver(home_dir=config_dir)
    step_runner = StepRunner(
        hpc_driver=hpc_driver,
        target=target,
        steps=steps,
        home_dir=config_dir,
        slurm_params_low_mem=slurm_params_low_mem,
        slurm_params_high_mem=slurm_params_high_mem,
        conf_exclude=conf_exclude,
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
