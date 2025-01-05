import argparse
import os
from typing import List
import sys

from orca_pipeline.hpc_driver import HPCDriver
from orca_pipeline.step_runner import StepRunner
from orca_pipeline.constants import DEFAULT_STEPS, SLURM_PARAMS_BIG_HIGH_MEM, SLURM_PARAMS_BIG_LOW_MEM
from orca_pipeline.chemistry import Reaction
from orca_pipeline.plotting.neb_trajectory import plot_reaction_profile
from run_pipeline import parse_yaml
import yaml


def main():

    parser = argparse.ArgumentParser(description="Run the ORCA pipeline.")
    parser.add_argument("config", type=str,
                        help="Path to the configuration file.")
    parser.add_argument("--bigmem", action="store_true",
                        help="Use bigmem nodes.")
    args = parser.parse_args()

    config = parse_yaml(args.config)

    if not config:
        print("No configuration provided. Exiting.")
        sys.exit(1)

    # it is assumed that it is SM TS1 P1 TS2 P2 ...... TSn Pn
    energy_files = config.get("energy_files", None)
    coords = config.get("coords", None)
    charge = config.get("charge", 0)
    mult = config.get("mult", 1)
    method = config.get("method", "r2scan-3c")
    sp_method = config.get("sp_method", "r2scanh def2-qzvpp d4")
    solvent = config.get("solvent", None)
    name = config.get("name", None)
    reactions = []
    for i, energy_file in enumerate(energy_files):
        rxn_coords = coords[i*3:i*3+3]
        if not os.path.exists(energy_file):
            print(f"Error: Energy file '{energy_file}' does not exist.")
            print(
                "Please ensure all energy files are present or provide a valid configuration.")
            sys.exit(1)
        reactions.append(Reaction.from_xyz(educt_filepath=rxn_coords[0], product_filepath=rxn_coords[1], transition_state_filepath=rxn_coords[2],
                                           nimages=8, method=method, charge=charge, mult=mult, solvent=solvent, name=name, fast=False, zoom=False, energy_file=energy_file, sp_method=sp_method))

    plot_reaction_profile(reactions, show=True, save=True, type="enthalpy")


if __name__ == "__main__":
    main()
