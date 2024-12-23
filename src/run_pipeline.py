#!/usr/bin/env python3
"""
run_pipeline.py

The main script that parses command-line arguments and orchestrates the 
pipeline using HPCDriver, StepRunner, and PipelineManager.
"""

import argparse
from typing import List

from hpc_driver import HPCDriver
from step_runner import StepRunner
from pipeline_manager import PipelineManager
from constants import DEFAULT_STEPS




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
    return steps_str.split(',')


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NEB optimization pipeline (OOP style).")
    parser.add_argument("--charge", "-c", type=int, default=0, help="Charge of the system")
    parser.add_argument("--mult", "-m", type=int, default=1, help="Multiplicity of the system")
    parser.add_argument("--solvent", "-s", type=str, default="", help="Solvent model to use")
    parser.add_argument("--Nimages", "-i", type=int, default=8, help="Number of images for NEB")
    parser.add_argument("--restart", type=str2bool, default=False, help="Restart from previous calculations")
    parser.add_argument(
        "--steps",
        type=parse_steps,
        default=DEFAULT_STEPS,
        help="Steps to run in the pipeline, comma separated"
    )

    args = parser.parse_args()

    cli_info = (
        f"Charge: {args.charge}\n"
        f"Multiplicity: {args.mult}\n"
        f"Solvent: {args.solvent}\n"
        f"Nimages: {args.Nimages}\n"
        f"Steps: {','.join(args.steps)}\n"
        f"Restart: {args.restart}"
    )
    print("[CLI] Starting Pipeline with parameters:")
    print(cli_info)

    # Initialize HPCDriver and StepRunner
    hpc_driver = HPCDriver()
    step_runner = StepRunner(hpc_driver)
    pipeline_manager = PipelineManager(step_runner)

    # Run the pipeline
    pipeline_manager.run_pipeline(
        charge=args.charge,
        mult=args.mult,
        solvent=args.solvent,
        Nimages=args.Nimages,
        restart=args.restart,
        steps=args.steps
    )


if __name__ == "__main__":
    main()
