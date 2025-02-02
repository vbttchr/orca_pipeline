#!/usr/bin/env python3
"""
hpc_driver.py

Contains the HPCDriver class, encapsulating SLURM-related operations like 
submitting jobs, checking job status, cancelling jobs, etc.
"""

import sys
import subprocess
import time
import os
from typing import List, Optional
from orca_pipeline.constants import SSUBO, CHECK_STATES


class HPCDriver:
    """
    Encapsulates SLURM-related operations like submitting jobs,
    checking job status, cancelling jobs, etc.
    """

    def __init__(self,
                 submit_cmd: List[str] = None,
                 check_states: List[str] = None,
                 retry_delay: int = 60) -> None:
        self.submit_cmd = submit_cmd or SSUBO
        self.check_states = check_states or CHECK_STATES
        self.retry_delay = retry_delay

    def run_subprocess(self,
                       command,
                       shell: bool = False,
                       capture_output: bool = True,
                       check: bool = True,
                       exit_on_error: bool = True,
                       timeout: int = 60,
                       cwd=None) -> Optional[subprocess.CompletedProcess]:
        """
        Wrapper for running a subprocess command.
        Returns CompletedProcess or None if it fails silently (exit_on_error=False).
        """
        if not cwd:
            cwd = os.getcwd()

        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                check=check,
                timeout=timeout,
                cwd=cwd
            )
            return result
        except subprocess.TimeoutExpired:
            print(f"Command '{command}' timed out after {timeout} seconds.")
            return None
        except subprocess.CalledProcessError as e:
            if exit_on_error:
                print(f"Command '{command}' failed with error: {e.stderr}")
                sys.exit(1)
            else:
                return None

    def submit_job(self, input_file: str, output_file: str, walltime: str = "24", mail: bool = False, job_type: str = "orca", version: int = 601, charge: int = 0, mult: int = 0, solvent: str = "", cwd=None) -> str:
        """
        Submits a job to SLURM using the configured submit command.
        Parses and returns the job ID from stdout.

       version = 601
       only freq can take other version as we currently need it for the qrc option. In the furture, this will either be more flexible or removed. Most recent orca version is preferable 

       charge and mult are only used for crest jobs.

        TODO: either change ssub scripts to always take same options or make this more flexibel
        """

        if not cwd:
            cwd = os.getcwd()
        command = []
        match job_type.lower():
            case "orca":
                command = ["ssubo", "-v", str(version), "-w", walltime, "-m",
                           str(mail), "-o", output_file, input_file]
            case "crest":
                command = ["ssubcrest", "-w", walltime, "-m",
                           str(mail), "-c", str(charge), "-u", str(mult-1), "-s", str(solvent), "-o",  output_file, input_file]
            case "fod":

                walltime = walltime+":00:00"

                nprocs = 1
                maxcore = 1000
                orca_out = input_file.replace(".inp", ".out")

                orca_path = self.shell_command("which orca").stdout.strip()
                print(f"Using ORCA path: {orca_path}")

                with open(input_file, 'r') as f:
                    for line in f:
                        if "nprocs" in line:
                            nprocs = int(line.split()[2])
                        if "maxcore" in line:
                            maxcore = int(line.split()[1])
                command = [
                    "sbatch", "-n", str(nprocs), "--mem-per-cpu", str(maxcore), f"--time={walltime}", f"--wrap={orca_path} {input_file} > {orca_out}"]
                print(f"Submitting FOD job with command: {command}")
        result = self.run_subprocess(command, cwd=cwd, timeout=1200)
        if not result:
            print(f"Failed to submit job with input '{input_file}'")
            sys.exit(1)

        for line in result.stdout.splitlines():
            if 'Submitted batch job' in line:
                job_id = line.split()[-1]
                print(f"Job submitted with ID: {job_id}")
                return job_id

        print(f"Failed to get job_id for {input_file}")
        sys.exit(1)

    def check_job_status(self, job_id: str, interval: int = 45, step: str = "", timeout: int = 1200) -> str:
        """
        Checks the status of a SLURM job at regular intervals until 
        it's out of the queue or recognized as completed/failed.
        """
        counter = 0
        while True:
            time.sleep(interval)
            print(f'Checking job {job_id} status {step}')
            squeue = self.run_subprocess(['squeue', '-j', job_id, '-h'],
                                         shell=False, exit_on_error=False,
                                         timeout=timeout)
            if squeue and squeue.stdout.strip():
                if counter % 10 == 0:
                    print(f'Job {job_id} is still in the queue.')
                counter += 1
            else:
                sacct = self.run_subprocess(['sacct', '-j', job_id, '--format=State', '--noheader'],
                                            shell=False, exit_on_error=False, timeout=timeout)
                if sacct and sacct.stdout.strip():
                    statuses = sacct.stdout.strip().split('\n')
                    latest_status = statuses[-1].strip()
                    print(f'Latest status for job {job_id}: {latest_status}')
                    if latest_status in self.check_states:
                        return latest_status
                    else:
                        print(
                            f'Job {job_id} ended with status: {latest_status}')
                        return latest_status
                else:
                    if sacct and sacct.stderr.strip():
                        print(f"Error from sacct: {sacct.stderr.strip()}")
                    print(f'Job {job_id} status could not be determined.')
                    return 'UNKNOWN'

    def scancel_job(self, job_id: str) -> None:
        """
        Cancels a running SLURM job by ID (no error if job is already stopped).
        """
        self.run_subprocess(['scancel', job_id],
                            exit_on_error=False, timeout=1200)

    def shell_command(self, command: str, cwd=None, timeout: int = 60) -> Optional[subprocess.CompletedProcess]:
        """
        Wrapper to run an arbitrary shell command for convenience (e.g., grep, cp, rm).
        """
        return self.run_subprocess(command, shell=True, check=False, exit_on_error=False, cwd=cwd, timeout=timeout)

    def grep_output(self, pattern: str, file_path: str, flags: str = "") -> str:
        """
        Wrapper around grep to return matched lines as a string.
        """
        command = f"grep {flags} '{pattern}' {file_path} "
        result = self.shell_command(command, timeout=1200)
        if result and result.stdout:
            return result.stdout.strip()
        return ""
