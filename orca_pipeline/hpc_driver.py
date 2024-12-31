#!/usr/bin/env python3
"""
hpc_driver.py

Contains the HPCDriver class, encapsulating SLURM-related operations like 
submitting jobs, checking job status, cancelling jobs, etc.
"""

import sys
import subprocess
import time
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
                       timeout: int = 60) -> Optional[subprocess.CompletedProcess]:
        """
        Wrapper for running a subprocess command.
        Returns CompletedProcess or None if it fails silently (exit_on_error=False).
        """
        try:
            result = subprocess.run(
                command,
                shell=shell,
                capture_output=capture_output,
                text=True,
                check=check,
                timeout=timeout
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

    def submit_job(self, input_file: str, output_file: str, walltime: str = "24") -> str:
        """
        Submits a job to SLURM using the configured submit command.
        Parses and returns the job ID from stdout.

        TODO: either change ssub scripts to always take same options or make this more flexibel
        """
        command = self.submit_cmd + \
            ["-w", walltime, "-o", output_file, input_file]
        result = self.run_subprocess(command)
        if not result:
            print(f"Failed to submit job with input '{input_file}'")
            sys.exit(1)

        for line in result.stdout.splitlines():
            if 'Submitted batch job' in line:
                job_id = line.split()[-1]
                print(f"Job submitted with ID: {job_id}")
                return job_id

        print(f"Failed to submit job {input_file}")
        sys.exit(1)

    def check_job_status(self, job_id: str, interval: int = 45, step: str = "", timeout: int = 60) -> str:
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
        self.run_subprocess(['scancel', job_id], exit_on_error=False)

    def shell_command(self, command: str) -> Optional[subprocess.CompletedProcess]:
        """
        Wrapper to run an arbitrary shell command for convenience (e.g., grep, cp, rm).
        """
        return self.run_subprocess(command, shell=True, check=False, exit_on_error=False)

    def grep_output(self, pattern: str, file_path: str) -> str:
        """
        Wrapper around grep to return matched lines as a string.
        """
        command = f"grep '{pattern}' {file_path}"
        result = self.shell_command(command)
        if result and result.stdout:
            return result.stdout.strip()
        return ""
