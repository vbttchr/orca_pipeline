import logging
import os
import re
import shutil
import subprocess
import sys
import time
import argparse
import concurrent.futures
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable

# Configuration
@dataclass
class PipelineConfig:
    """Configuration for NEB pipeline"""
    max_trials: int = 5
    retry_delay: int = 60
    freq_threshold: float = -50.0
    slurm_params_high: Dict[str, int] = field(default_factory=lambda: {'nprocs': 16, 'maxcore': 12000})
    slurm_params_low: Dict[str, int] = field(default_factory=lambda: {'nprocs': 24, 'maxcore': 2524})
    submit_command: List[str] = field(default_factory=lambda: ["ssubo", "-m", "No"])
    check_states: List[str] = field(default_factory=lambda: ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT'])

class JobStatus(Enum):
    """Job status enumeration"""
    PENDING = "PENDING"
    RUNNING = "RUNNING" 
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"

class SlurmJob:
    """Handles SLURM job submission and monitoring"""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
    def submit(self, input_file: str, output_file: str, walltime: str = "24") -> Optional[str]:
        """Submit job to SLURM"""
        cmd = self.config.submit_command + ["-w", walltime, "-o", output_file, input_file]
        result = self._run_cmd(cmd)
        if not result:
            return None
            
        job_id = None
        for line in result.stdout.splitlines():
            if 'Submitted batch job' in line:
                job_id = line.split()[-1]
                self.logger.info(f"Submitted job {job_id}")
                break
                
        return job_id

    def check_status(self, job_id: str, interval: int = 45) -> JobStatus:
        """Monitor job status"""
        counter = 0
        while True:
            time.sleep(interval)
            
            # Check queue
            queue_result = self._run_cmd(['squeue', '-j', job_id, '-h'], check=False)
            if queue_result and queue_result.stdout.strip():
                if counter % 10 == 0:
                    self.logger.info(f"Job {job_id} still in queue")
                counter += 1
                continue

            # Check completion status
            status_result = self._run_cmd(
                ['sacct', '-j', job_id, '--format=State', '--noheader'],
                check=False
            )
            
            if not status_result:
                return JobStatus.UNKNOWN
                
            status = status_result.stdout.strip().split('\n')[-1].strip()
            try:
                return JobStatus[status.upper()]
            except KeyError:
                self.logger.warning(f"Unknown job status: {status}")
                return JobStatus.UNKNOWN

    def _run_cmd(self, cmd: Union[str, List[str]], **kwargs) -> Optional[subprocess.CompletedProcess]:
        """Execute command with error handling"""
        try:
            return subprocess.run(
                cmd,
                shell=isinstance(cmd, str),
                capture_output=True,
                text=True,
                **kwargs
            )
        except subprocess.SubprocessError as e:
            self.logger.error(f"Command failed: {e}")
            if kwargs.get('check', True):
                raise
            return None

class NEBCalculation:
    """Handles NEB-specific calculations"""
    
    def __init__(self, config: PipelineConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.slurm = SlurmJob(config, logger)

    def optimize_reactants(self, charge: int, mult: int, solvent: str = "", xtb: bool = True) -> bool:
        """Optimize reactant and product structures"""
        trial = 0
        while trial < self.config.max_trials:
            trial += 1
            self.logger.info(f"Starting reactant optimization trial {trial}")
            
            if trial == 1:
                self._setup_opt_directory()
            
            if self._run_optimization_jobs(charge, mult, solvent, xtb):
                return True
                
            self._cleanup_failed_opt()
            
        return False

    def run_neb(self, charge: int, mult: int, n_images: int, solvent: str = "", xtb: bool = True) -> bool:
        """Run NEB calculation"""
        # NEB calculation implementation
        pass

    def run_ts_opt(self, charge: int, mult: int, solvent: str = "") -> bool:
        """Run transition state optimization"""
        # TS optimization implementation
        pass

class NEBPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.logger = self._setup_logger()
        self.calculator = NEBCalculation(self.config, self.logger)
        
    def _setup_logger(self) -> logging.Logger:
        """Configure logging"""
        logger = logging.getLogger("NEBPipeline")
        logger.setLevel(logging.INFO)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        fh = logging.FileHandler("neb_pipeline.log")
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger

    def run(self, charge: int = 0, mult: int = 1, solvent: str = "", 
            n_images: int = 32, steps: Optional[List[str]] = None) -> bool:
        """Run pipeline with specified parameters"""
        steps = steps or ["OPT_XTB", "NEB-TS_XTB", "TS", "IRC"]
        
        self.logger.info("Starting NEB pipeline")
        self.logger.info(f"Parameters: charge={charge}, mult={mult}, solvent={solvent}")
        
        try:
            for step in steps:
                self.logger.info(f"Running step: {step}")
                if not self._run_step(step, charge, mult, solvent, n_images):
                    self.logger.error(f"Step {step} failed")
                    return False
                self.logger.info(f"Completed step: {step}")
                
            self.logger.info("Pipeline completed successfully")
            return True
            
        except Exception as e:
            self.logger.exception("Pipeline failed")
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="NEB Pipeline")
    parser.add_argument("-c", "--charge", type=int, default=0)
    parser.add_argument("-m", "--mult", type=int, default=1)
    parser.add_argument("-s", "--solvent", type=str, default="")
    parser.add_argument("-i", "--images", type=int, default=32)
    parser.add_argument("--steps", type=str, nargs="+")
    
    args = parser.parse_args()
    
    pipeline = NEBPipeline()
    success = pipeline.run(
        charge=args.charge,
        mult=args.mult,
        solvent=args.solvent,
        n_images=args.images,
        steps=args.steps
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()