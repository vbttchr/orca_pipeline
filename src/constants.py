SSUBO= ["ssubo", "-m", "No"]
CHECK_STATES = ['COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT']
DEFAULT_STEPS = ["OPT", "FAST_NEB_TS", "TS", "IRC"]


MAX_TRIALS = 5
FREQ_THRESHOLD = -50
RETRY_DELAY = 60

SLURM_PARAMS_HIGH_MEM = {
    'nprocs': 16,
    'maxcore': 12000
}

SLURM_PARAMS_LOW_MEM = {
    'nprocs': 24,
    'maxcore': 2524
}

SLURM_PARAMS_XTB = {
    'nprocs': 12,
    'maxcore': 2000
}
