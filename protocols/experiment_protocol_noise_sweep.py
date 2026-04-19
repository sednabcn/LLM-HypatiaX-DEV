
from protocols.universal_protocol import run_protocol
from core.runners.common import run_task

def run():
    config = {"name": "noise_sweep"}
    return run_protocol(config, run_task)
