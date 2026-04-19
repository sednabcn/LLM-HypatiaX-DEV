
from protocols.universal_protocol import run_protocol
from core.runners.common import run_task

def run():
    config = {"name": "extrapolation_comparative"}
    return run_protocol(config, run_task)
