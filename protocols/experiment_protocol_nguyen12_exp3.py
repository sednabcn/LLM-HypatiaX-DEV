
from protocols.universal_protocol import run_protocol
from core.runners.common import run_task

def run():
    config = {"name": "nguyen12_exp3"}
    return run_protocol(config, run_task)
