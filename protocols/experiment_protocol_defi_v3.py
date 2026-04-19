
from protocols.universal_protocol import run_protocol
from core.runners.common import run_task

def run():
    config = {"name": "defi_v3"}
    return run_protocol(config, run_task)
