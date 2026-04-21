
from protocols.universal_protocol import run_protocol
from core.runners.common import run_task

def run():
    config = {"name": "instability_rf02_04"}
    return run_protocol(config, run_task)

if __name__ == "__main__":
    run()
