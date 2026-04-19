
from protocols._base import save_results, reproducibility_lock
from core.runners.run_dual_condition_benchmark import run_dual_condition_benchmark

def get_config():
    return {"name": "ablation_exp1", "n_cases": 15, "seed": 42}

def run():
    config = get_config()
    if not reproducibility_lock(config):
        return {"status": "skipped"}
    results = run_dual_condition_benchmark(config)
    save_results(results, config)
    return results
