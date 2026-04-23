# hypatiax/tools/utils/reproducibility.py

import os
import random
import numpy as np

def set_global_seed(seed: int = 42) -> None:
    """
    Call this at the top of every experiment script.
    Controls all randomness sources except LLM API.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch (only if installed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    # PySR (only if installed)
    try:
        import pysr
        os.environ["JULIA_SEED"] = str(seed)
    except ImportError:
        pass

    print(f"✅ Global seed set: {seed}")
