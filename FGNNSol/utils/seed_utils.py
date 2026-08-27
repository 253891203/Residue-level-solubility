import os, random
import numpy as np
import torch


def seed_everything(seed: int, deterministic_algorithms: bool = False) -> dict:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(deterministic_algorithms, warn_only=True)
    return {"seed": seed, "cudnn_deterministic": True, "cudnn_benchmark": False,
            "deterministic_algorithms": deterministic_algorithms}
