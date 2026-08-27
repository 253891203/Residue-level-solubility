from __future__ import annotations
import subprocess, sys
from pathlib import Path
import numpy as np
import torch


def str2bool(value):
    if isinstance(value, bool): return value
    value=value.lower()
    if value in {"true","1","yes","y"}: return True
    if value in {"false","0","no","n"}: return False
    raise ValueError(f"Expected boolean, got {value}")


def git_commit(root: Path):
    try: return subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError): return None


def runtime_versions():
    return {"python":sys.version,"torch":torch.__version__,"cuda_runtime":torch.version.cuda,
            "cuda_available":torch.cuda.is_available(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def regression_metrics(y, p):
    from sklearn.metrics import r2_score, mean_squared_error
    from scipy.stats import pearsonr
    y,p=np.asarray(y),np.asarray(p)
    return {"R2":float(r2_score(y,p)),"Pearson":float(pearsonr(y,p).statistic) if len(y)>1 and np.std(y) and np.std(p) else float("nan"),"RMSE":float(mean_squared_error(y,p)**.5)}
