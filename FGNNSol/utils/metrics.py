from __future__ import annotations
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, matthews_corrcoef, r2_score, mean_squared_error)


METRIC_NAMES = ["R2", "Pearson", "RMSE", "ACC", "Precision", "Recall", "F1", "AUC", "MCC"]


def calculate_metrics(y_true, y_pred, threshold=0.5):
    y, p = np.asarray(y_true, float), np.asarray(y_pred, float)
    if y.shape != p.shape or y.size == 0:
        raise ValueError(f"Invalid metric inputs: y={y.shape}, p={p.shape}")
    yc, pc = (y >= threshold).astype(int), (p >= threshold).astype(int)
    pearson = float(pearsonr(y, p).statistic) if y.size > 1 and np.std(y) and np.std(p) else float("nan")
    auc = float(roc_auc_score(yc, p)) if np.unique(yc).size == 2 else float("nan")
    return {"R2": float(r2_score(y, p)), "Pearson": pearson,
            "RMSE": float(mean_squared_error(y, p) ** 0.5),
            "ACC": float(accuracy_score(yc, pc)),
            "Precision": float(precision_score(yc, pc, zero_division=0)),
            "Recall": float(recall_score(yc, pc, zero_division=0)),
            "F1": float(f1_score(yc, pc, zero_division=0)), "AUC": auc,
            "MCC": float(matthews_corrcoef(yc, pc))}
