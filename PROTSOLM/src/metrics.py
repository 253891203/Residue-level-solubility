from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(labels, probs, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(labels).astype(int)
    y_prob = np.asarray(probs).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
    }
    try:
        metrics["AUC"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["AUC"] = float("nan")
    return metrics
