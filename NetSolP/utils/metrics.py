from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def binary_metrics(labels, probs, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(labels).astype(int)
    y_prob = np.asarray(probs).astype(float)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "ACC": float(accuracy_score(y_true, y_pred)),
        "PRE": float(precision_score(y_true, y_pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_true)) > 1 else 0.0,
        "AUC": float("nan"),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["AUC"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        pass
    return out


def youden_threshold(labels, probs) -> float:
    y_true = np.asarray(labels).astype(int)
    y_prob = np.asarray(probs).astype(float)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    finite = np.isfinite(thresholds)
    if not finite.any():
        return 0.5
    scores = tpr[finite] - fpr[finite]
    candidates = thresholds[finite]
    return float(candidates[int(np.argmax(scores))])
