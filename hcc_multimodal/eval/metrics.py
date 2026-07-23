"""Classification metrics for binary outcome evaluation."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def compute_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics for a binary prediction.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0/1).
    y_score:
        Predicted probabilities for the positive class.
    threshold:
        Decision threshold applied to y_score to produce hard labels.

    Returns
    -------
    dict with keys:
        auroc, auprc,
        sensitivity, specificity, ppv, npv, f1
    """
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")

    return {
        "auroc": roc_auc_score(y_true, y_score),
        "auprc": average_precision_score(y_true, y_score),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1_score(y_true, y_pred),
    }
