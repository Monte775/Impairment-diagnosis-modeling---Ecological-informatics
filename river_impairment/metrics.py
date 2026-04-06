"""Classification evaluation metrics used throughout the project."""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> Dict[str, float]:
    """Compute standard binary-classification metrics.

    Parameters
    ----------
    y_true : array-like of shape (n,)
        Ground-truth labels (0 or 1).
    y_pred : array-like of shape (n,)
        Predicted labels.
    y_proba : array-like of shape (n,)
        Predicted probability for the positive class.

    Returns
    -------
    dict
        ``{"AUC": …, "Accuracy": …, "Recall": …, "F1": …}``
    """
    return {
        "AUC": roc_auc_score(y_true, y_proba),
        "Accuracy": accuracy_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
    }
