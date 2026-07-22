import numpy as np
import torch
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    hamming_loss,
)


def logits_to_probabilities(logits: torch.Tensor) -> np.ndarray:
    probs = torch.sigmoid(logits)
    return probs.detach().cpu().numpy()


def probabilities_to_predictions(
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    return (probabilities >= threshold).astype(int)


def compute_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute multi-label classification metrics.

    Parameters
    ----------
    y_true:
        Ground-truth labels, shape (num_samples, num_classes)

    y_prob:
        Predicted probabilities, shape (num_samples, num_classes)

    threshold:
        Threshold used to convert probabilities into binary predictions.

    Returns
    -------
    Dictionary of metrics.
    """

    y_pred = probabilities_to_predictions(y_prob, threshold=threshold)

    metrics = {}

    metrics["micro_f1"] = f1_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    metrics["macro_f1"] = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["micro_precision"] = precision_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    metrics["macro_precision"] = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["micro_recall"] = recall_score(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    metrics["macro_recall"] = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)

    try:
        metrics["macro_auroc"] = roc_auc_score(
            y_true,
            y_prob,
            average="macro",
        )
    except ValueError:
        metrics["macro_auroc"] = np.nan

    try:
        metrics["macro_auprc"] = average_precision_score(
            y_true,
            y_prob,
            average="macro",
        )
    except ValueError:
        metrics["macro_auprc"] = np.nan

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    threshold: float = 0.5,
) -> dict:
    """
    Compute per-class AUROC, AUPRC, F1, precision, and recall.
    """

    y_pred = probabilities_to_predictions(y_prob, threshold=threshold)

    per_class = {}

    for idx, class_name in enumerate(class_names):
        class_result = {}

        class_result["f1"] = f1_score(
            y_true[:, idx],
            y_pred[:, idx],
            zero_division=0,
        )

        class_result["precision"] = precision_score(
            y_true[:, idx],
            y_pred[:, idx],
            zero_division=0,
        )

        class_result["recall"] = recall_score(
            y_true[:, idx],
            y_pred[:, idx],
            zero_division=0,
        )

        try:
            class_result["auroc"] = roc_auc_score(
                y_true[:, idx],
                y_prob[:, idx],
            )
        except ValueError:
            class_result["auroc"] = np.nan

        try:
            class_result["auprc"] = average_precision_score(
                y_true[:, idx],
                y_prob[:, idx],
            )
        except ValueError:
            class_result["auprc"] = np.nan

        per_class[class_name] = class_result

    return per_class