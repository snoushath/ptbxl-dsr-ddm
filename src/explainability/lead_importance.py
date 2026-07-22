"""
Lead importance utilities for ECG explainability.

Lead importance is computed by aggregating saliency values over time.
"""

from typing import List, Optional

import numpy as np
import pandas as pd


DEFAULT_LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


def compute_lead_importance(
    saliency: np.ndarray,
    aggregation: str = "mean",
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute lead importance from saliency maps.

    Parameters
    ----------
    saliency:
        Shape: (batch_size, leads, samples) or (leads, samples)

    aggregation:
        "mean" or "sum"

    normalize:
        If True, normalize lead scores so they sum to 1 per sample.

    Returns
    -------
    lead_scores:
        Shape: (batch_size, leads)
    """

    if saliency.ndim == 2:
        saliency = saliency[None, ...]

    if saliency.ndim != 3:
        raise ValueError(
            "saliency must have shape (batch_size, leads, samples) "
            "or (leads, samples)."
        )

    saliency = np.abs(saliency)

    if aggregation == "mean":
        scores = saliency.mean(axis=2)
    elif aggregation == "sum":
        scores = saliency.sum(axis=2)
    else:
        raise ValueError("aggregation must be either 'mean' or 'sum'.")

    if normalize:
        row_sum = scores.sum(axis=1, keepdims=True)
        scores = scores / (row_sum + 1e-8)

    return scores.astype(np.float32)


def average_lead_importance(
    lead_scores: np.ndarray,
) -> np.ndarray:
    """
    Average lead importance across samples.

    Input:
        (num_samples, 12)

    Output:
        (12,)
    """

    if lead_scores.ndim != 2:
        raise ValueError("lead_scores must have shape (num_samples, leads).")

    return lead_scores.mean(axis=0)


def lead_importance_to_dataframe(
    lead_scores: np.ndarray,
    lead_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert lead importance scores to a DataFrame.
    """

    if lead_names is None:
        lead_names = DEFAULT_LEAD_NAMES

    if lead_scores.ndim != 1:
        raise ValueError("lead_scores must be 1D.")

    if len(lead_scores) != len(lead_names):
        raise ValueError("lead_scores length must match lead_names.")

    df = pd.DataFrame({
        "lead": lead_names,
        "importance": lead_scores,
    })

    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    return df


def summarize_classwise_lead_importance(
    lead_scores: np.ndarray,
    targets: np.ndarray,
    lead_names: Optional[List[str]] = None,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Summarize lead importance for each diagnostic class.

    Returns long-format DataFrame:
        class_name | lead | mean_importance | std_importance
    """

    if lead_names is None:
        lead_names = DEFAULT_LEAD_NAMES

    if class_names is None:
        class_names = ["NORM", "MI", "STTC", "CD", "HYP"]

    if lead_scores.ndim != 2:
        raise ValueError("lead_scores must have shape (num_samples, leads).")

    if targets.ndim != 2:
        raise ValueError("targets must have shape (num_samples, classes).")

    records = []

    for class_idx, class_name in enumerate(class_names):
        mask = targets[:, class_idx] == 1

        if mask.sum() == 0:
            continue

        class_scores = lead_scores[mask]

        for lead_idx, lead_name in enumerate(lead_names):
            records.append({
                "class_name": class_name,
                "lead": lead_name,
                "mean_importance": float(class_scores[:, lead_idx].mean()),
                "std_importance": float(class_scores[:, lead_idx].std(ddof=1)),
                "num_samples": int(mask.sum()),
            })

    return pd.DataFrame(records)