"""
DDM explainability utilities.

This module extracts and summarizes attention weights from the
Diagnostic Dependency Module (DDM).

Expected model call:
    logits, ddm_attention = model(x, return_ddm_attention=True)

Expected ddm_attention shape:
    (batch_size, num_classes, num_classes)
"""

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


DEFAULT_CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]


@torch.no_grad()
def extract_ddm_attention(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    class_names: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract DDM attention weights for all samples in a dataloader.

    Supports the current PTBXLDataset batch format:
        batch["signal"]
        batch["label"]

    Returns
    -------
    attentions:
        Shape: (num_samples, num_classes, num_classes)

    logits:
        Shape: (num_samples, num_classes)

    targets:
        Shape: (num_samples, num_classes)
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    model.eval()

    all_attentions = []
    all_logits = []
    all_targets = []

    for batch in dataloader:
        if isinstance(batch, dict):
            x = batch["signal"].to(device)
            y = batch["label"].to(device)
        else:
            raise TypeError(
                "Expected dataloader batch to be a dictionary with "
                "'signal' and 'label' keys."
            )

        logits, attention = model(
            x,
            return_ddm_attention=True,
        )

        all_attentions.append(attention.detach().cpu())
        all_logits.append(logits.detach().cpu())
        all_targets.append(y.detach().cpu())

    attentions = torch.cat(all_attentions, dim=0).numpy()
    logits = torch.cat(all_logits, dim=0).numpy()
    targets = torch.cat(all_targets, dim=0).numpy()

    return attentions, logits, targets


def average_ddm_attention(
    attentions: np.ndarray,
) -> np.ndarray:
    """
    Compute average DDM attention matrix over all samples.

    Input shape:
        (num_samples, num_classes, num_classes)

    Output shape:
        (num_classes, num_classes)
    """

    if attentions.ndim != 3:
        raise ValueError(
            "attentions must have shape "
            "(num_samples, num_classes, num_classes)"
        )

    return attentions.mean(axis=0)


def classwise_ddm_attention(
    attentions: np.ndarray,
    targets: np.ndarray,
    class_index: int,
) -> np.ndarray:
    """
    Compute average DDM attention matrix for samples belonging to one class.

    This is useful for producing class-specific DDM heatmaps,
    e.g., average DDM behavior for MI-positive ECGs.

    Parameters
    ----------
    attentions:
        Shape: (num_samples, num_classes, num_classes)

    targets:
        Shape: (num_samples, num_classes)

    class_index:
        Index of target class.
    """

    if attentions.ndim != 3:
        raise ValueError("attentions must be 3D.")

    if targets.ndim != 2:
        raise ValueError("targets must be 2D.")

    mask = targets[:, class_index] == 1

    if mask.sum() == 0:
        raise ValueError(f"No positive samples found for class index {class_index}.")

    return attentions[mask].mean(axis=0)


def ddm_attention_to_dataframe(
    attention_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Convert a DDM attention matrix into a pandas DataFrame.

    Rows:
        source/query diagnostic class

    Columns:
        target/key diagnostic class
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    if attention_matrix.ndim != 2:
        raise ValueError("attention_matrix must be 2D.")

    if attention_matrix.shape[0] != len(class_names):
        raise ValueError("Number of rows does not match class_names.")

    if attention_matrix.shape[1] != len(class_names):
        raise ValueError("Number of columns does not match class_names.")

    return pd.DataFrame(
        attention_matrix,
        index=class_names,
        columns=class_names,
    )


def save_ddm_attention_csv(
    attention_matrix: np.ndarray,
    save_path: str,
    class_names: Optional[List[str]] = None,
) -> None:
    """
    Save DDM attention matrix as CSV.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = ddm_attention_to_dataframe(
        attention_matrix=attention_matrix,
        class_names=class_names,
    )

    df.to_csv(save_path)


def get_top_dependencies(
    attention_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    top_k: int = 10,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """
    Return the strongest diagnostic dependencies.

    Example output:
        source  target  weight
        MI      STTC    0.42
        STTC    MI      0.39
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    if attention_matrix.ndim != 2:
        raise ValueError("attention_matrix must be 2D.")

    records = []

    for i, source in enumerate(class_names):
        for j, target in enumerate(class_names):
            if exclude_self and i == j:
                continue

            records.append(
                {
                    "source": source,
                    "target": target,
                    "weight": float(attention_matrix[i, j]),
                }
            )

    df = pd.DataFrame(records)
    df = df.sort_values("weight", ascending=False).head(top_k)
    df = df.reset_index(drop=True)

    return df

def summarize_dominant_dependencies(
    attentions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Quantify dominant DDM dependency for each diagnostic class.

    For each positive class, this function:
    1. Selects samples where that class is present.
    2. Averages each sample's DDM attention over query rows.
    3. Finds the attended diagnostic class with maximum mean attention.
    4. Reports mean, SD, SE, and 95% CI.

    Returns
    -------
    summary_df:
        One row per diagnostic class.
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    if attentions.ndim != 3:
        raise ValueError("attentions must have shape (N, C, C).")

    if targets.ndim != 2:
        raise ValueError("targets must have shape (N, C).")

    if attentions.shape[1] != len(class_names):
        raise ValueError("attention class dimension does not match class_names.")

    records = []

    for class_idx, class_name in enumerate(class_names):
        mask = targets[:, class_idx] == 1

        if mask.sum() == 0:
            records.append({
                "positive_class": class_name,
                "num_samples": 0,
                "dominant_attended_class": None,
                "mean_attention": np.nan,
                "std_attention": np.nan,
                "se_attention": np.nan,
                "ci95_lower": np.nan,
                "ci95_upper": np.nan,
            })
            continue

        class_attentions = attentions[mask]

        sample_attended_scores = class_attentions.mean(axis=1)

        mean_scores = sample_attended_scores.mean(axis=0)

        dominant_idx = int(np.argmax(mean_scores))
        dominant_class = class_names[dominant_idx]

        dominant_values = sample_attended_scores[:, dominant_idx]

        mean_value = float(np.mean(dominant_values))
        std_value = float(np.std(dominant_values, ddof=1))
        se_value = float(std_value / np.sqrt(len(dominant_values)))

        ci95_lower = float(mean_value - 1.96 * se_value)
        ci95_upper = float(mean_value + 1.96 * se_value)

        records.append({
            "positive_class": class_name,
            "num_samples": int(mask.sum()),
            "dominant_attended_class": dominant_class,
            "mean_attention": mean_value,
            "std_attention": std_value,
            "se_attention": se_value,
            "ci95_lower": ci95_lower,
            "ci95_upper": ci95_upper,
        })

    return pd.DataFrame(records)


def summarize_all_class_dependencies(
    attentions: np.ndarray,
    targets: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Quantify attention assigned to every attended class
    for each positive diagnostic group.

    Returns long-format DataFrame:
        positive_class | attended_class | num_samples | mean | std | se | ci95
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    if attentions.ndim != 3:
        raise ValueError("attentions must have shape (N, C, C).")

    if targets.ndim != 2:
        raise ValueError("targets must have shape (N, C).")

    records = []

    for class_idx, positive_class in enumerate(class_names):
        mask = targets[:, class_idx] == 1

        if mask.sum() == 0:
            continue

        class_attentions = attentions[mask]

        sample_attended_scores = class_attentions.mean(axis=1)

        for attended_idx, attended_class in enumerate(class_names):
            values = sample_attended_scores[:, attended_idx]

            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1))
            se_value = float(std_value / np.sqrt(len(values)))

            records.append({
                "positive_class": positive_class,
                "attended_class": attended_class,
                "num_samples": int(mask.sum()),
                "mean_attention": mean_value,
                "std_attention": std_value,
                "se_attention": se_value,
                "ci95_lower": float(mean_value - 1.96 * se_value),
                "ci95_upper": float(mean_value + 1.96 * se_value),
            })

    return pd.DataFrame(records)