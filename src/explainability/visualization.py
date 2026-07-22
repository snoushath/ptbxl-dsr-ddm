"""
Visualization utilities for ECG explainability.

This module contains reusable plotting functions for:
1. DDM attention heatmaps
2. Lead importance bar plots
3. ECG saliency overlays

All figures are saved to disk if save_path is provided.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]
DEFAULT_LEAD_NAMES = [
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6"
]


def _prepare_save_path(save_path: Optional[str]) -> None:
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)


def plot_ddm_heatmap(
    attention_matrix: np.ndarray,
    class_names: Optional[List[str]] = None,
    title: str = "Diagnostic Dependency Module Attention",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot a DDM diagnostic dependency heatmap.

    Parameters
    ----------
    attention_matrix:
        Shape: (num_classes, num_classes)

    class_names:
        Diagnostic class names.

    title:
        Figure title.

    save_path:
        Optional path to save the figure.

    show:
        Whether to display the figure.
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    if attention_matrix.ndim != 2:
        raise ValueError("attention_matrix must be 2D.")

    if attention_matrix.shape[0] != len(class_names):
        raise ValueError("attention_matrix row count must match class_names.")

    if attention_matrix.shape[1] != len(class_names):
        raise ValueError("attention_matrix column count must match class_names.")

    _prepare_save_path(save_path)

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(attention_matrix, aspect="auto")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel("Attended diagnostic class")
    ax.set_ylabel("Query diagnostic class")
    ax.set_title(title)

    for i in range(attention_matrix.shape[0]):
        for j in range(attention_matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{attention_matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_lead_importance(
    lead_scores: np.ndarray,
    lead_names: Optional[List[str]] = None,
    title: str = "Lead Importance",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot ECG lead importance scores as a bar chart.

    Parameters
    ----------
    lead_scores:
        Shape: (12,)

    lead_names:
        Names of the 12 ECG leads.
    """

    if lead_names is None:
        lead_names = DEFAULT_LEAD_NAMES

    if lead_scores.ndim != 1:
        raise ValueError("lead_scores must be 1D.")

    if len(lead_scores) != len(lead_names):
        raise ValueError("lead_scores length must match lead_names.")

    _prepare_save_path(save_path)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(lead_names, lead_scores)

    ax.set_xlabel("ECG lead")
    ax.set_ylabel("Importance score")
    ax.set_title(title)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def plot_ecg_saliency(
    ecg: np.ndarray,
    saliency: np.ndarray,
    lead_names: Optional[List[str]] = None,
    title: str = "ECG Saliency Map",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot 12-lead ECG with saliency overlay.

    Parameters
    ----------
    ecg:
        Shape: (12, num_samples)

    saliency:
        Shape: (12, num_samples)
    """

    if lead_names is None:
        lead_names = DEFAULT_LEAD_NAMES

    if ecg.ndim != 2:
        raise ValueError("ecg must be 2D with shape (12, num_samples).")

    if saliency.ndim != 2:
        raise ValueError("saliency must be 2D with shape (12, num_samples).")

    if ecg.shape != saliency.shape:
        raise ValueError("ecg and saliency must have the same shape.")

    if ecg.shape[0] != len(lead_names):
        raise ValueError("Number of ECG leads must match lead_names.")

    _prepare_save_path(save_path)

    num_leads, num_samples = ecg.shape
    time = np.arange(num_samples)

    saliency = np.abs(saliency)
    max_saliency = saliency.max()

    if max_saliency > 0:
        saliency = saliency / max_saliency

    fig, axes = plt.subplots(
        num_leads,
        1,
        figsize=(12, 14),
        sharex=True,
    )

    for lead_idx, ax in enumerate(axes):
        ax.plot(time, ecg[lead_idx], linewidth=0.8)

        ax.scatter(
            time,
            ecg[lead_idx],
            c=saliency[lead_idx],
            s=4,
            alpha=0.8,
        )

        ax.set_ylabel(lead_names[lead_idx], rotation=0, labelpad=20)
        ax.grid(True, linewidth=0.3, alpha=0.5)

    axes[-1].set_xlabel("Time sample")

    fig.suptitle(title, y=0.995)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig


def column_normalize_attention(
    attention_matrix: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Column-normalize a DDM attention matrix.

    Original DDM attention is row-normalized because of softmax.
    This function is only for complementary visualization.

    Each column sums to 1 after normalization.
    """

    if attention_matrix.ndim != 2:
        raise ValueError("attention_matrix must be 2D.")

    column_sum = attention_matrix.sum(axis=0, keepdims=True)
    return attention_matrix / (column_sum + eps)


def plot_ddm_heatmap_grid(
    attention_matrices: dict,
    class_names: Optional[List[str]] = None,
    title: str = "Class-wise Diagnostic Dependency Attention",
    save_path: Optional[str] = None,
    show: bool = True,
    normalize: Optional[str] = None,
):
    """
    Plot multiple DDM heatmaps in a 2x3 grid.

    Parameters
    ----------
    attention_matrices:
        Dictionary where keys are subplot titles and values are
        attention matrices of shape (num_classes, num_classes).

        Example:
            {
                "NORM": norm_matrix,
                "MI": mi_matrix,
                "STTC": sttc_matrix,
                "CD": cd_matrix,
                "HYP": hyp_matrix,
                "Average": avg_matrix,
            }

    normalize:
        None:
            Use matrices as provided.

        "column":
            Column-normalize each matrix for complementary visualization.
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    if len(attention_matrices) > 6:
        raise ValueError("A 2x3 grid supports at most 6 attention matrices.")

    if normalize not in [None, "column"]:
        raise ValueError("normalize must be either None or 'column'.")

    processed_matrices = {}

    for name, matrix in attention_matrices.items():
        if matrix.ndim != 2:
            raise ValueError(f"Matrix for {name} must be 2D.")

        if matrix.shape != (len(class_names), len(class_names)):
            raise ValueError(
                f"Matrix for {name} must have shape "
                f"({len(class_names)}, {len(class_names)})."
            )

        if normalize == "column":
            matrix = column_normalize_attention(matrix)

        processed_matrices[name] = matrix

    vmin = min(matrix.min() for matrix in processed_matrices.values())
    vmax = max(matrix.max() for matrix in processed_matrices.values())

    _prepare_save_path(save_path)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        sharex=False,
        sharey=False,
    )

    axes = axes.flatten()

    last_im = None

    for ax_idx, ax in enumerate(axes):
        if ax_idx >= len(processed_matrices):
            ax.axis("off")
            continue

        subplot_title = list(processed_matrices.keys())[ax_idx]
        matrix = processed_matrices[subplot_title]

        last_im = ax.imshow(
            matrix,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(subplot_title)

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))

        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        ax.set_xlabel("Attended class")
        ax.set_ylabel("Query class")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    fig.suptitle(title, fontsize=16, y=0.98)

    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    if last_im is not None:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(last_im, cax=cbar_ax)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig

def plot_ddm_heatmap_grid_with_annotation(
    attention_matrices: dict,
    annotation_text: str,
    class_names: Optional[List[str]] = None,
    title: str = "Class-wise Diagnostic Dependency Attention",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot five DDM heatmaps in a 2x3 grid and use the sixth panel
    for text annotation.

    This version is intended for manuscript-ready visualization.
    """

    if class_names is None:
        class_names = DEFAULT_CLASS_NAMES

    if len(attention_matrices) != 5:
        raise ValueError("Expected exactly 5 class-wise attention matrices.")

    processed_matrices = {}

    for name, matrix in attention_matrices.items():
        if matrix.ndim != 2:
            raise ValueError(f"Matrix for {name} must be 2D.")

        if matrix.shape != (len(class_names), len(class_names)):
            raise ValueError(
                f"Matrix for {name} must have shape "
                f"({len(class_names)}, {len(class_names)})."
            )

        processed_matrices[name] = matrix

    vmin = min(matrix.min() for matrix in processed_matrices.values())
    vmax = max(matrix.max() for matrix in processed_matrices.values())

    _prepare_save_path(save_path)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(15, 9),
        sharex=False,
        sharey=False,
    )

    axes = axes.flatten()

    last_im = None

    for ax_idx, (subplot_title, matrix) in enumerate(processed_matrices.items()):
        ax = axes[ax_idx]

        last_im = ax.imshow(
            matrix,
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        ax.set_title(subplot_title)

        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))

        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

        if ax_idx in [3, 4]:
            ax.set_xlabel("Attended class")
        else:
            ax.set_xlabel("")

        if ax_idx in [0, 3]:
            ax.set_ylabel("Query class")
        else:
            ax.set_ylabel("")

        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    annotation_ax = axes[5]
    annotation_ax.axis("off")

    annotation_ax.text(
        0.02,
        0.95,
        annotation_text,
        ha="left",
        va="top",
        fontsize=11,
        wrap=True,
        transform=annotation_ax.transAxes,
    )

    fig.suptitle(title, fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    if last_im is not None:
        cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(last_im, cax=cbar_ax)

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig