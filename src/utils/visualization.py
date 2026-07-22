"""
Visualization utilities for ECG signals and batches.
"""

from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import matplotlib.pyplot as plt


DEFAULT_LEAD_NAMES = [
    "I", "II", "III",
    "aVR", "aVL", "aVF",
    "V1", "V2", "V3",
    "V4", "V5", "V6",
]


def _to_numpy(array: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    Convert torch.Tensor or np.ndarray to NumPy array.
    """
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()

    if isinstance(array, np.ndarray):
        return array

    raise TypeError(
        f"Expected np.ndarray or torch.Tensor, got {type(array)}"
    )


def plot_single_lead(
    signal: Union[np.ndarray, torch.Tensor],
    lead_index: int = 0,
    lead_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
):
    """
    Plot one ECG lead from one ECG signal.

    Expected signal shape:
        (leads, samples)
    """

    signal = _to_numpy(signal)

    if signal.ndim != 2:
        raise ValueError(
            f"Expected signal shape (leads, samples), got {signal.shape}"
        )

    if lead_names is None:
        lead_names = DEFAULT_LEAD_NAMES

    lead_name = lead_names[lead_index]

    plt.figure(figsize=(14, 4))
    plt.plot(signal[lead_index])
    plt.title(title or f"ECG Lead {lead_name}")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_12_leads(
    signal: Union[np.ndarray, torch.Tensor],
    lead_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
):
    """
    Plot all 12 ECG leads in a 4 x 3 layout.

    Expected signal shape:
        (12, samples)
    """

    signal = _to_numpy(signal)

    if signal.ndim != 2:
        raise ValueError(
            f"Expected signal shape (leads, samples), got {signal.shape}"
        )

    if signal.shape[0] != 12:
        raise ValueError(
            f"Expected 12 leads, got {signal.shape[0]}"
        )

    if lead_names is None:
        lead_names = DEFAULT_LEAD_NAMES

    fig, axes = plt.subplots(
        nrows=4,
        ncols=3,
        figsize=(15, 10),
        sharex=True,
    )

    axes = axes.flatten()

    for i in range(12):
        axes[i].plot(signal[i])
        axes[i].set_title(lead_names[i])
        axes[i].grid(True)

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_batch_single_lead(
    signals: Union[np.ndarray, torch.Tensor],
    labels: Optional[Union[np.ndarray, torch.Tensor]] = None,
    ecg_ids: Optional[Sequence] = None,
    lead_index: int = 0,
    lead_names: Optional[Sequence[str]] = None,
    max_samples: Optional[int] = None,
    class_names: Optional[Sequence[str]] = None,
    title: Optional[str] = None,
):
    """
    Plot one ECG lead from every sample in a batch.

    Expected signals shape:
        (batch_size, leads, samples)

    This function is useful for visually checking:
    - batch loading
    - preprocessing consistency
    - label alignment
    - corrupted or flat signals
    """

    signals = _to_numpy(signals)

    if signals.ndim != 3:
        raise ValueError(
            f"Expected signals shape (batch, leads, samples), got {signals.shape}"
        )

    if labels is not None:
        labels = _to_numpy(labels)

    if lead_names is None:
        lead_names = DEFAULT_LEAD_NAMES

    batch_size = signals.shape[0]
    lead_name = lead_names[lead_index]

    fig, axes = plt.subplots(
        nrows=batch_size,
        ncols=1,
        figsize=(15, 2.2 * batch_size),
        sharex=True,
    )

    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        signal_to_plot = signals[i, lead_index]

        if max_samples is not None:
            signal_to_plot = signal_to_plot[:max_samples]

        axes[i].plot(signal_to_plot)
        axes[i].grid(True)

        title_parts = [f"Sample {i}"]

        if ecg_ids is not None:
            title_parts.append(f"ECG ID: {ecg_ids[i]}")

        title_parts.append(f"Lead: {lead_name}")

        if labels is not None:
            label_vector = labels[i].astype(int).tolist()

            if class_names is not None:
                active_classes = [
                    class_names[j]
                    for j, value in enumerate(label_vector)
                    if value == 1
                ]
                label_text = active_classes if active_classes else ["None"]
            else:
                label_text = label_vector

            title_parts.append(f"Label: {label_text}")

        axes[i].set_title(" | ".join(title_parts), fontsize=10)

    axes[-1].set_xlabel("Samples")

    if title is not None:
        fig.suptitle(title, fontsize=14)

    plt.tight_layout()
    plt.show()