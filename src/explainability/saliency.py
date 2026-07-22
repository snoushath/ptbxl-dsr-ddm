"""
Saliency utilities for ECG explainability.

This module computes gradient-based saliency maps for multi-label
ECG classification models.

Expected input shape:
    x: (batch_size, leads, samples)

Expected model output:
    logits: (batch_size, num_classes)
"""

from typing import Optional

import numpy as np
import torch


def compute_vanilla_saliency(
    model: torch.nn.Module,
    x: torch.Tensor,
    target_class: int,
    device: torch.device,
    absolute: bool = True,
    normalize: bool = True,
) -> np.ndarray:
    """
    Compute vanilla gradient saliency for a target diagnostic class.

    Parameters
    ----------
    model:
        Trained PyTorch model.

    x:
        Input ECG tensor of shape (1, leads, samples) or
        (batch_size, leads, samples).

    target_class:
        Target class index.

    device:
        CPU or CUDA device.

    absolute:
        If True, use absolute gradients.

    normalize:
        If True, normalize saliency to [0, 1].

    Returns
    -------
    saliency:
        NumPy array with same shape as x:
        (batch_size, leads, samples)
    """

    model.eval()

    x = x.to(device)
    x = x.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)

    logits = model(x)

    if isinstance(logits, tuple):
        logits = logits[0]

    if logits.ndim != 2:
        raise ValueError("Expected model output shape (batch_size, num_classes).")

    if target_class < 0 or target_class >= logits.shape[1]:
        raise ValueError("target_class is out of range.")

    score = logits[:, target_class].sum()

    score.backward()

    saliency = x.grad.detach().cpu().numpy()

    if absolute:
        saliency = np.abs(saliency)

    if normalize:
        saliency = normalize_saliency(saliency)

    return saliency


def normalize_saliency(
    saliency: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalize saliency map to [0, 1] per sample.
    """

    if saliency.ndim != 3:
        raise ValueError("saliency must have shape (batch_size, leads, samples).")

    saliency_norm = np.zeros_like(saliency, dtype=np.float32)

    for idx in range(saliency.shape[0]):
        sample = saliency[idx]

        min_value = sample.min()
        max_value = sample.max()

        saliency_norm[idx] = (sample - min_value) / (max_value - min_value + eps)

    return saliency_norm.astype(np.float32)


def select_target_class(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> int:
    """
    Select target class for saliency.

    If labels are provided and contain at least one positive class,
    choose the positive class with highest predicted logit.

    Otherwise, choose the class with highest predicted logit.
    """

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch_size, num_classes).")

    if logits.shape[0] != 1:
        raise ValueError("select_target_class currently expects batch_size=1.")

    logits_1d = logits[0]

    if labels is not None:
        if labels.ndim == 2:
            labels_1d = labels[0]
        elif labels.ndim == 1:
            labels_1d = labels
        else:
            raise ValueError("labels must be 1D or 2D.")

        positive_indices = torch.where(labels_1d > 0.5)[0]

        if len(positive_indices) > 0:
            positive_logits = logits_1d[positive_indices]
            best_positive_idx = torch.argmax(positive_logits)
            return int(positive_indices[best_positive_idx].item())

    return int(torch.argmax(logits_1d).item())