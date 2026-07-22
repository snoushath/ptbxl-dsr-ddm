"""
Composable ECG preprocessing utilities.

This module only transforms already-loaded ECG signals.

It does NOT:
- load ECG/WFDB records
- read metadata CSV files
- create labels
- split train/validation/test data
- define PyTorch Dataset logic

Expected signal shape:
    (leads, samples)
"""

from typing import Callable, List
import numpy as np


class Compose:
    """
    Apply a sequence of preprocessing transforms.
    """

    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        for transform in self.transforms:
            signal = transform(signal)
        return signal


class EnsureFloat32:
    """
    Convert ECG signal to float32.
    """

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        return signal.astype(np.float32)


class PadOrCrop:
    """
    Pad or crop ECG signal to a fixed number of samples.

    Input:
        signal shape = (leads, samples)

    Output:
        signal shape = (leads, target_length)
    """

    def __init__(self, target_length: int):
        self.target_length = target_length

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim != 2:
            raise ValueError(
                f"Expected signal shape (leads, samples), got {signal.shape}"
            )

        leads, samples = signal.shape

        if samples == self.target_length:
            return signal.astype(np.float32)

        if samples > self.target_length:
            return signal[:, :self.target_length].astype(np.float32)

        pad_length = self.target_length - samples

        signal = np.pad(
            signal,
            pad_width=((0, 0), (0, pad_length)),
            mode="constant",
            constant_values=0.0,
        )

        return signal.astype(np.float32)


class PerLeadZScore:
    """
    Apply z-score normalization independently to each ECG lead.

    For each lead:
        normalized = (lead - lead_mean) / lead_std

    Input:
        signal shape = (leads, samples)

    Output:
        signal shape = (leads, samples)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim != 2:
            raise ValueError(
                f"Expected signal shape (leads, samples), got {signal.shape}"
            )

        mean = np.mean(signal, axis=1, keepdims=True)
        std = np.std(signal, axis=1, keepdims=True)

        signal = (signal - mean) / (std + self.eps)

        return signal.astype(np.float32)


class GlobalZScore:
    """
    Apply z-score normalization using the whole ECG recording.

    Input:
        signal shape = (leads, samples)

    Output:
        signal shape = (leads, samples)
    """

    def __init__(self, eps: float = 1e-8):
        self.eps = eps

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        mean = np.mean(signal)
        std = np.std(signal)

        signal = (signal - mean) / (std + self.eps)

        return signal.astype(np.float32)


class MinMaxNormalize:
    """
    Normalize ECG signal to a fixed range.

    Default output range:
        [-1, 1]
    """

    def __init__(
        self,
        min_value: float = -1.0,
        max_value: float = 1.0,
        eps: float = 1e-8,
    ):
        self.min_value = min_value
        self.max_value = max_value
        self.eps = eps

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        sig_min = np.min(signal)
        sig_max = np.max(signal)

        signal = (signal - sig_min) / (sig_max - sig_min + self.eps)
        signal = signal * (self.max_value - self.min_value) + self.min_value

        return signal.astype(np.float32)


class ClipSignal:
    """
    Clip ECG signal values to a fixed range.

    Useful after normalization to reduce extreme values.
    """

    def __init__(self, min_value: float = -5.0, max_value: float = 5.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        signal = np.clip(signal, self.min_value, self.max_value)
        return signal.astype(np.float32)


class ToChannelFirst:
    """
    Ensure ECG signal is in (leads, samples) format.

    If the signal is already (leads, samples), it is returned unchanged.
    If the signal is (samples, leads), it is transposed.

    This is useful as a safety transform.
    """

    def __init__(self, num_leads: int = 12):
        self.num_leads = num_leads

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        if signal.ndim != 2:
            raise ValueError(
                f"Expected 2D ECG signal, got shape {signal.shape}"
            )

        if signal.shape[0] == self.num_leads:
            return signal.astype(np.float32)

        if signal.shape[1] == self.num_leads:
            return signal.T.astype(np.float32)

        raise ValueError(
            f"Could not infer ECG lead dimension from shape {signal.shape}"
        )


class RemoveNaN:
    """
    Replace NaN and infinite values with finite numbers.
    """

    def __init__(
        self,
        nan_value: float = 0.0,
        posinf_value: float = 0.0,
        neginf_value: float = 0.0,
    ):
        self.nan_value = nan_value
        self.posinf_value = posinf_value
        self.neginf_value = neginf_value

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        signal = np.nan_to_num(
            signal,
            nan=self.nan_value,
            posinf=self.posinf_value,
            neginf=self.neginf_value,
        )

        return signal.astype(np.float32)