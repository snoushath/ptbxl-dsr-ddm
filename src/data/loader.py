"""
ECG signal loading utilities for PTB-XL.

This module is responsible only for loading ECG waveform data
from WFDB files. It does not handle labels, preprocessing, or
PyTorch Dataset logic.
"""

from pathlib import Path
from typing import Union, Tuple, List
import wfdb
import numpy as np
import pandas as pd


def load_ecg_record(
    dataset_path: Union[str, Path],
    filename: str
):
    """
    Load one ECG record using WFDB.

    Parameters
    ----------
    dataset_path : str or Path
        Root path of the PTB-XL dataset.
        Example:
        /home/students/nshaffi/coding/CAD using PTB-XL/dataset/ptbxl

    filename : str
        Relative WFDB filename from ptbxl_database.csv.
        Example:
        records100/00000/00001_lr

    Returns
    -------
    record : wfdb.Record
        WFDB record object containing signal and metadata.
    """

    dataset_path = Path(dataset_path)
    record_path = dataset_path / filename

    record = wfdb.rdrecord(str(record_path))

    return record


def load_ecg_signal(
    dataset_path: Union[str, Path],
    filename: str,
    transpose: bool = True
) -> Tuple[np.ndarray, int, List[str]]:
    """
    Load ECG signal as a NumPy array.

    Parameters
    ----------
    dataset_path : str or Path
        Root path of the PTB-XL dataset.

    filename : str
        Relative WFDB filename from ptbxl_database.csv.

    transpose : bool
        If True, returns signal as (leads, samples).
        If False, returns signal as (samples, leads).

    Returns
    -------
    signal : np.ndarray
        ECG signal array.

    fs : int
        Sampling frequency.

    lead_names : list of str
        Names of the ECG leads.
    """

    record = load_ecg_record(dataset_path, filename)

    signal = record.p_signal.astype(np.float32)
    fs = record.fs
    lead_names = record.sig_name

    if transpose:
        signal = signal.T

    return signal, fs, lead_names


def load_ecg_from_metadata_row(
    dataset_path: Union[str, Path],
    row: pd.Series,
    sampling_rate: int = 100,
    transpose: bool = True
) -> Tuple[np.ndarray, int, List[str]]:
    """
    Load ECG signal using one row from ptbxl_database.csv.

    Parameters
    ----------
    dataset_path : str or Path
        Root path of the PTB-XL dataset.

    row : pd.Series
        One row from ptbxl_database.csv.

    sampling_rate : int
        100 for low-resolution records.
        500 for high-resolution records.

    transpose : bool
        If True, returns signal as (leads, samples).

    Returns
    -------
    signal : np.ndarray
        ECG signal.

    fs : int
        Sampling frequency.

    lead_names : list of str
        ECG lead names.
    """

    if sampling_rate == 100:
        filename = row["filename_lr"]
    elif sampling_rate == 500:
        filename = row["filename_hr"]
    else:
        raise ValueError("sampling_rate must be either 100 or 500.")

    return load_ecg_signal(
        dataset_path=dataset_path,
        filename=filename,
        transpose=transpose
    )