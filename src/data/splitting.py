"""
Patient-wise splitting utilities for PTB-XL.

This module creates train/validation/test splits without patient leakage.
It does not load ECG signals, create labels, preprocess data, or create DataLoaders.
"""

from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def create_patient_wise_split(
    metadata: pd.DataFrame,
    patient_col: str = "patient_id",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create patient-wise train/validation/test split.

    Parameters
    ----------
    metadata : pd.DataFrame
        PTB-XL metadata dataframe.

    patient_col : str
        Column name containing patient IDs.

    train_size : float
        Fraction of patients assigned to training.

    val_size : float
        Fraction of patients assigned to validation.

    test_size : float
        Fraction of patients assigned to testing.

    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    train_metadata, val_metadata, test_metadata : pd.DataFrame
        Metadata splits with no patient overlap.
    """

    if patient_col not in metadata.columns:
        raise ValueError(f"Column '{patient_col}' not found in metadata.")

    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"train_size + val_size + test_size must equal 1.0, got {total}"
        )

    unique_patients = metadata[patient_col].dropna().unique()

    train_patients, temp_patients = train_test_split(
        unique_patients,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
    )

    relative_val_size = val_size / (val_size + test_size)

    val_patients, test_patients = train_test_split(
        temp_patients,
        train_size=relative_val_size,
        random_state=random_state,
        shuffle=True,
    )

    train_metadata = metadata[metadata[patient_col].isin(train_patients)].copy()
    val_metadata = metadata[metadata[patient_col].isin(val_patients)].copy()
    test_metadata = metadata[metadata[patient_col].isin(test_patients)].copy()

    return (
        train_metadata.reset_index(drop=True),
        val_metadata.reset_index(drop=True),
        test_metadata.reset_index(drop=True),
    )


def check_patient_overlap(
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    patient_col: str = "patient_id",
) -> bool:
    """
    Check whether there is patient overlap across splits.

    Returns
    -------
    bool
        True if there is no patient overlap.
        False if leakage exists.
    """

    train_patients = set(train_metadata[patient_col].dropna().unique())
    val_patients = set(val_metadata[patient_col].dropna().unique())
    test_patients = set(test_metadata[patient_col].dropna().unique())

    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    val_test_overlap = val_patients.intersection(test_patients)

    return (
        len(train_val_overlap) == 0
        and len(train_test_overlap) == 0
        and len(val_test_overlap) == 0
    )


def summarize_split(
    train_metadata: pd.DataFrame,
    val_metadata: pd.DataFrame,
    test_metadata: pd.DataFrame,
    patient_col: str = "patient_id",
) -> pd.DataFrame:
    """
    Summarize number of ECG records and unique patients in each split.
    """

    summary = pd.DataFrame({
        "split": ["train", "validation", "test"],
        "num_records": [
            len(train_metadata),
            len(val_metadata),
            len(test_metadata),
        ],
        "num_patients": [
            train_metadata[patient_col].nunique(),
            val_metadata[patient_col].nunique(),
            test_metadata[patient_col].nunique(),
        ],
    })

    return summary