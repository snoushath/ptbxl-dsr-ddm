"""
Label processing utilities for PTB-XL.

This module converts SCP diagnostic codes into machine-learning labels.
"""

import ast
import numpy as np
import pandas as pd


DIAGNOSTIC_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def parse_scp_codes(scp_codes):
    """
    Convert SCP codes from string to dictionary.

    Example
    -------
    "{'NORM': 100}" -> {'NORM': 100}
    """

    if isinstance(scp_codes, dict):
        return scp_codes

    return ast.literal_eval(scp_codes)


def load_scp_statements(dataset_path):
    """
    Load scp_statements.csv.
    """

    return pd.read_csv(f"{dataset_path}/scp_statements.csv", index_col=0)


def get_diagnostic_superclasses(scp_codes, scp_statements):
    """
    Convert detailed SCP codes into diagnostic superclasses.

    Example
    -------
    {'AMI': 100} -> ['MI']
    {'NORM': 100} -> ['NORM']
    """

    scp_codes = parse_scp_codes(scp_codes)

    classes = []

    for code in scp_codes.keys():
        if code in scp_statements.index:
            diagnostic_class = scp_statements.loc[code, "diagnostic_class"]

            if pd.notna(diagnostic_class):
                classes.append(diagnostic_class)

    return sorted(list(set(classes)))


def make_multihot_label(classes, class_order=DIAGNOSTIC_CLASSES):
    """
    Convert diagnostic classes into a multi-hot vector.

    Example
    -------
    ['NORM'] -> [1, 0, 0, 0, 0]
    ['MI', 'STTC'] -> [0, 1, 1, 0, 0]
    """

    label = np.zeros(len(class_order), dtype=np.float32)

    for cls in classes:
        if cls in class_order:
            label[class_order.index(cls)] = 1.0

    return label


def get_label_from_scp_codes(scp_codes, scp_statements):
    """
    Convert raw SCP codes directly into multi-hot diagnostic label.
    """

    classes = get_diagnostic_superclasses(scp_codes, scp_statements)
    return make_multihot_label(classes)