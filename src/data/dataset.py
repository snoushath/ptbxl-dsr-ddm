"""
PyTorch Dataset for PTB-XL ECG classification.

This module connects:
- loader.py
- labels.py
- preprocessing.py

It does not define preprocessing choices internally.
"""

from pathlib import Path
from typing import Optional, Union, Dict, Any

import torch
from torch.utils.data import Dataset
import pandas as pd

from src.data.loader import load_ecg_from_metadata_row
from src.data.labels import get_label_from_scp_codes


class PTBXLDataset(Dataset):
    """
    PyTorch Dataset for PTB-XL multi-label ECG classification.

    Each item returns:
        {
            "signal": torch.FloatTensor, shape (leads, samples),
            "label": torch.FloatTensor, shape (num_classes,),
            "ecg_id": int
        }
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        dataset_path: Union[str, Path],
        scp_statements: pd.DataFrame,
        sampling_rate: int = 100,
        transform: Optional[Any] = None,
    ):
        self.metadata = metadata.reset_index(drop=True)
        self.dataset_path = Path(dataset_path)
        self.scp_statements = scp_statements
        self.sampling_rate = sampling_rate
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.metadata.iloc[index]

        signal, fs, lead_names = load_ecg_from_metadata_row(
            dataset_path=self.dataset_path,
            row=row,
            sampling_rate=self.sampling_rate,
            transpose=True,
        )

        if self.transform is not None:
            signal = self.transform(signal)

        label = get_label_from_scp_codes(
            row["scp_codes"],
            self.scp_statements,
        )

        signal_tensor = torch.tensor(signal, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.float32)

        ecg_id = row["ecg_id"] if "ecg_id" in row else index

        return {
            "signal": signal_tensor,
            "label": label_tensor,
            "ecg_id": ecg_id,
            "fs": fs,
            "lead_names": lead_names,
        }