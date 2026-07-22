"""
1D-CNN baseline for PTB-XL multi-label ECG classification.

Input:
    x: shape (batch_size, 12, 1000)

Output:
    logits: shape (batch_size, num_classes)

Note:
    The model returns raw logits.
    Use BCEWithLogitsLoss during training.
"""

import torch
import torch.nn as nn


class CNN1D(nn.Module):
    """
    Simple 1D-CNN baseline model.
    """

    def __init__(
        self,
        in_channels: int = 12,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     return self.classifier(self.features(x))