"""
Gated Sequence model with Diagnosis-Specific Representation learning.

This module implements the Stage-I-only ablation:

    Gated Sequence backbone
        -> global average pooling
        -> learnable label-query interaction
        -> label-feature projection
        -> shared per-label classifier

The Diagnostic Dependency Module is intentionally excluded.

Input:
    x: Tensor of shape (batch_size, 12, 1000)

Output:
    logits: Tensor of shape (batch_size, num_classes)

The model returns raw logits. Use BCEWithLogitsLoss during training.
"""

import torch
import torch.nn as nn

from src.models.gated_sequence import GatedSequenceClassifier


class DiagnosisSpecificGatedSequenceClassifier(nn.Module):
    """
    Gated Sequence backbone followed by Stage-I DSR learning.

    This class mirrors DependencyGatedSequenceClassifier while
    intentionally removing only the Diagnostic Dependency Module.
    """

    def __init__(
        self,
        input_size: int = 12,
        d_model: int = 128,
        num_layers: int = 4,
        num_classes: int = 5,
        expansion: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        if input_size <= 0:
            raise ValueError("input_size must be a positive integer.")

        if d_model <= 0:
            raise ValueError("d_model must be a positive integer.")

        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        self.input_size = input_size
        self.d_model = d_model
        self.num_classes = num_classes

        self.input_projection = nn.Linear(
            input_size,
            d_model,
        )

        # Reuse the validated Gated Sequence blocks.
        self.sequence_blocks = GatedSequenceClassifier(
            input_size=input_size,
            d_model=d_model,
            num_layers=num_layers,
            num_classes=num_classes,
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        ).blocks

        # One learnable query for every diagnostic category.
        self.label_queries = nn.Parameter(
            torch.randn(num_classes, d_model)
        )

        # Retained for direct comparability with the complete
        # Gated Sequence + DSR + DDM model.
        self.label_feature_projection = nn.Linear(
            d_model,
            d_model,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def extract_global_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the pooled shared ECG representation.

        Args:
            x:
                ECG tensor of shape
                (batch_size, leads, samples).

        Returns:
            Shared ECG feature tensor of shape
            (batch_size, d_model).
        """
        if x.ndim != 3:
            raise ValueError(
                "Expected ECG input with shape "
                "(batch_size, leads, samples). "
                f"Received shape: {tuple(x.shape)}"
            )

        if x.size(1) != self.input_size:
            raise ValueError(
                f"Expected {self.input_size} ECG leads, "
                f"received {x.size(1)}."
            )

        # (batch, leads, samples)
        # -> (batch, samples, leads)
        x = x.transpose(1, 2)

        x = self.input_projection(x)

        sequence_features = self.sequence_blocks(x)

        global_feature = sequence_features.mean(dim=1)

        return global_feature

    def extract_label_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct diagnosis-specific representations.

        Returns:
            Tensor of shape
            (batch_size, num_classes, d_model).
        """
        global_feature = self.extract_global_features(x)

        batch_size = global_feature.size(0)

        label_queries = self.label_queries.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        expanded_global_feature = global_feature.unsqueeze(1)

        label_features = (
            label_queries * expanded_global_feature
        )

        label_features = self.label_feature_projection(
            label_features
        )

        return label_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate multi-label diagnostic logits.

        Returns:
            Raw logits with shape
            (batch_size, num_classes).
        """
        label_features = self.extract_label_features(x)

        logits = self.classifier(label_features).squeeze(-1)

        return logits