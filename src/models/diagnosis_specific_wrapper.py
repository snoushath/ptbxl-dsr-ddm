"""
Generic Diagnosis-Specific Representation (DSR) wrapper.

This module converts a global feature representation produced by an
ECG backbone into one diagnosis-specific representation per class.

The wrapper is intended for backbones that implement:

    extract_features(x) -> Tensor of shape (batch_size, feature_dim)

Supported existing backbones:
    - CNN1D
    - LSTMClassifier
    - TransformerClassifier

Input:
    x: Tensor of shape (batch_size, 12, 1000)

Output:
    logits: Tensor of shape (batch_size, num_classes)

The model returns raw logits. Use BCEWithLogitsLoss during training.
"""

import torch
import torch.nn as nn


class DiagnosisSpecificWrapper(nn.Module):
    """
    Add Diagnosis-Specific Representation learning to a backbone.

    This model implements:

        Backbone
            -> global feature
            -> learnable label-query interaction
            -> label-feature projection
            -> shared per-label classifier

    It intentionally excludes Diagnostic Dependency Modeling (DDM)
    and is therefore used as the Stage-I-only ablation model.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        if feature_dim <= 0:
            raise ValueError("feature_dim must be a positive integer.")

        if num_classes <= 0:
            raise ValueError("num_classes must be a positive integer.")

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # One trainable query for each diagnostic category.
        self.label_queries = nn.Parameter(
            torch.randn(num_classes, feature_dim)
        )

        # Retained to ensure a controlled comparison with
        # DependencyWrapper. The only removed component is DDM.
        self.label_feature_projection = nn.Linear(
            feature_dim,
            feature_dim,
        )

        # The same classifier is independently applied to every
        # diagnosis-specific representation.
        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def extract_label_features(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Construct diagnosis-specific representations.

        Args:
            x:
                ECG tensor of shape
                (batch_size, leads, samples).

        Returns:
            Diagnosis-specific feature tensor of shape
            (batch_size, num_classes, feature_dim).
        """
        if not hasattr(self.backbone, "extract_features"):
            raise AttributeError(
                "Backbone must implement an extract_features(x) method."
            )

        global_feature = self.backbone.extract_features(x)

        if global_feature.ndim != 2:
            raise ValueError(
                "Backbone extract_features(x) must return a 2D tensor "
                "with shape (batch_size, feature_dim). "
                f"Received shape: {tuple(global_feature.shape)}"
            )

        if global_feature.size(-1) != self.feature_dim:
            raise ValueError(
                "Backbone feature dimension does not match feature_dim. "
                f"Expected {self.feature_dim}, "
                f"received {global_feature.size(-1)}."
            )

        batch_size = global_feature.size(0)

        label_queries = self.label_queries.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        expanded_global_feature = global_feature.unsqueeze(1)

        # Element-wise diagnosis-specific modulation.
        label_features = label_queries * expanded_global_feature

        label_features = self.label_feature_projection(
            label_features
        )

        return label_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generate multi-label diagnostic logits.

        Args:
            x:
                ECG tensor of shape
                (batch_size, leads, samples).

        Returns:
            Raw logits of shape
            (batch_size, num_classes).
        """
        label_features = self.extract_label_features(x)

        logits = self.classifier(label_features).squeeze(-1)

        return logits