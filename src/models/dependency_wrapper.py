import torch
import torch.nn as nn

from src.models.diagnostic_dependency import DiagnosticDependencyModule


class DependencyWrapper(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = backbone
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.label_queries = nn.Parameter(
            torch.randn(num_classes, feature_dim)
        )

        self.label_feature_projection = nn.Linear(feature_dim, feature_dim)

        self.dependency_module = DiagnosticDependencyModule(
            num_classes=num_classes,
            feature_dim=feature_dim,
            hidden_dim=feature_dim,
            dropout=dropout,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self.backbone, "extract_features"):
            raise AttributeError(
                "Backbone must implement an extract_features(x) method."
            )

        global_feature = self.backbone.extract_features(x)

        batch_size = global_feature.size(0)

        label_queries = self.label_queries.unsqueeze(0).expand(
            batch_size,
            -1,
            -1,
        )

        global_feature = global_feature.unsqueeze(1)

        label_features = label_queries * global_feature

        label_features = self.label_feature_projection(label_features)

        label_features = self.dependency_module(label_features)

        logits = self.classifier(label_features).squeeze(-1)

        return logits