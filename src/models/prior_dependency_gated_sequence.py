import torch
import torch.nn as nn

from src.models.gated_sequence import GatedSequenceClassifier
from src.models.prior_guided_dependency import PriorGuidedDependencyModule


class PriorDependencyGatedSequenceClassifier(nn.Module):
    def __init__(
        self,
        prior_matrix: torch.Tensor,
        input_size: int = 12,
        d_model: int = 128,
        num_layers: int = 4,
        num_classes: int = 5,
        expansion: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.3,
        learnable_prior: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.d_model = d_model

        self.input_projection = nn.Linear(input_size, d_model)

        self.sequence_blocks = GatedSequenceClassifier(
            input_size=input_size,
            d_model=d_model,
            num_layers=num_layers,
            num_classes=num_classes,
            expansion=expansion,
            kernel_size=kernel_size,
            dropout=dropout,
        ).blocks

        self.label_queries = nn.Parameter(
            torch.randn(num_classes, d_model)
        )

        self.label_feature_projection = nn.Linear(d_model, d_model)

        self.dependency_module = PriorGuidedDependencyModule(
            prior_matrix=prior_matrix,
            feature_dim=d_model,
            hidden_dim=d_model,
            dropout=dropout,
            learnable_prior=learnable_prior,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)

        x = self.input_projection(x)

        sequence_features = self.sequence_blocks(x)

        global_feature = sequence_features.mean(dim=1)

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