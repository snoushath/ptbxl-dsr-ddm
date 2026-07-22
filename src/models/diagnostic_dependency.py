import torch
import torch.nn as nn


class DiagnosticDependencyModule(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.label_embeddings = nn.Parameter(
            torch.randn(num_classes, feature_dim)
        )

        self.query_projection = nn.Linear(feature_dim, hidden_dim)
        self.key_projection = nn.Linear(feature_dim, hidden_dim)
        self.value_projection = nn.Linear(feature_dim, hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(feature_dim)

    # def forward(self, label_features: torch.Tensor) -> torch.Tensor:
    #     """
    #     label_features shape:
    #         (batch_size, num_classes, feature_dim)

    #     output shape:
    #         (batch_size, num_classes, feature_dim)
    #     """

    #     residual = label_features

    #     x = self.norm(label_features)

    #     q = self.query_projection(x)
    #     k = self.key_projection(x)
    #     v = self.value_projection(x)

    #     attention_scores = torch.matmul(
    #         q,
    #         k.transpose(-2, -1),
    #     ) / (q.size(-1) ** 0.5)

    #     attention_weights = torch.softmax(attention_scores, dim=-1)

    #     dependency_features = torch.matmul(attention_weights, v)

    #     dependency_features = self.output_projection(dependency_features)
    #     dependency_features = self.dropout(dependency_features)

    #     return residual + dependency_features

    def forward(self,label_features: torch.Tensor,return_attention: bool = False,):
        """
        label_features shape:
            (batch_size, num_classes, feature_dim)

        output shape:
            (batch_size, num_classes, feature_dim)

        If return_attention=True:
            also returns attention_weights of shape
            (batch_size, num_classes, num_classes)
        """

        residual = label_features

        x = self.norm(label_features)

        q = self.query_projection(x)
        k = self.key_projection(x)
        v = self.value_projection(x)

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / (q.size(-1) ** 0.5)

        attention_weights = torch.softmax(attention_scores, dim=-1)

        dependency_features = torch.matmul(attention_weights, v)

        dependency_features = self.output_projection(dependency_features)
        dependency_features = self.dropout(dependency_features)

        output = residual + dependency_features

        if return_attention:
            return output, attention_weights

        return output