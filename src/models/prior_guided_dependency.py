import torch
import torch.nn as nn


class PriorGuidedDependencyModule(nn.Module):
    def __init__(
        self,
        prior_matrix: torch.Tensor,
        feature_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        learnable_prior: bool = True,
    ):
        super().__init__()

        if prior_matrix.ndim != 2:
            raise ValueError("prior_matrix must be 2D.")

        if prior_matrix.shape[0] != prior_matrix.shape[1]:
            raise ValueError("prior_matrix must be square.")

        self.num_classes = prior_matrix.shape[0]

        prior_matrix = prior_matrix.float()

        row_sum = prior_matrix.sum(dim=1, keepdim=True).clamp_min(1e-8)
        prior_matrix = prior_matrix / row_sum

        if learnable_prior:
            self.prior_logits = nn.Parameter(torch.log(prior_matrix + 1e-8))
        else:
            self.register_buffer("prior_logits", torch.log(prior_matrix + 1e-8))

        self.norm = nn.LayerNorm(feature_dim)

        self.query_projection = nn.Linear(feature_dim, hidden_dim)
        self.key_projection = nn.Linear(feature_dim, hidden_dim)
        self.value_projection = nn.Linear(feature_dim, hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, feature_dim)

        self.dropout = nn.Dropout(dropout)

        self.prior_strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, label_features: torch.Tensor) -> torch.Tensor:
        residual = label_features

        x = self.norm(label_features)

        q = self.query_projection(x)
        k = self.key_projection(x)
        v = self.value_projection(x)

        attention_scores = torch.matmul(
            q,
            k.transpose(-2, -1),
        ) / (q.size(-1) ** 0.5)

        prior_attention = torch.softmax(self.prior_logits, dim=-1)

        prior_bias = torch.log(prior_attention + 1e-8).unsqueeze(0)

        attention_scores = attention_scores + self.prior_strength * prior_bias

        attention_weights = torch.softmax(attention_scores, dim=-1)

        dependency_features = torch.matmul(attention_weights, v)

        dependency_features = self.output_projection(dependency_features)
        dependency_features = self.dropout(dependency_features)

        return residual + dependency_features

    def get_learned_prior(self) -> torch.Tensor:
        return torch.softmax(self.prior_logits.detach().cpu(), dim=-1)