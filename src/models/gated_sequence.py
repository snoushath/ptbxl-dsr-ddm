import torch
import torch.nn as nn


class MambaInspiredBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        expansion: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()

        hidden_dim = d_model * expansion

        self.norm = nn.LayerNorm(d_model)

        self.input_projection = nn.Linear(d_model, hidden_dim * 2)

        self.depthwise_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=hidden_dim,
        )

        self.activation = nn.SiLU()

        self.gate_projection = nn.Linear(hidden_dim, hidden_dim)

        self.output_projection = nn.Linear(hidden_dim, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.norm(x)

        x_proj, gate = self.input_projection(x).chunk(2, dim=-1)

        x_conv = self.depthwise_conv(
            x_proj.transpose(1, 2)
        ).transpose(1, 2)

        x_conv = self.activation(x_conv)

        gate = torch.sigmoid(self.gate_projection(gate))

        x = x_conv * gate

        x = self.output_projection(x)
        x = self.dropout(x)

        return x + residual


class GatedSequenceClassifier(nn.Module):
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

        self.input_projection = nn.Linear(input_size, d_model)

        self.blocks = nn.Sequential(
            *[
                MambaInspiredBlock(
                    d_model=d_model,
                    expansion=expansion,
                    kernel_size=kernel_size,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input:  (batch, leads, samples)
        # Convert: (batch, samples, leads)
        x = x.transpose(1, 2)

        x = self.input_projection(x)

        x = self.blocks(x)

        pooled = x.mean(dim=1)

        logits = self.classifier(pooled)

        return logits