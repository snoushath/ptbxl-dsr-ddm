import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int,
        max_len: int = 1000,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 12,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        num_classes: int = 5,
        dropout: float = 0.3,
        max_len: int = 1000,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, d_model)

        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.input_projection(x)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return pooled

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # Input from dataset:   (batch, leads, samples)
    #     # Transformer expects:  (batch, samples, features)
    #     x = x.transpose(1, 2)

    #     x = self.input_projection(x)
    #     x = self.positional_encoding(x)

    #     encoded = self.encoder(x)

    #     # Global average pooling over time
    #     pooled = encoded.mean(dim=1)

    #     logits = self.classifier(pooled)

    #     return logits
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits