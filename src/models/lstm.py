import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        num_classes: int = 5,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.hidden_multiplier = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * self.hidden_multiplier, num_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        output, _ = self.lstm(x)
        final_output = output[:, -1, :]
        return final_output

    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # Input from dataset: (batch, leads, samples)
    #     # LSTM expects:       (batch, samples, leads)
    #     x = x.transpose(1, 2)

    #     output, _ = self.lstm(x)

    #     # Use final time step representation
    #     final_output = output[:, -1, :]

    #     logits = self.classifier(final_output)

    #     return logits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.classifier(features)
        return logits
    
