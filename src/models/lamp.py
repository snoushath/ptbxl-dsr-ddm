from __future__ import annotations
from typing import Any
import torch
from torch import Tensor, nn

def count_trainable_parameters(model: nn.Module) -> int:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class LabelMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4,
                 feedforward_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads,
                                          dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(feedforward_dim, hidden_dim)
        )

    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        if states.ndim != 3:
            raise ValueError("states must have shape [B,C,D]")
        msg, weights = self.attn(states, states, states, need_weights=True,
                                 average_attn_weights=True)
        states = self.norm1(states + self.drop(msg))
        states = self.norm2(states + self.drop(self.ffn(states)))
        return states, weights

class LaMPHead(nn.Module):
    def __init__(self, feature_dim: int = 128, num_classes: int = 5,
                 hidden_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, feedforward_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        self.feature_dim = feature_dim
        self.label_embeddings = nn.Parameter(torch.empty(num_classes, hidden_dim))
        nn.init.normal_(self.label_embeddings, std=0.02)
        self.context = nn.Linear(feature_dim, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([
            LabelMessagePassingLayer(hidden_dim, num_heads,
                                     feedforward_dim, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.feature_dim:
            raise ValueError("features must have shape [B,feature_dim]")
        states = self.label_embeddings.unsqueeze(0).expand(features.size(0), -1, -1)
        states = self.input_norm(states + self.context(features).unsqueeze(1))
        attentions = []
        for layer in self.layers:
            states, attention = layer(states)
            attentions.append(attention)
        attention_stack = torch.stack(attentions, dim=1)
        return {
            "logits": self.classifier(states).squeeze(-1),
            "label_states": states,
            "attention_matrices": attention_stack,
            "final_attention": attention_stack[:, -1],
        }

class LaMPGatedSequence(nn.Module):
    KEYS = ("sequence_features", "features", "memory", "encoder_output",
            "hidden_states", "last_hidden_state")

    def __init__(self, backbone: nn.Module, feature_dim: int = 128,
                 num_classes: int = 5, hidden_dim: int = 128,
                 num_message_passing_layers: int = 2,
                 num_attention_heads: int = 4,
                 feedforward_dim: int = 256, dropout: float = 0.3,
                 temporal_layout: str = "auto", expected_num_leads: int = 12):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.temporal_layout = temporal_layout
        self.expected_num_leads = expected_num_leads
        self.lamp_head = LaMPHead(
            feature_dim, num_classes, hidden_dim,
            num_message_passing_layers, num_attention_heads,
            feedforward_dim, dropout
        )

    def _run_backbone(self, x: Tensor) -> Any:
        for name in ("forward_features", "extract_features", "encode"):
            fn = getattr(self.backbone, name, None)
            if callable(fn):
                return fn(x)
        return self.backbone(x)

    def _tensor(self, output: Any) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, dict):
            for key in self.KEYS:
                if isinstance(output.get(key), Tensor):
                    return output[key]
        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, Tensor):
                    return value
        raise TypeError("No feature tensor found in backbone output")

    def _pool(self, x: Tensor) -> Tensor:
        if x.ndim == 2:
            pooled = x
        elif x.ndim == 3:
            if self.temporal_layout == "BTD":
                pooled = x.mean(1)
            elif self.temporal_layout == "BDT":
                pooled = x.mean(2)
            elif x.shape[-1] == self.feature_dim:
                pooled = x.mean(1)
            elif x.shape[1] == self.feature_dim:
                pooled = x.mean(2)
            else:
                raise ValueError("Cannot infer temporal feature layout")
        else:
            raise ValueError("Backbone features must be rank 2 or 3")
        if pooled.shape[-1] != self.feature_dim:
            raise ValueError("Unexpected pooled feature dimension")
        return pooled

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        if x.ndim != 3 or x.shape[1] != self.expected_num_leads:
            raise ValueError("ECG input must have shape [B,12,T]")
        if not torch.isfinite(x).all():
            raise ValueError("ECG input contains non-finite values")
        features = self._pool(self._tensor(self._run_backbone(x)))
        output = self.lamp_head(features)
        output["features"] = features
        return output
