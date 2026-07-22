"""Query2Label-style dependency baseline for multi-label ECG classification."""
from __future__ import annotations
from typing import Any
import torch
from torch import Tensor, nn

def count_trainable_parameters(model: nn.Module) -> int:
    if not isinstance(model, nn.Module):
        raise TypeError("model must be an instance of torch.nn.Module.")
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class Query2LabelHead(nn.Module):
    def __init__(self, *, feature_dim: int = 128, num_classes: int = 5,
                 num_decoder_layers: int = 2, num_attention_heads: int = 4,
                 feedforward_dim: int = 256, dropout: float = 0.3,
                 activation: str = "gelu") -> None:
        super().__init__()
        if feature_dim <= 0 or num_classes <= 1 or num_decoder_layers <= 0:
            raise ValueError("Invalid Query2Label dimensions.")
        if feature_dim % num_attention_heads != 0:
            raise ValueError("feature_dim must be divisible by num_attention_heads.")
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.label_queries = nn.Parameter(torch.empty(num_classes, feature_dim))
        nn.init.normal_(self.label_queries, mean=0.0, std=0.02)
        layer = nn.TransformerDecoderLayer(
            d_model=feature_dim, nhead=num_attention_heads,
            dim_feedforward=feedforward_dim, dropout=dropout,
            activation=activation, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_decoder_layers)
        self.output_norm = nn.LayerNorm(feature_dim)
        self.classifier = nn.Linear(feature_dim, 1)

    def forward(self, memory: Tensor, *, memory_key_padding_mask: Tensor | None = None) -> dict[str, Tensor]:
        if memory.ndim != 3 or memory.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected memory [B,T,{self.feature_dim}], got {tuple(memory.shape)}.")
        if not torch.isfinite(memory).all():
            raise ValueError("memory contains non-finite values.")
        if memory_key_padding_mask is not None:
            if tuple(memory_key_padding_mask.shape) != (memory.shape[0], memory.shape[1]):
                raise ValueError("Invalid memory_key_padding_mask shape.")
            memory_key_padding_mask = memory_key_padding_mask.bool()
        queries = self.label_queries.unsqueeze(0).expand(memory.shape[0], -1, -1)
        label_features = self.decoder(
            tgt=queries, memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        label_features = self.output_norm(label_features)
        logits = self.classifier(label_features).squeeze(-1)
        return {"logits": logits, "label_features": label_features,
                "label_queries": queries, "memory": memory}

class Query2LabelGatedSequence(nn.Module):
    _KEYS = ("sequence_features", "features", "memory", "encoder_output", "hidden_states", "last_hidden_state")
    def __init__(self, *, backbone: nn.Module, feature_dim: int = 128,
                 num_classes: int = 5, num_decoder_layers: int = 2,
                 num_attention_heads: int = 4, feedforward_dim: int = 256,
                 dropout: float = 0.3, temporal_layout: str = "auto",
                 expected_num_leads: int = 12) -> None:
        super().__init__()
        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be a torch.nn.Module.")
        if temporal_layout not in {"auto", "BTD", "BDT"}:
            raise ValueError("temporal_layout must be auto, BTD, or BDT.")
        self.backbone = backbone
        self.feature_dim = feature_dim
        self.temporal_layout = temporal_layout
        self.expected_num_leads = expected_num_leads
        self.query2label_head = Query2LabelHead(
            feature_dim=feature_dim, num_classes=num_classes,
            num_decoder_layers=num_decoder_layers,
            num_attention_heads=num_attention_heads,
            feedforward_dim=feedforward_dim, dropout=dropout,
        )

    def _run_backbone(self, x: Tensor) -> Any:
        for name in ("forward_features", "extract_features", "encode"):
            method = getattr(self.backbone, name, None)
            if callable(method):
                return method(x)
        return self.backbone(x)

    def _extract_tensor(self, output: Any) -> Tensor:
        if isinstance(output, Tensor):
            return output
        if isinstance(output, dict):
            for key in self._KEYS:
                value = output.get(key)
                if isinstance(value, Tensor):
                    return value
        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, Tensor):
                    return value
        raise TypeError("Backbone output contains no recognized feature tensor.")

    def _to_btd(self, features: Tensor) -> Tensor:
        if features.ndim == 2:
            if features.shape[-1] != self.feature_dim:
                raise ValueError("Invalid pooled feature dimension.")
            return features.unsqueeze(1)
        if features.ndim != 3:
            raise ValueError("Backbone features must be rank 2 or 3.")
        if self.temporal_layout == "BTD":
            memory = features
        elif self.temporal_layout == "BDT":
            memory = features.transpose(1, 2)
        elif features.shape[-1] == self.feature_dim:
            memory = features
        elif features.shape[1] == self.feature_dim:
            memory = features.transpose(1, 2)
        else:
            raise ValueError("Cannot infer temporal feature layout.")
        if memory.shape[-1] != self.feature_dim:
            raise ValueError("Invalid memory feature dimension.")
        return memory

    def forward(self, x: Tensor, *, memory_key_padding_mask: Tensor | None = None) -> dict[str, Tensor]:
        if x.ndim != 3:
            raise ValueError("Expected ECG input [B,12,T].")
        if x.shape[1] != self.expected_num_leads:
            raise ValueError(f"Expected {self.expected_num_leads} leads.")
        if not torch.isfinite(x).all():
            raise ValueError("ECG input contains non-finite values.")
        memory = self._to_btd(self._extract_tensor(self._run_backbone(x)))
        return self.query2label_head(memory, memory_key_padding_mask=memory_key_padding_mask)
