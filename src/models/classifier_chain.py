"""
Classifier Chain baseline for multi-label ECG classification.

The model uses an existing ECG backbone to obtain a shared feature
representation and predicts the diagnostic labels sequentially.

Canonical label indices:
    0 -> NORM
    1 -> MI
    2 -> STTC
    3 -> CD
    4 -> HYP

Input shape:
    x: [batch_size, 12, 1000]

Output shape:
    logits: [batch_size, 5]

Important:
    The model returns raw logits. Do not apply sigmoid before
    BCEWithLogitsLoss.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn


DEFAULT_LABEL_NAMES: Tuple[str, ...] = (
    "NORM",
    "MI",
    "STTC",
    "CD",
    "HYP",
)


class ClassifierChainHead(nn.Module):
    """
    Sequential classifier-chain head.

    At chain position k, the corresponding classifier receives:

        [shared_features, previous_label_values]

    During training, previous ground-truth labels may be supplied through
    teacher forcing. During inference, previous predicted probabilities are
    used.

    Parameters
    ----------
    feature_dim:
        Dimension of the pooled backbone representation.

    num_classes:
        Number of multi-label outputs.

    hidden_dim:
        Optional hidden-layer dimension for each binary classifier.
        When None, each chain classifier is a single linear layer.

    dropout:
        Dropout applied within each binary classifier.

    label_order:
        Sequence containing the canonical class indices in the order in which
        they should be predicted. For example, [0, 1, 2, 3, 4] corresponds to
        NORM -> MI -> STTC -> CD -> HYP.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        label_order: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError("feature_dim must be greater than zero.")

        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1).")

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.hidden_dim = hidden_dim
        self.dropout = float(dropout)

        if label_order is None:
            label_order = list(range(num_classes))

        self.label_order = self._validate_label_order(
            label_order=label_order,
            num_classes=num_classes,
        )

        classifiers: List[nn.Module] = []

        for chain_position in range(num_classes):
            # Each later classifier receives all previous chain predictions.
            classifier_input_dim = feature_dim + chain_position

            if hidden_dim is None:
                classifier = nn.Linear(classifier_input_dim, 1)
            else:
                if hidden_dim <= 0:
                    raise ValueError("hidden_dim must be greater than zero.")

                classifier = nn.Sequential(
                    nn.Linear(classifier_input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )

            classifiers.append(classifier)

        self.classifiers = nn.ModuleList(classifiers)

    @staticmethod
    def _validate_label_order(
        label_order: Sequence[int],
        num_classes: int,
    ) -> Tuple[int, ...]:
        order = tuple(int(index) for index in label_order)

        if len(order) != num_classes:
            raise ValueError(
                "label_order must contain exactly one index for each class. "
                f"Expected {num_classes} entries, received {len(order)}."
            )

        expected = set(range(num_classes))
        received = set(order)

        if received != expected:
            raise ValueError(
                "label_order must be a permutation of "
                f"{list(range(num_classes))}. Received {list(order)}."
            )

        return order

    def forward(
        self,
        features: Tensor,
        targets: Optional[Tensor] = None,
        teacher_forcing: bool = False,
    ) -> Dict[str, Tensor]:
        """
        Generate classifier-chain logits.

        Parameters
        ----------
        features:
            Shared ECG representation with shape [B, feature_dim].

        targets:
            Optional ground-truth multi-label tensor with shape [B, C].
            Targets must follow the canonical class order.

        teacher_forcing:
            When True, previous ground-truth labels are passed to subsequent
            chain classifiers. This should normally be enabled during training
            and disabled during validation and testing.

        Returns
        -------
        dict
            {
                "logits": canonical-order logits [B, C],
                "chain_logits": chain-order logits [B, C],
                "chain_probabilities": chain-order probabilities [B, C]
            }
        """
        self._validate_inputs(
            features=features,
            targets=targets,
            teacher_forcing=teacher_forcing,
        )

        batch_size = features.size(0)

        chain_logits: List[Tensor] = []
        previous_values: List[Tensor] = []

        for chain_position, canonical_label_index in enumerate(self.label_order):
            if chain_position == 0:
                classifier_input = features
            else:
                previous_tensor = torch.cat(previous_values, dim=1)
                classifier_input = torch.cat(
                    [features, previous_tensor],
                    dim=1,
                )

            current_logit = self.classifiers[chain_position](classifier_input)
            chain_logits.append(current_logit)

            if teacher_forcing:
                if targets is None:
                    raise RuntimeError(
                        "targets must be provided when teacher_forcing=True."
                    )

                current_value = targets[
                    :,
                    canonical_label_index,
                ].unsqueeze(1).to(dtype=features.dtype)
            else:
                # Predicted probabilities are used as inputs to the next
                # classifier. The operation remains differentiable.
                current_value = torch.sigmoid(current_logit)

            previous_values.append(current_value)

        chain_logits_tensor = torch.cat(chain_logits, dim=1)
        chain_probabilities = torch.sigmoid(chain_logits_tensor)

        # Convert the outputs from chain order back to canonical label order.
        canonical_logits = torch.empty(
            batch_size,
            self.num_classes,
            dtype=chain_logits_tensor.dtype,
            device=chain_logits_tensor.device,
        )

        for chain_position, canonical_label_index in enumerate(self.label_order):
            canonical_logits[:, canonical_label_index] = (
                chain_logits_tensor[:, chain_position]
            )

        return {
            "logits": canonical_logits,
            "chain_logits": chain_logits_tensor,
            "chain_probabilities": chain_probabilities,
        }

    def _validate_inputs(
        self,
        features: Tensor,
        targets: Optional[Tensor],
        teacher_forcing: bool,
    ) -> None:
        if features.ndim != 2:
            raise ValueError(
                "features must have shape [B, feature_dim]. "
                f"Received shape {tuple(features.shape)}."
            )

        if features.size(1) != self.feature_dim:
            raise ValueError(
                f"Expected feature dimension {self.feature_dim}, "
                f"received {features.size(1)}."
            )

        if targets is not None:
            expected_shape = (features.size(0), self.num_classes)

            if tuple(targets.shape) != expected_shape:
                raise ValueError(
                    f"targets must have shape {expected_shape}. "
                    f"Received {tuple(targets.shape)}."
                )

        if teacher_forcing and targets is None:
            raise ValueError(
                "targets must be provided when teacher_forcing=True."
            )


class ClassifierChainGatedSequence(nn.Module):
    """
    Classifier Chain model using an externally supplied ECG backbone.

    The wrapper is intentionally compatible with several backbone interfaces.
    It searches for feature-producing methods in the following order:

        1. backbone.forward_features(x)
        2. backbone.extract_features(x)
        3. backbone.encode(x)
        4. backbone(x, return_features=True)
        5. backbone(x)

    The extracted output may be:

        - [B, D]
        - [B, T, D]
        - [B, D, T]
        - a dictionary containing a suitable feature tensor
        - a tuple whose first suitable tensor is a feature representation

    Parameters
    ----------
    backbone:
        Existing Gated Sequence backbone.

    feature_dim:
        Required dimension of the pooled backbone representation.

    num_classes:
        Number of diagnostic labels.

    hidden_dim:
        Optional hidden dimension for each chain classifier.

    dropout:
        Dropout used by the chain classifiers.

    label_order:
        Prediction order expressed using canonical label indices.

    temporal_layout:
        Layout used when a three-dimensional backbone tensor is returned:

        - "BTD": [batch, time, feature]
        - "BDT": [batch, feature, time]
        - "auto": infer the feature dimension using feature_dim
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int = 5,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.2,
        label_order: Optional[Sequence[int]] = None,
        temporal_layout: str = "auto",
    ) -> None:
        super().__init__()

        if not isinstance(backbone, nn.Module):
            raise TypeError("backbone must be an instance of torch.nn.Module.")

        if temporal_layout not in {"auto", "BTD", "BDT"}:
            raise ValueError(
                "temporal_layout must be one of: 'auto', 'BTD', or 'BDT'."
            )

        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.temporal_layout = temporal_layout

        self.chain_head = ClassifierChainHead(
            feature_dim=feature_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            label_order=label_order,
        )

    @property
    def label_order(self) -> Tuple[int, ...]:
        """Return the configured classifier-chain label order."""
        return self.chain_head.label_order

    def forward(
        self,
        x: Tensor,
        targets: Optional[Tensor] = None,
        teacher_forcing: Optional[bool] = None,
    ) -> Dict[str, Tensor]:
        """
        Run the backbone and classifier chain.

        Parameters
        ----------
        x:
            ECG tensor with expected shape [B, 12, 1000].

        targets:
            Optional ground-truth labels with shape [B, 5].

        teacher_forcing:
            When None, teacher forcing is enabled only when the model is in
            training mode and targets are supplied.

        Returns
        -------
        dict
            {
                "logits": [B, 5],
                "features": [B, feature_dim],
                "chain_logits": [B, 5],
                "chain_probabilities": [B, 5]
            }
        """
        self._validate_ecg_input(x)

        if teacher_forcing is None:
            teacher_forcing = self.training and targets is not None

        backbone_output = self._run_backbone(x)
        feature_tensor = self._select_feature_tensor(backbone_output)
        pooled_features = self._pool_features(feature_tensor)

        head_outputs = self.chain_head(
            features=pooled_features,
            targets=targets,
            teacher_forcing=teacher_forcing,
        )

        return {
            "logits": head_outputs["logits"],
            "features": pooled_features,
            "chain_logits": head_outputs["chain_logits"],
            "chain_probabilities": head_outputs["chain_probabilities"],
        }

    def _run_backbone(self, x: Tensor) -> Any:
        """
        Attempt to retrieve feature representations from the backbone.
        """
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)

        if hasattr(self.backbone, "extract_features"):
            return self.backbone.extract_features(x)

        if hasattr(self.backbone, "encode"):
            return self.backbone.encode(x)

        try:
            return self.backbone(x, return_features=True)
        except TypeError:
            return self.backbone(x)

    def _select_feature_tensor(self, output: Any) -> Tensor:
        """
        Extract a suitable feature tensor from a backbone output.
        """
        if isinstance(output, Tensor):
            return output

        if isinstance(output, dict):
            preferred_keys = (
                "features",
                "feature",
                "encoder_features",
                "sequence_features",
                "hidden_states",
                "representation",
                "embedding",
                "pooled_features",
            )

            for key in preferred_keys:
                value = output.get(key)

                if isinstance(value, Tensor):
                    return value

            for value in output.values():
                if isinstance(value, Tensor) and value.ndim in {2, 3}:
                    return value

            raise ValueError(
                "No suitable feature tensor was found in the backbone "
                f"dictionary. Available keys: {list(output.keys())}."
            )

        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, Tensor) and value.ndim in {2, 3}:
                    return value

            raise ValueError(
                "No suitable feature tensor was found in the backbone output."
            )

        raise TypeError(
            "Unsupported backbone output type: "
            f"{type(output).__name__}."
        )

    def _pool_features(self, features: Tensor) -> Tensor:
        """
        Convert backbone features to shape [B, feature_dim].
        """
        if features.ndim == 2:
            pooled = features

        elif features.ndim == 3:
            pooled = self._pool_temporal_features(features)

        else:
            raise ValueError(
                "Backbone features must have shape [B, D], [B, T, D], "
                f"or [B, D, T]. Received {tuple(features.shape)}."
            )

        if pooled.size(1) != self.feature_dim:
            raise ValueError(
                "The pooled backbone representation has an unexpected "
                f"dimension. Expected {self.feature_dim}, received "
                f"{pooled.size(1)}. Check feature_dim and temporal_layout."
            )

        return pooled

    def _pool_temporal_features(self, features: Tensor) -> Tensor:
        if self.temporal_layout == "BTD":
            return features.mean(dim=1)

        if self.temporal_layout == "BDT":
            return features.mean(dim=2)

        # Automatic layout inference.
        second_dimension = features.size(1)
        third_dimension = features.size(2)

        if third_dimension == self.feature_dim:
            # [B, T, D]
            return features.mean(dim=1)

        if second_dimension == self.feature_dim:
            # [B, D, T]
            return features.mean(dim=2)

        raise ValueError(
            "Unable to infer the temporal feature layout automatically. "
            f"Received shape {tuple(features.shape)} with feature_dim="
            f"{self.feature_dim}. Set temporal_layout explicitly to 'BTD' "
            "or 'BDT'."
        )

    @staticmethod
    def _validate_ecg_input(x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "ECG input must have shape [B, 12, 1000]. "
                f"Received {tuple(x.shape)}."
            )

        if x.size(1) != 12:
            raise ValueError(
                "Expected 12 ECG leads in dimension 1. "
                f"Received {x.size(1)} leads."
            )


def count_trainable_parameters(model: nn.Module) -> int:
    """
    Count trainable parameters in a PyTorch model.
    """
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )