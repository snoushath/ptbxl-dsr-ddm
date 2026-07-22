"""
ML-GCN dependency baseline for multi-label ECG classification.

This module implements a controlled ML-GCN-style baseline in which:

1. A shared ECG backbone extracts a patient-specific feature vector.
2. A graph convolutional network propagates information among diagnosis nodes.
3. The graph-refined diagnosis embeddings become class-specific classifier
   weights.
4. ECG logits are obtained by comparing the ECG feature vector with each
   diagnosis-specific classifier weight.

The label graph must be constructed using labels from the training split only.

Expected ECG input shape
------------------------
[B, 12, 1000]

Expected output shape
---------------------
[B, num_classes]

For the PTB-XL diagnostic superclass experiment:

    num_classes = 5
    classes = [NORM, MI, STTC, CD, HYP]
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Graph-construction utilities
# ---------------------------------------------------------------------------


def _as_float_tensor(
    value: Tensor | np.ndarray | Sequence[Sequence[float]],
) -> Tensor:
    """
    Convert an input matrix to a detached float32 tensor.
    """
    if isinstance(value, Tensor):
        return value.detach().clone().to(dtype=torch.float32)

    return torch.as_tensor(
        np.asarray(value),
        dtype=torch.float32,
    )


def build_label_cooccurrence_matrix(
    labels: Tensor | np.ndarray | Sequence[Sequence[float]],
    *,
    binary_threshold: float = 0.5,
    include_diagonal: bool = True,
) -> Tensor:
    """
    Construct a symmetric label co-occurrence count matrix.

    Parameters
    ----------
    labels:
        Training-label matrix with shape [N, C], where N is the number of
        training samples and C is the number of classes.

        Only training labels should be passed to this function. The function
        cannot determine whether labels came from training, validation, or
        test data, so preventing leakage is the caller's responsibility.

    binary_threshold:
        Label values greater than or equal to this value are treated as
        positive.

    include_diagonal:
        If True, diagonal entries contain the number of positive examples for
        each class. If False, the diagonal is set to zero.

    Returns
    -------
    Tensor
        Symmetric co-occurrence count matrix with shape [C, C].

    Notes
    -----
    For binary label matrix Y, the co-occurrence matrix is:

        M = Y^T Y

    Therefore:

        M[i, j]

    equals the number of training samples in which labels i and j are both
    positive.
    """
    labels_tensor = _as_float_tensor(labels)

    if labels_tensor.ndim != 2:
        raise ValueError(
            "labels must have shape [num_samples, num_classes]. "
            f"Received shape {tuple(labels_tensor.shape)}."
        )

    num_samples, num_classes = labels_tensor.shape

    if num_samples < 1:
        raise ValueError(
            "labels must contain at least one training sample."
        )

    if num_classes < 2:
        raise ValueError(
            "labels must contain at least two classes."
        )

    if not torch.isfinite(labels_tensor).all():
        raise ValueError(
            "labels contains NaN or infinite values."
        )

    if not 0.0 <= binary_threshold <= 1.0:
        raise ValueError(
            "binary_threshold must lie in [0, 1]."
        )

    binary_labels = (
        labels_tensor >= binary_threshold
    ).to(dtype=torch.float32)

    cooccurrence = binary_labels.transpose(0, 1) @ binary_labels

    if not include_diagonal:
        cooccurrence.fill_diagonal_(0.0)

    return cooccurrence


def build_conditional_adjacency_matrix(
    cooccurrence_matrix: Tensor | np.ndarray,
    *,
    threshold: float = 0.0,
    top_k: Optional[int] = None,
    include_self_loops: bool = True,
    symmetrize: bool = True,
    eps: float = 1e-8,
) -> Tensor:
    """
    Convert co-occurrence counts into a label-dependency adjacency matrix.

    For each source label i and target label j, the directed conditional
    relation is estimated as:

        P(j | i) = count(i and j) / count(i)

    where count(i) is obtained from the diagonal of the co-occurrence matrix.

    Parameters
    ----------
    cooccurrence_matrix:
        Square co-occurrence count matrix [C, C].

    threshold:
        Conditional probabilities below this value are removed.

    top_k:
        Optional maximum number of non-self neighbours retained per row.
        If None, all edges surviving the threshold are retained.

    include_self_loops:
        Whether to set diagonal entries to one.

    symmetrize:
        ML-GCN-style conditional relations are directed. For a standard
        symmetric GCN normalization, this implementation optionally converts
        the matrix to an undirected graph by averaging both directions.

    eps:
        Numerical stability constant.

    Returns
    -------
    Tensor
        Adjacency matrix with shape [C, C].
    """
    cooccurrence = _as_float_tensor(cooccurrence_matrix)

    if cooccurrence.ndim != 2:
        raise ValueError(
            "cooccurrence_matrix must be two-dimensional."
        )

    if cooccurrence.shape[0] != cooccurrence.shape[1]:
        raise ValueError(
            "cooccurrence_matrix must be square. "
            f"Received shape {tuple(cooccurrence.shape)}."
        )

    if cooccurrence.shape[0] < 2:
        raise ValueError(
            "cooccurrence_matrix must contain at least two classes."
        )

    if not torch.isfinite(cooccurrence).all():
        raise ValueError(
            "cooccurrence_matrix contains NaN or infinite values."
        )

    if torch.any(cooccurrence < 0):
        raise ValueError(
            "cooccurrence_matrix cannot contain negative counts."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must lie in [0, 1]."
        )

    num_classes = cooccurrence.shape[0]

    if top_k is not None:
        if not isinstance(top_k, int):
            raise TypeError("top_k must be an integer or None.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        if top_k > num_classes - 1:
            raise ValueError(
                "top_k cannot exceed num_classes - 1."
            )

    positive_counts = torch.diagonal(cooccurrence)

    if torch.any(positive_counts <= 0):
        missing_indices = (
            torch.where(positive_counts <= 0)[0]
            .cpu()
            .tolist()
        )

        raise ValueError(
            "Every class must have at least one positive training example. "
            f"Classes with zero positives: {missing_indices}."
        )

    conditional = cooccurrence / positive_counts.unsqueeze(1).clamp_min(eps)

    conditional = conditional.clamp(min=0.0, max=1.0)

    # Remove weak relations.
    conditional = torch.where(
        conditional >= threshold,
        conditional,
        torch.zeros_like(conditional),
    )

    # The diagonal is controlled explicitly below.
    conditional.fill_diagonal_(0.0)

    if top_k is not None:
        retained = torch.zeros_like(conditional)

        top_values, top_indices = torch.topk(
            conditional,
            k=top_k,
            dim=1,
        )

        retained.scatter_(
            dim=1,
            index=top_indices,
            src=top_values,
        )

        conditional = retained

    if symmetrize:
        adjacency = 0.5 * (
            conditional + conditional.transpose(0, 1)
        )
    else:
        adjacency = conditional

    if include_self_loops:
        adjacency.fill_diagonal_(1.0)
    else:
        adjacency.fill_diagonal_(0.0)

    return adjacency


def normalize_adjacency_matrix(
    adjacency_matrix: Tensor | np.ndarray,
    *,
    add_self_loops: bool = False,
    eps: float = 1e-8,
) -> Tensor:
    """
    Symmetrically normalize an adjacency matrix.

    The normalized matrix is:

        A_hat = D^(-1/2) A D^(-1/2)

    Parameters
    ----------
    adjacency_matrix:
        Square adjacency matrix [C, C].

    add_self_loops:
        If True, an identity matrix is added before normalization. Set this to
        False when self-loops have already been included during graph
        construction.

    eps:
        Numerical stability constant.

    Returns
    -------
    Tensor
        Symmetrically normalized adjacency matrix [C, C].
    """
    adjacency = _as_float_tensor(adjacency_matrix)

    if adjacency.ndim != 2:
        raise ValueError(
            "adjacency_matrix must be two-dimensional."
        )

    if adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError(
            "adjacency_matrix must be square. "
            f"Received shape {tuple(adjacency.shape)}."
        )

    if adjacency.shape[0] < 2:
        raise ValueError(
            "adjacency_matrix must contain at least two nodes."
        )

    if not torch.isfinite(adjacency).all():
        raise ValueError(
            "adjacency_matrix contains NaN or infinite values."
        )

    if torch.any(adjacency < 0):
        raise ValueError(
            "adjacency_matrix cannot contain negative values."
        )

    if add_self_loops:
        adjacency = adjacency + torch.eye(
            adjacency.shape[0],
            dtype=adjacency.dtype,
            device=adjacency.device,
        )

    degree = adjacency.sum(dim=1)

    if torch.any(degree <= 0):
        zero_degree_nodes = (
            torch.where(degree <= 0)[0]
            .cpu()
            .tolist()
        )

        raise ValueError(
            "Every graph node must have positive degree. "
            f"Zero-degree nodes: {zero_degree_nodes}."
        )

    inverse_sqrt_degree = torch.rsqrt(
        degree.clamp_min(eps)
    )

    normalized = (
        inverse_sqrt_degree.unsqueeze(1)
        * adjacency
        * inverse_sqrt_degree.unsqueeze(0)
    )

    return normalized


def build_normalized_label_graph(
    labels: Tensor | np.ndarray | Sequence[Sequence[float]],
    *,
    binary_threshold: float = 0.5,
    edge_threshold: float = 0.0,
    top_k: Optional[int] = None,
    include_self_loops: bool = True,
    symmetrize: bool = True,
) -> Dict[str, Tensor]:
    """
    Convenience function that creates all graph matrices from training labels.

    Returns a dictionary containing:

    - ``cooccurrence_matrix``
    - ``adjacency_matrix``
    - ``normalized_adjacency``
    """
    cooccurrence = build_label_cooccurrence_matrix(
        labels=labels,
        binary_threshold=binary_threshold,
        include_diagonal=True,
    )

    adjacency = build_conditional_adjacency_matrix(
        cooccurrence_matrix=cooccurrence,
        threshold=edge_threshold,
        top_k=top_k,
        include_self_loops=include_self_loops,
        symmetrize=symmetrize,
    )

    normalized = normalize_adjacency_matrix(
        adjacency_matrix=adjacency,
        add_self_loops=False,
    )

    return {
        "cooccurrence_matrix": cooccurrence,
        "adjacency_matrix": adjacency,
        "normalized_adjacency": normalized,
    }


# ---------------------------------------------------------------------------
# Graph neural network layers
# ---------------------------------------------------------------------------


class GraphConvolution(nn.Module):
    """
    Basic graph-convolution layer.

    Given node features X and normalized adjacency A, this layer computes:

        H = A X W + b

    Shapes
    ------
    node_features:
        [C, input_dim]

    normalized_adjacency:
        [C, C]

    output:
        [C, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,
    ) -> None:
        super().__init__()

        if input_dim < 1:
            raise ValueError("input_dim must be positive.")

        if output_dim < 1:
            raise ValueError("output_dim must be positive.")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)

        self.weight = nn.Parameter(
            torch.empty(self.input_dim, self.output_dim)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(self.output_dim)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
        self,
        node_features: Tensor,
        normalized_adjacency: Tensor,
    ) -> Tensor:
        if node_features.ndim != 2:
            raise ValueError(
                "node_features must have shape [num_nodes, input_dim]. "
                f"Received {tuple(node_features.shape)}."
            )

        if node_features.shape[1] != self.input_dim:
            raise ValueError(
                "Unexpected node feature dimension. "
                f"Expected {self.input_dim}, "
                f"received {node_features.shape[1]}."
            )

        if normalized_adjacency.ndim != 2:
            raise ValueError(
                "normalized_adjacency must be two-dimensional."
            )

        num_nodes = node_features.shape[0]

        if normalized_adjacency.shape != (
            num_nodes,
            num_nodes,
        ):
            raise ValueError(
                "normalized_adjacency must have shape "
                f"[{num_nodes}, {num_nodes}]. "
                f"Received {tuple(normalized_adjacency.shape)}."
            )

        if not torch.isfinite(node_features).all():
            raise ValueError(
                "node_features contains NaN or infinite values."
            )

        if not torch.isfinite(normalized_adjacency).all():
            raise ValueError(
                "normalized_adjacency contains NaN or infinite values."
            )

        support = node_features @ self.weight
        output = normalized_adjacency @ support

        if self.bias is not None:
            output = output + self.bias

        return output


class MLGCNHead(nn.Module):
    """
    ML-GCN classification head.

    Learnable label embeddings are propagated through graph-convolution layers.
    The resulting label representations are projected into the same space as
    the ECG feature vector and act as class-specific classifier weights.

    Parameters
    ----------
    feature_dim:
        Dimension of the pooled ECG representation.

    num_classes:
        Number of diagnosis labels.

    label_embedding_dim:
        Initial dimension of each learnable label embedding.

    gcn_hidden_dim:
        Hidden dimension of the first graph-convolution layer.

    dropout:
        Dropout probability applied between GCN layers.

    use_class_bias:
        Whether to learn one scalar bias per diagnosis.
    """

    def __init__(
        self,
        *,
        feature_dim: int,
        num_classes: int,
        label_embedding_dim: int = 128,
        gcn_hidden_dim: int = 256,
        dropout: float = 0.3,
        use_class_bias: bool = True,
        normalize_classifier_weights: bool = False,
    ) -> None:
        super().__init__()

        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2."
            )

        if label_embedding_dim < 1:
            raise ValueError(
                "label_embedding_dim must be positive."
            )

        if gcn_hidden_dim < 1:
            raise ValueError(
                "gcn_hidden_dim must be positive."
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must lie in [0, 1)."
            )

        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.label_embedding_dim = int(label_embedding_dim)
        self.gcn_hidden_dim = int(gcn_hidden_dim)
        self.normalize_classifier_weights = bool(
            normalize_classifier_weights
        )

        self.label_embeddings = nn.Parameter(
            torch.empty(
                self.num_classes,
                self.label_embedding_dim,
            )
        )

        self.gcn1 = GraphConvolution(
            input_dim=self.label_embedding_dim,
            output_dim=self.gcn_hidden_dim,
        )

        self.gcn2 = GraphConvolution(
            input_dim=self.gcn_hidden_dim,
            output_dim=self.feature_dim,
        )

        self.dropout = nn.Dropout(dropout)

        if use_class_bias:
            self.class_bias = nn.Parameter(
                torch.zeros(self.num_classes)
            )
        else:
            self.register_parameter(
                "class_bias",
                None,
            )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(
            self.label_embeddings
        )

        self.gcn1.reset_parameters()
        self.gcn2.reset_parameters()

        if self.class_bias is not None:
            nn.init.zeros_(self.class_bias)

    def generate_classifier_weights(
        self,
        normalized_adjacency: Tensor,
    ) -> Tensor:
        """
        Generate graph-refined class-specific classifier weights.

        Returns
        -------
        Tensor
            Classifier weight matrix [num_classes, feature_dim].
        """
        if normalized_adjacency.shape != (
            self.num_classes,
            self.num_classes,
        ):
            raise ValueError(
                "normalized_adjacency has an incompatible shape. "
                f"Expected {(self.num_classes, self.num_classes)}, "
                f"received {tuple(normalized_adjacency.shape)}."
            )

        label_features = self.gcn1(
            self.label_embeddings,
            normalized_adjacency,
        )

        label_features = F.leaky_relu(
            label_features,
            negative_slope=0.2,
        )

        label_features = self.dropout(
            label_features
        )

        classifier_weights = self.gcn2(
            label_features,
            normalized_adjacency,
        )

        if self.normalize_classifier_weights:
            classifier_weights = F.normalize(
                classifier_weights,
                p=2,
                dim=-1,
            )

        return classifier_weights

    def forward(
        self,
        ecg_features: Tensor,
        normalized_adjacency: Tensor,
    ) -> Dict[str, Tensor]:
        if ecg_features.ndim != 2:
            raise ValueError(
                "ecg_features must have shape [batch_size, feature_dim]. "
                f"Received {tuple(ecg_features.shape)}."
            )

        if ecg_features.shape[1] != self.feature_dim:
            raise ValueError(
                "Unexpected ECG feature dimension. "
                f"Expected {self.feature_dim}, "
                f"received {ecg_features.shape[1]}."
            )

        classifier_weights = (
            self.generate_classifier_weights(
                normalized_adjacency
            )
        )

        if self.normalize_classifier_weights:
            classification_features = F.normalize(
                ecg_features,
                p=2,
                dim=-1,
            )
        else:
            classification_features = ecg_features

        logits = (
            classification_features
            @ classifier_weights.transpose(0, 1)
        )

        if self.class_bias is not None:
            logits = logits + self.class_bias

        return {
            "logits": logits,
            "classifier_weights": classifier_weights,
            "label_embeddings": self.label_embeddings,
        }


# ---------------------------------------------------------------------------
# Complete ECG model
# ---------------------------------------------------------------------------


class MLGCNGatedSequence(nn.Module):
    """
    ML-GCN model using an externally supplied ECG feature backbone.

    The wrapper is intentionally backbone-agnostic. The supplied backbone
    should expose one of the following interfaces:

    - ``forward_features(x)``
    - ``extract_features(x)``
    - ``encode(x)``
    - ``forward(x)``

    The backbone may return:

    - pooled features [B, D];
    - temporal features [B, T, D];
    - temporal features [B, D, T];
    - a dictionary containing a feature tensor;
    - a tuple whose first suitable tensor is a feature representation.

    Parameters
    ----------
    backbone:
        ECG encoder.

    normalized_adjacency:
        Fixed normalized label graph [C, C]. It should be derived from
        training labels only.

    feature_dim:
        Dimension of the pooled ECG representation.

    num_classes:
        Number of labels.

    temporal_layout:
        One of ``"auto"``, ``"BTD"``, or ``"BDT"``.
    """

    _FEATURE_KEYS: Tuple[str, ...] = (
        "features",
        "pooled_features",
        "sequence_features",
        "temporal_features",
        "embeddings",
        "hidden_states",
        "encoder_output",
        "representation",
    )

    def __init__(
        self,
        *,
        backbone: nn.Module,
        normalized_adjacency: Tensor | np.ndarray,
        feature_dim: int,
        num_classes: int = 5,
        label_embedding_dim: int = 128,
        gcn_hidden_dim: int = 256,
        dropout: float = 0.3,
        temporal_layout: str = "auto",
        expected_num_leads: int = 12,
        normalize_classifier_weights: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(backbone, nn.Module):
            raise TypeError(
                "backbone must be an instance of torch.nn.Module."
            )

        if feature_dim < 1:
            raise ValueError("feature_dim must be positive.")

        if num_classes < 2:
            raise ValueError(
                "num_classes must be at least 2."
            )

        if expected_num_leads < 1:
            raise ValueError(
                "expected_num_leads must be positive."
            )

        temporal_layout = temporal_layout.upper()

        if temporal_layout not in {
            "AUTO",
            "BTD",
            "BDT",
        }:
            raise ValueError(
                "temporal_layout must be one of "
                "{'auto', 'BTD', 'BDT'}."
            )

        normalized_adjacency_tensor = _as_float_tensor(
            normalized_adjacency
        )

        self._validate_adjacency(
            normalized_adjacency_tensor,
            num_classes=num_classes,
        )

        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.temporal_layout = temporal_layout
        self.expected_num_leads = int(
            expected_num_leads
        )

        # The graph is fixed for a given experiment and is not optimized.
        self.register_buffer(
            "normalized_adjacency",
            normalized_adjacency_tensor,
            persistent=True,
        )

        self.ml_gcn_head = MLGCNHead(
            feature_dim=self.feature_dim,
            num_classes=self.num_classes,
            label_embedding_dim=label_embedding_dim,
            gcn_hidden_dim=gcn_hidden_dim,
            dropout=dropout,
            normalize_classifier_weights=(
                normalize_classifier_weights
            ),
        )

    @staticmethod
    def _validate_adjacency(
        adjacency: Tensor,
        *,
        num_classes: int,
    ) -> None:
        if adjacency.ndim != 2:
            raise ValueError(
                "normalized_adjacency must be two-dimensional."
            )

        if adjacency.shape != (
            num_classes,
            num_classes,
        ):
            raise ValueError(
                "normalized_adjacency must have shape "
                f"[{num_classes}, {num_classes}]. "
                f"Received {tuple(adjacency.shape)}."
            )

        if not torch.isfinite(adjacency).all():
            raise ValueError(
                "normalized_adjacency contains NaN or infinite values."
            )

        if torch.any(adjacency < 0):
            raise ValueError(
                "normalized_adjacency cannot contain negative values."
            )

        if torch.any(adjacency.sum(dim=1) <= 0):
            raise ValueError(
                "Every node in normalized_adjacency must have "
                "positive degree."
            )

    def _call_backbone(
        self,
        x: Tensor,
    ) -> Any:
        if hasattr(self.backbone, "forward_features"):
            return self.backbone.forward_features(x)

        if hasattr(self.backbone, "extract_features"):
            return self.backbone.extract_features(x)

        if hasattr(self.backbone, "encode"):
            return self.backbone.encode(x)

        return self.backbone(x)

    def _extract_tensor(
        self,
        output: Any,
    ) -> Tensor:
        if isinstance(output, Tensor):
            return output

        if isinstance(output, Mapping):
            for key in self._FEATURE_KEYS:
                value = output.get(key)

                if isinstance(value, Tensor):
                    return value

            for value in output.values():
                if isinstance(value, Tensor) and value.ndim in {
                    2,
                    3,
                }:
                    return value

            raise TypeError(
                "The backbone returned a dictionary without a suitable "
                "feature tensor."
            )

        if isinstance(output, (tuple, list)):
            for value in output:
                if isinstance(value, Tensor) and value.ndim in {
                    2,
                    3,
                }:
                    return value

            raise TypeError(
                "The backbone returned a tuple/list without a suitable "
                "feature tensor."
            )

        raise TypeError(
            "Unsupported backbone output type: "
            f"{type(output).__name__}."
        )

    def _pool_features(
        self,
        features: Tensor,
    ) -> Tensor:
        if features.ndim == 2:
            pooled = features

        elif features.ndim == 3:
            if self.temporal_layout == "BTD":
                if features.shape[-1] != self.feature_dim:
                    raise ValueError(
                        "For temporal_layout='BTD', the final dimension "
                        f"must equal feature_dim={self.feature_dim}. "
                        f"Received shape {tuple(features.shape)}."
                    )

                pooled = features.mean(dim=1)

            elif self.temporal_layout == "BDT":
                if features.shape[1] != self.feature_dim:
                    raise ValueError(
                        "For temporal_layout='BDT', the second dimension "
                        f"must equal feature_dim={self.feature_dim}. "
                        f"Received shape {tuple(features.shape)}."
                    )

                pooled = features.mean(dim=2)

            else:
                second_matches = (
                    features.shape[1] == self.feature_dim
                )
                third_matches = (
                    features.shape[2] == self.feature_dim
                )

                if third_matches and not second_matches:
                    # [B, T, D]
                    pooled = features.mean(dim=1)

                elif second_matches and not third_matches:
                    # [B, D, T]
                    pooled = features.mean(dim=2)

                elif second_matches and third_matches:
                    raise ValueError(
                        "Automatic temporal-layout detection is ambiguous "
                        f"for feature shape {tuple(features.shape)}. "
                        "Specify temporal_layout='BTD' or 'BDT'."
                    )

                else:
                    raise ValueError(
                        "Unable to identify feature dimension in backbone "
                        f"output shape {tuple(features.shape)}. "
                        f"Expected one temporal axis to equal "
                        f"feature_dim={self.feature_dim}."
                    )

        else:
            raise ValueError(
                "Backbone features must have shape [B, D], [B, T, D], "
                f"or [B, D, T]. Received {tuple(features.shape)}."
            )

        if pooled.shape[1] != self.feature_dim:
            raise ValueError(
                "Unexpected pooled feature dimension. "
                f"Expected {self.feature_dim}, "
                f"received {pooled.shape[1]}."
            )

        return pooled

    def extract_features(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Extract and pool ECG features.

        Parameters
        ----------
        x:
            ECG batch [B, 12, 1000].

        Returns
        -------
        Tensor
            Pooled ECG features [B, feature_dim].
        """
        if x.ndim != 3:
            raise ValueError(
                "Expected ECG input with shape "
                "[batch_size, leads, samples]. "
                f"Received {tuple(x.shape)}."
            )

        if x.shape[1] != self.expected_num_leads:
            raise ValueError(
                f"Expected {self.expected_num_leads} ECG leads, "
                f"received {x.shape[1]}."
            )

        if not torch.isfinite(x).all():
            raise ValueError(
                "ECG input contains NaN or infinite values."
            )

        backbone_output = self._call_backbone(x)
        feature_tensor = self._extract_tensor(
            backbone_output
        )

        return self._pool_features(
            feature_tensor
        )

    def forward(
        self,
        x: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Forward pass.

        Returns
        -------
        dict
            ``logits``:
                Raw multi-label logits [B, C].

            ``features``:
                Pooled ECG features [B, D].

            ``classifier_weights``:
                Graph-refined class-specific weights [C, D].

            ``label_embeddings``:
                Initial learnable diagnosis embeddings.

            ``normalized_adjacency``:
                Fixed normalized training-label graph [C, C].
        """
        features = self.extract_features(x)

        head_outputs = self.ml_gcn_head(
            ecg_features=features,
            normalized_adjacency=(
                self.normalized_adjacency
            ),
        )

        return {
            "logits": head_outputs["logits"],
            "features": features,
            "classifier_weights": head_outputs[
                "classifier_weights"
            ],
            "label_embeddings": head_outputs[
                "label_embeddings"
            ],
            "normalized_adjacency": (
                self.normalized_adjacency
            ),
        }


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------


def count_trainable_parameters(
    model: nn.Module,
) -> int:
    """
    Count trainable model parameters.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(
            "model must be an instance of torch.nn.Module."
        )

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


__all__ = [
    "build_label_cooccurrence_matrix",
    "build_conditional_adjacency_matrix",
    "normalize_adjacency_matrix",
    "build_normalized_label_graph",
    "GraphConvolution",
    "MLGCNHead",
    "MLGCNGatedSequence",
    "count_trainable_parameters",
]