"""
Statistical validation utilities for paired multi-label experiments.

The functions in this module support:

1. Aggregating repeated-seed metrics.
2. Computing paired seed-wise differences.
3. Paired t-tests and Wilcoxon signed-rank tests.
4. Paired effect-size calculation.
5. Holm correction for multiple comparisons.
6. Paired bootstrap confidence intervals for macro F1.

All paired comparisons assume that predictions correspond to the
same test samples in the same order.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
from sklearn.metrics import f1_score


def _as_1d_float_array(
    values: Sequence[float] | np.ndarray,
    name: str,
) -> np.ndarray:
    """Convert input to a finite one-dimensional float array."""
    array = np.asarray(values, dtype=float)

    if array.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional. "
            f"Received shape {array.shape}."
        )

    if array.size < 2:
        raise ValueError(
            f"{name} must contain at least two values."
        )

    if not np.all(np.isfinite(array)):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    return array


def validate_prediction_pair(
    y_true_a: np.ndarray,
    y_prob_a: np.ndarray,
    y_true_b: np.ndarray,
    y_prob_b: np.ndarray,
) -> None:
    """
    Validate two paired multi-label prediction sets.

    Each label/probability matrix must have shape:

        (num_samples, num_classes)

    Ground-truth labels must also be identical between models.
    """
    arrays = {
        "y_true_a": np.asarray(y_true_a),
        "y_prob_a": np.asarray(y_prob_a),
        "y_true_b": np.asarray(y_true_b),
        "y_prob_b": np.asarray(y_prob_b),
    }

    for name, array in arrays.items():
        if array.ndim != 2:
            raise ValueError(
                f"{name} must be two-dimensional. "
                f"Received shape {array.shape}."
            )

    if arrays["y_true_a"].shape != arrays["y_prob_a"].shape:
        raise ValueError(
            "Model A labels and probabilities must have "
            "identical shapes."
        )

    if arrays["y_true_b"].shape != arrays["y_prob_b"].shape:
        raise ValueError(
            "Model B labels and probabilities must have "
            "identical shapes."
        )

    if arrays["y_true_a"].shape != arrays["y_true_b"].shape:
        raise ValueError(
            "Paired models must contain the same number of "
            "samples and classes."
        )

    if not np.array_equal(
        arrays["y_true_a"],
        arrays["y_true_b"],
    ):
        raise ValueError(
            "Ground-truth labels are not identical between "
            "the paired prediction sets. Check test-set ordering."
        )

    if not np.all(np.isfinite(arrays["y_prob_a"])):
        raise ValueError(
            "Model A probabilities contain NaN or infinite values."
        )

    if not np.all(np.isfinite(arrays["y_prob_b"])):
        raise ValueError(
            "Model B probabilities contain NaN or infinite values."
        )


def aggregate_seed_metrics(
    metrics: pd.DataFrame,
    model_column: str = "model",
    metric_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Aggregate repeated-run metrics as mean and sample standard deviation.

    Args:
        metrics:
            Long-form DataFrame containing one row per model and seed.

        model_column:
            Column identifying the model configuration.

        metric_columns:
            Metric columns to aggregate. When omitted, all numeric
            columns except ``seed`` are used.

    Returns:
        Wide DataFrame with columns such as:
            macro_f1_mean
            macro_f1_std
    """
    if model_column not in metrics.columns:
        raise ValueError(
            f"Missing model column: {model_column}"
        )

    dataframe = metrics.copy()

    if metric_columns is None:
        metric_columns = [
            column
            for column in dataframe.select_dtypes(
                include=[np.number]
            ).columns
            if column != "seed"
        ]
    else:
        metric_columns = list(metric_columns)

    if not metric_columns:
        raise ValueError(
            "No numeric metric columns were supplied."
        )

    missing = [
        column
        for column in metric_columns
        if column not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"Missing metric columns: {missing}"
        )

    grouped = (
        dataframe
        .groupby(model_column, sort=False)[metric_columns]
        .agg(["mean", "std"])
    )

    grouped.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in grouped.columns
    ]

    return grouped.reset_index()


def paired_seed_differences(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
) -> np.ndarray:
    """
    Compute paired differences B - A across repeated seeds.
    """
    array_a = _as_1d_float_array(values_a, "values_a")
    array_b = _as_1d_float_array(values_b, "values_b")

    if array_a.shape != array_b.shape:
        raise ValueError(
            "Paired seed arrays must have identical shapes."
        )

    return array_b - array_a


def paired_effect_size_dz(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
) -> float:
    """
    Compute paired Cohen's dz:

        mean(B - A) / sample_std(B - A)

    Returns 0 when all paired differences are exactly zero.
    Returns positive infinity when the difference is constant and positive,
    and negative infinity when constant and negative.
    """
    differences = paired_seed_differences(
        values_a,
        values_b,
    )

    mean_difference = float(np.mean(differences))
    standard_deviation = float(
        np.std(differences, ddof=1)
    )

    if np.isclose(standard_deviation, 0.0):
        if np.isclose(mean_difference, 0.0):
            return 0.0

        return float(
            np.sign(mean_difference) * np.inf
        )

    return mean_difference / standard_deviation


def paired_seed_tests(
    values_a: Sequence[float] | np.ndarray,
    values_b: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    """
    Perform paired t-test and Wilcoxon signed-rank test.

    The reported mean difference is B - A.
    """
    array_a = _as_1d_float_array(values_a, "values_a")
    array_b = _as_1d_float_array(values_b, "values_b")

    if array_a.shape != array_b.shape:
        raise ValueError(
            "Paired seed arrays must have identical shapes."
        )

    differences = array_b - array_a

    t_result = ttest_rel(
        array_b,
        array_a,
        nan_policy="raise",
    )

    if np.allclose(differences, 0.0):
        wilcoxon_statistic = 0.0
        wilcoxon_pvalue = 1.0
    else:
        wilcoxon_result = wilcoxon(
            array_b,
            array_a,
            alternative="two-sided",
            zero_method="wilcox",
        )

        wilcoxon_statistic = float(
            wilcoxon_result.statistic
        )

        wilcoxon_pvalue = float(
            wilcoxon_result.pvalue
        )

    return {
        "mean_difference": float(
            np.mean(differences)
        ),
        "std_difference": float(
            np.std(differences, ddof=1)
        ),
        "paired_t_statistic": float(
            t_result.statistic
        ),
        "paired_t_pvalue": float(
            t_result.pvalue
        ),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_pvalue": wilcoxon_pvalue,
        "cohens_dz": float(
            paired_effect_size_dz(
                array_a,
                array_b,
            )
        ),
    }


def holm_correction(
    p_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Apply Holm's step-down correction.

    Returns the original and adjusted p-values in their original order.
    """
    values = np.asarray(p_values, dtype=float)

    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "p_values must be a non-empty one-dimensional array."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(
            "p_values contains NaN or infinite values."
        )

    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError(
            "Every p-value must lie within [0, 1]."
        )

    if not 0.0 < alpha < 1.0:
        raise ValueError(
            "alpha must lie strictly between 0 and 1."
        )

    number_of_tests = values.size
    order = np.argsort(values)
    sorted_values = values[order]

    adjusted_sorted = np.empty(
        number_of_tests,
        dtype=float,
    )

    running_maximum = 0.0

    for rank, p_value in enumerate(sorted_values):
        multiplier = number_of_tests - rank
        adjusted = min(
            multiplier * p_value,
            1.0,
        )

        running_maximum = max(
            running_maximum,
            adjusted,
        )

        adjusted_sorted[rank] = running_maximum

    adjusted_original = np.empty_like(
        adjusted_sorted
    )

    adjusted_original[order] = adjusted_sorted

    return pd.DataFrame(
        {
            "p_value": values,
            "holm_adjusted_p": adjusted_original,
            "reject_h0": (
                adjusted_original < alpha
            ),
        }
    )


def macro_f1_from_probabilities(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute macro F1 from multi-label probabilities."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_true.shape != y_prob.shape:
        raise ValueError(
            "y_true and y_prob must have identical shapes."
        )

    y_pred = (
        y_prob >= threshold
    ).astype(np.int64)

    return float(
        f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )
    )


def paired_bootstrap_macro_f1(
    y_true_a: np.ndarray,
    y_prob_a: np.ndarray,
    y_true_b: np.ndarray,
    y_prob_b: np.ndarray,
    *,
    threshold: float = 0.5,
    num_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> dict[str, Any]:
    """
    Compute a paired bootstrap confidence interval for:

        macro_F1(B) - macro_F1(A)

    The same resampled test indices are applied to both models.
    """
    validate_prediction_pair(
        y_true_a,
        y_prob_a,
        y_true_b,
        y_prob_b,
    )

    if num_bootstrap < 100:
        raise ValueError(
            "num_bootstrap must be at least 100."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must lie strictly between 0 and 1."
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must lie within [0, 1]."
        )

    y_true = np.asarray(y_true_a)
    probability_a = np.asarray(y_prob_a)
    probability_b = np.asarray(y_prob_b)

    num_samples = y_true.shape[0]

    observed_a = macro_f1_from_probabilities(
        y_true,
        probability_a,
        threshold,
    )

    observed_b = macro_f1_from_probabilities(
        y_true,
        probability_b,
        threshold,
    )

    observed_difference = observed_b - observed_a

    random_generator = np.random.default_rng(
        random_state
    )

    bootstrap_differences = np.empty(
        num_bootstrap,
        dtype=float,
    )

    for bootstrap_index in range(num_bootstrap):
        sampled_indices = random_generator.integers(
            low=0,
            high=num_samples,
            size=num_samples,
        )

        sampled_labels = y_true[sampled_indices]

        score_a = macro_f1_from_probabilities(
            sampled_labels,
            probability_a[sampled_indices],
            threshold,
        )

        score_b = macro_f1_from_probabilities(
            sampled_labels,
            probability_b[sampled_indices],
            threshold,
        )

        bootstrap_differences[
            bootstrap_index
        ] = score_b - score_a

    alpha = 1.0 - confidence_level

    confidence_interval_low = float(
        np.quantile(
            bootstrap_differences,
            alpha / 2.0,
        )
    )

    confidence_interval_high = float(
        np.quantile(
            bootstrap_differences,
            1.0 - alpha / 2.0,
        )
    )

    two_sided_pvalue = float(
        min(
            1.0,
            2.0
            * min(
                np.mean(
                    bootstrap_differences <= 0.0
                ),
                np.mean(
                    bootstrap_differences >= 0.0
                ),
            ),
        )
    )

    return {
        "macro_f1_a": observed_a,
        "macro_f1_b": observed_b,
        "difference_b_minus_a": observed_difference,
        "ci_low": confidence_interval_low,
        "ci_high": confidence_interval_high,
        "confidence_level": confidence_level,
        "bootstrap_pvalue": two_sided_pvalue,
        "num_bootstrap": int(num_bootstrap),
        "bootstrap_differences": bootstrap_differences,
    }