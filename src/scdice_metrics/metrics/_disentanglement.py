from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import scipy.spatial as ss
import xgboost as xgb
from fairlearn.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
)
from scipy.special import digamma
from scipy.stats import entropy
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import check_array


def _coerce_factors(factors: pd.DataFrame | Mapping[str, np.ndarray]) -> tuple[list[str], list[np.ndarray]]:
    if isinstance(factors, pd.DataFrame):
        factor_names = list(factors.columns)
        factor_arrays = [factors[col].to_numpy() for col in factor_names]
    elif isinstance(factors, Mapping):
        factor_names = list(factors.keys())
        factor_arrays = [np.asarray(factors[name]) for name in factor_names]
    else:
        raise TypeError("`factors` must be a pandas DataFrame or a mapping of factor names to arrays.")

    if len(factor_arrays) == 0:
        raise ValueError("`factors` must contain at least one factor.")

    n_samples = len(factor_arrays[0])
    if any(len(values) != n_samples for values in factor_arrays):
        raise ValueError("All factors must have the same number of samples.")

    return factor_names, factor_arrays


def _encode_factor(values: np.ndarray) -> np.ndarray:
    encoded, _ = pd.factorize(values, sort=True)
    encoded = encoded.astype(int, copy=False)
    if np.any(encoded < 0):
        raise ValueError("Factor values must not contain missing values.")
    return encoded


def encode_factors(factors: pd.DataFrame | Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Encode categorical factors as integer arrays.

    Parameters
    ----------
    factors
        Factor table or mapping where each column/entry represents a discrete factor.

    Returns
    -------
    Encoded factors keyed by factor name.
    """
    factor_names, factor_arrays = _coerce_factors(factors)
    return {name: _encode_factor(values) for name, values in zip(factor_names, factor_arrays, strict=True)}


def assign_latent_blocks_by_mi(
    latent: np.ndarray,
    factors: pd.DataFrame | Mapping[str, np.ndarray],
    random_state: int = 0,
) -> dict[str, object]:
    """Assign latent dimensions to factors using per-dimension mutual information.

    Each factor is matched to the single latent dimension with the highest mutual
    information score, following the heuristic used in `tmp/MIG_metrics.py`.

    Parameters
    ----------
    latent
        Array of shape `(n_samples, n_latent)`.
    factors
        Factor table or mapping where each column/entry represents a discrete factor.
    random_state
        Random seed forwarded to `mutual_info_classif`.

    Returns
    -------
    A dictionary containing selected dimensions, latent blocks, complement blocks,
    shared latent dimensions and the full factor-by-dimension MI matrix.
    """
    latent = check_array(latent, accept_sparse=False, ensure_2d=True)
    encoded_factors = encode_factors(factors)
    factor_names = list(encoded_factors)
    latent_dims = [latent[:, [idx]] for idx in range(latent.shape[1])]

    mi_matrix = np.zeros((len(factor_names), latent.shape[1]), dtype=float)
    selected_dimension_indices: dict[str, int] = {}
    factor_blocks: dict[str, np.ndarray] = {}
    complement_blocks: dict[str, np.ndarray] = {}

    for row_idx, factor_name in enumerate(factor_names):
        encoded_factor = encoded_factors[factor_name]
        for col_idx, latent_dim in enumerate(latent_dims):
            mi_matrix[row_idx, col_idx] = mutual_info_classif(
                latent_dim,
                encoded_factor,
                discrete_features=False,
                random_state=random_state,
            )[0]

        selected_idx = int(np.argmax(mi_matrix[row_idx]))
        selected_dimension_indices[factor_name] = selected_idx
        factor_blocks[factor_name] = latent_dims[selected_idx]
        complement_blocks[factor_name] = np.concatenate(
            [latent_dims[idx] for idx in range(latent.shape[1]) if idx != selected_idx],
            axis=1,
        )

    selected_indices = set(selected_dimension_indices.values())
    shared_indices = [idx for idx in range(latent.shape[1]) if idx not in selected_indices]
    shared_block = latent[:, shared_indices] if shared_indices else np.empty((latent.shape[0], 0))

    return {
        "factor_names": factor_names,
        "encoded_factors": encoded_factors,
        "mi_matrix": mi_matrix,
        "selected_dimension_indices": selected_dimension_indices,
        "factor_blocks": factor_blocks,
        "complement_blocks": complement_blocks,
        "shared_indices": shared_indices,
        "shared_block": shared_block,
    }


def mixed_ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Estimate mutual information with the Mixed-KSG estimator.

    Parameters
    ----------
    x
        Array of shape `(n_samples, n_features_x)`.
    y
        Array of shape `(n_samples,)` or `(n_samples, n_features_y)`.
    k
        Number of neighbors to use in the estimator.

    Returns
    -------
    Estimated mutual information.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if len(x) != len(y):
        raise ValueError("`x` and `y` must contain the same number of samples.")
    if k > len(x) - 1:
        raise ValueError("`k` must be smaller than the number of samples.")

    n_samples = len(x)
    if x.ndim == 1:
        x = x.reshape((n_samples, 1))
    if y.ndim == 1:
        y = y.reshape((n_samples, 1))

    data = np.concatenate((x, y), axis=1)
    tree_xy = ss.cKDTree(data)
    tree_x = ss.cKDTree(x)
    tree_y = ss.cKDTree(y)

    knn_distances = [tree_xy.query(point, k + 1, p=float("inf"))[0][k] for point in data]
    estimate = 0.0

    for idx in range(n_samples):
        local_k = k
        if knn_distances[idx] == 0:
            local_k = len(tree_xy.query_ball_point(data[idx], 1e-15, p=float("inf")))
            n_x = len(tree_x.query_ball_point(x[idx], 1e-15, p=float("inf")))
            n_y = len(tree_y.query_ball_point(y[idx], 1e-15, p=float("inf")))
        else:
            n_x = len(tree_x.query_ball_point(x[idx], knn_distances[idx] - 1e-15, p=float("inf")))
            n_y = len(tree_y.query_ball_point(y[idx], knn_distances[idx] - 1e-15, p=float("inf")))
        estimate += (digamma(local_k) + np.log(n_samples) - digamma(n_x) - digamma(n_y)) / n_samples

    return float(estimate)


def mig(
    latent: np.ndarray,
    factors: pd.DataFrame | Mapping[str, np.ndarray],
    random_state: int = 0,
) -> dict[str, float]:
    """Compute the mutual information gap (MIG) using max-per-dimension MI.

    Parameters
    ----------
    latent
        Array of shape `(n_samples, n_latent)`.
    factors
        Factor table or mapping where each column/entry represents a discrete factor.
    random_state
        Random seed forwarded to the MI estimator.

    Returns
    -------
    A dictionary with the MIG score and mean factor/complement MI summaries.
    """
    assignments = assign_latent_blocks_by_mi(latent, factors, random_state=random_state)

    factor_mi = []
    complement_mi = []
    factor_entropies = []

    for factor_name in assignments["factor_names"]:
        encoded_factor = assignments["encoded_factors"][factor_name]
        factor_block = assignments["factor_blocks"][factor_name]
        complement_block = assignments["complement_blocks"][factor_name]

        factor_score = mutual_info_classif(
            factor_block,
            encoded_factor,
            discrete_features=False,
            random_state=random_state,
        )[0]
        complement_scores = mutual_info_classif(
            complement_block,
            encoded_factor,
            discrete_features=False,
            random_state=random_state,
        )

        factor_mi.append(float(factor_score))
        complement_mi.append(float(np.max(complement_scores)))
        _, counts = np.unique(encoded_factor, return_counts=True)
        factor_entropies.append(float(entropy(counts)))

    mig_scores = [(mi - mi_not) / h for mi, mi_not, h in zip(factor_mi, complement_mi, factor_entropies, strict=True)]
    return {
        "score": float(np.mean(mig_scores)),
        "mean_factor_mi": float(np.mean(factor_mi)),
        "mean_complement_mi": float(np.mean(complement_mi)),
    }


def mixed_ksg_mig(
    latent: np.ndarray,
    factors: pd.DataFrame | Mapping[str, np.ndarray],
    k: int = 5,
    random_state: int = 0,
) -> dict[str, float]:
    """Compute Mixed-KSG variants of the MIG score.

    Parameters
    ----------
    latent
        Array of shape `(n_samples, n_latent)`.
    factors
        Factor table or mapping where each column/entry represents a discrete factor.
    k
        Number of neighbors for the Mixed-KSG estimator.
    random_state
        Random seed used during latent block assignment.

    Returns
    -------
    A dictionary with the `max_mig`, `concat_mig`, and `min_mig` summaries.
    """
    assignments = assign_latent_blocks_by_mi(latent, factors, random_state=random_state)
    factor_latents = assignments["factor_blocks"]
    factor_names = assignments["factor_names"]
    encoded_factors = assignments["encoded_factors"]

    factor_mi = []
    complement_mi = []
    max_other_mi = []
    min_other_mi = []
    factor_entropies = []

    for factor_name in factor_names:
        encoded_factor = encoded_factors[factor_name]
        factor_score = mixed_ksg_mi(factor_latents[factor_name], encoded_factor, k=k)
        complement_score = mixed_ksg_mi(assignments["complement_blocks"][factor_name], encoded_factor, k=k)
        competing_scores = [
            mixed_ksg_mi(factor_latents[other_name], encoded_factor, k=k)
            for other_name in factor_names
            if other_name != factor_name
        ]
        _, counts = np.unique(encoded_factor, return_counts=True)

        factor_mi.append(float(factor_score))
        complement_mi.append(float(complement_score))
        max_other_mi.append(float(np.max(competing_scores)))
        min_other_mi.append(float(np.min(competing_scores)))
        factor_entropies.append(float(entropy(counts)))

    max_mig_scores = [(mi - other) / h for mi, other, h in zip(factor_mi, max_other_mi, factor_entropies, strict=True)]
    concat_mig_scores = [
        (mi - other) / h for mi, other, h in zip(factor_mi, complement_mi, factor_entropies, strict=True)
    ]
    min_mig_scores = [(mi - other) / h for mi, other, h in zip(factor_mi, min_other_mi, factor_entropies, strict=True)]

    return {
        "max_mig": float(np.mean(max_mig_scores)),
        "concat_mig": float(np.mean(concat_mig_scores)),
        "min_mig": float(np.mean(min_mig_scores)),
    }


def _default_classifier(random_state: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        tree_method="hist",
        n_estimators=64,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=random_state,
    )


def _resolve_n_splits(y: np.ndarray, requested_splits: int) -> int:
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(requested_splits, int(np.min(counts)))
    if n_splits < 2:
        raise ValueError("At least two samples per class are required for stratified cross-validation.")
    return n_splits


def classifier_attribute_gap(
    latent: np.ndarray,
    factors: pd.DataFrame | Mapping[str, np.ndarray],
    cv_splits: int = 5,
    random_state: int = 94,
    classifier: xgb.XGBClassifier | None = None,
) -> dict[str, float]:
    """Compute classifier-based attribute gap metrics.

    Parameters
    ----------
    latent
        Array of shape `(n_samples, n_latent)`.
    factors
        Factor table or mapping where each column/entry represents a discrete factor.
    cv_splits
        Number of stratified cross-validation folds.
    random_state
        Random seed used for latent block assignment and classifier training.
    classifier
        Optional classifier prototype. If `None`, an `xgboost.XGBClassifier` is used.

    Returns
    -------
    A dictionary summarizing the average attribute gap and accuracies.
    """
    assignments = assign_latent_blocks_by_mi(latent, factors, random_state=random_state)
    encoded_factors = assignments["encoded_factors"]
    factor_latents = assignments["factor_blocks"]
    factor_names = assignments["factor_names"]
    full_blocks = [assignments["shared_block"]] + [factor_latents[name] for name in factor_names]
    classifier = classifier or _default_classifier(random_state)

    factor_accuracy = []
    complement_accuracy = []
    best_nonmatching_accuracy = []
    concat_gap = []
    max_gap = []

    for factor_name in factor_names:
        y = encoded_factors[factor_name]
        splitter = StratifiedKFold(
            n_splits=_resolve_n_splits(y, cv_splits),
            shuffle=True,
            random_state=random_state,
        )

        matched_scores = []
        complement_scores = []
        competitor_scores = []

        for train_idx, test_idx in splitter.split(factor_latents[factor_name], y):
            block_scores = []
            for block in full_blocks:
                estimator = clone(classifier)
                estimator.fit(block[train_idx], y[train_idx])
                block_scores.append(float(estimator.score(block[test_idx], y[test_idx])))

            matched_idx = int(np.argmax(block_scores))
            matched_scores.append(block_scores[matched_idx])
            competitor_scores.append(max(score for idx, score in enumerate(block_scores) if idx != matched_idx))

            complement = np.concatenate(
                [block for idx, block in enumerate(full_blocks) if idx != matched_idx],
                axis=1,
            )
            estimator = clone(classifier)
            estimator.fit(complement[train_idx], y[train_idx])
            complement_scores.append(float(estimator.score(complement[test_idx], y[test_idx])))

        mean_matched = float(np.mean(matched_scores))
        mean_complement = float(np.mean(complement_scores))
        mean_competitor = float(np.mean(competitor_scores))

        factor_accuracy.append(mean_matched)
        complement_accuracy.append(mean_complement)
        best_nonmatching_accuracy.append(mean_competitor)
        concat_gap.append(mean_matched - mean_complement)
        max_gap.append(mean_matched - mean_competitor)

    return {
        "concat_gap": float(np.mean(concat_gap)),
        "max_gap": float(np.mean(max_gap)),
        "mean_accuracy": float(np.mean(factor_accuracy)),
        "mean_complement_accuracy": float(np.mean(complement_accuracy)),
        "mean_competitor_accuracy": float(np.mean(best_nonmatching_accuracy)),
    }


def _binarize_target(target: np.ndarray) -> np.ndarray:
    target = np.asarray(target)
    unique_values = np.unique(target)
    if len(unique_values) == 2:
        return _encode_factor(target)
    if np.issubdtype(target.dtype, np.number):
        threshold = np.median(target)
        return np.where(target < threshold, 0, 1).astype(int, copy=False)
    raise ValueError("`target` must be binary or numeric so it can be binarized.")


def fairness_leakage(
    latent: np.ndarray,
    factors: pd.DataFrame | Mapping[str, np.ndarray],
    target: np.ndarray,
    cv_splits: int = 5,
    random_state: int = 94,
    classifier: xgb.XGBClassifier | None = None,
) -> dict[str, float]:
    """Measure factor leakage from complement latents using fairness metrics.

    Parameters
    ----------
    latent
        Array of shape `(n_samples, n_latent)`.
    factors
        Factor table or mapping where each column/entry represents a discrete factor.
    target
        Binary target array or numeric array that can be binarized by median split.
    cv_splits
        Number of stratified cross-validation folds.
    random_state
        Random seed used for latent block assignment and classifier training.
    classifier
        Optional classifier prototype. If `None`, an `xgboost.XGBClassifier` is used.

    Returns
    -------
    A dictionary summarizing prediction accuracy and fairness leakage scores.
    """
    assignments = assign_latent_blocks_by_mi(latent, factors, random_state=random_state)
    encoded_target = _binarize_target(target)
    classifier = classifier or _default_classifier(random_state)

    dp_differences = []
    dp_ratios = []
    eo_differences = []
    accuracies = []

    for factor_name in assignments["factor_names"]:
        sensitive_feature = assignments["encoded_factors"][factor_name]
        complement = assignments["complement_blocks"][factor_name]
        splitter = StratifiedKFold(
            n_splits=_resolve_n_splits(encoded_target, cv_splits),
            shuffle=True,
            random_state=random_state,
        )

        factor_dp_diff = []
        factor_dp_ratio = []
        factor_eo_diff = []
        factor_acc = []

        for train_idx, test_idx in splitter.split(complement, encoded_target):
            estimator = clone(classifier)
            estimator.fit(complement[train_idx], encoded_target[train_idx])
            predictions = estimator.predict(complement[test_idx])

            y_true = encoded_target[test_idx]
            sensitive_test = sensitive_feature[test_idx]
            factor_dp_diff.append(float(demographic_parity_difference(y_true, predictions, sensitive_features=sensitive_test)))
            factor_dp_ratio.append(float(demographic_parity_ratio(y_true, predictions, sensitive_features=sensitive_test)))
            factor_eo_diff.append(float(equalized_odds_difference(y_true, predictions, sensitive_features=sensitive_test)))
            factor_acc.append(float(estimator.score(complement[test_idx], y_true)))

        dp_differences.append(float(np.mean(factor_dp_diff)))
        dp_ratios.append(float(np.mean(factor_dp_ratio)))
        eo_differences.append(float(np.mean(factor_eo_diff)))
        accuracies.append(float(np.mean(factor_acc)))

    return {
        "accuracy": float(np.mean(accuracies)),
        "demographic_parity_difference": float(np.mean(dp_differences)),
        "demographic_parity_ratio": float(np.mean(dp_ratios)),
        "equalized_odds_difference": float(np.mean(eo_differences)),
    }
