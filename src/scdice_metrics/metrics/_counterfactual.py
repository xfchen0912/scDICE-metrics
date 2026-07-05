"""Metrics for cell-type-specific OOD swap prediction."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance

from scdice_metrics.metrics._perturbation import pearson_delta_reference_metrics

try:
    import scipy.sparse as sp
except ImportError:  # pragma: no cover
    sp = None  # type: ignore[assignment]


# utilities


def _is_sparse(X: Any) -> bool:
    return sp is not None and sp.issparse(X)


def _asarray_1d(x: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _n_features(X: Any, *, name: str) -> int:
    if not hasattr(X, "shape"):
        raise TypeError(f"{name} must be an array-like object with a shape attribute.")
    shape = X.shape
    if len(shape) == 1:
        return int(shape[0])
    if len(shape) == 2:
        return int(shape[1])
    raise ValueError(f"{name} must be one- or two-dimensional.")


def _validate_feature_dimensions(*arrays: tuple[Any, str]) -> None:
    if not arrays:
        return
    n_features = _n_features(arrays[0][0], name=arrays[0][1])
    for arr, arr_name in arrays[1:]:
        if _n_features(arr, name=arr_name) != n_features:
            raise ValueError(
                f"{arr_name} feature count ({_n_features(arr, name=arr_name)}) "
                f"does not match {arrays[0][1]} ({n_features})."
            )


def _mean_profile(X: Any, *, name: str) -> np.ndarray:
    """Return a dense gene profile from a vector or cell matrix."""
    if not hasattr(X, "shape"):
        raise TypeError(f"{name} must be an array-like object with a shape attribute.")

    shape = X.shape
    if len(shape) == 1:
        return _asarray_1d(np.asarray(X), name=name)
    if len(shape) != 2:
        raise ValueError(f"{name} must be one- or two-dimensional.")

    if shape[0] == 0:
        raise ValueError(f"{name} must not be empty.")

    if _is_sparse(X):
        profile = np.asarray(X.mean(axis=0)).ravel()
    else:
        dense = np.asarray(X, dtype=float)
        if not np.all(np.isfinite(dense)):
            raise ValueError(f"{name} contains non-finite values.")
        profile = dense.mean(axis=0)

    profile = np.asarray(profile, dtype=float).ravel()
    if profile.size == 0 or not np.all(np.isfinite(profile)):
        raise ValueError(f"{name} produced an invalid profile.")
    return profile


def _to_dense_cells(
    X: Any,
    *,
    name: str,
    max_cells: int | None,
    random_state: int,
) -> np.ndarray:
    """Validate, optionally subsample rows, then densify cell-level input."""
    if not hasattr(X, "shape") or len(X.shape) != 2:
        raise ValueError(f"{name} must be a two-dimensional cell matrix.")

    n_cells = int(X.shape[0])
    if n_cells == 0:
        raise ValueError(f"{name} must not be empty.")

    indices = np.arange(n_cells)
    if max_cells is not None and n_cells > max_cells:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(indices, size=max_cells, replace=False)

    if _is_sparse(X):
        subset = X[indices]
        dense = np.asarray(subset.toarray(), dtype=float)
    else:
        dense = np.asarray(X[indices], dtype=float)

    if not np.all(np.isfinite(dense)):
        raise ValueError(f"{name} contains non-finite values.")
    return dense


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("Pearson inputs must have the same shape.")
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float(np.nan)
    return float(stats.pearsonr(x, y)[0])


def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError("Spearman inputs must have the same shape.")
    if x.size < 2 or np.std(x) == 0.0 or np.std(y) == 0.0:
        return float(np.nan)
    return float(stats.spearmanr(x, y).correlation)


def _safe_cosine(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0.0:
        return float(np.nan)
    return float(np.dot(x, y) / denom)


def _resolve_gene_indices(gene_indices: np.ndarray | None, n_genes: int) -> slice | np.ndarray:
    if gene_indices is None:
        return slice(None)
    idx = np.asarray(gene_indices, dtype=int).ravel()
    if idx.size == 0:
        raise ValueError("gene_indices must not be empty when provided.")
    if np.any(idx < 0) or np.any(idx >= n_genes):
        raise ValueError("gene_indices contains out-of-range positions.")
    return idx


def _subset_profile(profile: np.ndarray, gene_indices: np.ndarray | None) -> np.ndarray:
    idx = _resolve_gene_indices(gene_indices, profile.size)
    return profile[idx]


def _delta_profiles(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    gene_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = _subset_profile(_mean_profile(observed, name="observed"), gene_indices)
    pred = _subset_profile(_mean_profile(predicted, name="predicted"), gene_indices)
    ref = _subset_profile(_mean_profile(reference, name="reference"), gene_indices)
    _validate_feature_dimensions(
        (obs, "observed profile"),
        (pred, "predicted profile"),
        (ref, "reference profile"),
    )
    return obs - ref, pred - ref, obs


# pseudobulk metrics


def pseudobulk_pearson(
    observed: Any,
    predicted: Any,
    gene_indices: np.ndarray | None = None,
) -> float:
    obs = _subset_profile(_mean_profile(observed, name="observed"), gene_indices)
    pred = _subset_profile(_mean_profile(predicted, name="predicted"), gene_indices)
    _validate_feature_dimensions((obs, "observed profile"), (pred, "predicted profile"))
    return _safe_pearson(obs, pred)


def pseudobulk_spearman(
    observed: Any,
    predicted: Any,
    gene_indices: np.ndarray | None = None,
) -> float:
    obs = _subset_profile(_mean_profile(observed, name="observed"), gene_indices)
    pred = _subset_profile(_mean_profile(predicted, name="predicted"), gene_indices)
    _validate_feature_dimensions((obs, "observed profile"), (pred, "predicted profile"))
    return _safe_spearman(obs, pred)


def pseudobulk_rmse(
    observed: Any,
    predicted: Any,
    gene_indices: np.ndarray | None = None,
) -> float:
    obs = _subset_profile(_mean_profile(observed, name="observed"), gene_indices)
    pred = _subset_profile(_mean_profile(predicted, name="predicted"), gene_indices)
    _validate_feature_dimensions((obs, "observed profile"), (pred, "predicted profile"))
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def pseudobulk_mae(
    observed: Any,
    predicted: Any,
    gene_indices: np.ndarray | None = None,
) -> float:
    obs = _subset_profile(_mean_profile(observed, name="observed"), gene_indices)
    pred = _subset_profile(_mean_profile(predicted, name="predicted"), gene_indices)
    _validate_feature_dimensions((obs, "observed profile"), (pred, "predicted profile"))
    return float(np.mean(np.abs(obs - pred)))


# delta/effect metrics


def systema_pearson_delta_metrics(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    top_k: int = 20,
) -> dict[str, float]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    obs = _mean_profile(observed, name="observed")
    pred = _mean_profile(predicted, name="predicted")
    ref = _mean_profile(reference, name="reference")
    _validate_feature_dimensions(
        (obs, "observed profile"),
        (pred, "predicted profile"),
        (ref, "reference profile"),
    )

    delta_true = obs - ref
    k_eff = min(top_k, obs.size)
    top_idx = np.argsort(-np.abs(delta_true))[:k_eff]

    raw = pearson_delta_reference_metrics(obs, pred, ref, top20_de_idxs=top_idx)
    out = {"all_genes": float(raw["corr_all_allpert"])}
    top_key = f"top{top_k}_true_effect"
    if "corr_20de_allpert" in raw:
        out[top_key] = float(raw["corr_20de_allpert"])
    return out


def delta_pearson(
    observed: Any,
    predicted: Any,
    reference: Any,
    gene_indices: np.ndarray | None = None,
) -> float:
    if gene_indices is None:
        return systema_pearson_delta_metrics(observed, predicted, reference)["all_genes"]
    delta_true, delta_pred, _ = _delta_profiles(observed, predicted, reference, gene_indices=gene_indices)
    return _safe_pearson(delta_true, delta_pred)


def delta_spearman(
    observed: Any,
    predicted: Any,
    reference: Any,
    gene_indices: np.ndarray | None = None,
) -> float:
    delta_true, delta_pred, _ = _delta_profiles(observed, predicted, reference, gene_indices=gene_indices)
    return _safe_spearman(delta_true, delta_pred)


def delta_cosine(
    observed: Any,
    predicted: Any,
    reference: Any,
    gene_indices: np.ndarray | None = None,
) -> float:
    delta_true, delta_pred, _ = _delta_profiles(observed, predicted, reference, gene_indices=gene_indices)
    return _safe_cosine(delta_true, delta_pred)


def delta_rmse(
    observed: Any,
    predicted: Any,
    reference: Any,
    gene_indices: np.ndarray | None = None,
) -> float:
    delta_true, delta_pred, _ = _delta_profiles(observed, predicted, reference, gene_indices=gene_indices)
    return float(np.sqrt(np.mean((delta_true - delta_pred) ** 2)))


def delta_mae(
    observed: Any,
    predicted: Any,
    reference: Any,
    gene_indices: np.ndarray | None = None,
) -> float:
    delta_true, delta_pred, _ = _delta_profiles(observed, predicted, reference, gene_indices=gene_indices)
    return float(np.mean(np.abs(delta_true - delta_pred)))


def delta_profile_metrics(
    observed: Any,
    predicted: Any,
    reference: Any,
    gene_indices: np.ndarray | None = None,
) -> dict[str, float]:
    return {
        "delta_pearson": delta_pearson(observed, predicted, reference, gene_indices=gene_indices),
        "delta_spearman": delta_spearman(observed, predicted, reference, gene_indices=gene_indices),
        "delta_cosine": delta_cosine(observed, predicted, reference, gene_indices=gene_indices),
        "delta_rmse": delta_rmse(observed, predicted, reference, gene_indices=gene_indices),
        "delta_mae": delta_mae(observed, predicted, reference, gene_indices=gene_indices),
    }


# differential-expression metrics


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float(np.nan)
    return float(numerator / denominator)


def signed_de_recovery(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    top_k: int = 50,
    min_abs_effect: float = 0.0,
) -> dict[str, float]:
    if top_k < 1:
        raise ValueError("top_k must be >= 1.")

    delta_true, delta_pred, _ = _delta_profiles(observed, predicted, reference)
    eligible = np.flatnonzero(np.abs(delta_true) >= min_abs_effect)
    if eligible.size == 0:
        nan = float(np.nan)
        return {
            "precision": nan,
            "recall": nan,
            "f1": nan,
            "jaccard": nan,
            "direction_accuracy_true_top": nan,
            "signed_precision": nan,
            "up_precision": nan,
            "down_precision": nan,
        }

    k_eff = min(top_k, eligible.size)
    ranked_true = eligible[np.argsort(-np.abs(delta_true[eligible]))[:k_eff]]
    ranked_pred = eligible[np.argsort(-np.abs(delta_pred[eligible]))[:k_eff]]

    true_set = set(ranked_true.tolist())
    pred_set = set(ranked_pred.tolist())
    intersection = true_set & pred_set
    union = true_set | pred_set

    precision = _safe_div(len(intersection), len(pred_set))
    recall = _safe_div(len(intersection), len(true_set))
    f1 = _safe_div(2.0 * precision * recall, precision + recall) if np.isfinite(precision + recall) else float(np.nan)
    jaccard = _safe_div(len(intersection), len(union))

    true_top = np.array(sorted(true_set), dtype=int)
    nonzero = delta_true[true_top] != 0.0
    if np.any(nonzero):
        direction_accuracy = float(
            np.mean(np.sign(delta_pred[true_top[nonzero]]) == np.sign(delta_true[true_top[nonzero]]))
        )
    else:
        direction_accuracy = float(np.nan)

    signed_matches = [
        g for g in pred_set if np.sign(delta_pred[g]) == np.sign(delta_true[g]) and delta_true[g] != 0.0
    ]
    signed_precision = _safe_div(len(signed_matches), len(pred_set))

    pred_up = [g for g in pred_set if delta_pred[g] > 0.0]
    pred_down = [g for g in pred_set if delta_pred[g] < 0.0]
    up_hits = [g for g in pred_up if g in true_set and delta_true[g] > 0.0]
    down_hits = [g for g in pred_down if g in true_set and delta_true[g] < 0.0]
    up_precision = _safe_div(len(up_hits), len(pred_up))
    down_precision = _safe_div(len(down_hits), len(pred_down))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "direction_accuracy_true_top": direction_accuracy,
        "signed_precision": signed_precision,
        "up_precision": up_precision,
        "down_precision": down_precision,
    }


# distribution metrics


def _pairwise_mean_distance(X: np.ndarray) -> float:
    if X.shape[0] < 2:
        return 0.0
    distances = cdist(X, X)
    n = X.shape[0]
    return float((distances.sum() - np.trace(distances)) / (n * (n - 1)))


def energy_distance(
    observed: Any,
    predicted: Any,
    *,
    max_cells: int | None = 2000,
    random_state: int = 0,
) -> float:
    obs = _to_dense_cells(observed, name="observed", max_cells=max_cells, random_state=random_state)
    pred = _to_dense_cells(predicted, name="predicted", max_cells=max_cells, random_state=random_state + 1)
    _validate_feature_dimensions((obs, "observed"), (pred, "predicted"))
    cross = float(cdist(obs, pred).mean())
    stat = 2.0 * cross - _pairwise_mean_distance(obs) - _pairwise_mean_distance(pred)
    return float(max(stat, 0.0))


def _median_bandwidth(X: np.ndarray, Y: np.ndarray) -> float:
    combined = np.vstack([X, Y])
    if combined.shape[0] < 2:
        return 1.0
    distances = cdist(combined, combined)
    positive = distances[distances > 0.0]
    if positive.size == 0:
        return 1.0
    return float(np.median(positive))


def _rbf_kernel(X: np.ndarray, Y: np.ndarray, bandwidth: float) -> np.ndarray:
    gamma = 1.0 / (2.0 * bandwidth**2)
    sq_dists = cdist(X, Y, metric="sqeuclidean")
    return np.exp(-gamma * sq_dists)


def mmd_rbf(
    observed: Any,
    predicted: Any,
    *,
    bandwidth: float | str = "median",
    biased: bool = False,
    max_cells: int | None = 2000,
    random_state: int = 0,
) -> float:
    obs = _to_dense_cells(observed, name="observed", max_cells=max_cells, random_state=random_state)
    pred = _to_dense_cells(predicted, name="predicted", max_cells=max_cells, random_state=random_state + 1)
    _validate_feature_dimensions((obs, "observed"), (pred, "predicted"))

    if isinstance(bandwidth, str):
        if bandwidth != "median":
            raise ValueError("Only bandwidth='median' is supported for string bandwidth values.")
        bw = _median_bandwidth(obs, pred)
    else:
        bw = float(bandwidth)
    if bw <= 0.0:
        raise ValueError("bandwidth must be positive.")

    k_xx = _rbf_kernel(obs, obs, bw)
    k_yy = _rbf_kernel(pred, pred, bw)
    k_xy = _rbf_kernel(obs, pred, bw)

    if biased:
        mmd = k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()
    else:
        n_x = obs.shape[0]
        n_y = pred.shape[0]
        sum_xx = (k_xx.sum() - np.trace(k_xx)) / (n_x * (n_x - 1)) if n_x > 1 else 0.0
        sum_yy = (k_yy.sum() - np.trace(k_yy)) / (n_y * (n_y - 1)) if n_y > 1 else 0.0
        mmd = sum_xx + sum_yy - 2.0 * k_xy.mean()
    return float(max(mmd, 0.0))


def mean_gene_wasserstein(
    observed: Any,
    predicted: Any,
    gene_indices: np.ndarray | None = None,
) -> float:
    obs = _to_dense_cells(observed, name="observed", max_cells=None, random_state=0)
    pred = _to_dense_cells(predicted, name="predicted", max_cells=None, random_state=0)
    _validate_feature_dimensions((obs, "observed"), (pred, "predicted"))

    gene_cols = (
        np.arange(obs.shape[1], dtype=int)
        if gene_indices is None
        else np.asarray(gene_indices, dtype=int).ravel()
    )
    if gene_cols.size == 0:
        raise ValueError("gene_indices must not be empty when provided.")
    if np.any(gene_cols < 0) or np.any(gene_cols >= obs.shape[1]):
        raise ValueError("gene_indices contains out-of-range positions.")

    distances = [wasserstein_distance(obs[:, g], pred[:, g]) for g in gene_cols]
    return float(np.mean(distances))
