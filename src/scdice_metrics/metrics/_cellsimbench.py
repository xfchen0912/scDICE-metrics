"""CellSimBench-style perturbation prediction metrics (pseudobulk / dual-delta)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score

from scdice_metrics.metrics._counterfactual import (
    _mean_profile,
    _safe_pearson,
    _validate_feature_dimensions,
)

DegSource = Literal["gt_half", "predictor_half", "external"]


@dataclass(frozen=True)
class DegProfile:
    """DEG weights and masks aligned to evaluation gene order (typically GT-half derived)."""

    weights: np.ndarray
    top_mask: np.ndarray
    source: DegSource = "gt_half"
    significant_mask: np.ndarray | None = None
    directions: np.ndarray | None = None

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=float).ravel()
        m = np.asarray(self.top_mask, dtype=bool).ravel()
        if w.shape != m.shape:
            raise ValueError("DegProfile weights and top_mask must have the same length.")
        if not np.all(np.isfinite(w)):
            raise ValueError("DegProfile weights must be finite.")
        object.__setattr__(self, "weights", w)
        object.__setattr__(self, "top_mask", m)


def deg_weights_from_scores(
    gene_order: list[str] | np.ndarray,
    scores: np.ndarray,
    score_gene_names: np.ndarray | list[str],
) -> np.ndarray:
    """Min-max(|score|) then square, max weight per gene (CellSimBench DataManager logic)."""
    gene_order = list(gene_order)
    abs_scores = np.abs(np.asarray(scores, dtype=float))
    gene_names = np.asarray(score_gene_names)
    if abs_scores.shape[0] != gene_names.shape[0]:
        raise ValueError("scores and score_gene_names must have the same length.")

    min_val = float(np.min(abs_scores))
    max_val = float(np.max(abs_scores))
    if max_val == min_val:
        normalized = np.zeros_like(abs_scores)
    else:
        normalized = (abs_scores - min_val) / (max_val - min_val)
    normalized = np.nan_to_num(normalized, nan=0.0)
    squared = np.square(normalized)

    weights_df = pd.DataFrame({"gene": gene_names, "weight": squared})
    aggregated = weights_df.groupby("gene", as_index=True)["weight"].max()
    aligned = aggregated.reindex(gene_order, fill_value=0.0)
    return aligned.to_numpy(dtype=float)


def deg_mask_from_pvals(
    gene_order: list[str] | np.ndarray,
    pvals: np.ndarray,
    pval_gene_names: np.ndarray | list[str],
    *,
    topn: int = 100,
    pval_threshold: float = 0.05,
) -> np.ndarray:
    """Top-N genes by p-value (CellSimBench ``get_deg_mask`` with ``topn`` set)."""
    gene_order = list(gene_order)
    pvals = np.asarray(pvals, dtype=float)
    gene_names = np.asarray(pval_gene_names)
    if pvals.shape[0] != gene_names.shape[0]:
        raise ValueError("pvals and pval_gene_names must have the same length.")

    pvals_df = pd.DataFrame({"gene": gene_names, "pval": pvals})
    if topn is not None:
        pvals_df_sorted = pvals_df.sort_values(by="pval", ascending=True).head(topn)
        pvals_df["significant"] = False
        pvals_df.loc[pvals_df.index.isin(pvals_df_sorted.index), "significant"] = True
        deg_mask_aggregated = pvals_df.groupby("gene")["significant"].any()
    else:
        pvals_aggregated = pvals_df.groupby("gene")["pval"].min()
        deg_mask_aggregated = pvals_aggregated < pval_threshold

    aligned = deg_mask_aggregated.reindex(gene_order, fill_value=False)
    return aligned.to_numpy(dtype=bool)


def interpolated_duplicate(
    tech_dup: np.ndarray,
    dataset_mean: np.ndarray,
    pvals: np.ndarray,
    *,
    pval_genes: np.ndarray | list[str] | None = None,
    gene_order: list[str] | np.ndarray | None = None,
) -> np.ndarray:
    """Per-gene blend: alpha * tech_dup + (1-alpha) * mean with alpha = 1 - pval."""
    tech = np.asarray(tech_dup, dtype=float).ravel()
    mean = np.asarray(dataset_mean, dtype=float).ravel()
    if tech.shape != mean.shape:
        raise ValueError("tech_dup and dataset_mean must have the same shape.")

    p = np.asarray(pvals, dtype=float).ravel()
    if p.size == tech.size:
        alpha = 1.0 - np.clip(p, 0.0, 1.0)
        alpha = np.nan_to_num(alpha, nan=0.0)
        return alpha * tech + (1.0 - alpha) * mean

    if pval_genes is None or gene_order is None:
        raise ValueError("pvals length mismatch requires pval_genes and gene_order.")
    gene_order = list(gene_order)
    alpha_full = np.zeros(len(gene_order), dtype=float)
    pvals_df = pd.DataFrame({"gene": np.asarray(pval_genes), "pval": p})
    alpha_by_gene = 1.0 - np.clip(pvals_df.groupby("gene")["pval"].min(), 0.0, 1.0)
    alpha_full = alpha_by_gene.reindex(gene_order, fill_value=0.0).to_numpy(dtype=float)
    return alpha_full * tech + (1.0 - alpha_full) * mean


def profile_mse(observed: Any, predicted: Any) -> float:
    obs = _mean_profile(observed, name="observed")
    pred = _mean_profile(predicted, name="predicted")
    _validate_feature_dimensions((obs, "observed"), (pred, "predicted"))
    return float(np.mean((obs - pred) ** 2))


def profile_mae(observed: Any, predicted: Any) -> float:
    obs = _mean_profile(observed, name="observed")
    pred = _mean_profile(predicted, name="predicted")
    _validate_feature_dimensions((obs, "observed"), (pred, "predicted"))
    return float(np.mean(np.abs(obs - pred)))


def profile_wmse(observed: Any, predicted: Any, weights: np.ndarray) -> float:
    obs = _mean_profile(observed, name="observed")
    pred = _mean_profile(predicted, name="predicted")
    _validate_feature_dimensions((obs, "observed"), (pred, "predicted"))
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape != obs.shape:
        raise ValueError("weights must match profile length.")
    total = float(np.sum(w))
    if total == 0.0:
        return float(np.nan)
    norm = w / total
    return float(np.sum(norm * (obs - pred) ** 2))


def profile_wmae(observed: Any, predicted: Any, weights: np.ndarray) -> float:
    obs = _mean_profile(observed, name="observed")
    pred = _mean_profile(predicted, name="predicted")
    _validate_feature_dimensions((obs, "observed"), (pred, "predicted"))
    w = np.asarray(weights, dtype=float).ravel()
    if w.shape != obs.shape:
        raise ValueError("weights must match profile length.")
    total = float(np.sum(w))
    if total == 0.0:
        return float(np.nan)
    norm = w / total
    return float(np.sum(norm * np.abs(obs - pred)))


def r2_delta(
    delta_true: np.ndarray,
    delta_pred: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    delta_true = np.asarray(delta_true, dtype=float).ravel()
    delta_pred = np.asarray(delta_pred, dtype=float).ravel()
    if delta_true.size < 2 or delta_pred.size < 2 or delta_true.shape != delta_pred.shape:
        return float(np.nan)
    if weights is not None:
        w = np.asarray(weights, dtype=float).ravel()
        if float(np.sum(w)) == 0.0:
            return float(np.nan)
        return float(max(r2_score(delta_true, delta_pred, sample_weight=w), -1.0))
    return float(max(r2_score(delta_true, delta_pred), -1.0))


def _delta_ctrl(observed: Any, predicted: Any, reference: Any) -> tuple[np.ndarray, np.ndarray]:
    obs = _mean_profile(observed, name="observed")
    pred = _mean_profile(predicted, name="predicted")
    ref = _mean_profile(reference, name="reference")
    _validate_feature_dimensions((obs, "observed"), (pred, "predicted"), (ref, "reference"))
    return obs - ref, pred - ref


def _delta_pert(observed: Any, predicted: Any, dataset_mean: Any) -> tuple[np.ndarray, np.ndarray]:
    if dataset_mean is None:
        raise ValueError("dataset_mean is required for deltapert metrics.")
    obs = _mean_profile(observed, name="observed")
    pred = _mean_profile(predicted, name="predicted")
    mean = _mean_profile(dataset_mean, name="dataset_mean")
    _validate_feature_dimensions((obs, "observed"), (pred, "predicted"), (mean, "dataset_mean"))
    return obs - mean, pred - mean


def pearson_deltactrl(observed: Any, predicted: Any, reference: Any, **_: Any) -> float:
    truth_d, pred_d = _delta_ctrl(observed, predicted, reference)
    return _safe_pearson(pred_d, truth_d)


def pearson_deltapert(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    dataset_mean: Any | None = None,
    **_: Any,
) -> float:
    if dataset_mean is None:
        return float(np.nan)
    truth_d, pred_d = _delta_pert(observed, predicted, dataset_mean)
    return _safe_pearson(pred_d, truth_d)


def pearson_deltactrl_degs(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    deg: DegProfile | None = None,
    **_: Any,
) -> float:
    if deg is None:
        return float(np.nan)
    truth_d, pred_d = _delta_ctrl(observed, predicted, reference)
    if int(np.sum(deg.top_mask)) <= 2:
        return float(np.nan)
    return _safe_pearson(pred_d[deg.top_mask], truth_d[deg.top_mask])


def pearson_deltapert_degs(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    dataset_mean: Any | None = None,
    deg: DegProfile | None = None,
    **_: Any,
) -> float:
    if dataset_mean is None or deg is None:
        return float(np.nan)
    truth_d, pred_d = _delta_pert(observed, predicted, dataset_mean)
    if int(np.sum(deg.top_mask)) <= 2:
        return float(np.nan)
    return _safe_pearson(pred_d[deg.top_mask], truth_d[deg.top_mask])


def r2_deltactrl(observed: Any, predicted: Any, reference: Any, **_: Any) -> float:
    truth_d, pred_d = _delta_ctrl(observed, predicted, reference)
    return r2_delta(truth_d, pred_d)


def r2_deltapert(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    dataset_mean: Any | None = None,
    **_: Any,
) -> float:
    if dataset_mean is None:
        return float(np.nan)
    truth_d, pred_d = _delta_pert(observed, predicted, dataset_mean)
    return r2_delta(truth_d, pred_d)


def r2_deltactrl_degs(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    deg: DegProfile | None = None,
    **_: Any,
) -> float:
    if deg is None:
        return float(np.nan)
    truth_d, pred_d = _delta_ctrl(observed, predicted, reference)
    if int(np.sum(deg.top_mask)) <= 2:
        return float(np.nan)
    return r2_delta(truth_d[deg.top_mask], pred_d[deg.top_mask])


def r2_deltapert_degs(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    dataset_mean: Any | None = None,
    deg: DegProfile | None = None,
    **_: Any,
) -> float:
    if dataset_mean is None or deg is None:
        return float(np.nan)
    truth_d, pred_d = _delta_pert(observed, predicted, dataset_mean)
    if int(np.sum(deg.top_mask)) <= 2:
        return float(np.nan)
    return r2_delta(truth_d[deg.top_mask], pred_d[deg.top_mask])


def weighted_r2_deltactrl(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    deg: DegProfile | None = None,
    **_: Any,
) -> float:
    if deg is None:
        return float(np.nan)
    truth_d, pred_d = _delta_ctrl(observed, predicted, reference)
    return r2_delta(truth_d, pred_d, weights=deg.weights)


def weighted_r2_deltapert(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    dataset_mean: Any | None = None,
    deg: DegProfile | None = None,
    **_: Any,
) -> float:
    if dataset_mean is None or deg is None:
        return float(np.nan)
    truth_d, pred_d = _delta_pert(observed, predicted, dataset_mean)
    return r2_delta(truth_d, pred_d, weights=deg.weights)


def mse_degs(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    deg: DegProfile | None = None,
    **_: Any,
) -> float:
    if deg is None:
        return float(np.nan)
    obs = _mean_profile(observed, name="observed")
    pred = _mean_profile(predicted, name="predicted")
    if int(np.sum(deg.top_mask)) == 0:
        return float(np.nan)
    return float(np.mean((obs[deg.top_mask] - pred[deg.top_mask]) ** 2))


def mae_degs(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    deg: DegProfile | None = None,
    **_: Any,
) -> float:
    if deg is None:
        return float(np.nan)
    obs = _mean_profile(observed, name="observed")
    pred = _mean_profile(predicted, name="predicted")
    if int(np.sum(deg.top_mask)) == 0:
        return float(np.nan)
    return float(np.mean(np.abs(obs[deg.top_mask] - pred[deg.top_mask])))


def wmse(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    deg: DegProfile | None = None,
    **_: Any,
) -> float:
    if deg is None:
        return float(np.nan)
    return profile_wmse(observed, predicted, deg.weights)


def wmae(
    observed: Any,
    predicted: Any,
    reference: Any,
    *,
    deg: DegProfile | None = None,
    **_: Any,
) -> float:
    if deg is None:
        return float(np.nan)
    return profile_wmae(observed, predicted, deg.weights)


def nir_scores(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    covariate_groups: dict[str, list[str]] | None = None,
    metric: str = "euclidean",
) -> dict[str, float]:
    """Nearest-Index Ratio within covariate groups (CellSimBench NIR)."""
    scores: dict[str, float] = {}
    if covariate_groups is None:
        keys = [k for k in predictions.index if k in ground_truth.index]
        covariate_groups = {"_all": keys}

    for _cov, pert_keys in covariate_groups.items():
        valid = [pk for pk in pert_keys if pk in predictions.index and pk in ground_truth.index]
        if len(valid) < 2:
            for pk in valid:
                scores[pk] = float(np.nan)
            continue
        pred_cov = predictions.loc[valid]
        truth_cov = ground_truth.loc[valid]
        dist = cdist(pred_cov.values, truth_cov.values, metric=metric)
        for i, pert_key in enumerate(valid):
            correct = dist[i, i]
            comparisons = [1 if correct < dist[i, j] else 0 for j in range(len(valid)) if j != i]
            scores[pert_key] = float(np.mean(comparisons)) if comparisons else float(np.nan)
    return scores


def pds_scores(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    *,
    covariate_groups: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    return nir_scores(
        predictions,
        ground_truth,
        covariate_groups=covariate_groups,
        metric="cityblock",
    )


def knn_jaccard_deltapert(
    predictions_deltamean: pd.DataFrame,
    ground_truth_deltamean: pd.DataFrame,
    *,
    k: int = 20,
) -> dict[str, float]:
    common = predictions_deltamean.index.intersection(ground_truth_deltamean.index)
    if len(common) < 2:
        return {key: float(np.nan) for key in common}

    pred = predictions_deltamean.loc[common]
    truth = ground_truth_deltamean.loc[common]
    n = len(common)
    k_eff = min(k, n - 1)
    pred_dist = cdist(pred.values, pred.values, metric="euclidean")
    truth_dist = cdist(truth.values, truth.values, metric="euclidean")
    pred_knn = np.argsort(pred_dist, axis=1)[:, 1 : k_eff + 1]
    truth_knn = np.argsort(truth_dist, axis=1)[:, 1 : k_eff + 1]

    out: dict[str, float] = {}
    for i, pert_key in enumerate(common):
        p_set = set(pred_knn[i])
        t_set = set(truth_knn[i])
        union = len(p_set | t_set)
        out[pert_key] = len(p_set & t_set) / union if union > 0 else float(np.nan)
    return out
