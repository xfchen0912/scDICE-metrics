"""SACCELERATOR-style supervised spatial domain metrics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    jaccard_score,
    matthews_corrcoef,
    normalized_mutual_info_score,
)

from scdice_metrics.metrics._nmi_ari import _check_label_pair


def _contingency_table(labels: np.ndarray, labels_pred: np.ndarray) -> pd.DataFrame:
    labels_arr, labels_pred_arr = _check_label_pair(labels, labels_pred)
    return pd.crosstab(labels_pred_arr, labels_arr)


def match_predicted_to_ground_truth(labels: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """Hungarian match predicted clusters to ground-truth domains (maximize overlap).

    Returns a copy of ``labels_pred`` with cluster ids replaced by matched GT domain labels.
    """
    table = _contingency_table(labels, labels_pred)
    if table.size == 0:
        return np.asarray(labels_pred).copy()

    cost = table.to_numpy(dtype=float)
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    pred_values = np.asarray(labels_pred).ravel()
    remap: dict[Any, Any] = {}
    for r, c in zip(row_ind, col_ind, strict=False):
        if r < len(table.index) and c < len(table.columns):
            remap[table.index[r]] = table.columns[c]
    return np.array([remap.get(v, v) for v in pred_values], dtype=object)


def spatial_ari(labels: np.ndarray, labels_pred: np.ndarray, *, match: bool = False) -> float:
    """Adjusted Rand index between spatial domain GT and predicted clusters."""
    labels_arr, labels_pred_arr = _check_label_pair(labels, labels_pred)
    if match:
        labels_pred_arr = match_predicted_to_ground_truth(labels_arr, labels_pred_arr)
    return float(adjusted_rand_score(labels_arr, labels_pred_arr))


def spatial_nmi(labels: np.ndarray, labels_pred: np.ndarray, *, match: bool = False) -> float:
    """Normalized mutual information between spatial domain GT and predicted clusters."""
    labels_arr, labels_pred_arr = _check_label_pair(labels, labels_pred)
    if match:
        labels_pred_arr = match_predicted_to_ground_truth(labels_arr, labels_pred_arr)
    return float(normalized_mutual_info_score(labels_arr, labels_pred_arr))


def gt_mixture_entropy(labels: np.ndarray, labels_pred: np.ndarray) -> float:
    """Weighted within-predicted-cluster entropy of ground-truth labels (SACCELERATOR Entropy).

    Lower is better (purer predicted domains).
    """
    labels_arr, labels_pred_arr = _check_label_pair(labels, labels_pred)
    df = pd.DataFrame({"pred": labels_pred_arr, "true": labels_arr})
    n = len(df)
    total_pred = df.groupby("pred", observed=True).size()
    counts = df.groupby(["pred", "true"], observed=True).size()

    metric = 0.0
    for pred in df["pred"].unique():
        cluster_size = total_pred.loc[pred]
        if cluster_size == 0:
            continue
        weight = cluster_size / n
        inner = 0.0
        if pred in counts.index.get_level_values(0):
            sub = counts.loc[pred]
            if isinstance(sub, pd.Series):
                for count in sub.values:
                    p = count / cluster_size
                    if p > 0:
                        inner -= p * math.log2(p)
            else:
                p = sub / cluster_size
                if p > 0:
                    inner -= p * math.log2(p)
        metric += weight * inner
    return float(metric)


def domain_specific_f1(labels: np.ndarray, labels_pred: np.ndarray) -> dict[str, float]:
    """Per ground-truth domain F1 after Hungarian matching (SACCELERATOR domain-specific-F1)."""
    labels_arr, labels_pred_arr = _check_label_pair(labels, labels_pred)
    table = _contingency_table(labels_arr, labels_pred_arr)
    n_gt = table.shape[1]
    n_pred = table.shape[0]

    tb = table.copy()
    if n_pred > n_gt:
        for i in range(n_pred - n_gt):
            tb[f"dummy{i}"] = 0

    cost = tb.to_numpy(dtype=float)
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)

    gt_names = list(table.columns)
    pred_names = list(table.index)
    out: dict[str, float] = {}
    f1_vals: list[float] = []

    for r, c in zip(row_ind, col_ind, strict=False):
        if c >= len(gt_names):
            continue
        gt_name = gt_names[c]
        if str(gt_name).startswith("dummy"):
            continue
        tp = cost[r, c]
        domain_size = float(table.iloc[:, c].sum()) if c < table.shape[1] else 0.0
        cluster_size = float(table.iloc[r, :].sum())
        if domain_size == 0 or cluster_size == 0:
            continue
        recall = tp / domain_size
        precision = tp / cluster_size
        if recall + precision == 0:
            f1 = 0.0
        else:
            f1 = 2.0 / (1.0 / recall + 1.0 / precision)
        key = str(gt_name)
        out[key] = float(f1)
        f1_vals.append(float(f1))

    out["macro"] = float(np.mean(f1_vals)) if f1_vals else float("nan")
    return out


def matched_mcc(labels: np.ndarray, labels_pred: np.ndarray) -> float:
    """MCC after Hungarian matching of predicted clusters to GT (SACCELERATOR MCC)."""
    labels_arr, labels_pred_arr = _check_label_pair(labels, labels_pred)
    matched = match_predicted_to_ground_truth(labels_arr, labels_pred_arr)
    gt_codes, _ = pd.factorize(labels_arr)
    pred_codes, _ = pd.factorize(matched)
    return float(matthews_corrcoef(gt_codes, pred_codes))


def matched_jaccard(labels: np.ndarray, labels_pred: np.ndarray) -> float:
    """Weighted Jaccard after Hungarian matching (SACCELERATOR jaccard)."""
    labels_arr, labels_pred_arr = _check_label_pair(labels, labels_pred)
    matched = match_predicted_to_ground_truth(labels_arr, labels_pred_arr)
    gt_codes, _ = pd.factorize(labels_arr)
    pred_codes, _ = pd.factorize(matched)
    return float(jaccard_score(gt_codes, pred_codes, average="weighted"))


def spatial_external_ari(
    labels: np.ndarray,
    labels_pred: np.ndarray,
    spatial: np.ndarray,
    **_: Any,
) -> float:
    """SpatialARI via external ClusteringMetrics (optional).

    Not bundled in core scdice-metrics. Returns NaN until an optional backend is installed.
    Use :func:`spatial_ari` for label-only ARI.
    """
    return float("nan")
