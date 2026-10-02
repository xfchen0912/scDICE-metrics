"""Multi-method spatial clustering consensus metrics (SACCELERATOR consensus layer)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import adjusted_rand_score

def cross_method_ari(labels_wide: pd.DataFrame) -> pd.DataFrame:
    """Pairwise adjusted Rand index between method columns."""
    methods = list(labels_wide.columns)
    n = len(methods)
    mat = np.full((n, n), np.nan)
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if i == j:
                mat[i, j] = 1.0
            else:
                mat[i, j] = adjusted_rand_score(labels_wide[mi], labels_wide[mj])
    return pd.DataFrame(mat, index=methods, columns=methods)


def _spot_entropy(labels: np.ndarray) -> float:
    labels = np.asarray(labels)
    _, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    if total == 0:
        return float("nan")
    ent = 0.0
    for c in counts:
        p = c / total
        if p > 0:
            ent -= p * math.log2(p)
    return float(ent)


def smoothness_entropy(
    labels_wide: pd.DataFrame,
    spatial: np.ndarray,
    *,
    k: int = 6,
) -> pd.Series:
    """Mean spatial kNN label entropy per method column (lower = smoother)."""
    coords = np.asarray(spatial, dtype=float)
    if coords.shape[0] != len(labels_wide):
        raise ValueError("spatial coordinates and labels_wide must have the same number of rows.")
    n = coords.shape[0]
    k_eff = min(k, n - 1)
    dist = cdist(coords, coords, metric="euclidean")
    nn_idx = np.argsort(dist, axis=1)[:, 1 : k_eff + 1]

    scores: dict[str, float] = {}
    for col in labels_wide.columns:
        vals = labels_wide[col].to_numpy()
        entropies = [_spot_entropy(vals[idx]) for idx in nn_idx]
        scores[str(col)] = float(np.mean(entropies)) if entropies else float("nan")
    return pd.Series(scores)


def cross_method_entropy(labels_wide: pd.DataFrame) -> pd.Series:
    """Per-row entropy of labels across methods (higher = more cross-method disagreement)."""
    rows = []
    for idx, row in labels_wide.iterrows():
        rows.append(_spot_entropy(row.to_numpy()))
    return pd.Series(rows, index=labels_wide.index, name="cross_method_entropy")


def summarize_consensus(
    labels_wide: pd.DataFrame,
    spatial: np.ndarray | None = None,
    *,
    k_neighbors: int = 6,
) -> dict[str, Any]:
    """Bundle consensus outputs for downstream tables/plots."""
    out: dict[str, Any] = {"cross_method_ari": cross_method_ari(labels_wide)}
    if spatial is not None:
        out["smoothness_entropy"] = smoothness_entropy(labels_wide, spatial, k=k_neighbors)
    out["cross_method_entropy_mean"] = float(cross_method_entropy(labels_wide).mean())
    return out
