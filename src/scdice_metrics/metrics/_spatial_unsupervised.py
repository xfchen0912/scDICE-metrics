"""Embedding-space cluster quality metrics (SACCELERATOR unsupervised group)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
)
from sklearn.utils import check_array

from scdice_metrics.metrics._spatial_clustering import _as_1d_array


def _embedding_matrix(embedding: np.ndarray) -> np.ndarray:
    X = check_array(embedding, accept_sparse=False)
    if X.ndim != 2:
        raise ValueError("`embedding` must be 2D.")
    return X


def calinski_harabasz(labels: np.ndarray, embedding: np.ndarray) -> float:
    """Calinski-Harabasz index on embedding coordinates."""
    labels_arr = _as_1d_array(labels, n_cells=_embedding_matrix(embedding).shape[0])
    return float(calinski_harabasz_score(_embedding_matrix(embedding), labels_arr))


def davies_bouldin(labels: np.ndarray, embedding: np.ndarray) -> float:
    """Davies-Bouldin index on embedding coordinates (lower is better)."""
    labels_arr = _as_1d_array(labels, n_cells=_embedding_matrix(embedding).shape[0])
    return float(davies_bouldin_score(_embedding_matrix(embedding), labels_arr))


def cluster_specific_silhouette(
    labels: np.ndarray,
    embedding: np.ndarray,
) -> dict[str, float]:
    """Mean silhouette width per predicted cluster."""
    X = _embedding_matrix(embedding)
    labels_arr = _as_1d_array(labels, n_cells=X.shape[0])
    sil = silhouette_samples(X, labels_arr)
    out: dict[str, float] = {}
    for label in np.unique(labels_arr):
        mask = labels_arr == label
        if np.sum(mask) == 0:
            continue
        out[str(label)] = float(np.mean(sil[mask]))
    out["macro"] = float(np.mean(sil)) if sil.size else float("nan")
    return out
