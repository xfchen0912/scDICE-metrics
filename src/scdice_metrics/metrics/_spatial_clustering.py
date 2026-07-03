from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array

from scdice_metrics.metrics._clustering import (
    cluster_labels_kmeans,
    cluster_labels_leiden,
    neighbors_from_spatial,
)
from scdice_metrics.nearest_neighbors import NeighborsResults


def _as_2d_float_array(spatial: np.ndarray) -> np.ndarray:
    spatial = np.asarray(spatial, dtype=float)
    if spatial.ndim != 2:
        raise ValueError("`spatial` must be a 2D array of shape (n_cells, n_dims).")
    if spatial.shape[0] == 0:
        raise ValueError("`spatial` must contain at least one cell.")
    return spatial


def _as_1d_array(labels: np.ndarray, n_cells: int) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        labels = labels.ravel()
    if labels.shape[0] != n_cells:
        raise ValueError("`labels` and `spatial` must have the same number of cells.")
    return labels


def _prepare_spatial_coords(spatial: np.ndarray, standardize: bool) -> np.ndarray:
    spatial_arr = _as_2d_float_array(spatial)
    if standardize:
        spatial_arr = StandardScaler().fit_transform(spatial_arr)
    return spatial_arr


def pas_from_neighbors(
    labels: np.ndarray,
    spatial_neighbors: NeighborsResults,
    k: int = 10,
) -> float:
    """Compute PAS using a precomputed spatial kNN graph (equivalent to SDMBench PAS)."""
    labels_arr = np.asarray(labels).ravel()
    n_cells = labels_arr.shape[0]
    if n_cells <= 1:
        return 0.0
    if k < 1:
        raise ValueError("`k` must be >= 1.")
    k_eff = min(k, n_cells - 1, spatial_neighbors.n_neighbors - 1)

    abnormal_count = 0
    for i in range(n_cells):
        nn_idx = spatial_neighbors.indices[i]
        nn_labels = labels_arr[nn_idx]
        keep = nn_idx != i
        if not np.any(keep):
            continue
        nn_labels = nn_labels[keep][:k_eff]
        if np.mean(nn_labels != labels_arr[i]) > 0.5:
            abnormal_count += 1
    return abnormal_count / float(n_cells)


def _pas_cdist(labels: np.ndarray, spatial_arr: np.ndarray, k: int) -> float:
    n_cells = spatial_arr.shape[0]
    k_eff = min(k, n_cells - 1)
    dmat = cdist(spatial_arr, spatial_arr, metric="euclidean")
    np.fill_diagonal(dmat, np.inf)
    nn_idx = np.argpartition(dmat, kth=k_eff - 1, axis=1)[:, :k_eff]
    mismatch = labels[nn_idx] != labels[:, None]
    abnormal = mismatch.sum(axis=1) > (k_eff / 2.0)
    return float(np.mean(abnormal))


def pas(
    labels: np.ndarray,
    spatial: np.ndarray,
    k: int = 10,
    standardize: bool = True,
    spatial_neighbors: NeighborsResults | None = None,
    reuse_spatial_neighbors: bool = True,
) -> float:
    """Compute PAS (Percentage of Abnormal Spots) for spatial clustering.

    When ``spatial_neighbors`` is provided and ``reuse_spatial_neighbors`` is True,
    PAS is evaluated on the precomputed kNN graph (faster, same definition as
    all-pairs distances for exact kNN).

    Parameters
    ----------
    labels
        Cluster labels of shape ``(n_cells,)``.
    spatial
        Spatial coordinates of shape ``(n_cells, n_dims)``.
    k
        Number of nearest spatial neighbors used for abnormality decision.
    standardize
        Whether to z-score spatial coordinates before computing distances
        (only used when falling back to coordinate-based computation).
    spatial_neighbors
        Optional precomputed spatial :class:`~scdice_metrics.nearest_neighbors.NeighborsResults`.
    reuse_spatial_neighbors
        If True and ``spatial_neighbors`` is set, reuse the neighbor graph.

    Returns
    -------
    float
        PAS score in ``[0, 1]``. Lower values indicate better spatial smoothness.
    """
    labels_arr = _as_1d_array(labels, n_cells=_as_2d_float_array(spatial).shape[0])

    if reuse_spatial_neighbors and spatial_neighbors is not None:
        if spatial_neighbors.n_neighbors < k + 1:
            raise ValueError(
                f"`spatial_neighbors` must include at least k+1={k + 1} neighbors "
                f"(has {spatial_neighbors.n_neighbors}). Increase `spatial_neighbors_k` in prepare."
            )
        return pas_from_neighbors(labels_arr, spatial_neighbors, k=k)

    spatial_arr = _prepare_spatial_coords(spatial, standardize=standardize)
    return _pas_cdist(labels_arr, spatial_arr, k=k)


def _chaos_kdtree(labels_arr: np.ndarray, spatial_arr: np.ndarray) -> float:
    dist_sum = 0.0
    for label in np.unique(labels_arr):
        cluster_spatial = spatial_arr[labels_arr == label]
        if cluster_spatial.shape[0] <= 2:
            continue
        tree = cKDTree(cluster_spatial)
        nn_dists, _ = tree.query(cluster_spatial, k=2)
        dist_sum += float(nn_dists[:, 1].sum())
    return dist_sum / float(spatial_arr.shape[0])


def _chaos_cdist(labels_arr: np.ndarray, spatial_arr: np.ndarray) -> float:
    dist_sum = 0.0
    for label in np.unique(labels_arr):
        mask = labels_arr == label
        cluster_spatial = spatial_arr[mask]
        if cluster_spatial.shape[0] <= 2:
            continue
        dmat = cdist(cluster_spatial, cluster_spatial, metric="euclidean")
        np.fill_diagonal(dmat, np.inf)
        dist_sum += float(np.min(dmat, axis=1).sum())
    return dist_sum / float(spatial_arr.shape[0])


def chaos(
    labels: np.ndarray,
    spatial: np.ndarray,
    standardize: bool = True,
    use_knn: bool = True,
) -> float:
    """Compute CHAOS score for spatial clustering continuity.

    Parameters
    ----------
    labels
        Cluster labels of shape ``(n_cells,)``.
    spatial
        Spatial coordinates of shape ``(n_cells, n_dims)``.
    standardize
        Whether to z-score spatial coordinates before computing distances.
    use_knn
        If True (default), use per-cluster ``cKDTree`` 1-NN instead of dense ``cdist``.

    Returns
    -------
    float
        CHAOS score. Lower values indicate tighter within-cluster spatial continuity.
    """
    spatial_arr = _prepare_spatial_coords(spatial, standardize=standardize)
    labels_arr = _as_1d_array(labels, n_cells=spatial_arr.shape[0])

    if use_knn:
        return _chaos_kdtree(labels_arr, spatial_arr)
    return _chaos_cdist(labels_arr, spatial_arr)


def spatial_cluster_labels_kmeans(
    X: np.ndarray,
    n_clusters: int,
    seed: int = 42,
) -> np.ndarray:
    """Assign spatial-domain clusters with k-means on an embedding."""
    X = check_array(X, accept_sparse=False, ensure_2d=True)
    return cluster_labels_kmeans(X, n_clusters, seed=seed)


def spatial_cluster_labels_leiden(
    neighbors: NeighborsResults,
    target_n_clusters: int | None = None,
    optimize_resolution: bool = True,
    resolution: float = 1.0,
    seed: int = 42,
    n_jobs: int = 1,
) -> np.ndarray:
    """Assign spatial-domain clusters with Leiden on a kNN graph."""
    return cluster_labels_leiden(
        neighbors,
        target_n_clusters=target_n_clusters,
        optimize_resolution=optimize_resolution,
        resolution=resolution,
        seed=seed,
        n_jobs=n_jobs,
    )


def spatial_cluster_labels_from_spatial_coords(
    spatial: np.ndarray,
    n_clusters: int | None = None,
    n_neighbors: int = 15,
    optimize_resolution: bool = True,
    resolution: float = 1.0,
    seed: int = 42,
    n_jobs: int = 1,
) -> np.ndarray:
    """Assign clusters with Leiden on a spatial-coordinate kNN graph."""
    spatial_arr = _as_2d_float_array(spatial)
    neighbors = neighbors_from_spatial(spatial_arr, n_neighbors=n_neighbors, n_jobs=n_jobs)
    return cluster_labels_leiden(
        neighbors,
        target_n_clusters=n_clusters,
        optimize_resolution=optimize_resolution,
        resolution=resolution,
        seed=seed,
        n_jobs=n_jobs,
    )
