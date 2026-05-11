from __future__ import annotations

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


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


def chaos(labels: np.ndarray, spatial: np.ndarray, standardize: bool = True) -> float:
    """Compute CHAOS score for spatial clustering continuity.

    This is adapted from SDMBench's CHAOS implementation. For each cluster,
    it sums each point's nearest-neighbor distance within the same cluster,
    then normalizes by the total number of cells.

    Parameters
    ----------
    labels
        Cluster labels of shape ``(n_cells,)``.
    spatial
        Spatial coordinates of shape ``(n_cells, n_dims)``.
    standardize
        Whether to z-score spatial coordinates before computing distances.

    Returns
    -------
    float
        CHAOS score. Lower values indicate tighter within-cluster spatial continuity.
    """
    spatial_arr = _as_2d_float_array(spatial)
    labels_arr = _as_1d_array(labels, n_cells=spatial_arr.shape[0])

    if standardize:
        spatial_arr = StandardScaler().fit_transform(spatial_arr)

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


def pas(labels: np.ndarray, spatial: np.ndarray, k: int = 10) -> float:
    """Compute PAS (Percentage of Abnormal Spots) for spatial clustering.

    This follows SDMBench's PAS definition: for each spot, if more than half of
    its ``k`` nearest spatial neighbors have a different cluster label, the spot
    is considered abnormal. PAS is the proportion of such spots.

    Parameters
    ----------
    labels
        Cluster labels of shape ``(n_cells,)``.
    spatial
        Spatial coordinates of shape ``(n_cells, n_dims)``.
    k
        Number of nearest spatial neighbors used for abnormality decision.

    Returns
    -------
    float
        PAS score in ``[0, 1]``. Lower values indicate better spatial smoothness.
    """
    spatial_arr = _as_2d_float_array(spatial)
    labels_arr = _as_1d_array(labels, n_cells=spatial_arr.shape[0])

    n_cells = spatial_arr.shape[0]
    if n_cells <= 1:
        return 0.0
    if k < 1:
        raise ValueError("`k` must be >= 1.")
    k_eff = min(k, n_cells - 1)

    dmat = cdist(spatial_arr, spatial_arr, metric="euclidean")
    np.fill_diagonal(dmat, np.inf)
    nn_idx = np.argpartition(dmat, kth=k_eff - 1, axis=1)[:, :k_eff]

    mismatch = labels_arr[nn_idx] != labels_arr[:, None]
    abnormal = mismatch.sum(axis=1) > (k_eff / 2.0)
    return float(np.mean(abnormal))
