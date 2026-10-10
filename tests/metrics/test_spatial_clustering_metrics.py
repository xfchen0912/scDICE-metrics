import numpy as np
import pytest
from scipy.spatial import distance_matrix
from sklearn.preprocessing import StandardScaler

from scdice_metrics.benchmark._spatial_prepare import build_spatial_neighbors
from scdice_metrics.metrics import chaos, pas, pas_from_neighbors


def test_chaos_prefers_spatially_compact_clusters():
    labels = np.array([0, 0, 0, 1, 1, 1])
    compact = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ]
    )
    mixed = np.array(
        [
            [0.0, 0.0],
            [10.0, 10.0],
            [0.1, 0.0],
            [10.1, 10.0],
            [0.0, 0.1],
            [10.0, 10.1],
        ]
    )

    chaos_compact = chaos(labels, compact)
    chaos_mixed = chaos(labels, mixed)

    assert chaos_compact < chaos_mixed


def test_chaos_skips_clusters_with_two_or_fewer_cells():
    labels = np.array(["A", "A", "B"], dtype=object)
    spatial = np.array([[0.0, 0.0], [1.0, 0.0], [100.0, 100.0]])
    assert chaos(labels, spatial) == 0.0


def test_pas_range_and_sanity():
    labels = np.array([0, 0, 1, 1, 1, 0])
    spatial = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.2, 10.0],
            [0.2, 0.0],
        ]
    )

    score = pas(labels, spatial, k=3)
    assert 0.0 <= score <= 1.0


def test_pas_reuses_precomputed_neighbors():
    labels = np.array([0, 0, 1, 1, 1, 0])
    spatial = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.2, 10.0],
            [0.2, 0.0],
        ]
    )
    k = 3
    neighbors_k = max(k + 1, 4)
    from anndata import AnnData

    ad = AnnData(X=np.zeros((spatial.shape[0], 1)))
    ad.obsm["spatial"] = spatial
    spatial_neighbors = build_spatial_neighbors(ad, "spatial", n_neighbors=neighbors_k)

    cdist_score = pas(labels, spatial, k=k, reuse_spatial_neighbors=False)
    neighbor_score = pas(
        labels,
        spatial,
        k=k,
        spatial_neighbors=spatial_neighbors,
        reuse_spatial_neighbors=True,
    )
    direct_score = pas_from_neighbors(labels, spatial_neighbors, k=k)

    assert cdist_score == neighbor_score == direct_score


def _sdmbench_fx_kNN(i, location_in, k, cluster_in):
    location_in = np.array(location_in)
    cluster_in = np.array(cluster_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    ind = np.argsort(dist_array)[:k]
    cluster_use = np.array(cluster_in)
    if np.sum(cluster_use[ind] != cluster_in[i]) > (k / 2):
        return 1
    return 0


def _sdmbench_fx_1NN(i, location_in):
    location_in = np.array(location_in)
    dist_array = distance_matrix(location_in[i, :][None, :], location_in)[0, :]
    dist_array[i] = np.inf
    return np.min(dist_array)


def _sdmbench_compute_PAS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = location
    results = [
        _sdmbench_fx_kNN(i, matched_location, k=10, cluster_in=clusterlabel)
        for i in range(matched_location.shape[0])
    ]
    return np.sum(results) / len(clusterlabel)


def _sdmbench_compute_CHAOS(clusterlabel, location):
    clusterlabel = np.array(clusterlabel)
    location = np.array(location)
    matched_location = StandardScaler().fit_transform(location)
    clusterlabel_unique = np.unique(clusterlabel)
    dist_val = np.zeros(len(clusterlabel_unique))
    count = 0
    for k in clusterlabel_unique:
        location_cluster = matched_location[clusterlabel == k, :]
        if len(location_cluster) <= 2:
            continue
        n_location_cluster = len(location_cluster)
        results = [_sdmbench_fx_1NN(i, location_cluster) for i in range(n_location_cluster)]
        dist_val[count] = np.sum(results)
        count = count + 1
    return np.sum(dist_val) / len(clusterlabel)


def test_pas_matches_sdmbench_exact_knn():
    rng = np.random.default_rng(0)
    spatial = rng.normal(size=(80, 2)) * np.array([8.0, 0.3])
    labels = np.array(["A"] * 40 + ["B"] * 40, dtype=object)
    flip = rng.random(80) < 0.25
    labels[flip] = np.where(labels[flip] == "A", "B", "A")
    expected = _sdmbench_compute_PAS(labels, spatial)
    assert pas(labels, spatial) == pytest.approx(expected)
    assert pas(labels, spatial, k=10, standardize=False) == pytest.approx(expected)
    assert pas(labels, spatial, k=10, standardize=True) != pytest.approx(expected)


def test_pas_small_n_keeps_sdmbench_fixed_k_threshold():
    labels = np.array([0, 0, 0, 1, 1, 1])
    spatial = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ]
    )
    expected = _sdmbench_compute_PAS(labels, spatial)
    assert expected == 0.0
    assert pas(labels, spatial, k=10) == pytest.approx(expected)


def test_chaos_matches_sdmbench_scaler_and_skip():
    rng = np.random.default_rng(1)
    spatial = rng.normal(size=(60, 2))
    labels = np.array([f"D{i // 15}" for i in range(60)], dtype=object)
    labels[0] = labels[1] = "tiny"
    expected = _sdmbench_compute_CHAOS(labels, spatial)
    assert chaos(labels, spatial) == pytest.approx(expected)
    assert chaos(labels, spatial, use_knn=False) == pytest.approx(expected)
