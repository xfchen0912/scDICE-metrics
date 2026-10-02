import numpy as np

from scdice_metrics.benchmark._spatial_prepare import build_spatial_neighbors
from scdice_metrics.metrics import chaos, pas, pas_from_neighbors


def test_chaos_prefers_spatially_compact_clusters():
    labels = np.array([0, 0, 1, 1])
    compact = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    mixed = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, 0.0], [10.1, 10.0]])

    chaos_compact = chaos(labels, compact)
    chaos_mixed = chaos(labels, mixed)

    assert chaos_compact < chaos_mixed


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
    neighbor_score = pas(labels, spatial, k=k, spatial_neighbors=spatial_neighbors)
    direct_score = pas_from_neighbors(labels, spatial_neighbors, k=k)

    assert cdist_score == neighbor_score == direct_score
