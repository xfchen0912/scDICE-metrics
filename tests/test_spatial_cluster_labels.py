import numpy as np
import pytest

from scdice_metrics.benchmark import Benchmarker, SpatialClustering, SpatialClusteringPrepare
from scdice_metrics.benchmark._spatial_prepare import compute_spatial_cluster_labels
from scdice_metrics.metrics import spatial_cluster_labels_kmeans, spatial_cluster_labels_leiden
from scdice_metrics.metrics._clustering import (
    explain_cluster_order,
    match_cluster_labels_to_ground_truth,
    reorder_cluster_labels,
    resolve_effective_cluster_order,
)
from scdice_metrics.nearest_neighbors import pynndescent
from tests.utils.data import dummy_benchmarker_adata


def test_reorder_cluster_labels_by_cell_count():
    labels = np.array([0, 0, 1, 1, 1, 2, 2])
    reordered = reorder_cluster_labels(labels, order="cell_count")
    expected = np.array([1, 1, 0, 0, 0, 2, 2])
    np.testing.assert_array_equal(reordered, expected)


def test_match_cluster_labels_to_ground_truth_by_count():
    gt = np.array(["A", "A", "A", "B", "B", "C", "C", "C", "C"])
    pred = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    matched, categories = match_cluster_labels_to_ground_truth(
        pred,
        gt,
        label_style="ground_truth_name",
    )
    assert list(matched) == ["A", "A", "A", "B", "B", "C", "C", "C", "C"]
    assert categories == ["C", "A", "B"]


def test_match_cluster_labels_to_ground_truth_index_style():
    gt = np.array(["A", "A", "A", "B", "B", "C", "C", "C", "C"])
    pred = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
    matched, categories = match_cluster_labels_to_ground_truth(
        pred,
        gt,
        label_style="index",
    )
    np.testing.assert_array_equal(matched, np.array([1, 1, 1, 2, 2, 0, 0, 0, 0]))
    assert categories == ["0", "1", "2"]


def test_apply_cluster_order_ground_truth_fallback():
    labels = np.array([0, 0, 1, 1, 1, 2, 2])
    ordered, categories = apply_cluster_order(labels, order="ground_truth", ground_truth=None)
    np.testing.assert_array_equal(ordered, reorder_cluster_labels(labels, order="cell_count"))
    assert categories is not None


def test_reorder_cluster_labels_custom_order():
    labels = np.array([0, 0, 1, 1, 1, 2, 2])
    reordered = reorder_cluster_labels(labels, order=[2, 0, 1])
    expected = np.array([1, 1, 2, 2, 2, 0, 0])
    np.testing.assert_array_equal(reordered, expected)


def test_reorder_cluster_labels_custom_order_unknown_label():
    labels = np.array([0, 1, 1, 2])
    with pytest.raises(ValueError, match="unknown label"):
        reorder_cluster_labels(labels, order=[3, 0, 1])


def test_reorder_cluster_labels_none():
    labels = np.array([3, 1, 3, 0])
    reordered = reorder_cluster_labels(labels, order="none")
    np.testing.assert_array_equal(reordered, labels)


def test_explain_cluster_order_ground_truth_fallback():
    labels = np.array([0, 0, 1, 1, 1, 2, 2])
    ordered = reorder_cluster_labels(labels, order="cell_count")
    lines = explain_cluster_order(labels, ordered, order="ground_truth", ground_truth=None)
    assert any("effective cluster_order: 'cell_count'" in line for line in lines)


def test_resolve_effective_cluster_order():
    assert resolve_effective_cluster_order("ground_truth", None) == "cell_count"
    assert resolve_effective_cluster_order("ground_truth", np.array([0, 1, 1])) == "ground_truth"
    assert resolve_effective_cluster_order([2, 0, 1], np.array([0, 1])) == "custom"


def test_spatial_cluster_labels_kmeans_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 5))
    labels = spatial_cluster_labels_kmeans(X, n_clusters=3)
    assert labels.shape == (40,)
    assert len(np.unique(labels)) <= 3


def test_spatial_cluster_labels_leiden_target_k():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 8))
    neighbors = pynndescent(X, n_neighbors=10, random_state=0, n_jobs=1)
    labels = spatial_cluster_labels_leiden(
        neighbors,
        target_n_clusters=4,
        optimize_resolution=True,
        n_jobs=1,
    )
    assert labels.shape == (60,)


def test_benchmarker_auto_spatial_cluster_per_embedding():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    ad.obsm["spatial"] = ad.X[:, :2]
    emb_keys = emb_keys[:2]

    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_clustering_metrics=SpatialClustering(hom=True, com=True, chaos=True, pas=True),
        spatial_clustering_prepare=SpatialClusteringPrepare(
            cluster_method="kmeans",
            target_n_clusters=3,
            optimize_resolution=False,
            log_prepare=False,
        ),
        compute_neighbors=False,
        progress_bar=False,
    )
    bm.prepare_spatial_clusters()
    key_a = bm.spatial_cluster_obs_keys[emb_keys[0]]
    key_b = bm.spatial_cluster_obs_keys[emb_keys[1]]
    assert key_a != key_b
    assert key_a in ad.obs
    assert key_b in ad.obs
    assert str(ad.obs[key_a].dtype) == "category"
    assert list(ad.obs[labels_key].cat.categories) == list(ad.obs[key_a].cat.categories)

    bm.benchmark()
    results = bm.get_results(clean_names=False, display_template="sdmbench")
    assert "hom" in results.columns
    assert "Total" in results.columns


def test_spatial_clustering_prepare_kmeans_with_custom_order():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    ad.obsm["spatial"] = ad.X[:, :2]
    emb_key = emb_keys[0]

    raw_cfg = SpatialClusteringPrepare(
        cluster_method="kmeans",
        target_n_clusters=3,
        cluster_order="none",
        log_prepare=False,
    )
    raw_labels = compute_spatial_cluster_labels(
        ad,
        emb_key,
        raw_cfg,
        spatial_label_key=labels_key,
        spatial_obsm_key="spatial",
    )
    custom_cfg = SpatialClusteringPrepare(
        cluster_method="kmeans",
        target_n_clusters=3,
        cluster_order=[int(x) for x in np.unique(raw_labels)[::-1]],
    )
    ordered_labels, _ = apply_cluster_order(
        raw_labels,
        order=custom_cfg.cluster_order,
        ground_truth=ad.obs[labels_key].to_numpy(),
        label_style=custom_cfg.cluster_label_style,
    )
    assert np.array_equal(
        raw_labels[raw_labels == raw_labels[0]],
        ordered_labels[ordered_labels == ordered_labels[0]],
    )


def test_benchmarker_attaches_spatial_neighbors_for_pas():
    from scdice_metrics.benchmark._core import _SPATIAL_NEIGHBORS

    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    ad.obsm["spatial"] = ad.X[:, :2]
    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys[:1],
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_clustering_metrics=SpatialClustering(pas=True, chaos=False, hom=False, com=False),
        spatial_clustering_prepare=SpatialClusteringPrepare(
            reuse_spatial_neighbors_for_pas=True,
            build_spatial_neighbors=True,
            log_prepare=False,
        ),
        spatial_obsm_key="spatial",
        compute_neighbors=False,
        progress_bar=False,
    )
    bm.prepare()
    assert bm._spatial_neighbors is not None
    emb_ad = bm._emb_adatas[emb_keys[0]]
    assert _SPATIAL_NEIGHBORS in emb_ad.uns
    assert emb_ad.uns[_SPATIAL_NEIGHBORS] is bm._spatial_neighbors
