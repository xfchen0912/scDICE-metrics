import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from scdice_metrics.benchmark import (
    BatchCorrection,
    Benchmarker,
    BenchmarkTemplate,
    BioConservation,
    Disentanglement,
    SpatialClustering,
)
from scdice_metrics.nearest_neighbors import jax_approx_min_k
from tests.utils.data import dummy_benchmarker_adata, dummy_disentanglement_benchmarker_adata


def test_benchmarker():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys,
        batch_correction_metrics=BatchCorrection(),
        bio_conservation_metrics=BioConservation(),
    )
    bm.benchmark()
    results = bm.get_results()
    assert isinstance(results, pd.DataFrame)
    bm.plot_results_table()


def test_benchmarker_default():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys,
    )
    bm.benchmark()
    results = bm.get_results()
    assert isinstance(results, pd.DataFrame)
    bm.plot_results_table()


def test_benchmarker_custom_metric_booleans():
    bioc = BioConservation(
        isolated_labels=False, nmi_ari_cluster_labels_leiden=False, silhouette_label=False, clisi_knn=True
    )
    bc = BatchCorrection(kbet_per_label=False, graph_connectivity=False, ilisi_knn=True)
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(ad, batch_key, labels_key, emb_keys, batch_correction_metrics=bc, bio_conservation_metrics=bioc)
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    assert isinstance(results, pd.DataFrame)
    assert "isolated_labels" not in results.columns
    assert "nmi_ari_cluster_labels_leiden" not in results.columns
    assert "silhouette_label" not in results.columns
    assert "clisi_knn" in results.columns
    assert "kbet_per_label" not in results.columns
    assert "graph_connectivity" not in results.columns
    assert "ilisi_knn" in results.columns
    assert "bras" in results.columns


def test_benchmarker_custom_metric_callable():
    bioc = BioConservation(clisi_knn={"perplexity": 10})
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(
        ad, batch_key, labels_key, emb_keys, bio_conservation_metrics=bioc, batch_correction_metrics=BatchCorrection()
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    assert "clisi_knn" in results.columns


def test_benchmarker_custom_near_neighs():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=BioConservation(),
        batch_correction_metrics=BatchCorrection(),
    )
    bm.prepare(neighbor_computer=jax_approx_min_k)
    bm.benchmark()
    results = bm.get_results()
    assert isinstance(results, pd.DataFrame)
    bm.plot_results_table()


@pytest.mark.parametrize("solver", ["arpack", "randomized"])
def test_benchmarker_different_solvers(solver):
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    bm = Benchmarker(ad, batch_key, labels_key, emb_keys, solver=solver)
    bm.benchmark()
    results = bm.get_results()
    assert isinstance(results, pd.DataFrame)
    bm.plot_results_table()


def test_benchmarker_disentanglement_metrics():
    ad, emb_keys, batch_key, labels_key = dummy_disentanglement_benchmarker_adata()
    classifier = LogisticRegression(max_iter=200)
    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        disentanglement_metrics=Disentanglement(
            mig=True,
            mixed_ksg_mig={"k": 3},
            classifier_attribute_gap={"classifier": classifier, "cv_splits": 3},
            fairness_leakage={"classifier": classifier, "cv_splits": 3},
        ),
        disentanglement_factor_keys=["factor_a", "factor_b"],
        leakage_target_key="response",
        progress_bar=False,
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    clean_results = bm.get_results()
    assert isinstance(results, pd.DataFrame)
    assert "mig_score" in results.columns
    assert "mixed_ksg_mig_max_mig" in results.columns
    assert "classifier_attribute_gap_concat_gap" in results.columns
    assert "fairness_leakage_accuracy" in results.columns
    assert "Disentanglement" in results.columns
    assert "MIG" in clean_results.columns
    assert "Leakage accuracy" in clean_results.columns
    assert "Total" not in clean_results.columns
    assert clean_results.loc["X_good", "Disentanglement"] > clean_results.loc["X_bad", "Disentanglement"]


def test_benchmarker_spatial_clustering_metrics():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    ad.obsm["spatial"] = ad.X[:, :2]
    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys[:2],
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_clustering_metrics=SpatialClustering(hom=True, com=True),
        spatial_cluster_key=None,
        spatial_obsm_key="spatial",
        compute_neighbors=False,
        progress_bar=False,
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False, display_template="legacy")
    sdmbench_results = bm.get_results(clean_names=False, display_template="sdmbench")
    clean_results = bm.get_results(display_template="sdmbench")
    assert isinstance(results, pd.DataFrame)
    assert "hom" in results.columns
    assert "com" in results.columns
    assert "chaos" in results.columns
    assert "pas" in results.columns
    assert "Accuracy" in sdmbench_results.columns
    assert bm.spatial_cluster_obs_keys[emb_keys[0]] in ad.obs
    assert bm.spatial_cluster_obs_keys[emb_keys[1]] in ad.obs
    assert "Continuity" in sdmbench_results.columns
    assert "HOM" in clean_results.columns
    assert "COM" in clean_results.columns
    assert "CHAOS" in clean_results.columns
    assert "PAS" in clean_results.columns
    assert "Accuracy" in clean_results.columns
    assert "Continuity" in clean_results.columns


def test_benchmarker_spatial_and_bio_label_keys():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    ad.obs["domain"] = (ad.obs[labels_key].astype(int) % 3).astype(str)
    ad.obsm["spatial"] = ad.X[:, :2]

    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys[:1],
        bio_conservation_metrics=BioConservation(
            nmi_ari_cluster_labels_kmeans=True,
            nmi_ari_cluster_labels_leiden=False,
            isolated_labels=False,
            silhouette_label=False,
            clisi_knn=False,
        ),
        batch_correction_metrics=None,
        spatial_clustering_metrics=SpatialClustering(hom=True, com=True, chaos=False, pas=False),
        spatial_label_key="domain",
        spatial_cluster_key=None,
        spatial_obsm_key="spatial",
        compute_neighbors=True,
        progress_bar=False,
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    assert "nmi_ari_cluster_labels_kmeans_nmi" in results.columns
    assert "hom" in results.columns
    assert "com" in results.columns


def test_benchmarker_custom_display_template():
    ad, emb_keys, batch_key, labels_key = dummy_benchmarker_adata()
    ad.obsm["spatial"] = ad.X[:, :2]
    custom = BenchmarkTemplate(
        name="custom_spatial",
        groups={"Label match": ("hom", "com"), "Geometry": ("chaos", "pas")},
        weights={"Label match": 0.6, "Geometry": 0.4},
    )
    bm = Benchmarker(
        ad,
        batch_key,
        labels_key,
        emb_keys,
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_clustering_metrics=SpatialClustering(hom=True, com=True, chaos=True, pas=True),
        spatial_obsm_key="spatial",
        display_template=custom,
        compute_neighbors=False,
        progress_bar=False,
    )
    bm.benchmark()
    results = bm.get_results(clean_names=False)
    assert "Label match" in results.columns
    assert "Geometry" in results.columns
    assert "Total" in results.columns
