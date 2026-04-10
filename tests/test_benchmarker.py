import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from scdice_metrics.benchmark import BatchCorrection, Benchmarker, BioConservation, Disentanglement
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
