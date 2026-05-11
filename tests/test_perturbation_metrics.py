import anndata as ad
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import scdice_metrics
from scdice_metrics.metrics._perturbation import (
    average_of_perturbation_centroids,
    calculate_centroid_accuracies,
    get_perts,
    pearson_delta_reference_metrics,
    score_centroids,
)

scdice_metrics.settings.jax_fix_no_kernel_image()


def test_pearson_delta_reference_metrics_all_genes():
    rng = np.random.default_rng(0)
    n = 50
    reference = rng.random(n)
    delta_true = rng.random(n)
    delta_pred = 2.0 * delta_true + 0.1 * rng.standard_normal(n)
    X_true = delta_true + reference
    X_pred = delta_pred + reference
    out = pearson_delta_reference_metrics(X_true, X_pred, reference)
    assert set(out.keys()) == {"corr_all_allpert"}
    exp, _ = stats.pearsonr(delta_true, delta_pred)
    assert np.isclose(out["corr_all_allpert"], exp, atol=1e-6)


def test_pearson_delta_reference_metrics_top20():
    rng = np.random.default_rng(1)
    n = 30
    reference = rng.random(n)
    delta_true = rng.random(n)
    delta_pred = delta_true + 0.05 * rng.standard_normal(n)
    X_true = delta_true + reference
    X_pred = delta_pred + reference
    top20 = np.arange(5, 25)
    out = pearson_delta_reference_metrics(X_true, X_pred, reference, top20_de_idxs=top20)
    assert "corr_20de_allpert" in out
    sub_t = (X_true - reference)[top20]
    sub_p = (X_pred - reference)[top20]
    exp20, _ = stats.pearsonr(sub_t, sub_p)
    assert np.isclose(out["corr_20de_allpert"], exp20, atol=1e-6)


def test_pearson_delta_reference_metrics_shape_error():
    with pytest.raises(ValueError, match="same shape"):
        pearson_delta_reference_metrics(np.ones(3), np.ones(3), np.ones(4))


def test_calculate_centroid_accuracies():
    rng = np.random.default_rng(42)
    genes = ["G1", "G2", "G3", "G4"]
    perts = ["P1", "P2", "P3"]
    methods = ["M1", "M2"]
    agg = pd.DataFrame(rng.random((len(perts) * len(methods), len(genes))), columns=genes)
    agg.index = pd.MultiIndex.from_product([perts, methods], names=["condition", "method"])
    post_gt = pd.DataFrame(rng.random((len(perts), len(genes))), index=perts, columns=genes)
    scores_df = calculate_centroid_accuracies(agg, post_gt)
    assert scores_df.shape == (len(perts), len(methods))
    assert ((scores_df.values >= 0) & (scores_df.values <= 1)).all()


def test_average_of_perturbation_centroids():
    n_cells = 6
    n_genes = 4
    X = np.arange(n_cells * n_genes, dtype=float).reshape(n_cells, n_genes)
    obs = pd.DataFrame(
        {
            "control": [1, 1, 0, 0, 0, 0],
            "condition": ["ctrl", "ctrl", "A", "A", "B", "B"],
        }
    )
    adata = ad.AnnData(X=X, obs=obs)
    ref = average_of_perturbation_centroids(adata, control_key="control", condition_key="condition")
    mean_a = X[2:4].mean(axis=0)
    mean_b = X[4:6].mean(axis=0)
    expected = (mean_a + mean_b) / 2.0
    assert np.allclose(ref, expected)


def test_average_of_perturbation_centroids_control_is_zero():
    n_genes = 3
    X = np.ones((4, n_genes))
    X[2:] *= 3.0
    obs = pd.DataFrame({"control": [0, 0, 1, 1], "condition": ["c", "c", "A", "B"]})
    adata = ad.AnnData(X=X, obs=obs)
    ref = average_of_perturbation_centroids(
        adata, control_key="control", condition_key="condition", control_is_one=False
    )
    assert np.allclose(ref, np.ones(n_genes) * 3.0)


def test_get_perts():
    test_perts = np.array(["P1", "P2", "P3", "P6"])
    phenotypes = {"A": ["P1", "P2"], "B": ["P3", "P4", "P5"], "C": ["P6", "P7"]}
    out = get_perts(test_perts, phenotypes, ["A", "B"])
    assert np.array_equal(out, np.array(["P1", "P2", "P3"]))


def test_score_centroids_shapes_and_labels():
    genes = ["g1", "g2"]
    perts = ["P1", "P2", "P3"]
    methods = ["M1", "M2"]
    rng = np.random.default_rng(0)
    post_gt = pd.DataFrame(rng.random((len(perts), len(genes))), index=perts, columns=genes)
    pred = pd.DataFrame(
        rng.random((len(perts) * len(methods), len(genes))),
        index=pd.MultiIndex.from_product([perts, methods], names=["condition", "method"]),
        columns=genes,
    )
    perts_dict = {"C1": ["P1", "P2"], "C2": ["P3"]}
    labels, scores_dict = score_centroids(post_gt, pred, perts_dict, methods)
    assert labels.shape == (3, 2)
    assert np.array_equal(labels[:, 0], [1, 1, 0])
    assert np.array_equal(labels[:, 1], [0, 0, 1])
    assert set(scores_dict.keys()) == {"M1", "M2"}
    assert scores_dict["M1"].shape == (3, 2)
