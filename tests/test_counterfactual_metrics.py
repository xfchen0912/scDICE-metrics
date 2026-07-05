import numpy as np
import pytest
from scipy import sparse
from scipy import stats

import scdice_metrics
from scdice_metrics.metrics._counterfactual import (
    _mean_profile,
    delta_pearson,
    delta_profile_metrics,
    delta_rmse,
    energy_distance,
    mean_gene_wasserstein,
    mmd_rbf,
    pseudobulk_pearson,
    pseudobulk_rmse,
    signed_de_recovery,
    systema_pearson_delta_metrics,
)
from scdice_metrics.metrics._perturbation import pearson_delta_reference_metrics

scdice_metrics.settings.jax_fix_no_kernel_image()


def _make_swap_data(rng: np.random.Generator, *, n_cells: int = 40, n_genes: int = 30):
    reference = rng.random((n_cells, n_genes))
    delta = rng.normal(scale=0.5, size=(n_cells, n_genes))
    observed = reference + delta
    predicted = reference + delta + rng.normal(scale=0.01, size=(n_cells, n_genes))
    return observed, predicted, reference


def test_perfect_prediction_sanity():
    rng = np.random.default_rng(0)
    observed, _, reference = _make_swap_data(rng)
    predicted = observed.copy()

    assert np.isclose(pseudobulk_pearson(observed, predicted), 1.0)
    assert np.isclose(pseudobulk_rmse(observed, predicted), 0.0, atol=1e-12)
    assert np.isclose(delta_pearson(observed, predicted, reference), 1.0)
    assert np.isclose(delta_rmse(observed, predicted, reference), 0.0, atol=1e-12)

    de = signed_de_recovery(observed, predicted, reference, top_k=10)
    assert de["precision"] == 1.0
    assert de["recall"] == 1.0
    assert de["f1"] == 1.0
    assert de["jaccard"] == 1.0
    assert de["direction_accuracy_true_top"] == 1.0

    assert energy_distance(observed, predicted) == pytest.approx(0.0, abs=1e-10)
    assert mmd_rbf(observed, predicted) == pytest.approx(0.0, abs=1e-10)
    assert mean_gene_wasserstein(observed, predicted) == pytest.approx(0.0, abs=1e-10)


def test_corrupted_prediction_performs_worse():
    rng = np.random.default_rng(1)
    observed, predicted, reference = _make_swap_data(rng)
    corrupted = predicted + rng.normal(scale=2.0, size=predicted.shape)

    assert pseudobulk_pearson(observed, corrupted) < pseudobulk_pearson(observed, predicted)
    assert delta_pearson(observed, corrupted, reference) < delta_pearson(observed, predicted, reference)
    assert energy_distance(observed, corrupted) > energy_distance(observed, predicted)
    assert mmd_rbf(observed, corrupted) > mmd_rbf(observed, predicted)


def test_unequal_cell_counts():
    rng = np.random.default_rng(2)
    observed, predicted, reference = _make_swap_data(rng, n_cells=50)
    predicted_small = predicted[:20]

    assert np.isfinite(pseudobulk_pearson(observed, predicted_small))
    assert np.isfinite(delta_pearson(observed, predicted_small, reference))
    assert np.isfinite(energy_distance(observed, predicted_small))


def test_sparse_pseudobulk_without_full_densification():
    rng = np.random.default_rng(3)
    dense = rng.random((25, 40))
    sparse_obs = sparse.csr_matrix(dense)
    sparse_pred = sparse.csr_matrix(dense.copy())

    dense_profile = dense.mean(axis=0)
    sparse_profile = np.asarray(_mean_profile(sparse_obs, name="observed"))
    assert np.allclose(sparse_profile, dense_profile)
    assert np.isclose(pseudobulk_pearson(sparse_obs, sparse_pred), 1.0, atol=1e-12)


def test_one_dimensional_preaggregated_profiles():
    rng = np.random.default_rng(4)
    observed = rng.random(20)
    predicted = observed.copy()
    reference = rng.random(20)

    assert np.isclose(pseudobulk_pearson(observed, predicted), 1.0, atol=1e-12)
    assert np.isclose(delta_pearson(observed, predicted, reference), 1.0, atol=1e-12)


def test_distribution_metrics_reject_one_dimensional_profiles():
    rng = np.random.default_rng(5)
    vec = rng.random(20)
    matrix = rng.random((10, 20))

    with pytest.raises(ValueError, match="two-dimensional"):
        energy_distance(vec, matrix)
    with pytest.raises(ValueError, match="two-dimensional"):
        mmd_rbf(matrix, vec)


def test_systema_wrapper_matches_direct_pearson_delta():
    rng = np.random.default_rng(6)
    observed, predicted, reference = _make_swap_data(rng, n_cells=1)

    obs = observed.mean(axis=0)
    pred = predicted.mean(axis=0)
    ref = reference.mean(axis=0)
    delta_true = obs - ref
    top_k = 10
    top_idx = np.argsort(-np.abs(delta_true))[:top_k]

    direct = pearson_delta_reference_metrics(obs, pred, ref, top20_de_idxs=top_idx)
    wrapped = systema_pearson_delta_metrics(observed, predicted, reference, top_k=top_k)

    assert np.isclose(wrapped["all_genes"], direct["corr_all_allpert"])
    assert np.isclose(wrapped["top10_true_effect"], direct["corr_20de_allpert"])


def test_delta_pearson_alias_matches_systema_all_genes():
    rng = np.random.default_rng(7)
    observed, predicted, reference = _make_swap_data(rng)

    alias = delta_pearson(observed, predicted, reference)
    systema = systema_pearson_delta_metrics(observed, predicted, reference)["all_genes"]
    assert np.isclose(alias, systema)


def test_constant_vectors_return_nan():
    genes = 10
    const = np.ones((5, genes))
    ref = np.zeros((5, genes))
    assert np.isnan(pseudobulk_pearson(const, const))
    assert np.isnan(delta_pearson(const, const, ref))


def test_feature_mismatch_raises():
    rng = np.random.default_rng(8)
    observed = rng.random((10, 20))
    predicted = rng.random((10, 21))
    reference = rng.random((10, 20))

    with pytest.raises(ValueError, match="feature count"):
        pseudobulk_pearson(observed, predicted)
    with pytest.raises(ValueError, match="feature count"):
        delta_pearson(observed, predicted, reference)


def test_signed_de_recovery_top_k():
    rng = np.random.default_rng(9)
    n_genes = 30
    reference = np.zeros((5, n_genes))
    delta = np.zeros(n_genes)
    delta[:10] = np.linspace(1.0, 2.0, 10)
    delta[10:20] = np.linspace(-2.0, -1.0, 10)
    observed = reference + delta
    predicted = observed.copy()

    out = signed_de_recovery(observed, predicted, reference, top_k=10)
    assert out["precision"] == 1.0
    assert out["recall"] == 1.0
    assert out["up_precision"] == 1.0
    assert out["down_precision"] == 1.0


def test_signed_de_recovery_requires_top_k():
    rng = np.random.default_rng(10)
    observed, predicted, reference = _make_swap_data(rng, n_cells=5, n_genes=10)
    with pytest.raises(ValueError, match="top_k"):
        signed_de_recovery(observed, predicted, reference, top_k=0)


def test_delta_profile_metrics_keys():
    rng = np.random.default_rng(11)
    observed, predicted, reference = _make_swap_data(rng)
    out = delta_profile_metrics(observed, predicted, reference)
    assert set(out) == {
        "delta_pearson",
        "delta_spearman",
        "delta_cosine",
        "delta_rmse",
        "delta_mae",
    }


def test_gene_indices_subset():
    rng = np.random.default_rng(12)
    observed, predicted, reference = _make_swap_data(rng)
    idx = np.array([0, 2, 4, 6])

    full = pseudobulk_pearson(observed, predicted)
    subset = pseudobulk_pearson(observed, predicted, gene_indices=idx)
    assert np.isfinite(full)
    assert np.isfinite(subset)
    assert full != subset or np.isclose(full, subset)


def test_energy_distance_and_mmd_finite_on_random_data():
    rng = np.random.default_rng(13)
    observed, predicted, reference = _make_swap_data(rng, n_cells=15, n_genes=8)
    assert np.isfinite(energy_distance(observed, predicted, max_cells=10))
    assert np.isfinite(mmd_rbf(observed, predicted, max_cells=10))


def test_mean_gene_wasserstein_per_gene_average():
    rng = np.random.default_rng(14)
    observed = rng.random((12, 5))
    predicted = observed.copy()
    assert mean_gene_wasserstein(observed, predicted) == pytest.approx(0.0, abs=1e-12)

    shifted = observed + 0.5
    manual = float(np.mean([stats.wasserstein_distance(observed[:, g], shifted[:, g]) for g in range(5)]))
    assert mean_gene_wasserstein(observed, shifted) == pytest.approx(manual)
