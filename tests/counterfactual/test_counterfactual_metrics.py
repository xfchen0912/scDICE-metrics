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
    delta_spearman,
    energy_distance,
    mean_gene_wasserstein,
    mmd_rbf,
    pseudobulk_log1p_delta_metrics,
    pseudobulk_log1p_reference_delta_metrics,
    pseudobulk_pearson,
    pseudobulk_rmse,
    signed_de_recovery,
    systema_pearson_delta_metrics,
    systema_reference_delta_metrics,
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


def test_systema_reference_delta_without_template_matches_control_referenced():
    rng = np.random.default_rng(11)
    observed, predicted, reference = _make_swap_data(rng, n_cells=1)

    with_none = systema_reference_delta_metrics(observed, predicted, reference, template=None, top_k=10)
    plain = systema_pearson_delta_metrics(observed, predicted, reference, top_k=10)

    assert np.isclose(with_none["all_genes"], plain["all_genes"])
    assert np.isclose(with_none["top10_true_effect"], plain["top10_true_effect"])


def test_systema_reference_delta_removes_the_shared_template():
    """Subtracting the template recovers the response correlation that the template masks."""
    rng = np.random.default_rng(12)
    n_genes = 40
    template = rng.normal(scale=5.0, size=n_genes)          # dominant shared/systematic part
    specific_true = rng.normal(scale=0.3, size=n_genes)
    specific_pred = specific_true + rng.normal(scale=0.3, size=n_genes)

    reference = rng.random(n_genes)
    delta_true = template + specific_true
    delta_pred = template + specific_pred

    observed = (reference + delta_true)[None, :]
    predicted = (reference + delta_pred)[None, :]
    ref = reference[None, :]

    raw = systema_pearson_delta_metrics(observed, predicted, ref, top_k=40)["all_genes"]
    removed = systema_reference_delta_metrics(
        observed, predicted, ref, template=template, top_k=40
    )["all_genes"]
    expected_specific = float(stats.pearsonr(specific_true, specific_pred)[0])

    # the shared template alone would look like near-perfect agreement ...
    assert raw > 0.99
    # ... but the perturbation-specific agreement is much lower, and that is what the
    # template-removed metric reports
    assert np.isclose(removed, expected_specific, atol=1e-9)
    assert removed < raw - 0.05


def test_systema_reference_delta_validates_template_length():
    rng = np.random.default_rng(13)
    observed, predicted, reference = _make_swap_data(rng, n_cells=1)
    with pytest.raises(ValueError, match="template has"):
        systema_reference_delta_metrics(
            observed, predicted, reference, template=np.zeros(3), top_k=5
        )


def _sparse_like(rng, n_cells, profile, *, drop_rate=0.8, noise=1.0):
    """Draw cells whose per-cell values are noisy/sparse around ``profile`` (CPM space)."""
    cells = np.broadcast_to(profile, (n_cells, profile.size)).astype(float).copy()
    cells += rng.normal(scale=noise, size=cells.shape)
    mask = rng.random(cells.shape) < drop_rate
    cells[mask] = 0.0
    return np.log1p(np.clip(cells, 0.0, None))


def test_pseudobulk_log1p_delta_is_the_consistent_aggregation():
    """Both sides are aggregated the same way: log1p(mean_i expm1(x_i)) == log1p(mean CPM)."""
    rng = np.random.default_rng(21)
    n_genes = 25
    ref_profile = rng.uniform(1.0, 20.0, size=n_genes)
    true_profile = ref_profile + rng.normal(scale=2.0, size=n_genes)
    pred_profile = true_profile + rng.normal(scale=0.1, size=n_genes)

    reference = _sparse_like(rng, 60, ref_profile)
    observed = _sparse_like(rng, 60, true_profile)
    predicted = _sparse_like(rng, 60, pred_profile)

    out = pseudobulk_log1p_delta_metrics(observed, predicted, reference, top_k=10)
    assert set(out) == {"spearman", "pearson", "pearson_top10", "rmse"}

    pb = lambda x: np.log1p(np.expm1(x).mean(axis=0))  # noqa: E731
    delta_true = pb(observed) - pb(reference)
    delta_pred = pb(predicted) - pb(reference)
    assert np.isclose(out["pearson"], float(stats.pearsonr(delta_true, delta_pred)[0]))
    assert np.isclose(out["spearman"], float(stats.spearmanr(delta_true, delta_pred).correlation))
    top = np.argsort(-np.abs(delta_true))[:10]
    assert np.isclose(out["pearson_top10"], float(stats.pearsonr(delta_true[top], delta_pred[top])[0]))
    # the reference cancels, so the rmse is a plain profile distance (reference-insensitive)
    expected_rmse = float(np.sqrt(np.mean((pb(observed) - pb(predicted)) ** 2)))
    assert np.isclose(out["rmse"], expected_rmse)


def test_pseudobulk_log1p_delta_fixes_the_jensen_gap_that_penalises_smooth_predictions():
    """A smooth predictor is judged on its response, not on the sparsity it cannot emit."""
    rng = np.random.default_rng(22)
    n_genes = 40
    ref_profile = rng.uniform(5.0, 40.0, size=n_genes)
    delta = rng.normal(scale=2.0, size=n_genes)
    true_profile = ref_profile + delta

    reference = _sparse_like(rng, 400, ref_profile, drop_rate=0.5, noise=0.3)
    observed = _sparse_like(rng, 400, true_profile, drop_rate=0.5, noise=0.3)
    # a "generative" predictor: every cell gets the same smooth expected profile
    predicted = np.broadcast_to(np.log1p(true_profile), (400, n_genes)).copy()

    per_cell = delta_spearman(observed, predicted, reference)
    consistent = pseudobulk_log1p_delta_metrics(observed, predicted, reference)["spearman"]

    # the Jensen gap alone destroys the per-cell-log1p delta even for a perfect response;
    # the residual gap to 1.0 is the finite-sample noise of the observed profile itself
    assert consistent > 0.8
    assert consistent > per_cell + 0.3


def test_pseudobulk_log1p_delta_accepts_profile_vectors_and_validates_top_k():
    rng = np.random.default_rng(23)
    reference = rng.random(10)
    observed = reference + 0.5
    predicted = reference + 0.5
    out = pseudobulk_log1p_delta_metrics(observed, predicted, reference)
    assert np.isclose(out["pearson"], 1.0)
    assert np.isclose(out["rmse"], 0.0)
    with pytest.raises(ValueError, match="top_k"):
        pseudobulk_log1p_delta_metrics(observed, predicted, reference, top_k=0)


def test_pseudobulk_log1p_reference_delta_without_template_matches_the_control_reference():
    """The 2x2's missing cell degenerates to the control-referenced PB metric."""
    rng = np.random.default_rng(31)
    observed, predicted, reference = _make_swap_data(rng, n_cells=25, n_genes=18)

    plain = pseudobulk_log1p_delta_metrics(observed, predicted, reference, top_k=10)
    removed = pseudobulk_log1p_reference_delta_metrics(
        observed, predicted, reference, template=None, top_k=10
    )

    assert set(removed) == {
        "spearman",
        "pearson",
        "pearson_top10_true_effect",
        "pearson_top10_specific_effect",
        "rmse",
    }
    assert np.isclose(removed["spearman"], plain["spearman"])
    assert np.isclose(removed["pearson"], plain["pearson"])
    assert np.isclose(removed["pearson_top10_true_effect"], plain["pearson_top10"])
    assert np.isclose(removed["rmse"], plain["rmse"])


def test_pseudobulk_log1p_reference_delta_removes_the_shared_template():
    """Subtracting the template exposes the specific agreement the template was masking."""
    rng = np.random.default_rng(32)
    n_genes = 40
    template = rng.normal(scale=5.0, size=n_genes)          # dominant shared response
    specific_true = rng.normal(scale=0.3, size=n_genes)
    specific_pred = specific_true + rng.normal(scale=0.3, size=n_genes)

    reference = rng.random(n_genes)
    observed = reference + template + specific_true
    predicted = reference + template + specific_pred

    raw = pseudobulk_log1p_delta_metrics(observed, predicted, reference, top_k=n_genes)["spearman"]
    removed = pseudobulk_log1p_reference_delta_metrics(
        observed, predicted, reference, template=template, top_k=n_genes
    )

    # the shared template alone looks like near-perfect agreement ...
    assert raw > 0.95
    # ... while the specific agreement is much weaker, and is what the shifted reference reports
    expected_specific = float(stats.spearmanr(specific_true, specific_pred).correlation)
    assert np.isclose(removed["spearman"], expected_specific, atol=1e-9)
    assert removed["spearman"] < raw - 0.05
    # the reference cancels out of the RMSE, so the two references agree exactly
    assert np.isclose(
        removed["rmse"],
        pseudobulk_log1p_delta_metrics(observed, predicted, reference)["rmse"],
    )


def test_pseudobulk_log1p_reference_delta_validates_template_length():
    rng = np.random.default_rng(32)
    observed, predicted, reference = _make_swap_data(rng, n_cells=1)
    with pytest.raises(ValueError, match="template has"):
        pseudobulk_log1p_reference_delta_metrics(
            observed, predicted, reference, template=np.zeros(3), top_k=5
        )
    with pytest.raises(ValueError, match="top_k"):
        pseudobulk_log1p_reference_delta_metrics(observed, predicted, reference, top_k=0)


def test_pseudobulk_log1p_reference_delta_selects_genes_before_and_after_the_shift():
    """`true_effect` must use the un-shifted |delta|; `specific_effect` the shifted one."""
    rng = np.random.default_rng(33)
    n_genes = 50
    reference_profile = rng.uniform(5.0, 40.0, size=n_genes)
    template = rng.normal(scale=5.0, size=n_genes)          # dominates the un-shifted |delta|
    specific = rng.normal(scale=0.3, size=n_genes)

    reference = _sparse_like(rng, 60, reference_profile)
    observed = _sparse_like(rng, 60, reference_profile + template + specific)
    predicted = np.broadcast_to(
        np.log1p(reference_profile + template + specific), (60, n_genes)
    ).copy()

    out = pseudobulk_log1p_reference_delta_metrics(
        observed, predicted, reference, template=template, top_k=10
    )
    assert not np.isclose(
        out["pearson_top10_true_effect"], out["pearson_top10_specific_effect"]
    )

    pb = lambda x: np.log1p(np.expm1(x).mean(0))  # noqa: E731
    obs_pb, pred_pb, ref_pb = pb(observed), pb(predicted), pb(reference)
    shifted = obs_pb - ref_pb - template
    for key, idx in (
        ("true_effect", np.argsort(-np.abs(obs_pb - ref_pb))[:10]),
        ("specific_effect", np.argsort(-np.abs(shifted))[:10]),
    ):
        np.testing.assert_allclose(
            out[f"pearson_top10_{key}"],
            float(stats.pearsonr(shifted[idx], (pred_pb - ref_pb - template)[idx])[0]),
            atol=1e-9,
        )


def test_systema_reference_delta_reports_specific_gene_selection():
    """Top-k by the specific effect is reported alongside the raw-DE selection."""
    rng = np.random.default_rng(14)
    n_genes = 60
    template = rng.normal(scale=5.0, size=n_genes)          # dominates the raw |delta|
    specific_true = rng.normal(scale=0.3, size=n_genes)
    specific_pred = specific_true + rng.normal(scale=0.3, size=n_genes)
    reference = rng.random(n_genes)

    observed = (reference + template + specific_true)[None, :]
    predicted = (reference + template + specific_pred)[None, :]
    ref = reference[None, :]

    out = systema_reference_delta_metrics(observed, predicted, ref, template=template, top_k=15)
    assert set(out) == {"all_genes", "top15_true_effect", "top15_specific_effect"}

    # The raw top-k list is template-dominated, so after removing the template the
    # residual agreement is much weaker than on the specific-effect gene list.
    assert out["top15_specific_effect"] > out["top15_true_effect"] + 0.05
    # ... and equals the plain Pearson between the true and predicted specific parts.
    top = np.argsort(-np.abs(specific_true))[:15]
    np.testing.assert_allclose(
        out["top15_specific_effect"],
        float(stats.pearsonr(specific_true[top], specific_pred[top])[0]),
        atol=1e-9,
    )
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
