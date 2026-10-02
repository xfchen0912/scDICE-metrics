"""Tests for split-half perturbation decomposition (Milestone 4)."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest

from scdice_metrics.decomposition import (
    perturbation_decomposition,
    split_half_perturbation_decomposition,
)
from scdice_metrics.decomposition._split_half import _cross_half_signal_energy, _split_half_delta_tensors


def _make_noisy_context_adata(
    *,
    context_offset: float,
    true_deltas: dict[str, float],
    noise_scale: float,
    cells_per_group: int = 40,
    control: str = "non-targeting",
    n_genes: int = 8,
    seed: int = 0,
) -> ad.AnnData:
    rng = np.random.default_rng(seed)
    ctrl_base = np.arange(n_genes, dtype=float) + context_offset
    rows: list[np.ndarray] = []
    obs_pert: list[str] = []

    for _ in range(cells_per_group):
        rows.append(ctrl_base + rng.normal(0.0, noise_scale, size=n_genes))
        obs_pert.append(control)

    for pert, delta_scalar in true_deltas.items():
        delta = np.full(n_genes, delta_scalar, dtype=float)
        for _ in range(cells_per_group):
            rows.append(ctrl_base + delta + rng.normal(0.0, noise_scale, size=n_genes))
            obs_pert.append(pert)

    x = np.stack(rows, axis=0)
    adata = ad.AnnData(X=x)
    adata.obs["gene"] = obs_pert
    adata.obsm["X_hvg"] = x.copy()
    return adata


def _synthetic_adatas(noise_scale: float) -> tuple[list[ad.AnnData], list[str]]:
    adatas = [
        _make_noisy_context_adata(
            context_offset=0.0,
            true_deltas={"P1": 2.0, "P2": 0.0},
            noise_scale=noise_scale,
            seed=1,
        ),
        _make_noisy_context_adata(
            context_offset=10.0,
            true_deltas={"P1": 2.0, "P2": 0.0},
            noise_scale=noise_scale,
            seed=2,
        ),
    ]
    return adatas, ["ctx0", "ctx1"]


def test_cross_half_signal_recovers_pure_beta():
    n_a, n_b, n_g = 2, 2, 4
    beta = np.array([[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]])
    beta -= beta.mean(axis=0, keepdims=True)
    d = np.broadcast_to(beta[None, :, :], (n_a, n_b, n_g)).copy()

    signal = _cross_half_signal_energy(d, d)
    assert signal["global"] == pytest.approx(0.0, abs=1e-10)
    assert signal["context"] == pytest.approx(0.0, abs=1e-10)
    assert signal["perturbation"] > 0.0
    assert signal["context:perturbation"] == pytest.approx(0.0, abs=1e-10)


def test_split_half_uses_shared_control_mean():
    from scdice_metrics.decomposition._split_half import _ContextCellGroups

    groups = [
        _ContextCellGroups(
            control_mean=np.array([5.0, 5.0]),
            perturbation_cells={"P1": np.array([[6.0, 6.0], [8.0, 8.0]])},
        )
    ]
    rng = np.random.default_rng(0)
    d1, d2 = _split_half_delta_tensors(groups, ["P1"], rng)
    assert np.allclose(d1[0, 0], [1.0, 1.0])
    assert np.allclose(d2[0, 0], [3.0, 3.0])


def test_split_half_end_to_end_fractions_sum_to_one():
    adatas, context_names = _synthetic_adatas(noise_scale=0.05)
    result = split_half_perturbation_decomposition(
        adatas,
        context_names=context_names,
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
        n_splits=10,
        random_state=0,
    )

    assert "noise" in result.component_fraction
    assert sum(result.component_fraction.values()) == pytest.approx(1.0, abs=1e-6)
    assert result.n_splits == 10
    assert result.noise_fraction == pytest.approx(result.component_fraction["noise"])
    assert "noise" not in result.naive_component_fraction


def test_naive_vs_split_half_noise():
    adatas, context_names = _synthetic_adatas(noise_scale=0.5)
    naive = perturbation_decomposition(
        adatas,
        context_names=context_names,
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
    )
    split = split_half_perturbation_decomposition(
        adatas,
        context_names=context_names,
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
        n_splits=20,
        random_state=0,
    )

    assert split.component_fraction["noise"] > 0.0
    assert split.naive_component_fraction["perturbation"] >= split.component_fraction["perturbation"]


def test_noise_increases_with_injected_noise():
    low_adatas, low_contexts = _synthetic_adatas(0.02)
    high_adatas, high_contexts = _synthetic_adatas(1.0)
    low = split_half_perturbation_decomposition(
        low_adatas,
        context_names=low_contexts,
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
        n_splits=15,
        random_state=1,
    )
    high = split_half_perturbation_decomposition(
        high_adatas,
        context_names=high_contexts,
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
        n_splits=15,
        random_state=1,
    )
    assert high.component_fraction["noise"] > low.component_fraction["noise"]


def test_clip_negative_signal():
    adatas, context_names = _synthetic_adatas(noise_scale=0.2)
    result = split_half_perturbation_decomposition(
        adatas,
        context_names=context_names,
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
        n_splits=5,
        random_state=0,
        clip_negative_signal=True,
    )
    assert result.noise_energy >= 0.0
    assert all(value >= 0.0 for value in result.component_fraction.values())


def test_reproducible_with_random_state():
    adatas, context_names = _synthetic_adatas(noise_scale=0.1)
    kwargs = dict(
        context_names=context_names,
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
        n_splits=8,
        random_state=42,
    )
    r1 = split_half_perturbation_decomposition(adatas, **kwargs)
    r2 = split_half_perturbation_decomposition(adatas, **kwargs)
    assert r1.signal_energy == r2.signal_energy
    assert r1.component_fraction == r2.component_fraction


def test_to_frame_includes_noise_row():
    adatas, context_names = _synthetic_adatas(noise_scale=0.1)
    result = split_half_perturbation_decomposition(
        adatas,
        context_names=context_names,
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
        n_splits=5,
        random_state=0,
    )
    frame = result.to_frame()
    assert "noise" in set(frame["component"])
    assert frame["fraction"].sum() == pytest.approx(1.0, abs=1e-6)
