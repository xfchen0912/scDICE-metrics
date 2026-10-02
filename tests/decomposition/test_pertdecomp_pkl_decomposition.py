"""Tests for perturbation-decomposition pseudobulk pkl -> AnnData loaders."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "notebooks"))

from replogle_decomposition_utils import (  # noqa: E402
    default_pertdecomp_pkl_paths,
    load_pertdecomp_decomposition_adatas,
    pertdecomp_pkl_to_anndata,
)

import scdice_metrics as sm  # noqa: E402


def _make_synthetic_pkl(*, seed: int, n_features: int = 32, n_perts: int = 8) -> dict:
    rng = np.random.default_rng(seed)
    ctrl_mean = rng.normal(size=n_features)
    deltas = {f"GENE{i}": rng.normal(scale=0.5, size=n_features) for i in range(n_perts)}
    cell_counts = {pert: int(40 + i) for i, pert in enumerate(deltas)}
    return {
        "deltas": deltas,
        "ctrl_mean": ctrl_mean,
        "cell_counts": cell_counts,
        "n_hvgs": n_features,
        "hvg_names": [f"g{i}" for i in range(n_features)],
        "dataset": "synthetic",
    }


def test_pertdecomp_pkl_to_anndata_preserves_pseudobulk_means():
    pkl_data = _make_synthetic_pkl(seed=0)
    adata = pertdecomp_pkl_to_anndata(pkl_data, context_name="CtxA")

    matrix = np.asarray(adata.obsm["X_hvg"])
    labels = adata.obs["gene"].astype(str).to_numpy()
    ctrl_mean = matrix[labels == "non-targeting"].mean(axis=0)

    for pert, delta in pkl_data["deltas"].items():
        mask = labels == pert
        assert mask.sum() == pkl_data["cell_counts"][pert]
        observed = matrix[mask].mean(axis=0) - ctrl_mean
        np.testing.assert_allclose(observed, delta, rtol=1e-7, atol=1e-7)


def test_pertdecomp_pkl_decomposition_end_to_end():
    adatas = [
        pertdecomp_pkl_to_anndata(_make_synthetic_pkl(seed=0), context_name="K562"),
        pertdecomp_pkl_to_anndata(_make_synthetic_pkl(seed=1), context_name="RPE1"),
    ]
    result = sm.perturbation_decomposition(
        adatas,
        context_names=["K562", "RPE1"],
        perturbation_key="gene",
        control="non-targeting",
        obsm_key="X_hvg",
        min_cells=5,
    )
    assert set(result.component_fraction) == {
        "global",
        "context",
        "perturbation",
        "context:perturbation",
    }
    assert sum(result.component_fraction.values()) == pytest.approx(1.0, rel=1e-4)


def test_pertdecomp_pkl_split_half_runs():
    adatas = [
        pertdecomp_pkl_to_anndata(_make_synthetic_pkl(seed=0), context_name="K562"),
        pertdecomp_pkl_to_anndata(_make_synthetic_pkl(seed=1), context_name="RPE1"),
    ]
    result = sm.split_half_perturbation_decomposition(
        adatas,
        context_names=["K562", "RPE1"],
        perturbation_key="gene",
        control="non-targeting",
        obsm_key="X_hvg",
        min_cells=5,
        n_splits=3,
        random_state=0,
    )
    assert "noise" in result.component_fraction
    assert sum(result.component_fraction.values()) == pytest.approx(1.0, rel=1e-4)


def _release_pkls_available() -> bool:
    paths = default_pertdecomp_pkl_paths()
    return all(path.exists() for path in paths.values())


@pytest.mark.skipif(not _release_pkls_available(), reason="perturbation-decomposition pkls not found")
def test_release_pertdecomp_pkls_end_to_end():
    adatas, context_names = load_pertdecomp_decomposition_adatas()
    result = sm.perturbation_decomposition(
        adatas,
        context_names=context_names,
        perturbation_key="gene",
        control="non-targeting",
        obsm_key="X_hvg",
        min_cells=5,
    )
    assert result.responses.shape[0] == 2
    assert result.responses.shape[1] >= 2
    assert sum(result.component_fraction.values()) == pytest.approx(1.0, rel=1e-4)
