"""Optional end-to-end smoke test on local Replogle K562/RPE1 essential screens.

Set environment variables before running:

    export REPLOGLE_K562_H5AD=/path/to/k562/replogle.h5ad
    export REPLOGLE_RPE1_H5AD=/path/to/rpe1/replogle.h5ad

Or place files under ``data/replogle_essential/{k562,rpe1}/replogle.h5ad`` relative to cwd.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs" / "notebooks"))

from replogle_decomposition_utils import (  # noqa: E402
    DECOMP_COLORS,
    default_replogle_paths,
    load_replogle_decomposition_adatas,
    plot_fraction_stacked_bars,
)

import scdice_metrics as sm  # noqa: E402


def _replogle_data_available() -> bool:
    k562, rpe1 = default_replogle_paths()
    return k562.exists() and rpe1.exists()


pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(not _replogle_data_available(), reason="Replogle h5ad files not found"),
]


@pytest.fixture(scope="module")
def replogle_adatas():
    adatas, _context_names = load_replogle_decomposition_adatas(
        k562_max_cells_per_pert=40,
        random_state=0,
    )
    return adatas


def test_prepare_replogle_responses(replogle_adatas):
    data = sm.prepare_perturbation_responses(
        replogle_adatas,
        context_names=["K562", "RPE1"],
        perturbation_key="gene",
        control="non-targeting",
        obsm_key="X_hvg",
        min_cells=30,
    )
    assert data.responses.shape[0] == 2
    assert data.responses.shape[1] >= 2
    assert data.responses.shape[2] >= 100


def test_replogle_perturbation_decomposition(replogle_adatas):
    result = sm.perturbation_decomposition(
        replogle_adatas,
        context_names=["K562", "RPE1"],
        perturbation_key="gene",
        control="non-targeting",
        obsm_key="X_hvg",
        min_cells=30,
    )
    assert set(result.component_fraction) == {
        "global",
        "context",
        "perturbation",
        "context:perturbation",
    }
    assert sum(result.component_fraction.values()) == pytest.approx(1.0, rel=1e-4)


def test_replogle_split_half_decomposition(replogle_adatas):
    result = sm.split_half_perturbation_decomposition(
        replogle_adatas,
        context_names=["K562", "RPE1"],
        perturbation_key="gene",
        control="non-targeting",
        obsm_key="X_hvg",
        min_cells=30,
        n_splits=5,
        random_state=0,
    )
    assert "noise" in result.component_fraction
    assert sum(result.component_fraction.values()) == pytest.approx(1.0, rel=1e-4)
    assert result.per_factor_b_transferability().shape[0] >= 2
