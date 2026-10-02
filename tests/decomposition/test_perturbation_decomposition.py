"""Tests for perturbation response preparation and high-level decomposition."""

from __future__ import annotations

import anndata as ad
import numpy as np
import pytest
from scipy import sparse

import scdice_metrics as sm
from scdice_metrics.decomposition import (
    PerturbationDecompositionResult,
    prepare_perturbation_responses,
    perturbation_decomposition,
    response_decomposition,
)


def _make_context_adata(
    *,
    context_offset: float,
    perturbations: dict[str, tuple[float, int]],
    control: str = "non-targeting",
    n_genes: int = 5,
    use_obsm: bool = True,
) -> ad.AnnData:
    """Create synthetic AnnData with known pseudobulk structure."""
    rows: list[np.ndarray] = []
    obs_pert: list[str] = []

    ctrl_values = np.arange(n_genes, dtype=float) + context_offset
    for _ in range(40):
        rows.append(ctrl_values + np.random.default_rng(0).normal(scale=0.01, size=n_genes))
        obs_pert.append(control)

    for pert_name, (pert_offset, n_cells) in perturbations.items():
        for _ in range(n_cells):
            rows.append(ctrl_values + pert_offset + np.random.default_rng(1).normal(scale=0.01, size=n_genes))
            obs_pert.append(pert_name)

    x = np.stack(rows, axis=0)
    adata = ad.AnnData(X=x)
    adata.obs["gene"] = obs_pert
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    if use_obsm:
        adata.obsm["X_hvg"] = x.copy()
        adata.uns["X_hvg"] = list(adata.var_names)
    return adata


def test_prepare_basic_shared_perturbations():
    adatas = [
        _make_context_adata(
            context_offset=0.0,
            perturbations={"PERT_A": (1.0, 35), "PERT_B": (2.0, 35), "PERT_C": (3.0, 35), "LOW": (4.0, 5)},
        ),
        _make_context_adata(
            context_offset=10.0,
            perturbations={"PERT_A": (1.5, 35), "PERT_B": (2.5, 35), "PERT_C": (3.5, 35), "ONLY_B": (5.0, 35)},
        ),
    ]
    data = prepare_perturbation_responses(
        adatas,
        context_names=["K562", "RPE1"],
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
    )

    assert data.contexts == ["K562", "RPE1"]
    assert data.perturbations == ["PERT_A", "PERT_B", "PERT_C"]
    assert data.responses.shape == (2, 3, 5)
    assert data.feature_names == ["g0", "g1", "g2", "g3", "g4"]
    assert np.all(data.cell_counts >= 30)


def test_matched_control_subtraction():
    adatas = [
        _make_context_adata(
            context_offset=0.0,
            perturbations={"PERT_A": (2.0, 35), "PERT_B": (0.0, 35)},
        ),
        _make_context_adata(
            context_offset=100.0,
            perturbations={"PERT_A": (4.0, 35), "PERT_B": (0.0, 35)},
        ),
    ]
    data = prepare_perturbation_responses(
        adatas,
        context_names=["ctx0", "ctx1"],
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
    )

    pert_a_idx = data.perturbations.index("PERT_A")
    expected = np.array(
        [
            [2.0, 2.0, 2.0, 2.0, 2.0],
            [4.0, 4.0, 4.0, 4.0, 4.0],
        ],
        dtype=float,
    )
    assert np.allclose(data.responses[:, pert_a_idx, :], expected, atol=0.05)


def test_obsm_key_none_uses_x():
    adata0 = _make_context_adata(
        context_offset=0.0,
        perturbations={"PERT_A": (1.0, 35), "PERT_B": (2.0, 35)},
        use_obsm=False,
    )
    adata1 = _make_context_adata(
        context_offset=5.0,
        perturbations={"PERT_A": (1.0, 35), "PERT_B": (2.0, 35)},
        use_obsm=False,
    )
    data = prepare_perturbation_responses(
        [adata0, adata1],
        context_names=["a", "b"],
        perturbation_key="gene",
        control="non-targeting",
        obsm_key=None,
        min_cells=30,
    )
    assert data.responses.shape == (2, 2, 5)
    assert data.feature_names == ["g0", "g1", "g2", "g3", "g4"]


def test_sparse_x_support():
    adata0 = _make_context_adata(
        context_offset=0.0,
        perturbations={"PERT_A": (1.0, 35), "PERT_B": (2.0, 35)},
        use_obsm=False,
    )
    adata1 = _make_context_adata(
        context_offset=1.0,
        perturbations={"PERT_A": (1.0, 35), "PERT_B": (2.0, 35)},
        use_obsm=False,
    )
    adata0.X = sparse.csr_matrix(adata0.X)
    adata1.X = sparse.csr_matrix(adata1.X)

    data = prepare_perturbation_responses(
        [adata0, adata1],
        context_names=["a", "b"],
        perturbation_key="gene",
        control="non-targeting",
        obsm_key=None,
        min_cells=30,
    )
    assert data.responses.shape == (2, 2, 5)


def test_deterministic_ordering():
    adatas = [
        _make_context_adata(context_offset=0.0, perturbations={"Z": (1.0, 35), "A": (2.0, 35)}),
        _make_context_adata(context_offset=1.0, perturbations={"A": (1.0, 35), "Z": (2.0, 35)}),
    ]
    data = prepare_perturbation_responses(
        adatas,
        context_names=["c0", "c1"],
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
    )
    assert data.perturbations == ["A", "Z"]


def test_validation_errors():
    adata = _make_context_adata(context_offset=0.0, perturbations={"PERT_A": (1.0, 35), "PERT_B": (2.0, 35)})

    with pytest.raises(ValueError, match="At least two contexts"):
        prepare_perturbation_responses(
            [adata],
            context_names=["only"],
            perturbation_key="gene",
            control="non-targeting",
        )

    with pytest.raises(ValueError, match="must be unique"):
        prepare_perturbation_responses(
            [adata, adata],
            context_names=["dup", "dup"],
            perturbation_key="gene",
            control="non-targeting",
        )

    with pytest.raises(ValueError, match="missing obs column"):
        prepare_perturbation_responses(
            [adata, adata],
            context_names=["a", "b"],
            perturbation_key="missing_col",
            control="non-targeting",
        )

    adata_no_ctrl = adata.copy()
    adata_no_ctrl.obs["gene"] = "PERT_A"
    with pytest.raises(ValueError, match="no cells with control label"):
        prepare_perturbation_responses(
            [adata, adata_no_ctrl],
            context_names=["a", "b"],
            perturbation_key="gene",
            control="non-targeting",
        )

    with pytest.raises(ValueError, match="missing obsm"):
        prepare_perturbation_responses(
            [adata, adata],
            context_names=["a", "b"],
            perturbation_key="gene",
            control="non-targeting",
            obsm_key="missing",
        )

    adata_one_shared = _make_context_adata(context_offset=0.0, perturbations={"PERT_A": (1.0, 35)})
    adata_other = _make_context_adata(context_offset=1.0, perturbations={"PERT_A": (1.0, 35), "PERT_B": (2.0, 35)})
    with pytest.raises(ValueError, match="Fewer than two shared perturbations"):
        prepare_perturbation_responses(
            [adata_one_shared, adata_other],
            context_names=["a", "b"],
            perturbation_key="gene",
            control="non-targeting",
            min_cells=30,
        )


def test_mismatched_feature_dimensions():
    adata0 = _make_context_adata(context_offset=0.0, perturbations={"P1": (1.0, 35), "P2": (2.0, 35)}, n_genes=5)
    adata1 = _make_context_adata(context_offset=0.0, perturbations={"P1": (1.0, 35), "P2": (2.0, 35)}, n_genes=6)
    with pytest.raises(ValueError, match="Mismatched feature dimensions"):
        prepare_perturbation_responses(
            [adata0, adata1],
            context_names=["a", "b"],
            perturbation_key="gene",
            control="non-targeting",
            obsm_key=None,
            min_cells=30,
        )


def test_warns_on_large_discard_fraction():
    adatas = [
        _make_context_adata(
            context_offset=0.0,
            perturbations={
                "P1": (1.0, 35),
                "P2": (2.0, 35),
                "P3": (3.0, 35),
                "P4": (4.0, 35),
                "P5": (5.0, 35),
            },
        ),
        _make_context_adata(
            context_offset=1.0,
            perturbations={"P1": (1.0, 35), "P2": (2.0, 35)},
        ),
    ]
    with pytest.warns(UserWarning, match="discarded after shared filtering"):
        prepare_perturbation_responses(
            adatas,
            context_names=["a", "b"],
            perturbation_key="gene",
            control="non-targeting",
            min_cells=30,
        )


def test_shared_only_false_warns():
    adata = _make_context_adata(context_offset=0.0, perturbations={"P1": (1.0, 35), "P2": (2.0, 35)})
    with pytest.warns(UserWarning, match="shared_only=False"):
        prepare_perturbation_responses(
            [adata, adata],
            context_names=["a", "b"],
            perturbation_key="gene",
            control="non-targeting",
            shared_only=False,
        )


def test_perturbation_decomposition_end_to_end():
    adatas = [
        _make_context_adata(
            context_offset=0.0,
            perturbations={"P1": (1.0, 35), "P2": (2.0, 35), "P3": (3.0, 35)},
        ),
        _make_context_adata(
            context_offset=10.0,
            perturbations={"P1": (1.5, 35), "P2": (2.5, 35), "P3": (3.5, 35)},
        ),
    ]
    result = perturbation_decomposition(
        adatas,
        context_names=["K562", "RPE1"],
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
    )

    assert isinstance(result, PerturbationDecompositionResult)
    assert result.factor_names == ("context", "perturbation")
    assert result.factor_a_names == ["K562", "RPE1"]
    assert result.factor_b_names == ["P1", "P2", "P3"]
    assert set(result.component_fraction) == {
        "global",
        "context",
        "perturbation",
        "context:perturbation",
    }
    assert sum(result.component_fraction.values()) == pytest.approx(1.0, rel=1e-5)
    assert result.reconstruction_error < 1e-10


def test_perturbation_decomposition_aliases():
    adatas = [
        _make_context_adata(context_offset=0.0, perturbations={"P1": (1.0, 35), "P2": (2.0, 35)}),
        _make_context_adata(context_offset=5.0, perturbations={"P1": (1.0, 35), "P2": (2.0, 35)}),
    ]
    result = perturbation_decomposition(
        adatas,
        context_names=["ctx0", "ctx1"],
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
    )

    assert np.shares_memory(result.context_effect, result.factor_a_effect)
    assert np.shares_memory(result.conserved_perturbation_effect, result.factor_b_effect)
    assert np.shares_memory(result.context_perturbation_interaction, result.interaction_effect)


def test_perturbation_decomposition_matches_manual_pipeline():
    adatas = [
        _make_context_adata(context_offset=0.0, perturbations={"P1": (1.0, 35), "P2": (2.0, 35)}),
        _make_context_adata(context_offset=3.0, perturbations={"P1": (1.0, 35), "P2": (2.0, 35)}),
    ]
    wrapped = perturbation_decomposition(
        adatas,
        context_names=["a", "b"],
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
    )
    data = prepare_perturbation_responses(
        adatas,
        context_names=["a", "b"],
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
    )
    manual = response_decomposition(
        data.responses,
        factor_a_names=data.contexts,
        factor_b_names=data.perturbations,
        feature_names=data.feature_names,
        factor_names=("context", "perturbation"),
    )

    assert wrapped.component_fraction == manual.component_fraction
    assert wrapped.total_energy == pytest.approx(manual.total_energy)
    assert np.allclose(wrapped.global_effect, manual.global_effect)


def test_root_package_exports_perturbation_decomposition():
    assert sm.perturbation_decomposition is perturbation_decomposition


def test_transferable_fraction_accepts_perturbation_alias():
    adatas = [
        _make_context_adata(context_offset=0.0, perturbations={"P1": (1.0, 35), "P2": (2.0, 35)}),
        _make_context_adata(context_offset=2.0, perturbations={"P1": (1.0, 35), "P2": (2.0, 35)}),
    ]
    result = perturbation_decomposition(
        adatas,
        context_names=["a", "b"],
        perturbation_key="gene",
        control="non-targeting",
        min_cells=30,
    )
    assert result.transferable_fraction("perturbation") == result.transferable_fraction("factor_b")
