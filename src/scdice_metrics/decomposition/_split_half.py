"""Split-half reproducibility noise correction for perturbation decomposition."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._core import response_decomposition
from ._perturbation import (
    _get_expression_matrix,
    _group_mean,
    _validate_inputs,
    prepare_perturbation_responses,
)
from ._result import as_perturbation_decomposition_result

if TYPE_CHECKING:
    from anndata import AnnData

    from ._result import SplitHalfDecompositionResult


@dataclass
class _ContextCellGroups:
    control_mean: np.ndarray
    perturbation_cells: dict[str, np.ndarray]


def _collect_context_cell_groups(
    adata: AnnData,
    *,
    context_name: str,
    perturbation_key: str,
    control: str,
    obsm_key: str | None,
    layer: str | None,
    use_raw: bool,
    min_cells: int,
) -> tuple[_ContextCellGroups, dict[str, int]]:
    matrix = _get_expression_matrix(
        adata,
        context_name=context_name,
        obsm_key=obsm_key,
        layer=layer,
        use_raw=use_raw,
    )
    perturbations = adata.obs[perturbation_key].astype(str).to_numpy()
    control_mask = perturbations == str(control)
    if not control_mask.any():
        raise ValueError(f"context {context_name!r} has no cells with control label {control!r}.")
    control_mean = _group_mean(matrix, control_mask)

    perturbation_cells: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for pert in sorted(set(perturbations) - {str(control)}):
        mask = perturbations == pert
        n_cells = int(mask.sum())
        if n_cells >= min_cells:
            perturbation_cells[pert] = matrix[mask]
            counts[pert] = n_cells

    return _ContextCellGroups(control_mean=control_mean, perturbation_cells=perturbation_cells), counts


def _split_half_delta_tensors(
    context_groups: Sequence[_ContextCellGroups],
    perturbations: Sequence[str],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    n_contexts = len(context_groups)
    n_perts = len(perturbations)
    n_features = context_groups[0].control_mean.shape[0]
    d_h1 = np.zeros((n_contexts, n_perts, n_features), dtype=float)
    d_h2 = np.zeros((n_contexts, n_perts, n_features), dtype=float)

    for ci, groups in enumerate(context_groups):
        ctrl_mean = groups.control_mean
        for pi, pert in enumerate(perturbations):
            cells = groups.perturbation_cells[pert]
            perm = rng.permutation(len(cells))
            half = len(cells) // 2
            if half == 0:
                raise ValueError(
                    f"Cannot split perturbation {pert!r} in context index {ci}: only {len(cells)} cells."
                )
            d_h1[ci, pi] = cells[perm[:half]].mean(axis=0) - ctrl_mean
            d_h2[ci, pi] = cells[perm[half : 2 * half]].mean(axis=0) - ctrl_mean

    return d_h1, d_h2


def _cross_half_signal_energy(
    d_h1: np.ndarray,
    d_h2: np.ndarray,
    *,
    factor_names: tuple[str, str] = ("context", "perturbation"),
) -> dict[str, float]:
    mu1 = d_h1.mean(axis=(0, 1))
    mu2 = d_h2.mean(axis=(0, 1))
    a1 = d_h1.mean(axis=1) - mu1
    a2 = d_h2.mean(axis=1) - mu2
    b1 = d_h1.mean(axis=0) - mu1
    b2 = d_h2.mean(axis=0) - mu2
    g1 = d_h1 - mu1 - a1[:, None, :] - b1[None, :, :]
    g2 = d_h2 - mu2 - a2[:, None, :] - b2[None, :, :]

    n_a, n_b = d_h1.shape[:2]
    fn_a, fn_b = factor_names
    interaction = f"{fn_a}:{fn_b}"
    return {
        "global": float(np.dot(mu1, mu2)),
        fn_a: float(np.mean([np.dot(a1[c], a2[c]) for c in range(n_a)])),
        fn_b: float(np.mean([np.dot(b1[p], b2[p]) for p in range(n_b)])),
        interaction: float(np.mean([np.dot(g1[c, p], g2[c, p]) for c in range(n_a) for p in range(n_b)])),
    }


def _average_signal_estimates(estimates: list[dict[str, float]]) -> dict[str, float]:
    keys = estimates[0].keys()
    return {key: float(np.mean([estimate[key] for estimate in estimates])) for key in keys}


def _build_corrected_fractions(
    signal_energy: dict[str, float],
    total_energy: float,
    *,
    clip_negative_signal: bool,
) -> tuple[dict[str, float], float]:
    if total_energy <= 0.0:
        return {key: 0.0 for key in (*signal_energy, "noise")}, 0.0

    adjusted = signal_energy
    if clip_negative_signal:
        adjusted = {key: max(0.0, value) for key, value in signal_energy.items()}

    noise_energy = float(total_energy - sum(adjusted.values()))
    if clip_negative_signal and noise_energy < 0.0:
        noise_energy = 0.0

    fractions = {key: float(value / total_energy) for key, value in adjusted.items()}
    fractions["noise"] = float(noise_energy / total_energy)
    return fractions, noise_energy


def split_half_perturbation_decomposition(
    adatas: Sequence[AnnData],
    *,
    context_names: Sequence[str],
    perturbation_key: str,
    control: str,
    obsm_key: str | None = "X_hvg",
    layer: str | None = None,
    use_raw: bool = False,
    min_cells: int = 30,
    shared_only: bool = True,
    n_splits: int = 50,
    random_state: int = 0,
    clip_negative_signal: bool = False,
) -> SplitHalfDecompositionResult:
    """Estimate reproducible decomposition fractions with split-half noise correction.

    Perturbation cells are randomly split within each context × perturbation group.
    Both halves subtract the same full-data context control mean (reference behavior).
    Reproducible component energy is estimated from cross-half inner products and
    averaged over ``n_splits`` resamples. Residual energy is reported as ``noise``.

    Parameters
    ----------
    adatas, context_names, perturbation_key, control, obsm_key, layer, use_raw, min_cells, shared_only
        Same as :func:`prepare_perturbation_responses`.
    n_splits
        Number of random split-half resamples.
    random_state
        Seed for the split-half RNG.
    clip_negative_signal
        If ``True``, clip negative cross-half signal estimates to zero before computing
        ``noise`` fractions. Raw estimates are always stored in ``signal_energy``.

    Returns
    -------
    SplitHalfDecompositionResult
        Full-data naive decomposition plus split-half ``signal_energy``, ``noise_energy``,
        and corrected ``component_fraction`` including a ``noise`` term.
    """
    from ._result import SplitHalfDecompositionResult

    if n_splits < 1:
        raise ValueError("`n_splits` must be at least 1.")
    if min_cells < 2:
        raise ValueError("`min_cells` must be at least 2 for split-half estimation.")

    data = prepare_perturbation_responses(
        adatas,
        context_names=context_names,
        perturbation_key=perturbation_key,
        control=control,
        obsm_key=obsm_key,
        layer=layer,
        use_raw=use_raw,
        min_cells=min_cells,
        shared_only=shared_only,
    )

    naive = response_decomposition(
        data.responses,
        factor_a_names=data.contexts,
        factor_b_names=data.perturbations,
        feature_names=data.feature_names,
        factor_names=("context", "perturbation"),
    )
    naive_result = as_perturbation_decomposition_result(naive)

    adatas = list(adatas)
    context_names = list(context_names)
    _validate_inputs(adatas, context_names, perturbation_key)

    context_groups: list[_ContextCellGroups] = []
    for context_name, adata in zip(context_names, adatas, strict=True):
        groups, _counts = _collect_context_cell_groups(
            adata,
            context_name=context_name,
            perturbation_key=perturbation_key,
            control=control,
            obsm_key=obsm_key,
            layer=layer,
            use_raw=use_raw,
            min_cells=min_cells,
        )
        context_groups.append(groups)

    shared = data.perturbations
    if min_cells // 2 < 5:
        warnings.warn(
            "Split-half groups are small after halving; consider increasing `min_cells`.",
            stacklevel=2,
        )

    rng = np.random.default_rng(random_state)
    split_estimates: list[dict[str, float]] = []
    for _ in range(n_splits):
        d_h1, d_h2 = _split_half_delta_tensors(context_groups, shared, rng)
        split_estimates.append(_cross_half_signal_energy(d_h1, d_h2))

    signal_energy = _average_signal_estimates(split_estimates)
    if any(value < 0.0 for value in signal_energy.values()):
        warnings.warn(
            "Negative cross-half signal estimates detected; see `signal_energy` for raw values.",
            stacklevel=2,
        )

    corrected_fraction, noise_energy = _build_corrected_fractions(
        signal_energy,
        naive_result.total_energy,
        clip_negative_signal=clip_negative_signal,
    )

    return SplitHalfDecompositionResult(
        global_effect=naive_result.global_effect,
        factor_a_effect=naive_result.factor_a_effect,
        factor_b_effect=naive_result.factor_b_effect,
        interaction_effect=naive_result.interaction_effect,
        component_energy=naive_result.component_energy,
        component_fraction=corrected_fraction,
        total_energy=naive_result.total_energy,
        reconstruction_error=naive_result.reconstruction_error,
        factor_a_names=naive_result.factor_a_names,
        factor_b_names=naive_result.factor_b_names,
        feature_names=naive_result.feature_names,
        factor_names=naive_result.factor_names,
        signal_energy=signal_energy,
        noise_energy=noise_energy,
        n_splits=n_splits,
        naive_component_fraction=naive_result.component_fraction,
        naive_component_energy=naive_result.component_energy,
    )
