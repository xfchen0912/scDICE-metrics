"""Perturb-seq pseudobulk response preparation for multi-context AnnData inputs."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import sparse

if TYPE_CHECKING:
    from anndata import AnnData

from ._result import PerturbationDecompositionResult


@dataclass
class PerturbationResponseData:
    """Context × perturbation response tensor and metadata."""

    responses: np.ndarray
    contexts: list[str]
    perturbations: list[str]
    feature_names: list[str]
    cell_counts: np.ndarray
    control_counts: dict[str, int]
    obsm_key: str | None
    layer: str | None
    use_raw: bool


def _validate_inputs(
    adatas: Sequence[AnnData],
    context_names: Sequence[str],
    perturbation_key: str,
) -> None:
    if len(adatas) != len(context_names):
        raise ValueError(
            f"`len(adatas)` ({len(adatas)}) must equal `len(context_names)` ({len(context_names)})."
        )
    if len(adatas) < 2:
        raise ValueError("At least two contexts are required.")
    if len(set(context_names)) != len(context_names):
        raise ValueError("`context_names` must be unique.")
    for context_name, adata in zip(context_names, adatas, strict=True):
        if perturbation_key not in adata.obs.columns:
            raise ValueError(
                f"context {context_name!r} is missing obs column {perturbation_key!r}."
            )


def _resolve_feature_names(adata: AnnData, obsm_key: str | None, n_features: int) -> list[str]:
    if obsm_key is not None and obsm_key in adata.uns:
        stored = adata.uns[obsm_key]
        if isinstance(stored, (list, tuple, np.ndarray)) and len(stored) == n_features:
            return [str(name) for name in stored]
    if obsm_key is None and adata.n_vars == n_features:
        return [str(name) for name in adata.var_names]
    return [f"f{i}" for i in range(n_features)]


def _get_expression_matrix(
    adata: AnnData,
    *,
    context_name: str,
    obsm_key: str | None,
    layer: str | None,
    use_raw: bool,
) -> np.ndarray:
    if obsm_key is not None:
        if obsm_key not in adata.obsm:
            raise ValueError(f"context {context_name!r} is missing obsm[{obsm_key!r}].")
        matrix = adata.obsm[obsm_key]
    elif use_raw:
        if adata.raw is None:
            raise ValueError(f"context {context_name!r} has no `.raw` but `use_raw=True`.")
        matrix = adata.raw.X
    elif layer is not None:
        if layer not in adata.layers:
            raise ValueError(f"context {context_name!r} is missing layer {layer!r}.")
        matrix = adata.layers[layer]
    else:
        matrix = adata.X

    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            f"context {context_name!r} expression matrix must be 2-dimensional; got shape {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"context {context_name!r} expression matrix contains non-finite values.")
    return matrix


def _group_mean(matrix: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0:
        raise ValueError("Cannot compute a group mean over zero cells.")
    return matrix[mask].mean(axis=0)


def _context_deltas(
    adata: AnnData,
    *,
    context_name: str,
    perturbation_key: str,
    control: str,
    obsm_key: str | None,
    layer: str | None,
    use_raw: bool,
    min_cells: int,
) -> tuple[dict[str, np.ndarray], dict[str, int], int, list[str], int]:
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
    n_control = int(control_mask.sum())
    feature_names = _resolve_feature_names(adata, obsm_key, matrix.shape[1])

    deltas: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    for pert in sorted(set(perturbations) - {str(control)}):
        mask = perturbations == pert
        n_cells = int(mask.sum())
        if n_cells >= min_cells:
            deltas[pert] = _group_mean(matrix, mask) - control_mean
            counts[pert] = n_cells

    return deltas, counts, n_control, feature_names, matrix.shape[1]


def _maybe_warn_cell_count_imbalance(cell_counts: np.ndarray) -> None:
    positive = cell_counts[cell_counts > 0]
    if positive.size == 0:
        return
    ratio = float(positive.max() / positive.min())
    if ratio > 10.0:
        warnings.warn(
            f"Strong imbalance in perturbation cell counts (max/min={ratio:.1f}).",
            stacklevel=3,
        )


def _maybe_warn_discarded_perturbations(
    *,
    n_shared: int,
    per_context_counts: list[int],
) -> None:
    max_per_context = max(per_context_counts)
    if max_per_context == 0:
        return
    retained_fraction = n_shared / max_per_context
    if retained_fraction < 0.5:
        warnings.warn(
            "More than half of per-context perturbations were discarded after shared filtering "
            f"({n_shared}/{max_per_context} retained).",
            stacklevel=3,
        )


def prepare_perturbation_responses(
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
) -> PerturbationResponseData:
    """Build a context × perturbation × feature response tensor from AnnData objects.

    Each AnnData corresponds to one cellular context. Perturbation responses are
    pseudobulk means relative to the matched control in the same context.
    """
    if min_cells < 1:
        raise ValueError("`min_cells` must be at least 1.")
    if not shared_only:
        warnings.warn(
            "`shared_only=False` is not supported for balanced decomposition; using shared perturbations.",
            stacklevel=2,
        )

    adatas = list(adatas)
    context_names = list(context_names)
    _validate_inputs(adatas, context_names, perturbation_key)

    per_context: list[dict[str, np.ndarray]] = []
    per_context_counts: list[dict[str, int]] = []
    control_counts: dict[str, int] = {}
    feature_names: list[str] | None = None
    n_features: int | None = None

    for context_name, adata in zip(context_names, adatas, strict=True):
        deltas, counts, n_control, names, n_feat = _context_deltas(
            adata,
            context_name=context_name,
            perturbation_key=perturbation_key,
            control=control,
            obsm_key=obsm_key,
            layer=layer,
            use_raw=use_raw,
            min_cells=min_cells,
        )
        per_context.append(deltas)
        per_context_counts.append(counts)
        control_counts[context_name] = n_control

        if n_features is None:
            n_features = n_feat
            feature_names = names
        elif n_feat != n_features:
            raise ValueError(
                f"Mismatched feature dimensions across contexts: expected {n_features}, "
                f"got {n_feat} in context {context_name!r}."
            )
        elif names != feature_names:
            feature_names = [f"f{i}" for i in range(n_features)]

    assert feature_names is not None and n_features is not None

    shared = sorted(set.intersection(*(set(d.keys()) for d in per_context)))
    if len(shared) < 2:
        raise ValueError(
            f"Fewer than two shared perturbations remain after filtering (found {len(shared)})."
        )

    _maybe_warn_discarded_perturbations(
        n_shared=len(shared),
        per_context_counts=[len(d) for d in per_context_counts],
    )

    n_contexts = len(context_names)
    n_perts = len(shared)
    responses = np.zeros((n_contexts, n_perts, n_features), dtype=float)
    cell_counts = np.zeros((n_contexts, n_perts), dtype=int)

    for ci, deltas in enumerate(per_context):
        for pi, pert in enumerate(shared):
            responses[ci, pi] = deltas[pert]
            cell_counts[ci, pi] = per_context_counts[ci][pert]

    _maybe_warn_cell_count_imbalance(cell_counts)

    if not np.all(np.isfinite(responses)):
        raise ValueError("Constructed response tensor contains non-finite values.")

    return PerturbationResponseData(
        responses=responses,
        contexts=context_names,
        perturbations=shared,
        feature_names=feature_names,
        cell_counts=cell_counts,
        control_counts=control_counts,
        obsm_key=obsm_key,
        layer=layer,
        use_raw=use_raw,
    )


def perturbation_decomposition(
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
) -> PerturbationDecompositionResult:
    """Prepare pseudobulk perturbation responses and run full four-component decomposition.

    This is the high-level Perturb-seq entry point. It builds a balanced
    context × perturbation × feature tensor with matched controls, then decomposes
    responses into global, context, conserved perturbation, and interaction components.

    Parameters
    ----------
    adatas
        One AnnData object per cellular context.
    context_names
        Context label for each AnnData, in the same order as ``adatas``.
    perturbation_key
        ``obs`` column containing perturbation identifiers.
    control
        Control perturbation label within ``perturbation_key``.
    obsm_key
        Key in ``obsm`` for expression features. Default ``\"X_hvg\"``.
        Set to ``None`` to use ``X``, ``layer``, or ``raw``.
    layer
        Optional ``layers`` key when ``obsm_key`` is ``None``.
    use_raw
        Use ``adata.raw.X`` when ``obsm_key`` is ``None`` and ``layer`` is ``None``.
    min_cells
        Minimum cells required to retain a perturbation within a context.
    shared_only
        Retain perturbations shared across all contexts (required for balanced design).

    Returns
    -------
    PerturbationDecompositionResult
        Decomposition with ``factor_names=(\"context\", \"perturbation\")`` and
        convenience aliases ``context_effect``, ``conserved_perturbation_effect``,
        and ``context_perturbation_interaction``.
    """
    from ._core import response_decomposition
    from ._result import as_perturbation_decomposition_result

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
    result = response_decomposition(
        data.responses,
        factor_a_names=data.contexts,
        factor_b_names=data.perturbations,
        feature_names=data.feature_names,
        factor_names=("context", "perturbation"),
    )
    return as_perturbation_decomposition_result(result)
