"""Generic balanced two-factor response decomposition."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ._result import ResponseDecompositionResult, _interaction_key


def _validate_response_tensor(
    responses: np.ndarray,
    *,
    factor_a_names: Sequence[str] | None,
    factor_b_names: Sequence[str] | None,
    feature_names: Sequence[str] | None,
) -> np.ndarray:
    responses = np.asarray(responses, dtype=float)
    if responses.ndim != 3:
        raise ValueError(f"`responses` must be 3-dimensional; got shape {responses.shape}.")
    n_a, n_b, n_g = responses.shape
    if n_a < 2 or n_b < 2:
        raise ValueError(f"Each factor must have at least 2 levels; got A={n_a}, B={n_b}.")
    if n_g < 1:
        raise ValueError("Feature dimension must be at least 1.")
    if not np.all(np.isfinite(responses)):
        raise ValueError("`responses` contains non-finite values.")
    if factor_a_names is not None and len(factor_a_names) != n_a:
        raise ValueError(f"`factor_a_names` length {len(factor_a_names)} != factor A size {n_a}.")
    if factor_b_names is not None and len(factor_b_names) != n_b:
        raise ValueError(f"`factor_b_names` length {len(factor_b_names)} != factor B size {n_b}.")
    if feature_names is not None and len(feature_names) != n_g:
        raise ValueError(f"`feature_names` length {len(feature_names)} != feature size {n_g}.")
    return responses


def _row_energy(vector: np.ndarray) -> float:
    return float(np.sum(vector**2))


def _energy_dict(
    global_effect: np.ndarray,
    factor_a_effect: np.ndarray,
    factor_b_effect: np.ndarray,
    interaction_effect: np.ndarray,
    responses: np.ndarray,
    factor_names: tuple[str, str],
) -> tuple[dict[str, float], float]:
    interaction = _interaction_key(factor_names)
    fn_a, fn_b = factor_names
    component_energy = {
        "global": _row_energy(global_effect),
        fn_a: float(np.mean(np.sum(factor_a_effect**2, axis=1))),
        fn_b: float(np.mean(np.sum(factor_b_effect**2, axis=1))),
        interaction: float(np.mean(np.sum(interaction_effect**2, axis=-1))),
    }
    total_energy = float(np.mean(np.sum(responses**2, axis=-1)))
    return component_energy, total_energy


def _component_fractions(component_energy: dict[str, float], total_energy: float) -> dict[str, float]:
    if total_energy <= 0.0:
        return {key: 0.0 for key in component_energy}
    return {key: float(value / total_energy) for key, value in component_energy.items()}


def response_decomposition(
    responses: np.ndarray,
    *,
    factor_a_names: Sequence[str] | None = None,
    factor_b_names: Sequence[str] | None = None,
    feature_names: Sequence[str] | None = None,
    factor_names: tuple[str, str] = ("factor_a", "factor_b"),
) -> ResponseDecompositionResult:
    """Decompose a balanced two-factor response tensor into mu, alpha, beta, and gamma.

    Parameters
    ----------
    responses
        Array of shape ``(n_a, n_b, n_features)``.
    factor_a_names
        Optional labels for factor A levels.
    factor_b_names
        Optional labels for factor B levels.
    feature_names
        Optional feature labels.
    factor_names
        Names used for reporting energies and fractions.

    Returns
    -------
    ResponseDecompositionResult
        Decomposed components, energies, fractions, and diagnostics.
    """
    if len(factor_names) != 2:
        raise ValueError("`factor_names` must be a pair of strings.")

    responses = _validate_response_tensor(
        responses,
        factor_a_names=factor_a_names,
        factor_b_names=factor_b_names,
        feature_names=feature_names,
    )

    global_effect = responses.mean(axis=(0, 1))
    factor_a_effect = responses.mean(axis=1) - global_effect
    factor_b_effect = responses.mean(axis=0) - global_effect
    interaction_effect = (
        responses - global_effect - factor_a_effect[:, None, :] - factor_b_effect[None, :, :]
    )

    reconstructed = (
        global_effect + factor_a_effect[:, None, :] + factor_b_effect[None, :, :] + interaction_effect
    )
    reconstruction_error = float(np.max(np.abs(responses - reconstructed)))

    component_energy, total_energy = _energy_dict(
        global_effect,
        factor_a_effect,
        factor_b_effect,
        interaction_effect,
        responses,
        factor_names,
    )
    component_fraction = _component_fractions(component_energy, total_energy)

    return ResponseDecompositionResult(
        global_effect=global_effect,
        factor_a_effect=factor_a_effect,
        factor_b_effect=factor_b_effect,
        interaction_effect=interaction_effect,
        component_energy=component_energy,
        component_fraction=component_fraction,
        total_energy=total_energy,
        reconstruction_error=reconstruction_error,
        factor_a_names=factor_a_names,
        factor_b_names=factor_b_names,
        feature_names=feature_names,
        factor_names=factor_names,
    )
