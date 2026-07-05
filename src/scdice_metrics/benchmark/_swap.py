"""Swap-task generation for cell-type-specific OOD benchmarks."""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import quote

import numpy as np
import pandas as pd
from anndata import AnnData


@dataclass(frozen=True)
class SwapSpec:
    task_id: str
    swap_type: Literal["condition", "niche"]
    cell_type: str

    source_mask: np.ndarray
    target_mask: np.ndarray
    train_mask: np.ndarray
    context_mask: np.ndarray

    source_value: Any
    target_value: Any

    fixed_key: str | None = None
    fixed_value: Any | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


def _escape_task_value(value: Any) -> str:
    return quote(str(value), safe="")


def _unique_non_null(values: pd.Series) -> list[Any]:
    return sorted(values.dropna().unique(), key=str)


def _validate_obs_columns(obs: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in obs.columns]
    if missing:
        raise ValueError(f"Missing required obs columns: {', '.join(missing)}")


def _format_task_id(
    swap_type: str,
    cell_type: str,
    source_value: Any,
    target_value: Any,
    *,
    fixed_key: str | None = None,
    fixed_value: Any | None = None,
) -> str:
    parts = [
        swap_type,
        f"cell_type={_escape_task_value(cell_type)}",
        f"source={_escape_task_value(source_value)}",
        f"target={_escape_task_value(target_value)}",
    ]
    if fixed_key is not None and fixed_value is not None:
        parts.append(f"{fixed_key}={_escape_task_value(fixed_value)}")
    return "__".join(parts)


def _stable_task_suffix(*parts: Any) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:8]


def _resolve_cell_types(obs: pd.DataFrame, cell_type_key: str, cell_types: Sequence[str] | None) -> list[Any]:
    if cell_types is not None:
        return list(cell_types)
    return _unique_non_null(obs[cell_type_key])


def _resolve_fixed_values(
    obs: pd.DataFrame,
    other_key: str,
    fixed_values: Sequence[Any] | None,
) -> list[Any]:
    if fixed_values is not None:
        deduped = list(dict.fromkeys(fixed_values))
        return deduped
    return _unique_non_null(obs[other_key])


def _build_masks(
    obs: pd.DataFrame,
    *,
    cell_type: Any,
    cell_type_key: str,
    swap_key: str,
    other_key: str,
    source_value: Any,
    target_value: Any,
    match_other_factor: bool,
    fixed_value: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cell_type_mask = obs[cell_type_key] == cell_type
    source_mask = cell_type_mask & (obs[swap_key] == source_value)
    target_mask = cell_type_mask & (obs[swap_key] == target_value)

    if match_other_factor:
        if fixed_value is None:
            raise ValueError("fixed_value must be provided when match_other_factor=True.")
        other_mask = obs[other_key] == fixed_value
        source_mask &= other_mask
        target_mask &= other_mask

    train_mask = ~target_mask.to_numpy(dtype=bool, copy=False)

    context_mask = (obs[swap_key] == target_value) & (obs[cell_type_key] != cell_type)
    if match_other_factor:
        context_mask &= obs[other_key] == fixed_value

    return (
        source_mask.to_numpy(dtype=bool, copy=False),
        target_mask.to_numpy(dtype=bool, copy=False),
        train_mask,
        context_mask.to_numpy(dtype=bool, copy=False),
    )


def _build_metadata(
    *,
    swap_type: Literal["condition", "niche"],
    cell_type: Any,
    swap_key: str,
    source_value: Any,
    target_value: Any,
    fixed_key: str | None,
    fixed_value: Any | None,
    match_other_factor: bool,
    source_mask: np.ndarray,
    target_mask: np.ndarray,
    context_mask: np.ndarray,
) -> dict[str, Any]:
    return {
        "swap_type": swap_type,
        "cell_type": cell_type,
        "swap_key": swap_key,
        "source_value": source_value,
        "target_value": target_value,
        "fixed_key": fixed_key,
        "fixed_value": fixed_value,
        "match_other_factor": match_other_factor,
        "n_source": int(source_mask.sum()),
        "n_target": int(target_mask.sum()),
        "n_context": int(context_mask.sum()),
        "split_type": "target_out",
        "context_mode": "target_context_available",
    }


def make_swap_specs(
    adata: AnnData,
    *,
    swap_type: Literal["condition", "niche"],
    source_value: Any,
    target_value: Any,
    cell_types: Sequence[str] | None = None,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    niche_key: str = "niche",
    match_other_factor: bool = True,
    fixed_values: Sequence[Any] | None = None,
    min_source_cells: int = 20,
    min_target_cells: int = 20,
) -> list[SwapSpec]:
    """Generate swap tasks for condition or niche OOD benchmarks."""
    if swap_type not in {"condition", "niche"}:
        raise ValueError("swap_type must be 'condition' or 'niche'.")
    if min_source_cells < 1 or min_target_cells < 1:
        raise ValueError("min_source_cells and min_target_cells must be >= 1.")
    if source_value == target_value:
        raise ValueError("source_value and target_value must differ.")

    if swap_type == "condition":
        swap_key = condition_key
        other_key = niche_key
        if match_other_factor:
            warnings.warn(
                "Condition swap with match_other_factor=True generates niche-matched tasks. "
                "Use match_other_factor=False for CellDISECT-compatible marginal condition transfer.",
                UserWarning,
                stacklevel=2,
            )
    else:
        swap_key = niche_key
        other_key = condition_key
        if not match_other_factor:
            raise ValueError("Niche swap with match_other_factor=False is unsupported in v0.2.0.")

    if not match_other_factor:
        if fixed_values is not None:
            raise ValueError("fixed_values must be None when match_other_factor=False.")

    obs = adata.obs
    _validate_obs_columns(obs, [cell_type_key, condition_key, niche_key])

    selected_cell_types = _resolve_cell_types(obs, cell_type_key, cell_types)
    if match_other_factor:
        selected_fixed_values = _resolve_fixed_values(obs, other_key, fixed_values)
        combinations = [(cell_type, fixed_value) for cell_type in selected_cell_types for fixed_value in selected_fixed_values]
    else:
        combinations = [(cell_type, None) for cell_type in selected_cell_types]

    specs: list[SwapSpec] = []
    seen_ids: set[str] = set()

    for cell_type, fixed_value in combinations:
        source_mask, target_mask, train_mask, context_mask = _build_masks(
            obs,
            cell_type=cell_type,
            cell_type_key=cell_type_key,
            swap_key=swap_key,
            other_key=other_key,
            source_value=source_value,
            target_value=target_value,
            match_other_factor=match_other_factor,
            fixed_value=fixed_value,
        )

        n_source = int(source_mask.sum())
        n_target = int(target_mask.sum())
        if n_source < min_source_cells or n_target < min_target_cells:
            continue

        fixed_key = other_key if match_other_factor else None
        task_id = _format_task_id(
            swap_type,
            str(cell_type),
            source_value,
            target_value,
            fixed_key=fixed_key,
            fixed_value=fixed_value,
        )
        if task_id in seen_ids:
            suffix = _stable_task_suffix(
                swap_type,
                cell_type,
                source_value,
                target_value,
                fixed_key,
                fixed_value,
            )
            task_id = f"{task_id}__hash={suffix}"
        seen_ids.add(task_id)

        metadata = _build_metadata(
            swap_type=swap_type,
            cell_type=cell_type,
            swap_key=swap_key,
            source_value=source_value,
            target_value=target_value,
            fixed_key=fixed_key,
            fixed_value=fixed_value,
            match_other_factor=match_other_factor,
            source_mask=source_mask,
            target_mask=target_mask,
            context_mask=context_mask,
        )

        specs.append(
            SwapSpec(
                task_id=task_id,
                swap_type=swap_type,
                cell_type=str(cell_type),
                source_mask=source_mask,
                target_mask=target_mask,
                train_mask=train_mask,
                context_mask=context_mask,
                source_value=source_value,
                target_value=target_value,
                fixed_key=fixed_key,
                fixed_value=fixed_value,
                metadata=metadata,
            )
        )

    if not specs:
        raise ValueError("No valid swap tasks were generated for the requested settings.")
    return specs


def summarize_swap_specs(specs: Sequence[SwapSpec]) -> pd.DataFrame:
    """Return one summary row per generated swap task."""
    rows: list[dict[str, Any]] = []
    for spec in specs:
        rows.append(
            {
                "task_id": spec.task_id,
                "swap_type": spec.swap_type,
                "cell_type": spec.cell_type,
                "source_value": spec.source_value,
                "target_value": spec.target_value,
                "fixed_key": spec.fixed_key,
                "fixed_value": spec.fixed_value,
                "n_source": spec.metadata.get("n_source", int(spec.source_mask.sum())),
                "n_target": spec.metadata.get("n_target", int(spec.target_mask.sum())),
                "n_context": spec.metadata.get("n_context", int(spec.context_mask.sum())),
            }
        )
    return pd.DataFrame(rows)
