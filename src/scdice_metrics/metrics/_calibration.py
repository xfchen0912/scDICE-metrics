"""Metric calibration utilities (CellSimBench dynamic range fraction)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

METRIC_DIRECTION: dict[str, dict[str, Any]] = {
    "mse": {"higher_is_better": False, "perfect": 0.0},
    "mse_degs": {"higher_is_better": False, "perfect": 0.0},
    "wmse": {"higher_is_better": False, "perfect": 0.0},
    "mae": {"higher_is_better": False, "perfect": 0.0},
    "mae_degs": {"higher_is_better": False, "perfect": 0.0},
    "wmae": {"higher_is_better": False, "perfect": 0.0},
    "pearson_deltactrl": {"higher_is_better": True, "perfect": 1.0},
    "pearson_deltactrl_degs": {"higher_is_better": True, "perfect": 1.0},
    "pearson_deltapert": {"higher_is_better": True, "perfect": 1.0},
    "pearson_deltapert_degs": {"higher_is_better": True, "perfect": 1.0},
    "r2_deltactrl": {"higher_is_better": True, "perfect": 1.0},
    "r2_deltactrl_degs": {"higher_is_better": True, "perfect": 1.0},
    "r2_deltapert": {"higher_is_better": True, "perfect": 1.0},
    "r2_deltapert_degs": {"higher_is_better": True, "perfect": 1.0},
    "weighted_r2_deltactrl": {"higher_is_better": True, "perfect": 1.0},
    "weighted_r2_deltapert": {"higher_is_better": True, "perfect": 1.0},
    "nir": {"higher_is_better": True, "perfect": 1.0},
    "pds": {"higher_is_better": True, "perfect": 1.0},
    "knn_jaccard_deltapert": {"higher_is_better": True, "perfect": 1.0},
}

DELTAPERT_METRIC_PREFIXES = ("pearson_deltapert", "r2_deltapert", "weighted_r2_deltapert", "knn_jaccard_deltapert")


def dynamic_range_fraction(
    pos: float,
    baseline: float,
    *,
    higher_is_better: bool,
    perfect: float,
    clip: tuple[float, float] = (-1.0, 1.0),
) -> float:
    """Dynamic range fraction (DRF) between baseline and perfect performance."""
    if np.isnan(pos) or np.isnan(baseline):
        return float(np.nan)
    eps = 1e-6
    if higher_is_better:
        denom = (perfect - baseline) + eps
        drf = (pos - baseline) / denom
    else:
        drf = (baseline - pos) / (baseline + eps)
    lo, hi = clip
    return float(np.clip(drf, lo, hi))


def _null_baseline_method_for_metric(metric: str) -> str:
    if any(metric.startswith(prefix) for prefix in DELTAPERT_METRIC_PREFIXES):
        return "control"
    return "dataset_mean"


def score_relative_to_baselines(
    long_df: pd.DataFrame,
    *,
    ceiling_method: str = "technical_duplicate",
    null_baseline_method: str | None = None,
    metric_config: Mapping[str, Mapping[str, Any]] | None = None,
    drf_column: str = "drf",
) -> pd.DataFrame:
    """Attach DRF scores by comparing each method to a null baseline and optional ceiling."""
    required = {"method", "task_id", "metric", "value"}
    missing = required - set(long_df.columns)
    if missing:
        raise ValueError(f"long_df missing columns: {sorted(missing)}")

    config = dict(METRIC_DIRECTION)
    if metric_config:
        config.update(metric_config)

    rows: list[dict[str, Any]] = []
    for (task_id, metric), group in long_df.groupby(["task_id", "metric"], dropna=False):
        cfg = config.get(str(metric))
        if cfg is None:
            continue
        null_method = null_baseline_method or _null_baseline_method_for_metric(str(metric))
        by_method = group.set_index("method")["value"]
        if null_method not in by_method.index:
            continue
        baseline_val = float(by_method[null_method])
        ceiling_val = float(by_method[ceiling_method]) if ceiling_method in by_method.index else float(np.nan)

        for method, val in by_method.items():
            if method in {null_method, ceiling_method}:
                continue
            pos = float(val)
            drf_vs_null = dynamic_range_fraction(
                pos,
                baseline_val,
                higher_is_better=bool(cfg["higher_is_better"]),
                perfect=float(cfg["perfect"]),
            )
            drf_vs_ceiling = float(np.nan)
            if not np.isnan(ceiling_val):
                drf_vs_ceiling = dynamic_range_fraction(
                    ceiling_val,
                    baseline_val,
                    higher_is_better=bool(cfg["higher_is_better"]),
                    perfect=float(cfg["perfect"]),
                )
            rows.append(
                {
                    "task_id": task_id,
                    "metric": metric,
                    "method": method,
                    "value": pos,
                    "null_baseline_method": null_method,
                    "ceiling_method": ceiling_method,
                    drf_column: drf_vs_null,
                    f"{drf_column}_ceiling": drf_vs_ceiling,
                }
            )
    return pd.DataFrame(rows)


def drf_all_versions(
    long_df: pd.DataFrame,
    *,
    metric_config: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Compute drf_mean, drf_ctrl, drf_interpolated-style columns via ceiling method swaps."""
    frames = [
        score_relative_to_baselines(
            long_df,
            ceiling_method="technical_duplicate",
            drf_column="drf_mean",
            metric_config=metric_config,
        ),
        score_relative_to_baselines(
            long_df,
            ceiling_method="interpolated_duplicate",
            null_baseline_method="control",
            drf_column="drf_ctrl",
            metric_config=metric_config,
        ),
        score_relative_to_baselines(
            long_df,
            ceiling_method="interpolated_duplicate",
            drf_column="drf_interpolated",
            metric_config=metric_config,
        ),
    ]
    return pd.concat([f for f in frames if not f.empty], ignore_index=True)
