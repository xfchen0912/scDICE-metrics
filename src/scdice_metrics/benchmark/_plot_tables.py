"""Benchmark result table plotting backends."""

from __future__ import annotations

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

TableStyle = Literal["plottable", "funkyheatmap"]

_AGGREGATE_SCORE_GROUP = "Aggregate score"


def _import_funky_heatmap():
    try:
        from funkyheatmappy import funky_heatmap
    except ImportError as exc:
        raise ImportError(
            "plot_results_table(style='funkyheatmap') requires the optional dependency "
            "'funkyheatmappy'. Install it with: pip install funkyheatmappy"
        ) from exc
    return funky_heatmap


def _palette_name(cmap: str | object, *, fallback: str) -> str:
    if isinstance(cmap, str):
        return cmap
    name = getattr(cmap, "name", None)
    return name if isinstance(name, str) and name else fallback


def build_funkyheatmap_column_specs(
    *,
    method_col: str,
    metric_cols: list[str],
    score_cols: list[str],
    metric_groups: pd.Series,
    circle_palette: str,
    score_palette: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Build ``column_info`` and ``column_groups`` for funkyheatmappy."""
    rows: list[dict[str, object]] = [
        {
            "id": method_col,
            "group": np.nan,
            "name": method_col,
            "geom": "text",
            "options": {"width": 6, "legend": False, "label": method_col},
            "palette": np.nan,
        }
    ]
    for col in metric_cols:
        rows.append(
            {
                "id": col,
                "group": metric_groups[col],
                "name": col.replace(" ", "\n", 1),
                "geom": "circle",
                "options": {"width": 1, "legend": False},
                "palette": circle_palette,
            }
        )
    for col in score_cols:
        rows.append(
            {
                "id": col,
                "group": _AGGREGATE_SCORE_GROUP,
                "name": col.replace(" ", "\n", 1),
                "geom": "bar",
                "options": {"width": 4, "legend": False},
                "palette": score_palette,
            }
        )

    column_info = pd.DataFrame(rows).set_index("id")
    group_names: list[str] = []
    for col in metric_cols:
        group = str(metric_groups[col])
        if group not in group_names:
            group_names.append(group)
    if score_cols and _AGGREGATE_SCORE_GROUP not in group_names:
        group_names.append(_AGGREGATE_SCORE_GROUP)

    column_groups = (
        pd.DataFrame({"group": group_names, "level1": group_names}) if group_names else None
    )
    return column_info, column_groups


def plot_funkyheatmap_table(
    plot_df: pd.DataFrame,
    *,
    method_col: str,
    metric_cols: list[str],
    score_cols: list[str],
    metric_groups: pd.Series,
    min_max_scale: bool,
    show: bool = True,
    save_dir: str | None = None,
    circle_cmap: str | object = "Blues",
    score_cmap: str | object = "Greens",
) -> Figure:
    """Render benchmark results with funkyheatmappy."""
    funky_heatmap = _import_funky_heatmap()

    data = plot_df.copy()
    data["id"] = data[method_col].astype(str)
    data[method_col] = data[method_col].astype(str)

    column_info, column_groups = build_funkyheatmap_column_specs(
        method_col=method_col,
        metric_cols=metric_cols,
        score_cols=list(score_cols),
        metric_groups=metric_groups,
        circle_palette=_palette_name(circle_cmap, fallback="Blues"),
        score_palette=_palette_name(score_cmap, fallback="Greens"),
    )

    with plt.rc_context({"svg.fonttype": "none"}):
        fig = funky_heatmap(
            data,
            column_info=column_info,
            column_groups=column_groups,
            scale_column=not min_max_scale,
            add_abc=False,
        )

    if show:
        plt.show()
    if save_dir is not None:
        fig.savefig(
            os.path.join(save_dir, "scib_results.svg"),
            facecolor=fig.get_facecolor(),
            dpi=300,
        )

    return fig
