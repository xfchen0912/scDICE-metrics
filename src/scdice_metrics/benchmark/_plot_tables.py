"""Benchmark result table plotting backends."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Literal, TypedDict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

TableStyle = Literal["plottable", "funkyheatmap"]

_METRIC_TYPE_ROW = "Metric Type"
_AGGREGATE_SCORE_LABEL = "Aggregate score"
_AGGREGATE_SCORE_GROUP = _AGGREGATE_SCORE_LABEL

# Poster-style funkyheatmap layout (figures/poster/scripts/embed_combo_table.py)
METHOD_COL_WIDTH = 3.6
METRIC_COL_WIDTH = 1.35
SCORE_COL_WIDTH = 2.6
GAP_COL_WIDTH = 1.0

FUNKY_POSITION_ARGS: dict[str, float] = dict(
    row_height=0.86,
    row_space=0.09,
    row_bigspace=0.4,
    col_width=0.86,
    col_space=0.08,
    col_bigspace=0.3,
    col_annot_offset=1.2,
    col_annot_angle=35,
    expand_xmin=0,
    expand_xmax=1.2,
    expand_ymin=0,
    expand_ymax=0,
)


class FunkyColumnSpec(TypedDict):
    id: str
    label: str
    group: str
    kind: Literal["metric", "score", "gap"]
    width: float


def _import_funky_heatmap():
    try:
        from funkyheatmappy import funky_heatmap
    except ImportError as exc:
        raise ImportError(
            "plot_results_table(style='funkyheatmap') requires the optional dependency "
            "'funkyheatmappy'. Install it with: pip install funkyheatmappy"
        ) from exc
    return funky_heatmap


def _import_calculate_column_positions():
    from funkyheatmappy.calculate_column_positions import calculate_column_positions

    return calculate_column_positions


def _palette_name(cmap: str | object, *, fallback: str) -> str:
    if isinstance(cmap, str):
        return cmap
    name = getattr(cmap, "name", None)
    return name if isinstance(name, str) and name else fallback


def _is_flat(values: pd.Series) -> bool:
    s = pd.to_numeric(values, errors="coerce").astype(float).dropna()
    if s.empty:
        return True
    return bool(s.max() - s.min() <= 1e-3 * max(1.0, abs(s.median())))


def circle_score_series(
    values: pd.Series,
    *,
    lower_is_better: bool = False,
    num_stds: float = 2.5,
    use_oriented_values: bool = False,
) -> pd.Series:
    """Map raw metric values to 0-1 circle colour scores (``plottable.cmap.normed_cmap`` style)."""
    v = pd.to_numeric(values, errors="coerce").astype(float)
    if use_oriented_values:
        out = v.clip(0.0, 1.0)
        return out.where(v.notna())

    median, std = v.median(), v.std()
    vmin, vmax = median - num_stds * std, median + num_stds * std
    if _is_flat(v) or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        out = pd.Series(0.5, index=v.index)
    else:
        out = ((v - vmin) / (vmax - vmin)).clip(0.0, 1.0)
    out = out.where(v.notna())
    return 1.0 - out if lower_is_better else out


def build_circle_score_frame(
    plot_df: pd.DataFrame,
    metric_cols: Sequence[str],
    *,
    lower_is_better_cols: frozenset[str] | set[str],
    min_max_scale: bool,
    num_stds: float = 2.5,
    oriented_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Circle colour scores aligned with :meth:`Benchmarker.plot_results_table` (plottable)."""
    scores = pd.DataFrame(index=plot_df.index)
    for col in metric_cols:
        if min_max_scale:
            scores[col] = plot_df[col]
            continue
        use_oriented = col in lower_is_better_cols and oriented_frame is not None
        source = oriented_frame[col] if use_oriented and col in oriented_frame.columns else plot_df[col]
        scores[col] = circle_score_series(
            source,
            lower_is_better=col in lower_is_better_cols and not use_oriented,
            num_stds=num_stds,
            use_oriented_values=use_oriented,
        )
    return scores


def split_benchmark_results_table(results: pd.DataFrame) -> tuple[pd.Series, list[str], list[str]]:
    """Split a :meth:`Benchmarker.get_results` frame into metric vs aggregate columns."""
    if _METRIC_TYPE_ROW not in results.index:
        raise ValueError(f"results table must contain a {_METRIC_TYPE_ROW!r} row")
    metric_type = results.loc[_METRIC_TYPE_ROW]
    score_cols = [c for c in results.columns if metric_type[c] == _AGGREGATE_SCORE_LABEL]
    metric_cols = [c for c in results.columns if c not in score_cols]
    return metric_type, metric_cols, score_cols


def build_funkyheatmap_column_specs(
    *,
    method_col: str,
    columns: Sequence[FunkyColumnSpec],
    circle_palette: str,
    score_palette: str,
    method_width: float = METHOD_COL_WIDTH,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Build ``column_info`` and ``column_groups`` for funkyheatmappy."""
    rows: list[dict[str, object]] = [
        {
            "id": method_col,
            "group": np.nan,
            "name": method_col,
            "geom": "text",
            "options": {"width": method_width, "legend": False, "label": method_col},
            "palette": np.nan,
        }
    ]
    for col in columns:
        cid = col["id"]
        name = col["label"].replace(" ", "\n", 1) if col["label"] else ""
        if col["kind"] == "gap":
            rows.append(
                {
                    "id": cid,
                    "group": col["group"],
                    "name": "",
                    "geom": "text",
                    "options": {"width": col["width"], "legend": False, "label": cid},
                    "palette": np.nan,
                }
            )
        elif col["kind"] == "metric":
            rows.append(
                {
                    "id": cid,
                    "group": col["group"],
                    "name": name,
                    "geom": "circle",
                    "options": {"width": col["width"], "legend": False},
                    "palette": circle_palette,
                }
            )
        else:
            rows.append(
                {
                    "id": cid,
                    "group": col["group"],
                    "name": name,
                    "geom": "bar",
                    "options": {"width": col["width"], "legend": False},
                    "palette": score_palette,
                }
            )

    column_info = pd.DataFrame(rows).set_index("id", drop=False)
    group_names: list[str] = []
    for col in columns:
        group = col["group"]
        if group not in group_names:
            group_names.append(group)
    column_groups = (
        pd.DataFrame({"group": group_names, "level1": group_names}) if group_names else None
    )
    return column_info, column_groups


def _hex_palettes(*cmap_names: str, steps: int = 101) -> dict[str, list[str]]:
    palettes: dict[str, list[str]] = {}
    for name in cmap_names:
        cmap = matplotlib.colormaps[name]
        palettes[name] = [matplotlib.colors.to_hex(cmap(i / (steps - 1))) for i in range(steps)]
    return palettes


def _hide_degenerate_patches(fig: Figure) -> None:
    """Hide zero-size patches so ``bbox_inches='tight'`` does not fail on empty circles."""
    renderer = fig.canvas.get_renderer()
    for ax in fig.axes:
        for patch in ax.patches:
            if not patch.get_visible():
                continue
            try:
                patch.get_window_extent(renderer)
            except Exception:
                patch.set_visible(False)


def _compact_legends(
    fig: Figure,
    *,
    width_in: float = 1.9,
    margin_in: float = 0.25,
    vgap_in: float = 0.30,
    aspect: float = 2.2,
) -> None:
    legend_axes = fig.axes[1:]
    if not legend_axes:
        return
    fig.canvas.draw()
    fig.set_layout_engine("none")

    fig_width, fig_height = fig.get_size_inches()
    table = fig.axes[0].get_position()
    x = table.x1 + margin_in / fig_width
    y = table.y1
    for ax in legend_axes:
        height = width_in / aspect / fig_height
        y -= height
        ax.set_position([x, y, width_in / fig_width, height])
        y -= vgap_in / fig_height


def _add_block_separators(
    fig: Figure,
    columns: Sequence[FunkyColumnSpec],
    *,
    method_col: str,
    method_width: float,
) -> None:
    calculate_column_positions = _import_calculate_column_positions()
    ids = [method_col] + [c["id"] for c in columns]
    groups: list = [None] + [c["group"] for c in columns]
    widths = [method_width] + [c["width"] for c in columns]
    pos_info = pd.DataFrame(
        {"id": ids, "group": groups, "width": widths, "overlay": False}
    ).set_index("id", drop=False)
    pos = calculate_column_positions(
        pos_info, FUNKY_POSITION_ARGS["col_space"], FUNKY_POSITION_ARGS["col_bigspace"]
    )

    boundaries: list[float] = []
    for i in range(1, len(ids)):
        if groups[i] != groups[i - 1]:
            boundaries.append(
                (float(pos.loc[ids[i - 1], "xmax"]) + float(pos.loc[ids[i], "xmin"])) / 2
            )
    if not boundaries:
        return

    ax = fig.axes[0]
    ys = [
        float(value)
        for coll in ax.collections
        for seg in coll.get_segments()
        for value in (seg[0][1], seg[1][1])
    ]
    if not ys:
        return
    ymin, ymax = min(ys), max(ys)
    for x in boundaries:
        ax.plot(
            [x, x],
            [ymin, ymax],
            color="black",
            linewidth=0.8,
            linestyle="solid",
            solid_capstyle="butt",
            zorder=6,
        )


def _assemble_funky_data(
    plot_df: pd.DataFrame,
    *,
    method_col: str,
    columns: Sequence[FunkyColumnSpec],
    circle_values: pd.DataFrame,
    bar_values: pd.DataFrame,
) -> pd.DataFrame:
    data = pd.DataFrame(index=plot_df.index)
    data["id"] = plot_df[method_col].astype(str)
    data[method_col] = plot_df[method_col].astype(str)
    for col in columns:
        cid = col["id"]
        if col["kind"] == "gap":
            data[cid] = ""
        elif col["kind"] == "metric":
            data[cid] = circle_values[cid]
        else:
            data[cid] = bar_values[cid].fillna(0.0)
    return data


def render_funkyheatmap_layout(
    plot_df: pd.DataFrame,
    *,
    method_col: str,
    columns: Sequence[FunkyColumnSpec],
    circle_values: pd.DataFrame,
    bar_values: pd.DataFrame,
    circle_cmap: str | object = "PRGn",
    score_cmap: str | object = "YlGnBu",
    method_width: float = METHOD_COL_WIDTH,
    show_legends: bool = True,
) -> Figure:
    """Low-level funkyheatmappy render (poster-style layout)."""
    funky_heatmap = _import_funky_heatmap()
    circle_palette = _palette_name(circle_cmap, fallback="PRGn")
    score_palette = _palette_name(score_cmap, fallback="YlGnBu")

    data = _assemble_funky_data(
        plot_df,
        method_col=method_col,
        columns=columns,
        circle_values=circle_values,
        bar_values=bar_values,
    )
    column_info, column_groups = build_funkyheatmap_column_specs(
        method_col=method_col,
        columns=columns,
        circle_palette=circle_palette,
        score_palette=score_palette,
        method_width=method_width,
    )
    palettes = _hex_palettes(circle_palette, score_palette)

    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        fig = funky_heatmap(
            data,
            column_info=column_info,
            column_groups=column_groups,
            palettes=palettes,
            position_args=FUNKY_POSITION_ARGS,
            scale_column=False,
            add_abc=False,
        )

    _add_block_separators(fig, columns, method_col=method_col, method_width=method_width)
    _hide_degenerate_patches(fig)
    if show_legends:
        _compact_legends(fig)
    return fig


def combine_benchmark_results_for_funkyheatmap(
    dataset_results: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]],
    *,
    method_order: Sequence[str] | None = None,
    aggregate_scope: Literal["overall", "per_dataset", "per_dataset_and_total"] = "overall",
    total_label: str = "Total",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[FunkyColumnSpec], list[str]]:
    """Merge several :meth:`Benchmarker.get_results` tables into one wide layout.

    Returns ``(plot_df, circle_values, bar_values, column_specs, method_keys)``.
    ``plot_df`` includes a ``Method`` column for display labels (initially equal to keys).
    """
    if isinstance(dataset_results, Mapping):
        blocks = list(dataset_results.items())
    else:
        blocks = list(dataset_results)
    if not blocks:
        raise ValueError("dataset_results must contain at least one dataset")

    parsed: list[tuple[str, pd.DataFrame, pd.Series, list[str], list[str]]] = []
    metric_names: list[str] | None = None
    for label, frame in blocks:
        metric_groups, metric_cols, score_cols = split_benchmark_results_table(frame)
        if metric_names is None:
            metric_names = list(metric_cols)
        elif list(metric_cols) != metric_names:
            raise ValueError(
                "all datasets must share the same metric columns for a combined funkyheatmap"
            )
        values = frame.drop(_METRIC_TYPE_ROW, errors="ignore").astype(np.float64)
        parsed.append((label, values, metric_groups, metric_cols, score_cols))

    method_keys: list[str]
    if method_order is not None:
        method_keys = [m for m in method_order if any(m in values.index for _, values, _, _, _ in parsed)]
    else:
        union: list[str] = []
        for _, values, _, _, _ in parsed:
            for key in values.index:
                if key not in union:
                    union.append(key)
        method_keys = union

    columns: list[FunkyColumnSpec] = []
    plot = pd.DataFrame(index=method_keys)
    circle = pd.DataFrame(index=method_keys)
    bar = pd.DataFrame(index=method_keys)

    for label, values, metric_groups, metric_cols, score_cols in parsed:
        block = values.reindex(method_keys)
        for col in metric_cols:
            cid = f"{label}|{col}"
            plot[cid] = block[col]
            columns.append(
                {
                    "id": cid,
                    "label": col,
                    "group": label,
                    "kind": "metric",
                    "width": METRIC_COL_WIDTH,
                }
            )
        if aggregate_scope in ("per_dataset", "per_dataset_and_total"):
            for col in score_cols:
                cid = f"{label}|{col}"
                plot[cid] = block[col]
                bar[cid] = block[col]
                columns.append(
                    {
                        "id": cid,
                        "label": col,
                        "group": _AGGREGATE_SCORE_GROUP,
                        "kind": "score",
                        "width": SCORE_COL_WIDTH,
                    }
                )

    if aggregate_scope in ("overall", "per_dataset_and_total"):
        score_cols = parsed[0][4]
        per_dataset_oriented = []
        for label, values, _, metric_cols, _ in parsed:
            block = values.reindex(method_keys)
            per_dataset_oriented.append(
                block[metric_cols].mean(axis=1, skipna=True).rename(label)
            )
        oriented_block = pd.concat(per_dataset_oriented, axis=1)
        total = oriented_block.mean(axis=1, skipna=True)
        total[oriented_block.isna().any(axis=1)] = np.nan
        cid = "__total" if aggregate_scope == "overall" else total_label
        plot[cid] = total
        bar[cid] = total
        columns.append(
            {
                "id": cid,
                "label": total_label if aggregate_scope != "overall" else "Total",
                "group": _AGGREGATE_SCORE_GROUP,
                "kind": "score",
                "width": SCORE_COL_WIDTH,
            }
        )

    plot["Method"] = plot.index.astype(str)
    return plot, circle, bar, columns, method_keys


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
    circle_cmap: str | object = "PRGn",
    score_cmap: str | object = "YlGnBu",
    circle_values: pd.DataFrame | None = None,
    bar_values: pd.DataFrame | None = None,
    lower_is_better_cols: frozenset[str] | set[str] | None = None,
    circle_num_stds: float = 2.5,
    oriented_frame: pd.DataFrame | None = None,
    show_legends: bool = True,
) -> Figure:
    """Render benchmark results with funkyheatmappy (single-dataset layout)."""
    columns: list[FunkyColumnSpec] = []
    for col in metric_cols:
        columns.append(
            {
                "id": col,
                "label": col,
                "group": str(metric_groups[col]),
                "kind": "metric",
                "width": METRIC_COL_WIDTH,
            }
        )
    for col in score_cols:
        columns.append(
            {
                "id": col,
                "label": col,
                "group": _AGGREGATE_SCORE_GROUP,
                "kind": "score",
                "width": SCORE_COL_WIDTH,
            }
        )

    if circle_values is None:
        circle_values = build_circle_score_frame(
            plot_df,
            metric_cols,
            lower_is_better_cols=lower_is_better_cols or frozenset(),
            min_max_scale=min_max_scale,
            num_stds=circle_num_stds,
            oriented_frame=oriented_frame,
        )
    if bar_values is None:
        bar_values = plot_df[list(score_cols)]

    fig = render_funkyheatmap_layout(
        plot_df,
        method_col=method_col,
        columns=columns,
        circle_values=circle_values,
        bar_values=bar_values,
        circle_cmap=circle_cmap,
        score_cmap=score_cmap,
        show_legends=show_legends,
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


def plot_funkyheatmap_multi_dataset_table(
    dataset_results: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]],
    *,
    method_col: str = "Method",
    min_max_scale: bool = False,
    aggregate_scope: Literal["overall", "per_dataset", "per_dataset_and_total"] = "overall",
    method_order: Sequence[str] | None = None,
    method_labels: Mapping[str, str] | None = None,
    lower_is_better_cols: frozenset[str] | set[str] | None = None,
    circle_num_stds: float = 2.5,
    oriented_by_dataset: Mapping[str, pd.DataFrame] | None = None,
    show: bool = True,
    save_dir: str | None = None,
    save_path: str | None = None,
    circle_cmap: str | object = "PRGn",
    score_cmap: str | object = "YlGnBu",
    show_legends: bool = True,
) -> Figure:
    """Outer wrapper: one funkyheatmap table with a column block per dataset.

    Each value in ``dataset_results`` must be a :meth:`Benchmarker.get_results` frame
    (including the ``Metric Type`` row).
    """
    plot_df, circle, bar, columns, method_keys = combine_benchmark_results_for_funkyheatmap(
        dataset_results,
        method_order=method_order,
        aggregate_scope=aggregate_scope,
    )
    if method_labels:
        plot_df[method_col] = [method_labels.get(str(k), str(k)) for k in method_keys]
    else:
        plot_df[method_col] = plot_df.index.astype(str)

    lower_cols = lower_is_better_cols or frozenset()
    for col_spec in columns:
        if col_spec["kind"] != "metric":
            continue
        cid = col_spec["id"]
        metric_label = col_spec["label"]
        series = plot_df[cid]
        if min_max_scale:
            circle[cid] = series
            continue
        oriented_series = None
        if oriented_by_dataset is not None and "|" in cid:
            block_key = cid.split("|", 1)[0]
            oriented_frame = oriented_by_dataset.get(block_key)
            if oriented_frame is not None and metric_label in oriented_frame.columns:
                oriented_series = oriented_frame[metric_label].reindex(plot_df.index)
        if metric_label in lower_cols and oriented_series is not None:
            circle[cid] = circle_score_series(oriented_series, use_oriented_values=True)
        else:
            circle[cid] = circle_score_series(
                series,
                lower_is_better=metric_label in lower_cols,
                num_stds=circle_num_stds,
            )

    for col_spec in columns:
        if col_spec["kind"] == "score":
            cid = col_spec["id"]
            bar[cid] = plot_df[cid]

    fig = render_funkyheatmap_layout(
        plot_df,
        method_col=method_col,
        columns=columns,
        circle_values=circle,
        bar_values=bar,
        circle_cmap=circle_cmap,
        score_cmap=score_cmap,
        show_legends=show_legends,
    )

    if show:
        plt.show()
    if save_dir is not None:
        fig.savefig(
            os.path.join(save_dir, "scib_results.svg"),
            facecolor=fig.get_facecolor(),
            dpi=300,
        )
    if save_path is not None:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")

    return fig
