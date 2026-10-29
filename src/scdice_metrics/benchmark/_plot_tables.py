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
POSTER_METHOD_COL_WIDTH = 6.0
METRIC_COL_WIDTH = 1.35
SCORE_COL_WIDTH = 2.6
GAP_COL_WIDTH = 1.0
PLOTTABLE_METRIC_COL_WIDTH = 0.62
PLOTTABLE_SCORE_COL_WIDTH = 1.5
PLOTTABLE_GAP_COL_WIDTH = 0.6
DEFAULT_SUBGROUP_AGG_LABELS: dict[str, str] = {
    "Accuracy": "Acc",
    "Continuity": "Cont",
    "Score": "Total",
}

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


class FunkyColumnSpec(TypedDict, total=False):
    id: str
    label: str
    group: str
    kind: Literal["metric", "score", "gap"]
    width: float
    pwidth: float
    metric_key: str


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


def orient_column_higher_is_better(
    values: pd.Series, *, lower_is_better: bool = False
) -> pd.Series:
    """Min-max a metric column across methods; higher always means better."""
    v = pd.to_numeric(values, errors="coerce").astype(float)
    if _is_flat(v):
        return pd.Series(1.0, index=v.index).where(v.notna())
    lo, hi = v.min(), v.max()
    if hi <= lo:
        out = pd.Series(1.0, index=v.index)
    else:
        out = (v - lo) / (hi - lo)
    out = 1.0 - out if lower_is_better else out
    return out.where(v.notna())


def _metric_subgroups(
    metric_keys: Sequence[str],
    metric_subgroups: Sequence[tuple[str, Sequence[str]]] | None,
) -> list[tuple[str, tuple[str, ...]]]:
    if metric_subgroups is None:
        return [("Score", tuple(metric_keys))]
    return [(name, tuple(keys)) for name, keys in metric_subgroups]


def _resolve_method_keys(
    blocks: Sequence[tuple[str, str, pd.DataFrame]],
    method_order: Sequence[str] | None,
) -> list[str]:
    if method_order is not None:
        present = set()
        for _, _, frame in blocks:
            present.update(frame.index.astype(str))
        return [str(m) for m in method_order if str(m) in present]
    union: list[str] = []
    for _, _, frame in blocks:
        for key in frame.index.astype(str):
            if key not in union:
                union.append(key)
    return union


def combine_wide_benchmark_metrics(
    blocks: Sequence[tuple[str, str, pd.DataFrame]],
    *,
    metric_keys: Sequence[str],
    metric_labels: Mapping[str, str] | None = None,
    metric_subgroups: Sequence[tuple[str, Sequence[str]]] | None = None,
    gap_between_subgroups: bool = False,
    lower_is_better: frozenset[str] | set[str] = frozenset(),
    method_order: Sequence[str] | None = None,
    split_aggregate: bool = False,
    aggregate_scope: Literal["overall", "per_dataset", "per_dataset_and_total"] = "overall",
    subgroup_aggregate_labels: Mapping[str, str] | None = None,
    total_label: str = "Total",
    circle_num_stds: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[FunkyColumnSpec], list[str]]:
    """Merge wide metric tables (methods × metrics) into one poster-style layout.

    Each block is ``(block_id, column_group_header, frame)`` where ``frame`` is indexed
    by method keys and holds raw metric values. Returns display values, circle colour
    scores, bar scores (min-max oriented), column specs, and method keys (unsorted).
    """
    if not blocks:
        raise ValueError("blocks must contain at least one dataset")
    labels = dict(metric_labels or {})
    agg_labels = dict(DEFAULT_SUBGROUP_AGG_LABELS)
    if subgroup_aggregate_labels:
        agg_labels.update(subgroup_aggregate_labels)
    subgroups = _metric_subgroups(metric_keys, metric_subgroups)
    lower = set(lower_is_better)
    method_keys = _resolve_method_keys(blocks, method_order)

    plot = pd.DataFrame(index=method_keys, dtype=float)
    circle = pd.DataFrame(index=method_keys, dtype=float)
    oriented = pd.DataFrame(index=method_keys, dtype=float)
    columns: list[FunkyColumnSpec] = []

    for block_id, group_header, frame in blocks:
        block = frame.reindex(method_keys)
        seen_sub = 0
        for sub_name, sub_keys in subgroups:
            if gap_between_subgroups and seen_sub > 0:
                gap_id = f"{block_id}|__gap{seen_sub}"
                columns.append(
                    {
                        "id": gap_id,
                        "label": "",
                        "group": group_header,
                        "kind": "gap",
                        "width": GAP_COL_WIDTH,
                        "pwidth": PLOTTABLE_GAP_COL_WIDTH,
                    }
                )
            for metric in sub_keys:
                if metric not in block.columns:
                    raise KeyError(f"metric {metric!r} missing from block {block_id!r}")
                cid = f"{block_id}|{metric}"
                series = block[metric]
                plot[cid] = series
                circle[cid] = circle_score_series(
                    series,
                    lower_is_better=metric in lower,
                    num_stds=circle_num_stds,
                )
                oriented[cid] = orient_column_higher_is_better(
                    series, lower_is_better=metric in lower
                )
                columns.append(
                    {
                        "id": cid,
                        "label": labels.get(metric, metric),
                        "group": group_header,
                        "kind": "metric",
                        "width": METRIC_COL_WIDTH,
                        "pwidth": PLOTTABLE_METRIC_COL_WIDTH,
                        "metric_key": metric,
                    }
                )
            seen_sub += 1
            if split_aggregate:
                agg_id = f"{block_id}|__{sub_name}"
                oriented[agg_id] = oriented[
                    [f"{block_id}|{m}" for m in sub_keys]
                ].mean(axis=1)

    if aggregate_scope in ("per_dataset", "per_dataset_and_total"):
        for block_id, _, _ in blocks:
            for sub_name, _ in subgroups:
                if not split_aggregate and sub_name != subgroups[0][0]:
                    continue
                src = f"{block_id}|__{sub_name}" if split_aggregate else None
                if src is None:
                    src_cols = [f"{block_id}|{m}" for m in metric_keys]
                    src = None
                    values = oriented[src_cols].mean(axis=1)
                else:
                    values = oriented[src]
                cid = f"{block_id}|__{sub_name}" if split_aggregate else f"{block_id}|__Score"
                label = (
                    f"{block_id} {agg_labels.get(sub_name, sub_name)}"
                    if split_aggregate
                    else block_id
                )
                plot[cid] = values
                oriented[cid] = values
                circle[cid] = values
                columns.append(
                    {
                        "id": cid,
                        "label": label,
                        "group": _AGGREGATE_SCORE_GROUP,
                        "kind": "score",
                        "width": SCORE_COL_WIDTH,
                        "pwidth": PLOTTABLE_SCORE_COL_WIDTH,
                    }
                )

    agg_cols: list[str] = []
    if aggregate_scope in ("overall", "per_dataset_and_total"):
        if split_aggregate and aggregate_scope == "overall":
            for sub_name, _ in subgroups:
                per_ds = [f"{bid}|__{sub_name}" for bid, _, _ in blocks]
                overall = oriented[per_ds].mean(axis=1)
                overall[oriented[per_ds].isna().any(axis=1)] = np.nan
                cid = f"__agg_{sub_name}"
                oriented[cid] = overall
                plot[cid] = overall
                circle[cid] = overall
                agg_cols.append(cid)
                bar_label = (
                    agg_labels.get(sub_name, sub_name)
                    if len(subgroups) > 1
                    else total_label
                )
                columns.append(
                    {
                        "id": cid,
                        "label": bar_label,
                        "group": _AGGREGATE_SCORE_GROUP,
                        "kind": "score",
                        "width": SCORE_COL_WIDTH,
                        "pwidth": PLOTTABLE_SCORE_COL_WIDTH,
                    }
                )
        elif aggregate_scope == "overall" and not split_aggregate:
            per_ds = []
            for block_id, _, _ in blocks:
                per_ds.append(
                    oriented[[f"{block_id}|{m}" for m in metric_keys]].mean(axis=1).rename(block_id)
                )
            block = pd.concat(per_ds, axis=1)
            total = block.mean(axis=1, skipna=True)
            total[block.isna().any(axis=1)] = np.nan
            sub_name = subgroups[0][0]
            cid = f"__agg_{sub_name}"
            plot[cid] = total
            oriented[cid] = total
            circle[cid] = total
            agg_cols.append(cid)
            columns.append(
                {
                    "id": cid,
                    "label": total_label,
                    "group": _AGGREGATE_SCORE_GROUP,
                    "kind": "score",
                    "width": SCORE_COL_WIDTH,
                    "pwidth": PLOTTABLE_SCORE_COL_WIDTH,
                }
            )
        elif aggregate_scope == "per_dataset_and_total":
            per_ds = []
            for block_id, _, _ in blocks:
                cols = [f"{block_id}|__{s}" for s, _ in subgroups] if split_aggregate else [
                    f"{block_id}|{m}" for m in metric_keys
                ]
                per_ds.append(oriented[cols].mean(axis=1).rename(block_id))
            block = pd.concat(per_ds, axis=1)
            total = block.mean(axis=1, skipna=True)
            total[block.isna().any(axis=1)] = np.nan
            cid = total_label if total_label != "Total" else "__total"
            plot[cid] = total
            oriented[cid] = total
            circle[cid] = total
            agg_cols.append(cid)
            columns.append(
                {
                    "id": cid,
                    "label": total_label,
                    "group": _AGGREGATE_SCORE_GROUP,
                    "kind": "score",
                    "width": SCORE_COL_WIDTH,
                    "pwidth": PLOTTABLE_SCORE_COL_WIDTH,
                }
            )

    if agg_cols and "__total" in oriented.columns:
        sort_col = "__total"
    elif agg_cols:
        sort_col = agg_cols[0]
    else:
        score_ids = [c["id"] for c in columns if c["kind"] == "score"]
        sort_col = score_ids[0] if score_ids else None
    if sort_col and sort_col in oriented.columns:
        order = oriented[sort_col].sort_values(ascending=False, na_position="last").index
        plot = plot.loc[order]
        circle = circle.loc[order]
        oriented = oriented.loc[order]
        method_keys = list(order)

    ids = [c["id"] for c in columns]
    for frame in (plot, circle, oriented):
        for col in columns:
            if col["kind"] == "gap" and col["id"] not in frame.columns:
                frame[col["id"]] = np.nan
    plot = plot[ids]
    circle = circle[ids]
    bar = oriented[ids]
    plot["Method"] = plot.index.astype(str)
    return plot, circle, bar, columns, method_keys


def sort_and_select_table_columns(
    plot_df: pd.DataFrame,
    circle_values: pd.DataFrame,
    bar_values: pd.DataFrame,
    columns: Sequence[FunkyColumnSpec],
    *,
    sort_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[FunkyColumnSpec]]:
    """Reorder rows by an aggregate column and align value frames to ``columns``."""
    cols = list(columns)
    ids = [c["id"] for c in cols]
    if sort_col is None:
        score_ids = [c["id"] for c in cols if c["kind"] == "score"]
        sort_col = "__total" if "__total" in bar_values.columns else (score_ids[0] if score_ids else None)
    if sort_col and sort_col in bar_values.columns:
        order = bar_values[sort_col].sort_values(ascending=False, na_position="last").index
        plot_df = plot_df.loc[order]
        circle_values = circle_values.loc[order]
        bar_values = bar_values.loc[order]
    for frame in (plot_df, circle_values, bar_values):
        for col in cols:
            if col["kind"] == "gap" and col["id"] not in frame.columns:
                frame[col["id"]] = np.nan
    plot_df = plot_df[ids]
    circle_values = circle_values[ids]
    bar_values = bar_values[ids]
    return plot_df, circle_values, bar_values, cols


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
    metric_subgroups: Sequence[tuple[str, Sequence[str]]] | None = None,
    gap_between_subgroups: bool = False,
    split_aggregate: bool = False,
    lower_is_better: frozenset[str] | set[str] | None = None,
    metric_labels: Mapping[str, str] | None = None,
    circle_num_stds: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[FunkyColumnSpec], list[str]]:
    """Merge several :meth:`Benchmarker.get_results` tables into one wide layout.

    Returns ``(plot_df, circle_values, bar_values, column_specs, method_keys)``.
    ``plot_df`` includes a ``Method`` column for display labels (initially equal to keys).
    """
    if isinstance(dataset_results, Mapping):
        items = list(dataset_results.items())
    else:
        items = list(dataset_results)
    if not items:
        raise ValueError("dataset_results must contain at least one dataset")

    metric_names: list[str] | None = None
    wide_blocks: list[tuple[str, str, pd.DataFrame]] = []
    for label, frame in items:
        _, metric_cols, score_cols = split_benchmark_results_table(frame)
        if metric_names is None:
            metric_names = list(metric_cols)
        elif list(metric_cols) != metric_names:
            raise ValueError(
                "all datasets must share the same metric columns for a combined funkyheatmap"
            )
        values = frame.drop(_METRIC_TYPE_ROW, errors="ignore").astype(np.float64)
        wide_blocks.append((label, label, values))

    assert metric_names is not None
    lower = lower_is_better or frozenset()
    plot, circle, bar, columns, method_keys = combine_wide_benchmark_metrics(
        wide_blocks,
        metric_keys=metric_names,
        metric_labels=metric_labels,
        metric_subgroups=metric_subgroups,
        gap_between_subgroups=gap_between_subgroups,
        lower_is_better=lower,
        method_order=method_order,
        split_aggregate=split_aggregate,
        aggregate_scope=aggregate_scope,
        total_label=total_label,
        circle_num_stds=circle_num_stds,
    )

    if aggregate_scope in ("per_dataset", "per_dataset_and_total") and not split_aggregate:
        for label, frame in items:
            _, metric_cols, score_cols = split_benchmark_results_table(frame)
            values = frame.drop(_METRIC_TYPE_ROW, errors="ignore").astype(np.float64)
            block = values.reindex(method_keys)
            for col in score_cols:
                cid = f"{label}|{col}"
                if cid not in plot.columns:
                    plot[cid] = block[col]
                    bar[cid] = block[col]
                    circle[cid] = block[col]
                    columns.append(
                        {
                            "id": cid,
                            "label": col,
                            "group": _AGGREGATE_SCORE_GROUP,
                            "kind": "score",
                            "width": SCORE_COL_WIDTH,
                            "pwidth": PLOTTABLE_SCORE_COL_WIDTH,
                        }
                    )

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
    metric_subgroups: Sequence[tuple[str, Sequence[str]]] | None = None,
    gap_between_subgroups: bool = False,
    split_aggregate: bool = False,
    show: bool = True,
    save_dir: str | None = None,
    save_path: str | None = None,
    circle_cmap: str | object = "PRGn",
    score_cmap: str | object = "YlGnBu",
    show_legends: bool = True,
    method_col_width: float = METHOD_COL_WIDTH,
) -> Figure:
    """Outer wrapper: one funkyheatmap table with a column block per dataset.

    Each value in ``dataset_results`` must be a :meth:`Benchmarker.get_results` frame
    (including the ``Metric Type`` row).
    """
    plot_df, circle, bar, columns, method_keys = combine_benchmark_results_for_funkyheatmap(
        dataset_results,
        method_order=method_order,
        aggregate_scope=aggregate_scope,
        metric_subgroups=metric_subgroups,
        gap_between_subgroups=gap_between_subgroups,
        split_aggregate=split_aggregate,
        lower_is_better=lower_is_better_cols or frozenset(),
        circle_num_stds=circle_num_stds,
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
        method_width=method_col_width,
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


def _blank_formatter(precision: int = 2):
    def fmt(value):
        return "" if pd.isna(value) else f"{float(value):.{precision}f}"

    return fmt


def _color_lookup(raw_col: pd.Series, oriented_col: pd.Series, cmap_name: str):
    cmap = matplotlib.colormaps[cmap_name]
    lut = {
        round(float(r), 8): float(o)
        for r, o in zip(raw_col, oriented_col, strict=True)
        if not pd.isna(r) and not pd.isna(o)
    }

    def fn(value):
        if pd.isna(value):
            return (1.0, 1.0, 1.0, 0.0)
        good = lut.get(round(float(value), 8))
        return (1.0, 1.0, 1.0, 0.0) if good is None else cmap(good)

    return fn


def _star_simulated_plottable_cells(table, raw: pd.DataFrame, simulated: pd.DataFrame, columns) -> None:
    if simulated is None or not simulated.to_numpy().any():
        return
    row_of = {key: i for i, key in enumerate(raw.index)}
    for col in columns:
        if col["kind"] != "metric":
            continue
        cid = col["id"]
        if cid not in simulated.columns or cid not in table.columns:
            continue
        cells = {cell.index[0]: cell for cell in table.get_column(cid).cells}
        for key, row in row_of.items():
            if bool(simulated.loc[key, cid]) and row in cells:
                cell = cells[row]
                cell.text.set_text(f"{cell.text.get_text()}*")


def _add_simulated_footnote(
    fig: Figure, simulated: pd.DataFrame | None, *, y: float = 0.004, va: str = "bottom"
) -> None:
    if simulated is None or not simulated.to_numpy().any():
        return
    fig.text(
        0.005,
        y,
        "* simulated placeholder (layout preview only)",
        fontsize=7,
        ha="left",
        va=va,
        color="#6B6B6B",
    )


def plot_plottable_multi_dataset_table(
    plot_df: pd.DataFrame,
    circle_values: pd.DataFrame,
    bar_values: pd.DataFrame,
    columns: Sequence[FunkyColumnSpec],
    *,
    method_col: str = "Method",
    circle_cmap: str | object = "PRGn",
    score_cmap: str | object = "YlGnBu",
    text_fontsize: int = 10,
    highlight_method_keys: frozenset[str] | set[str] | None = None,
    highlight_color: str = "#D62728",
    simulated: pd.DataFrame | None = None,
    show: bool = True,
    save_path: str | os.PathLike[str] | None = None,
) -> Figure:
    """Render a combined benchmark table with ``plottable`` (poster-style borders/gaps)."""
    from plottable import ColumnDefinition, Table
    from plottable.plots import bar

    circle_name = _palette_name(circle_cmap, fallback="PRGn")
    score_name = _palette_name(score_cmap, fallback="YlGnBu")
    score_cmap_obj = matplotlib.colormaps[score_name]

    metric_ids = [c["id"] for c in columns if c["kind"] == "metric"]
    score = circle_values
    plot_body = pd.DataFrame(index=plot_df.index)
    plot_body[method_col] = plot_df[method_col].astype(str)
    for col in columns:
        cid = col["id"]
        if col["kind"] == "gap":
            plot_body[cid] = np.nan
        elif col["kind"] == "metric":
            plot_body[cid] = plot_df[cid]
        else:
            plot_body[cid] = bar_values[cid]

    col_defs = [
        ColumnDefinition(
            method_col,
            width=2.5,
            textprops={"ha": "left", "weight": "bold"},
        ),
    ]
    seen_groups: list[str] = []
    score_id = next((c["id"] for c in columns if c["kind"] == "score"), None)
    for col in columns:
        cid = col["id"]
        starts_block = col["group"] not in seen_groups
        if starts_block:
            seen_groups.append(col["group"])
        left_border = "left" if (starts_block and len(seen_groups) > 1) else None
        pwidth = col.get("pwidth", col["width"])
        if col["kind"] == "gap":
            plot_body[cid] = np.nan
            col_defs.append(
                ColumnDefinition(
                    cid,
                    title="",
                    width=pwidth,
                    group=col["group"],
                    textprops={"ha": "center"},
                    formatter=_blank_formatter(2),
                    border=left_border,
                )
            )
        elif col["kind"] == "metric":
            col_defs.append(
                ColumnDefinition(
                    cid,
                    title=col["label"],
                    width=pwidth,
                    group=col["group"],
                    textprops={"ha": "center", "bbox": {"boxstyle": "circle", "pad": 0.22}},
                    cmap=_color_lookup(plot_df[cid], score[cid], circle_name),
                    formatter=_blank_formatter(2),
                    border=left_border,
                )
            )
        else:
            plot_body[cid] = bar_values[cid]
            col_defs.append(
                ColumnDefinition(
                    cid,
                    title=col["label"],
                    width=pwidth,
                    group=col["group"],
                    plot_fn=bar,
                    plot_kw={
                        "cmap": score_cmap_obj,
                        "plot_bg_bar": False,
                        "annotate": True,
                        "height": 0.9,
                        "formatter": "{:.2f}",
                    },
                    border="left" if (cid == score_id or left_border) else None,
                )
            )

    if highlight_method_keys:
        for i, key in enumerate(plot_body.index):
            if str(key) in highlight_method_keys:
                col_defs[0].textprops = col_defs[0].textprops or {}
                # plottable applies column textprops uniformly; row highlight via index order below
                pass

    width = 2.5 + sum(c_def.width for c_def in col_defs[1:]) + 0.6
    height = 1.9 + 0.34 * len(plot_body)
    fig, ax = plt.subplots(figsize=(width, height))
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42}):
        table = Table(
            plot_body,
            cell_kw={"linewidth": 0, "edgecolor": "k"},
            column_definitions=col_defs,
            ax=ax,
            row_dividers=True,
            footer_divider=True,
            textprops={"fontsize": text_fontsize, "ha": "center"},
            row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
            col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
            column_border_kw={"linewidth": 1, "linestyle": "-"},
            index_col=method_col,
        )
        table.autoset_fontcolors(colnames=plot_body.columns)
        if highlight_method_keys:
            method_cells = table.get_column(method_col).cells
            for cell in method_cells:
                row_key = plot_body.index[cell.index[0]]
                if str(row_key) in highlight_method_keys:
                    cell.text.set_color(highlight_color)
                    cell.text.set_weight("bold")
    if simulated is not None:
        _star_simulated_plottable_cells(table, plot_df, simulated, columns)
        _add_simulated_footnote(fig, simulated, y=0.004, va="bottom")

    if show:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")
    return fig


def plot_multi_dataset_benchmark_table(
    dataset_results: Mapping[str, pd.DataFrame] | Sequence[tuple[str, pd.DataFrame]] | None = None,
    *,
    wide_blocks: Sequence[tuple[str, str, pd.DataFrame]] | None = None,
    metric_keys: Sequence[str] | None = None,
    metric_labels: Mapping[str, str] | None = None,
    style: TableStyle = "funkyheatmap",
    method_col: str = "Method",
    min_max_scale: bool = False,
    aggregate_scope: Literal["overall", "per_dataset", "per_dataset_and_total"] = "overall",
    method_order: Sequence[str] | None = None,
    method_labels: Mapping[str, str] | None = None,
    lower_is_better_cols: frozenset[str] | set[str] | None = None,
    circle_num_stds: float = 2.5,
    metric_subgroups: Sequence[tuple[str, Sequence[str]]] | None = None,
    gap_between_subgroups: bool = False,
    split_aggregate: bool = False,
    show: bool = True,
    save_path: str | os.PathLike[str] | None = None,
    circle_cmap: str | object = "PRGn",
    score_cmap: str | object = "YlGnBu",
    show_legends: bool = True,
    method_col_width: float = METHOD_COL_WIDTH,
    highlight_method_keys: frozenset[str] | set[str] | None = None,
    highlight_color: str = "#D62728",
    simulated: pd.DataFrame | None = None,
    footnote_y: float | None = None,
) -> Figure:
    """Combined multi-dataset benchmark table (``plottable`` or ``funkyheatmap``)."""
    if wide_blocks is not None:
        if not metric_keys:
            raise ValueError("metric_keys is required when wide_blocks is set")
        plot_df, circle, bar, columns, method_keys = combine_wide_benchmark_metrics(
            wide_blocks,
            metric_keys=metric_keys,
            metric_labels=metric_labels,
            metric_subgroups=metric_subgroups,
            gap_between_subgroups=gap_between_subgroups,
            lower_is_better=lower_is_better_cols or frozenset(),
            method_order=method_order,
            split_aggregate=split_aggregate,
            aggregate_scope=aggregate_scope,
            circle_num_stds=circle_num_stds,
        )
    elif dataset_results is not None:
        plot_df, circle, bar, columns, method_keys = combine_benchmark_results_for_funkyheatmap(
            dataset_results,
            method_order=method_order,
            aggregate_scope=aggregate_scope,
            metric_subgroups=metric_subgroups,
            gap_between_subgroups=gap_between_subgroups,
            split_aggregate=split_aggregate,
            lower_is_better=lower_is_better_cols or frozenset(),
            metric_labels=metric_labels,
            circle_num_stds=circle_num_stds,
        )
    else:
        raise ValueError("provide dataset_results or wide_blocks")
    if method_labels:
        plot_df[method_col] = [method_labels.get(str(k), str(k)) for k in method_keys]
    else:
        plot_df[method_col] = [str(k) for k in method_keys]

    if min_max_scale:
        for col in columns:
            if col["kind"] == "metric":
                circle[col["id"]] = plot_df[col["id"]]

    if style == "plottable":
        return plot_plottable_multi_dataset_table(
            plot_df,
            circle,
            bar,
            columns,
            method_col=method_col,
            circle_cmap=circle_cmap,
            score_cmap=score_cmap,
            highlight_method_keys=highlight_method_keys,
            highlight_color=highlight_color,
            simulated=simulated,
            show=show,
            save_path=save_path,
        )

    if style != "funkyheatmap":
        raise ValueError("style must be 'plottable' or 'funkyheatmap'")

    fig = render_funkyheatmap_layout(
        plot_df,
        method_col=method_col,
        columns=columns,
        circle_values=circle,
        bar_values=bar,
        circle_cmap=circle_cmap,
        score_cmap=score_cmap,
        show_legends=show_legends,
        method_width=method_col_width,
    )
    if simulated is not None:
        _add_simulated_footnote(
            fig,
            simulated,
            y=0.995 if footnote_y is None else footnote_y,
            va="top",
        )
    if show:
        plt.show()
    if save_path is not None:
        fig.savefig(save_path, facecolor=fig.get_facecolor(), dpi=300, bbox_inches="tight")
    return fig
