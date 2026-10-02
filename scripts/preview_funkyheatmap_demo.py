#!/usr/bin/env python3
"""Preview funkyheatmap tables on synthetic benchmark results."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scdice_metrics.benchmark._plot_tables import (
    plot_funkyheatmap_multi_dataset_table,
    plot_funkyheatmap_table,
)

OUT_DIR = Path(__file__).resolve().parent.parent / "docs" / "_preview_funkyheatmap"
METRIC_TYPE = "Metric Type"
AGG = "Aggregate score"


def _results_frame(
    methods: list[str],
    metric_names: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a minimal get_results()-shaped table."""
    data = {name: rng.uniform(0.35, 0.95, size=len(methods)) for name in metric_names}
    data["Bio conservation"] = np.mean([data[n] for n in metric_names], axis=0)
    df = pd.DataFrame(data, index=methods)
    df.loc[METRIC_TYPE] = ["Bio conservation"] * len(metric_names) + [AGG]
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    methods = ["KALEIDO", "scVI", "scANVI", "Scanorama", "Unintegrated"]
    metric_names = ["Acc", "NMI", "ARI"]

    single = _results_frame(methods, metric_names, rng)
    plot_df = single.drop(METRIC_TYPE).astype(float)
    plot_df["Method"] = plot_df.index
    metric_cols = ["Acc", "NMI", "ARI"]
    score_cols = ["Bio conservation"]
    metric_groups = single.loc[METRIC_TYPE]

    fig1 = plot_funkyheatmap_table(
        plot_df,
        method_col="Method",
        metric_cols=metric_cols,
        score_cols=score_cols,
        metric_groups=metric_groups,
        min_max_scale=False,
        show=False,
        circle_cmap="PRGn",
        score_cmap="YlGnBu",
    )
    p1 = OUT_DIR / "demo_single_dataset_funkyheatmap.pdf"
    fig1.savefig(p1, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig1)
    print(f"wrote {p1}")

    ds_a = _results_frame(methods, metric_names, rng)
    ds_b = _results_frame(methods, metric_names, rng)
    ds_b.loc["KALEIDO", "Acc"] = 0.97
    ds_b.loc["KALEIDO", "NMI"] = 0.94

    fig2 = plot_funkyheatmap_multi_dataset_table(
        [
            ("Mouse brain\n(coronal)", ds_a),
            ("Human NSCLC\n(CosMx)", ds_b),
        ],
        aggregate_scope="overall",
        min_max_scale=False,
        show=False,
        circle_cmap="PRGn",
        score_cmap="YlGnBu",
    )
    p2 = OUT_DIR / "demo_multi_dataset_funkyheatmap.pdf"
    fig2.savefig(p2, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
