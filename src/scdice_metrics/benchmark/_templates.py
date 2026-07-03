from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import pandas as pd

BenchmarkMode = Literal["auto", "legacy", "scib", "sdmbench", "full"]

# Raw metric keys as stored in Benchmarker._results (before clean_names).
BIO_CONSERVATION_METRICS = (
    "isolated_labels",
    "nmi_ari_cluster_labels_leiden_nmi",
    "nmi_ari_cluster_labels_leiden_ari",
    "nmi_ari_cluster_labels_kmeans_nmi",
    "nmi_ari_cluster_labels_kmeans_ari",
    "silhouette_label",
    "clisi_knn",
)
BATCH_CORRECTION_METRICS = (
    "bras",
    "ilisi_knn",
    "kbet_per_label",
    "graph_connectivity",
    "pcr_comparison",
)
DISENTANGLEMENT_METRICS = (
    "mig_score",
    "mig_mean_factor_mi",
    "mig_mean_complement_mi",
    "mixed_ksg_mig_max_mig",
    "mixed_ksg_mig_concat_mig",
    "mixed_ksg_mig_min_mig",
    "classifier_attribute_gap_concat_gap",
    "classifier_attribute_gap_max_gap",
    "classifier_attribute_gap_mean_accuracy",
    "classifier_attribute_gap_mean_complement_accuracy",
    "classifier_attribute_gap_mean_competitor_accuracy",
    "fairness_leakage_accuracy",
    "fairness_leakage_demographic_parity_difference",
    "fairness_leakage_demographic_parity_ratio",
    "fairness_leakage_equalized_odds_difference",
)
SPATIAL_ACCURACY_METRICS = ("hom", "com")
SPATIAL_CONTINUITY_METRICS = ("chaos", "pas")

# Metrics where smaller raw values indicate better performance.
LOWER_IS_BETTER_METRICS = frozenset(SPATIAL_CONTINUITY_METRICS)


def orient_metrics_higher_is_better(
    df: pd.DataFrame,
    lower_is_better: frozenset[str] = LOWER_IS_BETTER_METRICS,
) -> pd.DataFrame:
    """Min-max each metric across embeddings and flip so higher always means better.

    Used for per-group aggregate scores (scIB-style), independent of ``min_max_scale``
    on the displayed per-metric table values.
    """
    out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)
    for metric in df.index:
        row = pd.to_numeric(df.loc[metric], errors="coerce").astype(float)
        lo, hi = row.min(), row.max()
        if hi > lo:
            norm = (row - lo) / (hi - lo)
        else:
            norm = pd.Series(1.0, index=row.index)
        if metric in lower_is_better:
            norm = 1.0 - norm
        out.loc[metric] = norm
    return out

_RUN_COLLECTION_TO_METRICS: dict[str, tuple[str, ...]] = {
    "Bio conservation": BIO_CONSERVATION_METRICS,
    "Batch correction": BATCH_CORRECTION_METRICS,
    "Disentanglement": DISENTANGLEMENT_METRICS,
    "Spatial clustering": SPATIAL_ACCURACY_METRICS + SPATIAL_CONTINUITY_METRICS,
}


@dataclass(frozen=True)
class BenchmarkTemplate:
    """Display and aggregation template for :class:`~scdice_metrics.benchmark.Benchmarker`.

    Maps individual metric keys to named groups shown in ``get_results`` and
    ``plot_results_table``. ``parent_groups`` is metadata only (plot uses flat
    ``_METRIC_TYPE`` groups like scIB-metrics, not nested headers).

    Parameters
    ----------
    name
        Short identifier for this template.
    groups
        Mapping from display group name to metric keys (raw names in results).
    weights
        Relative weights for aggregate score columns. Normalized to sum to 1.
        If ``None``, groups are weighted equally.
    parent_groups
        Optional mapping from display group to a parent section header used only
        for table grouping (e.g. ``{"Accuracy": "Spatial clustering"}``).
    lower_is_better
        Metric keys for which oriented scores use ``1 - x`` after min-max scaling.
    """

    name: str
    groups: dict[str, tuple[str, ...]]
    weights: dict[str, float] | None = None
    parent_groups: dict[str, str] = field(default_factory=dict)
    lower_is_better: frozenset[str] = LOWER_IS_BETTER_METRICS

    def __post_init__(self) -> None:
        object.__setattr__(self, "groups", {k: tuple(v) for k, v in self.groups.items()})

    def metric_to_group(self, metric_key: str, fallback: str | None = None) -> str:
        for group, metrics in self.groups.items():
            if metric_key in metrics:
                return group
        return fallback if fallback is not None else metric_key

    def resolve_weights(self, active_groups: Sequence[str]) -> dict[str, float]:
        if not active_groups:
            return {}
        if self.weights is not None:
            weights = {g: self.weights[g] for g in active_groups if g in self.weights}
            missing = [g for g in active_groups if g not in weights]
            if missing:
                for g in missing:
                    weights[g] = 1.0
        else:
            weights = dict.fromkeys(active_groups, 1.0)
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("Template weights must sum to a positive value.")
        return {g: w / total for g, w in weights.items()}

    def active_groups(self, present_metrics: set[str]) -> dict[str, list[str]]:
        active: dict[str, list[str]] = {}
        for group, metrics in self.groups.items():
            matched = [m for m in metrics if m in present_metrics]
            if matched:
                active[group] = matched
        return active

    @classmethod
    def scib(cls, weights: Mapping[str, float] | None = None) -> BenchmarkTemplate:
        """scIB-style template: Bio conservation + Batch correction."""
        default_weights = {"Bio conservation": 0.6, "Batch correction": 0.4}
        merged = default_weights if weights is None else {**default_weights, **dict(weights)}
        return cls(
            name="scib",
            groups={
                "Bio conservation": BIO_CONSERVATION_METRICS,
                "Batch correction": BATCH_CORRECTION_METRICS,
            },
            weights=merged,
        )

    @classmethod
    def sdmbench(cls, weights: Mapping[str, float] | None = None) -> BenchmarkTemplate:
        """SDMBench-style template: spatial Accuracy + Continuity."""
        default_weights = {"Accuracy": 0.5, "Continuity": 0.5}
        merged = default_weights if weights is None else {**default_weights, **dict(weights)}
        return cls(
            name="sdmbench",
            groups={
                "Accuracy": SPATIAL_ACCURACY_METRICS,
                "Continuity": SPATIAL_CONTINUITY_METRICS,
            },
            weights=merged,
            parent_groups={"Accuracy": "Spatial clustering", "Continuity": "Spatial clustering"},
        )

    @classmethod
    def full(cls, weights: Mapping[str, float] | None = None) -> BenchmarkTemplate:
        """Combined template for integration + spatial + disentanglement benchmarks."""
        default_weights = {
            "Bio conservation": 0.3,
            "Batch correction": 0.2,
            "Accuracy": 0.2,
            "Continuity": 0.15,
            "Disentanglement": 0.15,
        }
        merged = default_weights if weights is None else {**default_weights, **dict(weights)}
        return cls(
            name="full",
            groups={
                "Bio conservation": BIO_CONSERVATION_METRICS,
                "Batch correction": BATCH_CORRECTION_METRICS,
                "Accuracy": SPATIAL_ACCURACY_METRICS,
                "Continuity": SPATIAL_CONTINUITY_METRICS,
                "Disentanglement": DISENTANGLEMENT_METRICS,
            },
            weights=merged,
            parent_groups={"Accuracy": "Spatial clustering", "Continuity": "Spatial clustering"},
        )

    @classmethod
    def legacy_from_collections(cls, collection_names: Sequence[str]) -> BenchmarkTemplate:
        """One display group per enabled metric collection (previous default behavior)."""
        groups = {name: _RUN_COLLECTION_TO_METRICS[name] for name in collection_names if name in _RUN_COLLECTION_TO_METRICS}
        return cls(name="legacy", groups=groups, weights=None)

    @classmethod
    def from_mode(cls, mode: BenchmarkMode, collection_names: Sequence[str]) -> BenchmarkTemplate:
        if mode == "legacy":
            return cls.legacy_from_collections(collection_names)
        if mode == "scib":
            return cls.scib()
        if mode == "sdmbench":
            return cls.sdmbench()
        if mode == "full":
            return cls.full()
        if mode == "auto":
            return infer_template(collection_names)
        raise ValueError(f"Unknown benchmark mode: {mode!r}")


def infer_template(collection_names: Sequence[str]) -> BenchmarkTemplate:
    """Pick a template from enabled metric collections."""
    names = set(collection_names)
    if names == {"Bio conservation", "Batch correction"}:
        return BenchmarkTemplate.scib()
    if names == {"Spatial clustering"}:
        return BenchmarkTemplate.sdmbench()
    if names <= {"Bio conservation", "Batch correction", "Spatial clustering", "Disentanglement"}:
        # Build a template containing only groups that have at least one enabled collection.
        groups: dict[str, tuple[str, ...]] = {}
        weights: dict[str, float] = {}
        parent_groups: dict[str, str] = {}
        if "Bio conservation" in names:
            groups["Bio conservation"] = BIO_CONSERVATION_METRICS
            weights["Bio conservation"] = 0.3
        if "Batch correction" in names:
            groups["Batch correction"] = BATCH_CORRECTION_METRICS
            weights["Batch correction"] = 0.2
        if "Spatial clustering" in names:
            groups["Accuracy"] = SPATIAL_ACCURACY_METRICS
            groups["Continuity"] = SPATIAL_CONTINUITY_METRICS
            weights["Accuracy"] = 0.25
            weights["Continuity"] = 0.15
            parent_groups = {"Accuracy": "Spatial clustering", "Continuity": "Spatial clustering"}
        if "Disentanglement" in names:
            groups["Disentanglement"] = DISENTANGLEMENT_METRICS
            weights["Disentanglement"] = 0.1
        if not groups:
            return BenchmarkTemplate.legacy_from_collections(collection_names)
        return BenchmarkTemplate(
            name="auto",
            groups=groups,
            weights=weights,
            parent_groups=parent_groups,
        )
    return BenchmarkTemplate.legacy_from_collections(collection_names)


def resolve_template(
    template: BenchmarkTemplate | BenchmarkMode | None,
    collection_names: Sequence[str],
    default_mode: BenchmarkMode = "auto",
) -> BenchmarkTemplate:
    if template is None:
        return BenchmarkTemplate.from_mode(default_mode, collection_names)
    if isinstance(template, str):
        return BenchmarkTemplate.from_mode(template, collection_names)
    return template
