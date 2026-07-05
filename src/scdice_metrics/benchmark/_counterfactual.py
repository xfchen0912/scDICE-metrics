"""Counterfactual OOD swap benchmarker."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scdice_metrics.benchmark._progress import iter_progress, print_status
from scdice_metrics.metrics._counterfactual import (
    _n_features,
    _validate_feature_dimensions,
    delta_cosine,
    delta_mae,
    delta_pearson,
    delta_rmse,
    delta_spearman,
    energy_distance,
    mean_gene_wasserstein,
    mmd_rbf,
    pseudobulk_mae,
    pseudobulk_pearson,
    pseudobulk_rmse,
    pseudobulk_spearman,
    signed_de_recovery,
    systema_pearson_delta_metrics,
)

MetricType = bool | dict[str, Any]
MetricFn = Callable[..., float | dict[str, float]]

REFERENCE_METRICS = {
    "systema_pearson_delta",
    "delta_pearson",
    "delta_spearman",
    "delta_cosine",
    "delta_rmse",
    "delta_mae",
    "signed_de_recovery",
}

COUNTERFACTUAL_METRIC_INFO: dict[str, dict[str, Any]] = {
    "pseudobulk_pearson": {
        "display_name": "Pseudobulk Pearson",
        "group": "Expression",
        "higher_is_better": True,
        "requires_reference": False,
        "supports_gene_indices": True,
    },
    "pseudobulk_spearman": {
        "display_name": "Pseudobulk Spearman",
        "group": "Expression",
        "higher_is_better": True,
        "requires_reference": False,
        "supports_gene_indices": True,
    },
    "pseudobulk_rmse": {
        "display_name": "Pseudobulk RMSE",
        "group": "Expression",
        "higher_is_better": False,
        "requires_reference": False,
        "supports_gene_indices": True,
    },
    "pseudobulk_mae": {
        "display_name": "Pseudobulk MAE",
        "group": "Expression",
        "higher_is_better": False,
        "requires_reference": False,
        "supports_gene_indices": True,
    },
    "systema_pearson_delta_all_genes": {
        "display_name": "Systema Pearson-delta (all genes)",
        "group": "Effect fidelity",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "delta_pearson": {
        "display_name": "Delta Pearson",
        "group": "Effect fidelity",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": True,
    },
    "delta_spearman": {
        "display_name": "Delta Spearman",
        "group": "Effect fidelity",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": True,
    },
    "delta_cosine": {
        "display_name": "Delta Cosine",
        "group": "Effect fidelity",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": True,
    },
    "delta_rmse": {
        "display_name": "Delta RMSE",
        "group": "Effect fidelity",
        "higher_is_better": False,
        "requires_reference": True,
        "supports_gene_indices": True,
    },
    "delta_mae": {
        "display_name": "Delta MAE",
        "group": "Effect fidelity",
        "higher_is_better": False,
        "requires_reference": True,
        "supports_gene_indices": True,
    },
    "energy_distance": {
        "display_name": "Energy Distance",
        "group": "Distribution",
        "higher_is_better": False,
        "requires_reference": False,
        "supports_gene_indices": False,
    },
    "mmd_rbf": {
        "display_name": "MMD (RBF)",
        "group": "Distribution",
        "higher_is_better": False,
        "requires_reference": False,
        "supports_gene_indices": False,
    },
    "mean_gene_wasserstein": {
        "display_name": "Mean Gene Wasserstein",
        "group": "Distribution",
        "higher_is_better": False,
        "requires_reference": False,
        "supports_gene_indices": True,
    },
}

SIGNED_DE_DISPLAY = {
    "precision": "Signed DE Precision",
    "recall": "Signed DE Recall",
    "f1": "Signed DE F1",
    "jaccard": "Signed DE Jaccard",
    "direction_accuracy_true_top": "Signed DE Direction Accuracy",
    "signed_precision": "Signed DE Signed Precision",
    "up_precision": "Signed DE Up Precision",
    "down_precision": "Signed DE Down Precision",
}

METRIC_FUNCTIONS: dict[str, MetricFn] = {
    "pseudobulk_pearson": pseudobulk_pearson,
    "pseudobulk_spearman": pseudobulk_spearman,
    "pseudobulk_rmse": pseudobulk_rmse,
    "pseudobulk_mae": pseudobulk_mae,
    "systema_pearson_delta": systema_pearson_delta_metrics,
    "delta_pearson": delta_pearson,
    "delta_spearman": delta_spearman,
    "delta_cosine": delta_cosine,
    "delta_rmse": delta_rmse,
    "delta_mae": delta_mae,
    "signed_de_recovery": signed_de_recovery,
    "energy_distance": energy_distance,
    "mmd_rbf": mmd_rbf,
    "mean_gene_wasserstein": mean_gene_wasserstein,
}

PARTITION_COLUMNS = ["swap_type", "match_other_factor"]


@dataclass
class CounterfactualTask:
    task_id: str
    observed: Any
    predicted: Mapping[str, Any]
    reference: Any
    gene_names: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Counterfactual:
    pseudobulk_pearson: MetricType = True
    pseudobulk_spearman: MetricType = False
    pseudobulk_rmse: MetricType = True
    pseudobulk_mae: MetricType = False

    systema_pearson_delta: MetricType = field(default_factory=lambda: {"top_k": 20})
    delta_pearson: MetricType = False
    delta_spearman: MetricType = True
    delta_cosine: MetricType = False
    delta_rmse: MetricType = True
    delta_mae: MetricType = False

    signed_de_recovery: MetricType = field(default_factory=lambda: {"top_k": 50})

    energy_distance: MetricType = True
    mmd_rbf: MetricType = False
    mean_gene_wasserstein: MetricType = False


def _n_units(X: Any) -> int:
    shape = getattr(X, "shape", None)
    if shape is None:
        raise TypeError("Input must have a shape attribute.")
    if len(shape) == 1:
        return 1
    if len(shape) == 2:
        return int(shape[0])
    raise ValueError("Input must be one- or two-dimensional.")


def _validate_tasks(tasks: Sequence[CounterfactualTask]) -> None:
    if not tasks:
        raise ValueError("At least one CounterfactualTask is required.")

    seen: set[str] = set()
    for task in tasks:
        if not task.task_id:
            raise ValueError("CounterfactualTask.task_id must be non-empty.")
        if task.task_id in seen:
            raise ValueError(f"Duplicate CounterfactualTask.task_id: {task.task_id}")
        seen.add(task.task_id)
        if task.reference is None:
            raise ValueError(f"CounterfactualTask.reference is required for task {task.task_id!r}.")
        if not task.predicted:
            raise ValueError(f"CounterfactualTask.predicted must contain at least one method for task {task.task_id!r}.")

        n_genes = _n_features(task.observed, name="observed")
        _validate_feature_dimensions(
            (task.observed, "observed"),
            (task.reference, "reference"),
        )
        for method, predicted in task.predicted.items():
            _validate_feature_dimensions((predicted, f"predicted[{method}]"))
            if _n_features(predicted, name=f"predicted[{method}]") != n_genes:
                raise ValueError(f"Feature count mismatch for method {method!r} in task {task.task_id!r}.")

        if task.gene_names is not None and len(task.gene_names) != n_genes:
            raise ValueError(
                f"gene_names length ({len(task.gene_names)}) does not match feature count ({n_genes}) "
                f"for task {task.task_id!r}."
            )


def _metric_kwargs(config: MetricType) -> dict[str, Any]:
    return dict(config) if isinstance(config, dict) else {}


def _iter_enabled_metrics(config: Counterfactual) -> list[tuple[str, dict[str, Any]]]:
    enabled: list[tuple[str, dict[str, Any]]] = []
    for metric_name, use_metric_or_kwargs in asdict(config).items():
        if use_metric_or_kwargs:
            enabled.append((metric_name, _metric_kwargs(use_metric_or_kwargs)))
    return enabled


def _metric_info(metric_key: str) -> dict[str, Any]:
    if metric_key in COUNTERFACTUAL_METRIC_INFO:
        return COUNTERFACTUAL_METRIC_INFO[metric_key]
    if metric_key.startswith("systema_pearson_delta_top") and metric_key.endswith("_true_effect"):
        return {
            "display_name": "Systema Pearson-delta (top-k true effect)",
            "group": "Effect fidelity",
            "higher_is_better": True,
            "requires_reference": True,
            "supports_gene_indices": False,
        }
    if metric_key.startswith("signed_de_recovery_"):
        suffix = metric_key.removeprefix("signed_de_recovery_")
        return {
            "display_name": SIGNED_DE_DISPLAY.get(suffix, metric_key),
            "group": "DE recovery",
            "higher_is_better": True,
            "requires_reference": True,
            "supports_gene_indices": False,
        }
    raise KeyError(f"Unknown metric key: {metric_key}")


def _flatten_metric_outputs(metric_name: str, raw: float | dict[str, float], kwargs: dict[str, Any]) -> dict[str, float]:
    if metric_name == "systema_pearson_delta":
        if not isinstance(raw, dict):
            raise TypeError("systema_pearson_delta_metrics must return a dictionary.")
        top_k = int(kwargs.get("top_k", 20))
        top_key = f"top{top_k}_true_effect"
        if top_key not in raw:
            raise KeyError(f"Missing {top_key!r} in systema_pearson_delta output.")
        return {
            "systema_pearson_delta_all_genes": float(raw["all_genes"]),
            f"systema_pearson_delta_top{top_k}_true_effect": float(raw[top_key]),
        }
    if metric_name == "signed_de_recovery":
        if not isinstance(raw, dict):
            raise TypeError("signed_de_recovery must return a dictionary.")
        return {f"signed_de_recovery_{key}": float(value) for key, value in raw.items()}
    if isinstance(raw, dict):
        return {str(key): float(value) for key, value in raw.items()}
    return {metric_name: float(raw)}


def _run_metric(
    metric_name: str,
    observed: Any,
    predicted: Any,
    reference: Any,
    kwargs: dict[str, Any],
) -> dict[str, float]:
    metric_fn = METRIC_FUNCTIONS[metric_name]
    if metric_name in REFERENCE_METRICS:
        raw = metric_fn(observed, predicted, reference, **kwargs)
    else:
        raw = metric_fn(observed, predicted, **kwargs)
    return _flatten_metric_outputs(metric_name, raw, kwargs)


def _aggregate(values: pd.Series, aggregate: str) -> float:
    if aggregate == "median":
        return float(values.median())
    if aggregate == "mean":
        return float(values.mean())
    raise ValueError("aggregate must be 'mean' or 'median'.")


def _task_metadata_row(task: CounterfactualTask) -> dict[str, Any]:
    row = dict(task.metadata)
    row.setdefault("task_id", task.task_id)
    return row


class CounterfactualBenchmarker:
    """Benchmark counterfactual swap predictions across tasks and methods."""

    def __init__(
        self,
        tasks: list[CounterfactualTask],
        counterfactual_metrics: Counterfactual | None = None,
        *,
        progress_bar: bool = True,
    ) -> None:
        _validate_tasks(tasks)
        self.tasks = tasks
        self.counterfactual_metrics = counterfactual_metrics or Counterfactual()
        self._progress_bar = progress_bar
        self._benchmarked = False
        self._results = pd.DataFrame()

    def benchmark(self) -> None:
        """Run configured metrics for every task and method."""
        if self._benchmarked:
            warnings.warn(
                "The benchmark has already been run. Running it again will overwrite the previous results.",
                UserWarning,
            )

        enabled_metrics = _iter_enabled_metrics(self.counterfactual_metrics)
        rows: list[dict[str, Any]] = []

        print_status("[bold cyan]Running counterfactual benchmark metrics…[/]", disable=not self._progress_bar)
        for task in iter_progress(
            self.tasks,
            description="[cyan]Counterfactual tasks",
            disable=not self._progress_bar,
            total=len(self.tasks),
        ):
            metadata = _task_metadata_row(task)
            for method, predicted in task.predicted.items():
                for metric_name, metric_kwargs in enabled_metrics:
                    values = _run_metric(
                        metric_name,
                        task.observed,
                        predicted,
                        task.reference,
                        metric_kwargs,
                    )
                    for metric_key, metric_value in values.items():
                        info = _metric_info(metric_key)
                        rows.append(
                            {
                                "method": method,
                                "task_id": task.task_id,
                                "metric": metric_key,
                                "display_name": info["display_name"],
                                "metric_group": info["group"],
                                "value": metric_value,
                                "higher_is_better": info["higher_is_better"],
                                "n_observed": _n_units(task.observed),
                                "n_predicted": _n_units(predicted),
                                "n_reference": _n_units(task.reference),
                                **metadata,
                            }
                        )

        self._results = pd.DataFrame(rows)
        self._benchmarked = True
        print_status("[bold green]✅ Counterfactual benchmark complete.[/]", disable=not self._progress_bar)

    def _require_results(self) -> pd.DataFrame:
        if not self._benchmarked:
            raise RuntimeError("Call benchmark() before requesting results.")
        return self._results

    def get_results(
        self,
        *,
        long_format: bool = True,
        aggregate: str = "median",
    ) -> pd.DataFrame:
        """Return task-level or aggregated benchmark results."""
        df = self._require_results().copy()
        if long_format:
            return df

        for col in PARTITION_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan

        grouped = (
            df.groupby(PARTITION_COLUMNS + ["method", "metric"], dropna=False)["value"]
            .agg(lambda s: _aggregate(s, aggregate))
            .reset_index()
        )
        wide = grouped.pivot_table(
            index=PARTITION_COLUMNS + ["method"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        display_map = df.drop_duplicates("metric").set_index("metric")["display_name"]
        wide = wide.rename(columns=lambda metric: display_map.get(metric, metric))
        return wide.sort_index()

    def get_swap_summary(
        self,
        *,
        by_cell_type: bool = False,
        aggregate: str = "median",
    ) -> pd.DataFrame:
        """Summarize benchmark results with swap-aware partitions."""
        df = self._require_results().copy()
        for col in PARTITION_COLUMNS + ["cell_type"]:
            if col not in df.columns:
                df[col] = np.nan

        if by_cell_type:
            group_cols = PARTITION_COLUMNS + ["method", "cell_type", "metric"]
            summary = (
                df.groupby(group_cols, dropna=False)["value"]
                .agg(lambda s: _aggregate(s, aggregate))
                .reset_index()
            )
        else:
            stage1 = (
                df.groupby(PARTITION_COLUMNS + ["method", "cell_type", "metric"], dropna=False)["value"]
                .agg(lambda s: _aggregate(s, aggregate))
                .reset_index()
            )
            summary = (
                stage1.groupby(PARTITION_COLUMNS + ["method", "metric"], dropna=False)["value"]
                .agg(lambda s: _aggregate(s, aggregate))
                .reset_index()
            )

        display_map = df.drop_duplicates("metric").set_index("metric")["display_name"]
        summary["display_name"] = summary["metric"].map(lambda metric: display_map.get(metric, metric))
        return summary
