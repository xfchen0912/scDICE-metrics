"""Counterfactual OOD swap benchmarker."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from scdice_metrics.benchmark._progress import iter_progress, print_status
from scdice_metrics.metrics._cellsimbench import (
    DegProfile,
    knn_jaccard_deltapert,
    mae_degs,
    mse_degs,
    nir_scores,
    pds_scores,
    pearson_deltactrl,
    pearson_deltactrl_degs,
    pearson_deltapert,
    pearson_deltapert_degs,
    r2_deltactrl,
    r2_deltactrl_degs,
    r2_deltapert,
    r2_deltapert_degs,
    weighted_r2_deltactrl,
    weighted_r2_deltapert,
    wmae,
    wmse,
)
from scdice_metrics.metrics._counterfactual import (
    _mean_profile,
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
    pseudobulk_log1p_delta_metrics,
    pseudobulk_log1p_reference_delta_metrics,
    signed_de_recovery,
    systema_pearson_delta_metrics,
    systema_reference_delta_metrics,
)

MetricType = bool | dict[str, Any]
MetricFn = Callable[..., float | dict[str, float]]

REFERENCE_METRICS = {
    "systema_pearson_delta",
    "systema_reference_delta",
    "delta_pearson",
    "delta_spearman",
    "delta_cosine",
    "delta_rmse",
    "delta_mae",
    "signed_de_recovery",
    "pseudobulk_log1p_delta",
    "pseudobulk_log1p_reference_delta",
    "pearson_deltactrl",
    "pearson_deltactrl_degs",
    "pearson_deltapert",
    "pearson_deltapert_degs",
    "r2_deltactrl",
    "r2_deltactrl_degs",
    "r2_deltapert",
    "r2_deltapert_degs",
    "weighted_r2_deltactrl",
    "weighted_r2_deltapert",
    "mse_degs",
    "mae_degs",
    "wmse",
    "wmae",
}

PANEL_METRICS = frozenset({"nir", "pds", "knn_jaccard_deltapert"})

CELLSIMBENCH_TASK_KWARGS_METRICS = frozenset(
    {
        "pearson_deltactrl",
        "pearson_deltactrl_degs",
        "pearson_deltapert",
        "pearson_deltapert_degs",
        "r2_deltactrl",
        "r2_deltactrl_degs",
        "r2_deltapert",
        "r2_deltapert_degs",
        "weighted_r2_deltactrl",
        "weighted_r2_deltapert",
        "mse_degs",
        "mae_degs",
        "wmse",
        "wmae",
    }
)

#: Reference metrics that additionally require a per-task ``template`` (a per-gene
#: reference shift). The value is the ``CounterfactualTask`` attribute to read it from, so
#: the same task can carry a template measured in the per-cell space and one measured in the
#: consistent-aggregation (pseudobulk) space.
TEMPLATE_METRICS = {
    "systema_reference_delta": "template",
    "pseudobulk_log1p_reference_delta": "template_pseudobulk",
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
    "systema_reference_delta_all_genes": {
        "display_name": "Systema reference Pearson-delta (template-removed)",
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
    "pearson_deltactrl": {
        "display_name": "Pearson (Δ Ctrl)",
        "group": "CellSimBench effect",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "pearson_deltactrl_degs": {
        "display_name": "Pearson (Δ Ctrl DEG top-100)",
        "group": "CellSimBench effect",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "pearson_deltapert": {
        "display_name": "Pearson (Δ Pert)",
        "group": "CellSimBench effect",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "pearson_deltapert_degs": {
        "display_name": "Pearson (Δ Pert DEG top-100)",
        "group": "CellSimBench effect",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "r2_deltactrl": {
        "display_name": "R² (Δ Ctrl)",
        "group": "CellSimBench effect",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "r2_deltactrl_degs": {
        "display_name": "R² (Δ Ctrl DEG top-100)",
        "group": "CellSimBench effect",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "r2_deltapert": {
        "display_name": "R² (Δ Pert)",
        "group": "CellSimBench effect",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "r2_deltapert_degs": {
        "display_name": "R² (Δ Pert DEG top-100)",
        "group": "CellSimBench effect",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "weighted_r2_deltactrl": {
        "display_name": "Weighted R² (Δ Ctrl)",
        "group": "CellSimBench effect",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "weighted_r2_deltapert": {
        "display_name": "Weighted R² (Δ Pert)",
        "group": "CellSimBench effect",
        "higher_is_better": True,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "mse_degs": {
        "display_name": "MSE (DEG top-100)",
        "group": "CellSimBench expression",
        "higher_is_better": False,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "mae_degs": {
        "display_name": "MAE (DEG top-100)",
        "group": "CellSimBench expression",
        "higher_is_better": False,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "wmse": {
        "display_name": "WMSE",
        "group": "CellSimBench expression",
        "higher_is_better": False,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "wmae": {
        "display_name": "WMAE",
        "group": "CellSimBench expression",
        "higher_is_better": False,
        "requires_reference": True,
        "supports_gene_indices": False,
    },
    "nir": {
        "display_name": "NIR",
        "group": "CellSimBench identity",
        "higher_is_better": True,
        "requires_reference": False,
        "supports_gene_indices": False,
    },
    "pds": {
        "display_name": "PDS",
        "group": "CellSimBench identity",
        "higher_is_better": True,
        "requires_reference": False,
        "supports_gene_indices": False,
    },
    "knn_jaccard_deltapert": {
        "display_name": "KNN Jaccard (Δ Pert)",
        "group": "CellSimBench manifold",
        "higher_is_better": True,
        "requires_reference": False,
        "supports_gene_indices": False,
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
    "systema_reference_delta": systema_reference_delta_metrics,
    "pseudobulk_log1p_delta": pseudobulk_log1p_delta_metrics,
    "pseudobulk_log1p_reference_delta": pseudobulk_log1p_reference_delta_metrics,
    "delta_pearson": delta_pearson,
    "delta_spearman": delta_spearman,
    "delta_cosine": delta_cosine,
    "delta_rmse": delta_rmse,
    "delta_mae": delta_mae,
    "signed_de_recovery": signed_de_recovery,
    "energy_distance": energy_distance,
    "mmd_rbf": mmd_rbf,
    "mean_gene_wasserstein": mean_gene_wasserstein,
    "pearson_deltactrl": pearson_deltactrl,
    "pearson_deltactrl_degs": pearson_deltactrl_degs,
    "pearson_deltapert": pearson_deltapert,
    "pearson_deltapert_degs": pearson_deltapert_degs,
    "r2_deltactrl": r2_deltactrl,
    "r2_deltactrl_degs": r2_deltactrl_degs,
    "r2_deltapert": r2_deltapert,
    "r2_deltapert_degs": r2_deltapert_degs,
    "weighted_r2_deltactrl": weighted_r2_deltactrl,
    "weighted_r2_deltapert": weighted_r2_deltapert,
    "mse_degs": mse_degs,
    "mae_degs": mae_degs,
    "wmse": wmse,
    "wmae": wmae,
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
    #: Optional per-gene reference shift for ``systema_reference_delta``: the average
    #: response of the *training* items (Systema's perturbed-centroid template). ``None``
    #: makes the template-removed metric degenerate to the plain control-referenced one.
    template: Any | None = None
    #: Same idea as ``template`` but measured in the consistent-aggregation space, for
    #: ``pseudobulk_log1p_reference_delta`` (the two templates are *not* interchangeable:
    #: one lives in ``mean_i log1p``, the other in ``log1p(mean_i)``).
    template_pseudobulk: Any | None = None
    #: Fold-specific dataset mean profile for CellSimBench ``deltapert`` metrics.
    dataset_mean: Any | None = None
    #: GT-half DEG weights and top-gene mask aligned to evaluation genes.
    deg: DegProfile | None = None


@dataclass(frozen=True)
class Counterfactual:
    pseudobulk_pearson: MetricType = True
    pseudobulk_spearman: MetricType = False
    pseudobulk_rmse: MetricType = True
    pseudobulk_mae: MetricType = False

    systema_pearson_delta: MetricType = field(default_factory=lambda: {"top_k": 20})
    #: Reference shifted by ``CounterfactualTask.template`` (Systema's perturbed-centroid
    #: reference). Off by default: it needs a per-task template to be supplied.
    systema_reference_delta: MetricType = False
    delta_pearson: MetricType = False
    delta_spearman: MetricType = True
    delta_cosine: MetricType = False
    delta_rmse: MetricType = True
    delta_mae: MetricType = False

    signed_de_recovery: MetricType = field(default_factory=lambda: {"top_k": 50})

    #: Effect metrics after aggregating both sides identically (``log1p(pseudobulk CPM)``).
    #: Removes the Jensen-gap mismatch between ``mean_i log1p`` (observations) and
    #: ``log1p(mean_i)`` (generative models). See docs §4.3.1.
    pseudobulk_log1p_delta: MetricType = False
    #: The same consistent aggregation with Systema's reference shift applied, i.e. the
    #: ``log1p(pseudobulk) x perturbed-centroid`` cell of the 2x2 in docs §4.7.
    pseudobulk_log1p_reference_delta: MetricType = False

    energy_distance: MetricType = True
    mmd_rbf: MetricType = False
    mean_gene_wasserstein: MetricType = False

    pearson_deltactrl: MetricType = False
    pearson_deltactrl_degs: MetricType = False
    pearson_deltapert: MetricType = False
    pearson_deltapert_degs: MetricType = False
    r2_deltactrl: MetricType = False
    r2_deltactrl_degs: MetricType = False
    r2_deltapert: MetricType = False
    r2_deltapert_degs: MetricType = False
    weighted_r2_deltactrl: MetricType = False
    weighted_r2_deltapert: MetricType = False
    mse_degs: MetricType = False
    mae_degs: MetricType = False
    wmse: MetricType = False
    wmae: MetricType = False
    nir: MetricType = False
    pds: MetricType = False
    knn_jaccard_deltapert: MetricType = field(default_factory=lambda: {"k": 20})

    @classmethod
    def cellsimbench(cls) -> Counterfactual:
        """Enable the CellSimBench paper metric subset (swap/OOD tasks still use task fields)."""
        return cls(
            pseudobulk_pearson=False,
            pseudobulk_rmse=False,
            pseudobulk_mae=False,
            systema_pearson_delta=False,
            delta_spearman=False,
            delta_rmse=False,
            delta_mae=False,
            signed_de_recovery=False,
            energy_distance=False,
            pearson_deltactrl=True,
            pearson_deltapert=True,
            r2_deltactrl=True,
            r2_deltapert=True,
            weighted_r2_deltactrl=True,
            weighted_r2_deltapert=True,
            mse_degs=True,
            wmse=True,
        )


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

        if task.template is not None:
            template = np.asarray(task.template).ravel()
            if template.size != n_genes:
                raise ValueError(
                    f"template length ({template.size}) does not match feature count ({n_genes}) "
                    f"for task {task.task_id!r}."
                )
            if not np.all(np.isfinite(template)):
                raise ValueError(f"template contains non-finite values for task {task.task_id!r}.")

        if task.template_pseudobulk is not None:
            template_pb = np.asarray(task.template_pseudobulk).ravel()
            if template_pb.size != n_genes:
                raise ValueError(
                    f"template_pseudobulk length ({template_pb.size}) does not match feature "
                    f"count ({n_genes}) for task {task.task_id!r}."
                )
            if not np.all(np.isfinite(template_pb)):
                raise ValueError(
                    f"template_pseudobulk contains non-finite values for task {task.task_id!r}."
                )

        if task.dataset_mean is not None:
            _validate_feature_dimensions((task.dataset_mean, "dataset_mean"))
            if _n_features(task.dataset_mean, name="dataset_mean") != n_genes:
                raise ValueError(
                    f"dataset_mean feature count does not match observed for task {task.task_id!r}."
                )

        if task.deg is not None:
            deg = task.deg
            if deg.weights.size != n_genes or deg.top_mask.size != n_genes:
                raise ValueError(f"deg weights/mask length mismatch for task {task.task_id!r}.")
            if task.deg.source == "predictor_half":
                warnings.warn(
                    f"Task {task.task_id!r} uses deg.source='predictor_half'; "
                    "CellSimBench recommends GT-half DEGs for evaluation.",
                    UserWarning,
                    stacklevel=2,
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
    if metric_key.startswith("pseudobulk_log1p_delta_"):
        return {
            "display_name": "Pseudobulk-log1p " + metric_key.removeprefix("pseudobulk_log1p_delta_").replace("_", " "),
            "group": "Effect fidelity (consistent aggregation)",
            "higher_is_better": not metric_key.endswith("_rmse"),
            "requires_reference": True,
            "supports_gene_indices": False,
        }
    if metric_key.startswith("pseudobulk_log1p_reference_delta_"):
        return {
            "display_name": "Pseudobulk-log1p reference "
            + metric_key.removeprefix("pseudobulk_log1p_reference_delta_").replace("_", " "),
            "group": "Effect fidelity (consistent aggregation, template-removed)",
            "higher_is_better": not metric_key.endswith("_rmse"),
            "requires_reference": True,
            "supports_gene_indices": False,
        }
    if metric_key.startswith("systema_pearson_delta_top") and metric_key.endswith("_true_effect"):
        return {
            "display_name": "Systema Pearson-delta (top-k true effect)",
            "group": "Effect fidelity",
            "higher_is_better": True,
            "requires_reference": True,
            "supports_gene_indices": False,
        }
    if metric_key.startswith("systema_reference_delta_top") and metric_key.endswith("_true_effect"):
        return {
            "display_name": "Systema reference Pearson-delta (template-removed, top-k true effect)",
            "group": "Effect fidelity",
            "higher_is_better": True,
            "requires_reference": True,
            "supports_gene_indices": False,
        }
    if metric_key.startswith("systema_reference_delta_top") and metric_key.endswith("_specific_effect"):
        return {
            "display_name": "Systema reference Pearson-delta (template-removed, top-k specific effect)",
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
    if metric_name == "systema_reference_delta":
        if not isinstance(raw, dict):
            raise TypeError("systema_reference_delta_metrics must return a dictionary.")
        top_k = int(kwargs.get("top_k", 20))
        keys = ("true_effect", "specific_effect")
        missing = [k for k in keys if f"top{top_k}_{k}" not in raw]
        if missing:
            raise KeyError(f"Missing {missing} in systema_reference_delta output.")
        out = {
            "systema_reference_delta_all_genes": float(raw["all_genes"]),
        }
        for key in keys:
            out[f"systema_reference_delta_top{top_k}_{key}"] = float(raw[f"top{top_k}_{key}"])
        return out
    if metric_name == "signed_de_recovery":
        if not isinstance(raw, dict):
            raise TypeError("signed_de_recovery must return a dictionary.")
        return {f"signed_de_recovery_{key}": float(value) for key, value in raw.items()}
    if metric_name == "pseudobulk_log1p_delta":
        if not isinstance(raw, dict):
            raise TypeError("pseudobulk_log1p_delta_metrics must return a dictionary.")
        return {f"pseudobulk_log1p_delta_{key}": float(value) for key, value in raw.items()}
    if metric_name == "pseudobulk_log1p_reference_delta":
        if not isinstance(raw, dict):
            raise TypeError(
                "pseudobulk_log1p_reference_delta_metrics must return a dictionary."
            )
        return {
            f"pseudobulk_log1p_reference_delta_{key}": float(value)
            for key, value in raw.items()
        }
    if isinstance(raw, dict):
        return {str(key): float(value) for key, value in raw.items()}
    return {metric_name: float(raw)}


def _cellsimbench_kwargs(task: CounterfactualTask | None) -> dict[str, Any]:
    if task is None:
        return {}
    return {"dataset_mean": task.dataset_mean, "deg": task.deg}


def _run_metric(
    metric_name: str,
    observed: Any,
    predicted: Any,
    reference: Any,
    kwargs: dict[str, Any],
    *,
    task: CounterfactualTask | None = None,
) -> dict[str, float]:
    metric_fn = METRIC_FUNCTIONS[metric_name]
    cs_kwargs = _cellsimbench_kwargs(task) if metric_name in CELLSIMBENCH_TASK_KWARGS_METRICS else {}
    if metric_name in TEMPLATE_METRICS:
        template_field = TEMPLATE_METRICS[metric_name]
        template = getattr(task, template_field, None) if task is not None else None
        raw = metric_fn(observed, predicted, reference, template=template, **cs_kwargs, **kwargs)
    elif metric_name in REFERENCE_METRICS:
        raw = metric_fn(observed, predicted, reference, **cs_kwargs, **kwargs)
    else:
        raw = metric_fn(observed, predicted, **kwargs)
    return _flatten_metric_outputs(metric_name, raw, kwargs)


def _covariate_groups(task_ids: Sequence[str], metadata_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for task_id, meta in zip(task_ids, metadata_rows, strict=True):
        cov = meta.get("covariate", "_all")
        groups.setdefault(str(cov), []).append(task_id)
    return groups


def _append_panel_metric_rows(
    tasks: Sequence[CounterfactualTask],
    enabled: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> None:
    if not enabled:
        return

    methods: set[str] = set()
    for task in tasks:
        methods.update(task.predicted.keys())

    for method in methods:
        pred_profiles: dict[str, np.ndarray] = {}
        truth_profiles: dict[str, np.ndarray] = {}
        pred_deltamean: dict[str, np.ndarray] = {}
        truth_deltamean: dict[str, np.ndarray] = {}
        metadata_by_task: dict[str, dict[str, Any]] = {}

        for task in tasks:
            if method not in task.predicted:
                continue
            tid = task.task_id
            metadata_by_task[tid] = _task_metadata_row(task)
            truth_profiles[tid] = _mean_profile(task.observed, name="observed")
            pred_profiles[tid] = _mean_profile(task.predicted[method], name="predicted")
            if task.dataset_mean is not None:
                mean = _mean_profile(task.dataset_mean, name="dataset_mean")
                truth_deltamean[tid] = truth_profiles[tid] - mean
                pred_deltamean[tid] = pred_profiles[tid] - mean

        if len(pred_profiles) < 2:
            continue

        task_ids = list(pred_profiles.keys())
        meta_rows = [metadata_by_task[tid] for tid in task_ids]
        cov_groups = _covariate_groups(task_ids, meta_rows)

        pred_df = pd.DataFrame.from_dict(pred_profiles, orient="index")
        truth_df = pd.DataFrame.from_dict(truth_profiles, orient="index")

        panel_scores: dict[str, dict[str, float]] = {}
        if "nir" in enabled:
            panel_scores["nir"] = nir_scores(pred_df, truth_df, covariate_groups=cov_groups)
        if "pds" in enabled:
            panel_scores["pds"] = pds_scores(pred_df, truth_df, covariate_groups=cov_groups)
        if "knn_jaccard_deltapert" in enabled and len(pred_deltamean) >= 2:
            k = int(enabled["knn_jaccard_deltapert"].get("k", 20))
            pred_dm = pd.DataFrame.from_dict(pred_deltamean, orient="index")
            truth_dm = pd.DataFrame.from_dict(truth_deltamean, orient="index")
            panel_scores["knn_jaccard_deltapert"] = knn_jaccard_deltapert(pred_dm, truth_dm, k=k)

        for metric_name, scores in panel_scores.items():
            info = _metric_info(metric_name)
            for task in tasks:
                if method not in task.predicted:
                    continue
                tid = task.task_id
                if tid not in scores:
                    continue
                rows.append(
                    {
                        "method": method,
                        "task_id": tid,
                        "metric": metric_name,
                        "display_name": info["display_name"],
                        "metric_group": info["group"],
                        "value": float(scores[tid]),
                        "higher_is_better": info["higher_is_better"],
                        "n_observed": _n_units(task.observed),
                        "n_predicted": _n_units(task.predicted[method]),
                        "n_reference": _n_units(task.reference),
                        **_task_metadata_row(task),
                    }
                )


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
        per_task_enabled = [(n, k) for n, k in enabled_metrics if n not in PANEL_METRICS]
        panel_enabled = {
            n: k for n, k in enabled_metrics if n in PANEL_METRICS
        }
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
                for metric_name, metric_kwargs in per_task_enabled:
                    if metric_name not in METRIC_FUNCTIONS:
                        continue
                    values = _run_metric(
                        metric_name,
                        task.observed,
                        predicted,
                        task.reference,
                        metric_kwargs,
                        task=task,
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

        _append_panel_metric_rows(self.tasks, panel_enabled, rows)

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
