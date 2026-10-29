import numpy as np
import pytest
from scipy import sparse

from scdice_metrics.benchmark._counterfactual import (
    Counterfactual,
    CounterfactualBenchmarker,
    CounterfactualTask,
)
from scdice_metrics.benchmark._swap import make_swap_specs
from tests.utils.swap_fixtures import make_swap_benchmark_adata
from scdice_metrics.metrics import DegProfile


def _expression_from_mask(adata, mask):
    return adata[mask].X


def _build_task_from_spec(adata, spec, predictions, *, metadata_extra=None):
    metadata = dict(spec.metadata)
    if metadata_extra:
        metadata.update(metadata_extra)
    return CounterfactualTask(
        task_id=spec.task_id,
        observed=_expression_from_mask(adata, spec.target_mask),
        predicted=predictions,
        reference=_expression_from_mask(adata, spec.source_mask),
        gene_names=adata.var_names,
        metadata=metadata,
    )


def _perfect_predictions(adata, spec):
    target = _expression_from_mask(adata, spec.target_mask)
    if sparse.issparse(target):
        noisy = target.copy()
        noisy.data = noisy.data + 0.01
    else:
        noisy = target + 0.01
    return {"model_a": target, "model_b": noisy}


@pytest.fixture
def swap_adata():
    return make_swap_benchmark_adata(seed=0, cells_per_combo=5)


def test_multiple_methods_and_long_format_results(swap_adata):
    specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    task = _build_task_from_spec(swap_adata, specs[0], _perfect_predictions(swap_adata, specs[0]))
    bm = CounterfactualBenchmarker(
        tasks=[task],
        counterfactual_metrics=Counterfactual(
            pseudobulk_pearson=True,
            delta_rmse=True,
            systema_pearson_delta={"top_k": 10},
            signed_de_recovery={"top_k": 10},
            energy_distance=False,
            mmd_rbf=False,
        ),
        progress_bar=False,
    )
    bm.benchmark()
    results = bm.get_results(long_format=True)
    assert set(results["method"]) == {"model_a", "model_b"}
    assert "systema_pearson_delta_top10_true_effect" in set(results["metric"])
    assert "signed_de_recovery_precision" in set(results["metric"])
    assert results["swap_type"].iloc[0] == "condition"


def test_metadata_propagation_and_unequal_prediction_sizes(swap_adata):
    specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    target = _expression_from_mask(swap_adata, specs[0].target_mask)
    source = _expression_from_mask(swap_adata, specs[0].source_mask)
    task = CounterfactualTask(
        task_id=specs[0].task_id,
        observed=target,
        predicted={"small": target[:3], "full": target},
        reference=source,
        metadata={**specs[0].metadata, "dataset": "fixture"},
    )
    bm = CounterfactualBenchmarker(
        tasks=[task],
        counterfactual_metrics=Counterfactual(pseudobulk_pearson=True, energy_distance=False),
        progress_bar=False,
    )
    bm.benchmark()
    results = bm.get_results(long_format=True)
    assert (results["dataset"] == "fixture").all()
    assert set(results["n_predicted"]) == {3, target.shape[0]}


def test_benchmarker_routes_independent_deg_profile_to_signed_de_metric():
    observed = np.tile(np.asarray([2.0, -1.0, 0.5, 0.0]), (3, 1))
    reference = np.zeros_like(observed)
    task = CounterfactualTask(
        task_id="independent-de",
        observed=observed,
        predicted={"model": observed.copy()},
        reference=reference,
        deg=DegProfile(
            weights=np.asarray([1.0, 0.8, 0.5, 0.0]),
            top_mask=np.asarray([True, True, True, False]),
            significant_mask=np.asarray([True, True, False, False]),
            directions=np.asarray([1.0, -1.0, 1.0, 0.0]),
            effects=np.asarray([3.0, -2.0, 1.0, 0.0]),
        ),
    )
    metrics = {
        name: False for name in Counterfactual.__dataclass_fields__
    }
    metrics["signed_de_recovery"] = {"top_k": 3}
    benchmarker = CounterfactualBenchmarker(
        tasks=[task],
        counterfactual_metrics=Counterfactual(**metrics),
        progress_bar=False,
    )
    benchmarker.benchmark()
    names = set(benchmarker.get_results(long_format=True)["metric"])

    assert names == {
        "signed_de_recovery_topk_overlap",
        "signed_de_recovery_direction_accuracy",
        "signed_de_recovery_lfc_spearman",
        "signed_de_recovery_significant_deg_recall",
    }


def test_wide_summary_respects_partitions(swap_adata):
    condition_specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast", "Myeloid"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    niche_specs = make_swap_specs(
        swap_adata,
        swap_type="niche",
        source_value="immune_rich",
        target_value="fibrotic",
        cell_types=["Fibroblast"],
        match_other_factor=True,
        fixed_values=["CRC"],
        min_source_cells=5,
        min_target_cells=5,
    )
    tasks = [
        _build_task_from_spec(swap_adata, spec, _perfect_predictions(swap_adata, spec))
        for spec in condition_specs + niche_specs
    ]
    bm = CounterfactualBenchmarker(
        tasks=tasks,
        counterfactual_metrics=Counterfactual(pseudobulk_pearson=True, energy_distance=False),
        progress_bar=False,
    )
    bm.benchmark()
    wide = bm.get_results(long_format=False, aggregate="median")
    assert wide.index.nlevels == 3
    assert ("condition", False, "model_a") in wide.index
    assert ("niche", True, "model_a") in wide.index


def test_swap_summary_macro_aggregation_and_separation(swap_adata):
    marginal_specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast", "Myeloid"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    matched_specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=True,
        fixed_values=["fibrotic"],
        min_source_cells=5,
        min_target_cells=5,
    )
    tasks = [
        _build_task_from_spec(swap_adata, spec, _perfect_predictions(swap_adata, spec))
        for spec in marginal_specs + matched_specs
    ]
    bm = CounterfactualBenchmarker(
        tasks=tasks,
        counterfactual_metrics=Counterfactual(pseudobulk_pearson=True, energy_distance=False),
        progress_bar=False,
    )
    bm.benchmark()

    summary = bm.get_swap_summary(by_cell_type=False, aggregate="median")
    assert len(summary) >= 2
    assert set(summary["match_other_factor"]) == {True, False}

    by_cell_type = bm.get_swap_summary(by_cell_type=True, aggregate="median")
    assert "cell_type" in by_cell_type.columns
    assert len(by_cell_type) >= len(summary)


def test_duplicate_task_ids_rejected(swap_adata):
    specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    pred = _perfect_predictions(swap_adata, specs[0])
    task_a = _build_task_from_spec(swap_adata, specs[0], pred)
    task_b = _build_task_from_spec(swap_adata, specs[0], pred)
    with pytest.raises(ValueError, match="Duplicate CounterfactualTask.task_id"):
        CounterfactualBenchmarker(tasks=[task_a, task_b], progress_bar=False)


def test_mixed_swap_types_produce_separate_summary_rows(swap_adata):
    condition_specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    niche_specs = make_swap_specs(
        swap_adata,
        swap_type="niche",
        source_value="immune_rich",
        target_value="fibrotic",
        cell_types=["Fibroblast"],
        match_other_factor=True,
        fixed_values=["CRC"],
        min_source_cells=5,
        min_target_cells=5,
    )
    tasks = [
        _build_task_from_spec(swap_adata, spec, _perfect_predictions(swap_adata, spec))
        for spec in condition_specs + niche_specs
    ]
    bm = CounterfactualBenchmarker(
        tasks=tasks,
        counterfactual_metrics=Counterfactual(pseudobulk_pearson=True, energy_distance=False),
        progress_bar=False,
    )
    bm.benchmark()
    summary = bm.get_swap_summary(by_cell_type=False)
    assert set(summary["swap_type"]) == {"condition", "niche"}


def test_get_results_requires_benchmark():
    specs = make_swap_specs(
        make_swap_benchmark_adata(),
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    adata = make_swap_benchmark_adata()
    task = _build_task_from_spec(adata, specs[0], _perfect_predictions(adata, specs[0]))
    bm = CounterfactualBenchmarker(tasks=[task], progress_bar=False)
    with pytest.raises(RuntimeError, match="benchmark"):
        bm.get_results()


def test_explicit_reference_modes_have_distinct_metric_names(swap_adata):
    spec = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )[0]
    task = _build_task_from_spec(swap_adata, spec, _perfect_predictions(swap_adata, spec))
    n_genes = task.observed.shape[1]
    task.cross_context_template = np.zeros(n_genes)
    task.perturbed_centroid_reference = np.asarray(task.reference.mean(axis=0)).ravel()
    bm = CounterfactualBenchmarker(
        tasks=[task],
        counterfactual_metrics=Counterfactual(
            pseudobulk_pearson=False,
            pseudobulk_rmse=False,
            systema_pearson_delta=False,
            signed_de_recovery=False,
            energy_distance=False,
            knn_jaccard_deltapert=False,
            cross_context_residual_delta={"top_k": 5},
            systema_perturbed_centroid_delta={"top_k": 5},
        ),
        progress_bar=False,
    )
    bm.benchmark()
    metrics = set(bm.get_results(long_format=True)["metric"])
    assert "cross_context_residual_delta_all_genes" in metrics
    assert "systema_perturbed_centroid_delta_all_genes" in metrics


def test_task_validates_explicit_reference_vector_lengths(swap_adata):
    spec = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )[0]
    task = _build_task_from_spec(swap_adata, spec, _perfect_predictions(swap_adata, spec))
    task.perturbed_centroid_reference = np.zeros(3)
    with pytest.raises(ValueError, match="perturbed_centroid_reference length"):
        CounterfactualBenchmarker(tasks=[task], progress_bar=False)
