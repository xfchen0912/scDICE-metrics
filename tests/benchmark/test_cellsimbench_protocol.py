import numpy as np
import pytest

from scdice_metrics.benchmark._counterfactual import (
    Counterfactual,
    CounterfactualBenchmarker,
    CounterfactualTask,
)
from scdice_metrics.metrics._cellsimbench import DegProfile


def _task(*, with_mean: bool = True, with_deg: bool = True) -> CounterfactualTask:
    rng = np.random.default_rng(0)
    n_genes = 30
    ref = rng.random(n_genes)
    mean = rng.random(n_genes)
    obs = (ref + rng.normal(scale=0.1, size=n_genes)).reshape(1, -1)
    pred = (obs.ravel() + rng.normal(scale=0.01, size=n_genes)).reshape(1, -1)
    ref = ref.reshape(1, -1)
    mean = mean.reshape(1, -1)
    weights = np.linspace(0, 1, n_genes) ** 2
    top_mask = np.zeros(n_genes, dtype=bool)
    top_mask[np.argsort(-weights)[:10]] = True
    deg = DegProfile(weights=weights, top_mask=top_mask)
    return CounterfactualTask(
        task_id="covA_pert1",
        observed=obs,
        predicted={"model": pred, "dataset_mean": mean, "control": ref},
        reference=ref,
        dataset_mean=mean if with_mean else None,
        deg=deg if with_deg else None,
        metadata={"covariate": "covA"},
    )


def test_default_counterfactual_excludes_cellsimbench_columns():
    task = _task()
    bm = CounterfactualBenchmarker(tasks=[task], counterfactual_metrics=Counterfactual())
    bm.benchmark()
    metrics = set(bm.get_results()["metric"])
    assert "weighted_r2_deltapert" not in metrics
    assert "pearson_deltapert" not in metrics


def test_cellsimbench_factory_runs_deltapert_metrics():
    task = _task()
    bm = CounterfactualBenchmarker(
        tasks=[task],
        counterfactual_metrics=Counterfactual.cellsimbench(),
        progress_bar=False,
    )
    bm.benchmark()
    df = bm.get_results()
    assert "weighted_r2_deltapert" in set(df["metric"])
    model_row = df[(df["method"] == "model") & (df["metric"] == "weighted_r2_deltapert")]
    assert len(model_row) == 1
    assert model_row.iloc[0]["value"] == pytest.approx(1.0, abs=0.05)


def test_panel_nir_two_tasks():
    t1 = _task()
    rng = np.random.default_rng(1)
    n_genes = 30
    ref = rng.random(n_genes)
    mean = rng.random(n_genes)
    obs2 = (ref + rng.normal(scale=0.2, size=n_genes) + 1.0).reshape(1, -1)
    pred2 = obs2.copy()
    t2 = CounterfactualTask(
        task_id="covA_pert2",
        observed=obs2,
        predicted={"model": pred2},
        reference=ref.reshape(1, -1),
        dataset_mean=mean.reshape(1, -1),
        metadata={"covariate": "covA"},
    )
    cfg = Counterfactual(nir=True, energy_distance=False, pseudobulk_pearson=False, pseudobulk_rmse=False)
    bm = CounterfactualBenchmarker(tasks=[t1, t2], counterfactual_metrics=cfg, progress_bar=False)
    bm.benchmark()
    df = bm.get_results()
    nir_rows = df[df["metric"] == "nir"]
    assert len(nir_rows) >= 2
