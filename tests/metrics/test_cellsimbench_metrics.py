import numpy as np
import pandas as pd
import pytest

from scdice_metrics.metrics._calibration import (
    _null_baseline_method_for_metric,
    drf_all_versions,
    dynamic_range_fraction,
    score_relative_to_baselines,
)
from scdice_metrics.metrics._cellsimbench import (
    DegProfile,
    deg_mask_from_pvals,
    deg_weights_from_scores,
    knn_jaccard_deltapert,
    nir_scores,
    pearson_deltapert,
    r2_deltapert,
    weighted_r2_deltapert,
    wmse,
)


def test_deg_weights_and_mask():
    genes = ["g1", "g2", "g3"]
    weights = deg_weights_from_scores(genes, np.array([1.0, 2.0, 1.0]), ["g1", "g2", "g3"])
    assert weights.shape == (3,)
    assert weights[1] > weights[0]

    mask = deg_mask_from_pvals(
        genes,
        np.array([0.01, 0.5, 0.2]),
        ["g1", "g2", "g3"],
        topn=2,
    )
    assert mask.sum() == 2
    assert mask[0]


def test_perfect_deltapert_pearson():
    rng = np.random.default_rng(0)
    obs = rng.random(20)
    pred = obs.copy()
    ref = rng.random(20)
    mean = rng.random(20)
    assert pearson_deltapert(obs, pred, ref, dataset_mean=mean) == pytest.approx(1.0)


def test_deltapert_without_mean_is_nan():
    obs = np.ones(5)
    pred = np.ones(5)
    ref = np.zeros(5)
    assert np.isnan(pearson_deltapert(obs, pred, ref, dataset_mean=None))


def test_weighted_r2_and_wmse():
    obs = np.array([1.0, 2.0, 3.0, 4.0])
    pred = obs.copy()
    ref = np.zeros(4)
    mean = np.ones(4)
    deg = DegProfile(weights=np.array([0.0, 0.2, 0.8, 0.0]), top_mask=np.array([False, True, True, False]))
    assert weighted_r2_deltapert(obs, pred, ref, dataset_mean=mean, deg=deg) == pytest.approx(1.0)
    assert wmse(obs, pred, ref, deg=deg) == pytest.approx(0.0)


def test_nir_perfect_identity():
    pred = pd.DataFrame({"a": [0.0, 1.0], "b": [1.0, 0.0]}, index=["t1", "t2"])
    truth = pred.copy()
    scores = nir_scores(pred, truth, covariate_groups={"c": ["t1", "t2"]})
    assert scores["t1"] == pytest.approx(1.0)
    assert scores["t2"] == pytest.approx(1.0)


def test_knn_jaccard_identical_graphs():
    pred = pd.DataFrame(np.eye(4), index=[f"p{i}" for i in range(4)])
    truth = pred.copy()
    scores = knn_jaccard_deltapert(pred, truth, k=2)
    assert all(v == pytest.approx(1.0) for v in scores.values())


def test_drf_higher_is_better():
    drf = dynamic_range_fraction(0.6, 0.2, higher_is_better=True, perfect=1.0)
    assert drf == pytest.approx(0.5, rel=1e-4)


def test_drf_lower_is_better():
    drf = dynamic_range_fraction(0.2, 0.5, higher_is_better=False, perfect=0.0)
    assert drf == pytest.approx(0.6, rel=1e-4)


def test_score_relative_to_baselines():
    long_df = pd.DataFrame(
        [
            {"task_id": "t1", "method": "dataset_mean", "metric": "r2_deltapert", "value": 0.1},
            {"task_id": "t1", "method": "control", "metric": "r2_deltapert", "value": 0.1},
            {"task_id": "t1", "method": "model", "metric": "r2_deltapert", "value": 0.5},
        ]
    )
    out = score_relative_to_baselines(long_df, ceiling_method="control")
    model_rows = out[out["method"] == "model"]
    assert len(model_rows) == 1
    assert np.isfinite(model_rows.iloc[0]["drf"])


def test_null_baseline_deltactrl_uses_mean_for_drf_ctrl():
    assert _null_baseline_method_for_metric("r2_deltactrl", variant="ctrl") == "dataset_mean"
    assert _null_baseline_method_for_metric("pearson_deltactrl_degs", variant="ctrl") == "dataset_mean"
    assert _null_baseline_method_for_metric("weighted_r2_deltactrl", variant="ctrl") == "dataset_mean"
    assert _null_baseline_method_for_metric("r2_deltapert", variant="ctrl") == "control"
    assert _null_baseline_method_for_metric("r2_deltapert", variant="mean") == "control"
    assert _null_baseline_method_for_metric("r2_deltactrl", variant="mean") == "dataset_mean"


def test_drf_ctrl_deltactrl_uses_mean_baseline():
    """CellSimBench MEAN_BASELINE_METRICS_FOR_CTRL: control is NaN on deltactrl."""
    rows = [
        {"task_id": "t1", "method": m, "metric": k, "value": v}
        for k, d in {
            "r2_deltapert": {
                "dataset_mean": 0.05,
                "control": 0.20,
                "technical_duplicate": 0.72,
                "interpolated_duplicate": 0.55,
                "model": 0.61,
            },
            "r2_deltactrl": {
                "dataset_mean": 0.30,
                "control": np.nan,
                "technical_duplicate": 0.80,
                "interpolated_duplicate": 0.62,
                "model": 0.68,
            },
        }.items()
        for m, v in d.items()
    ]
    out = drf_all_versions(pd.DataFrame(rows))

    ctrl_pert = out[(out["metric"] == "r2_deltapert") & (out["method"] == "model")]
    assert ctrl_pert["null_baseline_method"].dropna().eq("control").all()
    pert_ceiling = ctrl_pert["drf_ctrl_ceiling"].dropna().iloc[0]
    assert pert_ceiling == pytest.approx(0.4375, rel=1e-3)

    ctrl_ctrl = out[(out["metric"] == "r2_deltactrl") & (out["method"] == "model")]
    assert ctrl_ctrl["null_baseline_method"].dropna().eq("dataset_mean").all()
    deltactrl_ceiling = ctrl_ctrl["drf_ctrl_ceiling"].dropna().iloc[0]
    assert deltactrl_ceiling == pytest.approx(0.4571, rel=1e-3)
    assert np.isfinite(ctrl_ctrl["drf_ctrl"].dropna().iloc[0])
