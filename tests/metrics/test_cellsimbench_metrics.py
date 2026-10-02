import numpy as np
import pandas as pd
import pytest

from scdice_metrics.metrics._calibration import dynamic_range_fraction, score_relative_to_baselines
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
