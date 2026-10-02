import numpy as np
import pandas as pd
import pytest

from scdice_metrics.metrics._spatial_clustering import chaos, pas
from scdice_metrics.metrics._spatial_supervised import (
    domain_specific_f1,
    gt_mixture_entropy,
    matched_jaccard,
    matched_mcc,
    spatial_ari,
)
from scdice_metrics.benchmark._consensus import cross_method_ari, smoothness_entropy


def test_perfect_match_spatial_ari():
    labels = np.array(["A", "A", "B", "B"])
    pred = labels.copy()
    assert spatial_ari(labels, pred) == pytest.approx(1.0)


def test_gt_mixture_entropy_pure_clusters():
    labels = np.array(["A", "A", "B", "B"])
    pred = labels.copy()
    assert gt_mixture_entropy(labels, pred) == pytest.approx(0.0, abs=1e-12)


def test_domain_specific_f1_perfect():
    labels = np.array(["A", "A", "B", "B"])
    pred = labels.copy()
    scores = domain_specific_f1(labels, pred)
    assert scores["macro"] == pytest.approx(1.0)
    assert scores["A"] == pytest.approx(1.0)


def test_matched_scores_perfect():
    labels = np.array(["A", "A", "B", "B"])
    pred = labels.copy()
    assert matched_mcc(labels, pred) == pytest.approx(1.0)
    assert matched_jaccard(labels, pred) == pytest.approx(1.0)


def test_pas_chaos_finite():
    rng = np.random.default_rng(0)
    n = 40
    spatial = rng.random((n, 2))
    labels = np.array(["0"] * 20 + ["1"] * 20)
    assert 0.0 <= pas(labels, spatial) <= 1.0
    assert chaos(labels, spatial) >= 0.0


def test_consensus_cross_method_ari():
    wide = pd.DataFrame({"m1": ["a", "a", "b"], "m2": ["a", "a", "b"]})
    mat = cross_method_ari(wide)
    assert mat.loc["m1", "m2"] == pytest.approx(1.0)


def test_smoothness_entropy():
    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float)
    wide = pd.DataFrame({"m1": ["a", "a", "b", "b"]})
    s = smoothness_entropy(wide, coords, k=2)
    assert np.isfinite(s["m1"])
