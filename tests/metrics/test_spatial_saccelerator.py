import math

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
    assert scores["B"] == pytest.approx(1.0)


def _saccelerator_domain_specific_f1(gt, pred):
    """SACCELERATOR domain-specific-f1.r: dummy-pad, LSAP, unmatched GT → NA."""
    from scipy.optimize import linear_sum_assignment

    gt = np.asarray(gt).ravel()
    pred = np.asarray(pred).ravel()
    table = pd.crosstab(pred, gt)
    n_gt = table.shape[1]
    n_pred = table.shape[0]
    gt_names = list(table.columns)
    tb = table.copy()
    if n_pred > n_gt:
        for i in range(n_pred - n_gt):
            tb[f"dummy{i}"] = 0
    cost = tb.to_numpy(dtype=float)
    row_ind, col_ind = linear_sum_assignment(cost, maximize=True)
    assigned: dict[str, float] = {}
    for r, c in zip(row_ind, col_ind, strict=False):
        if c >= n_gt:
            continue
        tp = float(cost[r, c])
        domain_size = float(table.iloc[:, c].sum())
        cluster_size = float(table.iloc[r, :].sum())
        recall = tp / domain_size
        precision = tp / cluster_size
        if recall + precision == 0:
            f1 = 0.0
        else:
            f1 = 2.0 / (1.0 / recall + 1.0 / precision)
        assigned[str(gt_names[c])] = float(f1)
    out = {str(name): assigned.get(str(name), float("nan")) for name in gt_names}
    out["macro"] = float(np.mean(np.asarray([out[str(n)] for n in gt_names], dtype=float)))
    return out


def test_domain_specific_f1_unmatched_gt_is_nan():
    """n_gt > n_clust: unmatched domains stay in the dict as NaN (SACCELERATOR NA)."""
    names = [f"domain{chr(65 + i)}" for i in range(5)]
    gt = np.array(names * 100)
    mapping = {
        "domainA": "c1",
        "domainB": "c1",
        "domainC": "c2",
        "domainD": "c2",
        "domainE": "c3",
    }
    pred = np.array([mapping[g] for g in gt], dtype=object)
    scores = domain_specific_f1(gt, pred)
    expected = _saccelerator_domain_specific_f1(gt, pred)
    assert set(scores) == set(expected)
    assert {"domainA", "domainB", "domainC", "domainD", "domainE", "macro"} <= set(scores)
    n_unmatched = sum(np.isnan(scores[k]) for k in scores if k != "macro")
    assert n_unmatched == 2
    assert np.isnan(scores["macro"])
    for key, value in expected.items():
        if np.isnan(value):
            assert np.isnan(scores[key])
        else:
            assert scores[key] == pytest.approx(value)


def test_domain_specific_f1_extra_clusters_keep_all_domains():
    gt = np.array(["A"] * 40 + ["B"] * 40)
    pred = np.array(["c0"] * 40 + ["c1"] * 25 + ["c2"] * 15)
    scores = domain_specific_f1(gt, pred)
    expected = _saccelerator_domain_specific_f1(gt, pred)
    assert not any(k.startswith("dummy") for k in scores)
    assert scores["A"] == pytest.approx(expected["A"])
    assert scores["B"] == pytest.approx(expected["B"])
    assert np.isfinite(scores["macro"])
    assert scores["macro"] == pytest.approx(expected["macro"])


def test_matched_scores_perfect():
    labels = np.array(["A", "A", "B", "B"])
    pred = labels.copy()
    assert matched_mcc(labels, pred) == pytest.approx(1.0)
    assert matched_jaccard(labels, pred) == pytest.approx(1.0)


def _saccelerator_mcc_jaccard(gt, pred):
    """SACCELERATOR MCC.py / jaccard.py: category codes then Hungarian on integers."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.metrics import jaccard_score, matthews_corrcoef

    domains = pd.Series(pred).astype("category").cat.codes
    groundtruth = pd.Series(gt).astype("category").cat.codes
    table = pd.crosstab(domains, groundtruth)
    row_ind, col_ind = linear_sum_assignment(table, maximize=True)
    mapped = domains.map(dict(zip(row_ind, col_ind)))
    if mapped.isna().any():
        return None, None
    return (
        float(matthews_corrcoef(groundtruth, mapped)),
        float(jaccard_score(groundtruth, mapped, average="weighted")),
    )


def test_matched_scores_saccelerator_adversarial_encoding():
    """First-appearance order of matched labels differs from GT (review 2026-10-02)."""
    gt = np.array(["D0"] * 40 + ["D1"] * 100 + ["D0"] * 100)
    pred = np.array(["X"] * 140 + ["Y"] * 100)
    # Frozen SACCELERATOR MCC.py / jaccard.py output on this pair.
    assert matched_mcc(gt, pred) == pytest.approx(0.7142857142857143)
    assert matched_jaccard(gt, pred) == pytest.approx(0.7142857142857143)
    orig_mcc, orig_jac = _saccelerator_mcc_jaccard(gt, pred)
    assert orig_mcc == pytest.approx(matched_mcc(gt, pred))
    assert orig_jac == pytest.approx(matched_jaccard(gt, pred))


def test_matched_scores_random_agree_with_saccelerator():
    rng = np.random.default_rng(1)
    for _ in range(12):
        k = int(rng.integers(3, 8))
        n = k * 80
        gl = [f"domain{chr(65 + i)}" for i in range(k)]
        gt = rng.choice(gl, size=n)
        cl = [f"clu{i}" for i in range(k)]
        rng.shuffle(cl)
        mapping = dict(zip(gl, cl))
        pred = np.array([mapping[g] for g in gt], dtype=object)
        noise = rng.random(n) < 0.25
        pred[noise] = rng.choice(cl, size=int(noise.sum()))
        orig_mcc, orig_jac = _saccelerator_mcc_jaccard(gt, pred)
        assert orig_mcc is not None
        assert matched_mcc(gt, pred) == pytest.approx(orig_mcc, abs=1e-12)
        assert matched_jaccard(gt, pred) == pytest.approx(orig_jac, abs=1e-12)


def test_matched_scores_unmatched_cluster_stays_finite():
    gt = np.array(["A"] * 50 + ["B"] * 50)
    pred = np.array(["c0"] * 50 + ["c1"] * 30 + ["c2"] * 20)
    orig_mcc, _ = _saccelerator_mcc_jaccard(gt, pred)
    assert orig_mcc is None
    assert np.isfinite(matched_mcc(gt, pred))
    assert np.isfinite(matched_jaccard(gt, pred))


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


def _saccelerator_calc_entropy(labels):
    """SACCELERATOR Smoothness_entropy.r ``calc_entropy``: natural log, drop zeros."""
    _, counts = np.unique(np.asarray(labels), return_counts=True)
    p = counts.astype(float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def test_smoothness_entropy_matches_saccelerator_natural_log():
    from scipy.spatial.distance import cdist

    rng = np.random.default_rng(0)
    n = 80
    coords = rng.normal(size=(n, 2))
    labels = rng.choice(["A", "B", "C"], size=n)
    wide = pd.DataFrame({"m1": labels})
    k = 6
    dist = cdist(coords, coords)
    nn_idx = np.argsort(dist, axis=1)[:, 1 : k + 1]
    expected_ln = float(np.mean([_saccelerator_calc_entropy(labels[idx]) for idx in nn_idx]))
    got = float(smoothness_entropy(wide, coords, k=k)["m1"])
    assert got == pytest.approx(expected_ln)
    assert expected_ln > 0
    expected_log2 = expected_ln / math.log(2)
    assert abs(got - expected_log2) > 0.1
