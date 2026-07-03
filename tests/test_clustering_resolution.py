from unittest.mock import patch

import numpy as np
import pytest

from scdice_metrics.metrics._clustering import (
    MIN_LEIDEN_RESOLUTION,
    _clamp_resolution,
    _pick_closest_candidate,
    search_leiden_resolution,
)


def test_clamp_resolution_floor():
    assert _clamp_resolution(-0.1) == MIN_LEIDEN_RESOLUTION
    assert _clamp_resolution(0.2) == 0.2


def test_pick_closest_candidate_prefers_target_k():
    candidates = [
        (0.05, np.array([0, 0, 0, 0]), 1),
        (0.4, np.array([0, 0, 1, 1, 2, 2, 3]), 4),
        (0.8, np.array([0, 1, 2, 3, 4, 5, 6]), 7),
        (1.2, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]), 9),
    ]
    res, labels, k = _pick_closest_candidate(candidates, target_k=7)
    assert k == 7
    assert len(np.unique(labels)) == 7
    assert res == 0.8


def test_pick_closest_candidate_avoids_single_cluster_when_closer_exists():
    candidates = [
        (0.05, np.array([0, 0, 0, 0]), 1),
        (0.3, np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 6]), 7),
    ]
    _, _, k = _pick_closest_candidate(candidates, target_k=7)
    assert k == 7


def test_search_leiden_resolution_exact_match_no_warning():
    rng = np.random.default_rng(0)
    n = 30
    labels_by_res = {
        0.1: np.repeat(np.arange(7), 4)[:n],
        0.2: np.repeat(np.arange(10), 3)[:n],
    }

    def fake_leiden(graph, resolution, seed=42):
        res_key = round(float(resolution), 1)
        if res_key in labels_by_res:
            return labels_by_res[res_key].copy()
        return np.zeros(n, dtype=int)

    class DummyGraph:
        pass

    with patch(
        "scdice_metrics.metrics._clustering.compute_clustering_leiden",
        side_effect=fake_leiden,
    ):
        res, labels = search_leiden_resolution(DummyGraph(), target_k=7, res_start=0.1, res_epochs=10)
    assert len(np.unique(labels)) == 7
    assert res == pytest.approx(0.1)


def test_search_leiden_resolution_fallback_closest_k():
    n = 20

    def fake_leiden(graph, resolution, seed=42):
        res = float(np.clip(resolution, 0.05, 3.0))
        if res <= 0.15:
            return np.zeros(n, dtype=int)
        if res <= 0.5:
            return np.repeat(np.arange(8), 2)[:n]
        return np.repeat(np.arange(10), 2)[:n]

    class DummyGraph:
        pass

    with patch(
        "scdice_metrics.metrics._clustering.compute_clustering_leiden",
        side_effect=fake_leiden,
    ):
        res, labels = search_leiden_resolution(
            DummyGraph(),
            target_k=7,
            res_start=0.1,
            res_epochs=3,
        )
    assert res >= MIN_LEIDEN_RESOLUTION
    assert abs(len(np.unique(labels)) - 7) <= abs(8 - 7)
