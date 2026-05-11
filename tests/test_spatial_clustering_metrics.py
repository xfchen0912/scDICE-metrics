import numpy as np

from scdice_metrics.metrics import chaos, pas


def test_chaos_prefers_spatially_compact_clusters():
    labels = np.array([0, 0, 1, 1])
    compact = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0]])
    mixed = np.array([[0.0, 0.0], [10.0, 10.0], [0.1, 0.0], [10.1, 10.0]])

    chaos_compact = chaos(labels, compact)
    chaos_mixed = chaos(labels, mixed)

    assert chaos_compact < chaos_mixed


def test_pas_range_and_sanity():
    labels = np.array([0, 0, 1, 1, 1, 0])
    spatial = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.2, 10.0],
            [0.2, 0.0],
        ]
    )

    score = pas(labels, spatial, k=3)
    assert 0.0 <= score <= 1.0
