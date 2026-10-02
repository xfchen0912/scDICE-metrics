"""Tests for generic two-factor response decomposition (Milestone 1)."""

from __future__ import annotations

import numpy as np
import pytest

from scdice_metrics.decomposition import ResponseDecompositionResult, response_decomposition


def _reconstruct(result: ResponseDecompositionResult) -> np.ndarray:
    mu = result.global_effect
    alpha = result.factor_a_effect
    beta = result.factor_b_effect
    gamma = result.interaction_effect
    return mu + alpha[:, None, :] + beta[None, :, :] + gamma


def test_reconstruction_random_tensor():
    rng = np.random.default_rng(0)
    d = rng.standard_normal((3, 4, 10))
    result = response_decomposition(d)
    assert np.allclose(d, _reconstruct(result))
    assert result.reconstruction_error < 1e-10


def test_centering_constraints():
    rng = np.random.default_rng(1)
    d = rng.standard_normal((3, 5, 8))
    result = response_decomposition(d)

    assert np.allclose(result.factor_a_effect.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(result.factor_b_effect.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(result.interaction_effect.mean(axis=1), 0.0, atol=1e-10)
    assert np.allclose(result.interaction_effect.mean(axis=0), 0.0, atol=1e-10)


def test_pure_factor_a():
    rng = np.random.default_rng(2)
    n_a, n_b, n_g = 3, 4, 20
    alpha = rng.standard_normal((n_a, n_g))
    alpha -= alpha.mean(axis=0, keepdims=True)
    d = np.broadcast_to(alpha[:, None, :], (n_a, n_b, n_g)).copy()

    result = response_decomposition(d)
    fr = result.component_fraction
    assert fr["factor_a"] == pytest.approx(1.0, abs=1e-6)
    assert fr["factor_b"] == pytest.approx(0.0, abs=1e-6)
    assert fr["factor_a:factor_b"] == pytest.approx(0.0, abs=1e-6)
    assert fr["global"] == pytest.approx(0.0, abs=1e-6)


def test_pure_factor_b():
    rng = np.random.default_rng(3)
    n_a, n_b, n_g = 3, 5, 15
    beta = rng.standard_normal((n_b, n_g))
    beta -= beta.mean(axis=0, keepdims=True)
    d = np.broadcast_to(beta[None, :, :], (n_a, n_b, n_g)).copy()

    result = response_decomposition(d)
    fr = result.component_fraction
    assert fr["factor_b"] == pytest.approx(1.0, abs=1e-6)
    assert fr["factor_a"] == pytest.approx(0.0, abs=1e-6)
    assert fr["factor_a:factor_b"] == pytest.approx(0.0, abs=1e-6)


def test_pure_interaction():
    n_a, n_b, n_g = 2, 2, 5
    gamma = np.zeros((n_a, n_b, n_g))
    pattern = np.array([1.0, -1.0, 0.5, -0.5, 0.25])
    gamma[0, 0] = pattern
    gamma[0, 1] = -pattern
    gamma[1, 0] = -pattern
    gamma[1, 1] = pattern

    result = response_decomposition(gamma)
    interaction_key = "factor_a:factor_b"
    assert result.component_fraction[interaction_key] == pytest.approx(1.0, abs=1e-6)
    assert result.component_fraction["factor_a"] == pytest.approx(0.0, abs=1e-6)
    assert result.component_fraction["factor_b"] == pytest.approx(0.0, abs=1e-6)
    assert result.component_fraction["global"] == pytest.approx(0.0, abs=1e-6)


def test_known_mixture_orthogonal_components():
    n_a, n_b, n_g = 2, 2, 4
    mu = np.array([2.0, 0.0, 0.0, 0.0])
    alpha = np.zeros((n_a, n_g))
    alpha[0] = np.array([0.0, 2.0, 0.0, 0.0])
    alpha[1] = np.array([0.0, -2.0, 0.0, 0.0])
    beta = np.zeros((n_b, n_g))
    beta[0] = np.array([0.0, 0.0, 3.0, 0.0])
    beta[1] = np.array([0.0, 0.0, -3.0, 0.0])
    gamma = np.zeros((n_a, n_b, n_g))
    gamma[0, 0, 3] = 4.0
    gamma[0, 1, 3] = -4.0
    gamma[1, 0, 3] = -4.0
    gamma[1, 1, 3] = 4.0

    d = mu + alpha[:, None, :] + beta[None, :, :] + gamma
    result = response_decomposition(d)

    e_mu = 4.0
    e_alpha = 4.0
    e_beta = 9.0
    e_gamma = 16.0
    e_total = 33.0
    expected = {
        "global": e_mu / e_total,
        "factor_a": e_alpha / e_total,
        "factor_b": e_beta / e_total,
        "factor_a:factor_b": e_gamma / e_total,
    }
    for key, val in expected.items():
        assert result.component_fraction[key] == pytest.approx(val, rel=1e-5, abs=1e-5)

    energy_sum = sum(result.component_energy.values())
    assert energy_sum == pytest.approx(result.total_energy, rel=1e-5)


def test_energy_partition_additivity():
    rng = np.random.default_rng(4)
    d = rng.standard_normal((4, 3, 12))
    result = response_decomposition(d)
    energy_sum = sum(result.component_energy.values())
    assert energy_sum == pytest.approx(result.total_energy, rel=1e-5, abs=1e-10)


def test_interaction_and_transferable_fractions():
    rng = np.random.default_rng(5)
    d = rng.standard_normal((3, 4, 6))
    result = response_decomposition(d)

    fn_a, fn_b = result.factor_names
    interaction_key = f"{fn_a}:{fn_b}"
    denom_if = (
        result.component_energy[fn_a] + result.component_energy[fn_b] + result.component_energy[interaction_key]
    )
    expected_if = result.component_energy[interaction_key] / denom_if if denom_if > 0 else 0.0
    assert result.interaction_fraction == pytest.approx(expected_if)

    denom_tf_b = result.component_energy[fn_b] + result.component_energy[interaction_key]
    expected_tf_b = result.component_energy[fn_b] / denom_tf_b if denom_tf_b > 0 else 0.0
    assert result.transferable_fraction("factor_b") == pytest.approx(expected_tf_b)


def test_per_factor_b_transferability():
    rng = np.random.default_rng(6)
    d = rng.standard_normal((2, 3, 5))
    names = ["p0", "p1", "p2"]
    result = response_decomposition(d, factor_b_names=names)

    df = result.per_factor_b_transferability()
    assert list(df["factor_b"]) == names
    assert np.all((df["transferability"] >= 0.0) & (df["transferability"] <= 1.0))


def test_custom_factor_names_in_fractions():
    result = response_decomposition(
        np.ones((2, 2, 3)),
        factor_names=("context", "perturbation"),
    )
    assert set(result.component_fraction) == {"global", "context", "perturbation", "context:perturbation"}


def test_to_frame():
    d = np.random.default_rng(7).standard_normal((2, 2, 4))
    result = response_decomposition(d)
    frame = result.to_frame()
    assert len(frame) == 4
    assert set(frame.columns) == {"component", "energy", "fraction"}


def test_validation_errors():
    with pytest.raises(ValueError, match="3-dimensional"):
        response_decomposition(np.zeros((2, 3)))

    with pytest.raises(ValueError, match="at least 2 levels"):
        response_decomposition(np.zeros((1, 3, 2)))

    with pytest.raises(ValueError, match="non-finite"):
        response_decomposition(np.array([[[np.nan, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]))

    with pytest.raises(ValueError, match="factor_a_names"):
        response_decomposition(np.zeros((2, 2, 2)), factor_a_names=["a"])

    with pytest.raises(ValueError, match="Unknown factor"):
        response_decomposition(np.zeros((2, 2, 2))).transferable_fraction("unknown")
