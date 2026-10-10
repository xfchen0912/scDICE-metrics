"""Result types for balanced two-factor response decomposition."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _interaction_key(factor_names: tuple[str, str]) -> str:
    return f"{factor_names[0]}:{factor_names[1]}"


@dataclass
class ResponseDecompositionResult:
    """Balanced two-factor decomposition of a response tensor."""

    global_effect: np.ndarray
    factor_a_effect: np.ndarray
    factor_b_effect: np.ndarray
    interaction_effect: np.ndarray

    component_energy: dict[str, float]
    component_fraction: dict[str, float]

    total_energy: float
    reconstruction_error: float

    factor_a_names: Sequence[str] | None
    factor_b_names: Sequence[str] | None
    feature_names: Sequence[str] | None
    factor_names: tuple[str, str]

    @property
    def interaction_fraction(self) -> float:
        """Fraction of non-global structure attributable to interaction (§7.1)."""
        fn_a, fn_b = self.factor_names
        interaction = self.component_energy[_interaction_key(self.factor_names)]
        denom = self.component_energy[fn_a] + self.component_energy[fn_b] + interaction
        if denom <= 0.0:
            return 0.0
        return float(interaction / denom)

    def transferable_fraction(self, factor: str) -> float:
        """Conserved main-effect fraction among main effect + interaction (§7.2)."""
        fn_a, fn_b = self.factor_names
        interaction = self.component_energy[_interaction_key(self.factor_names)]
        if factor in ("factor_a", fn_a):
            main = self.component_energy[fn_a]
        elif factor in ("factor_b", fn_b):
            main = self.component_energy[fn_b]
        else:
            raise ValueError(
                f"Unknown factor {factor!r}; expected 'factor_a', 'factor_b', {fn_a!r}, or {fn_b!r}."
            )
        denom = main + interaction
        if denom <= 0.0:
            return 0.0
        return float(main / denom)

    def per_factor_b_transferability(self) -> pd.DataFrame:
        """Per level of factor B: energy-based transferability (§7.3)."""
        fn_a, fn_b = self.factor_names
        beta = self.factor_b_effect
        gamma = self.interaction_effect
        n_a = gamma.shape[0]
        names = list(self.factor_b_names) if self.factor_b_names is not None else [f"{fn_b}_{i}" for i in range(beta.shape[0])]

        rows: list[dict[str, object]] = []
        for j, name in enumerate(names):
            beta_energy = float(np.sum(beta[j] ** 2))
            gamma_energy = float(np.mean([np.sum(gamma[a, j] ** 2) for a in range(n_a)]))
            denom = beta_energy + gamma_energy
            tf = 0.0 if denom <= 0.0 else beta_energy / denom
            rows.append(
                {
                    fn_b: name,
                    "beta_energy": beta_energy,
                    "interaction_energy": gamma_energy,
                    "transferability": float(tf),
                }
            )
        return pd.DataFrame(rows)

    def to_frame(self) -> pd.DataFrame:
        """Tidy table of component energies and fractions."""
        rows = [
            {
                "component": component,
                "energy": self.component_energy[component],
                "fraction": self.component_fraction[component],
            }
            for component in self.component_energy
        ]
        return pd.DataFrame(rows)


@dataclass
class PerturbationDecompositionResult(ResponseDecompositionResult):
    """Perturb-seq decomposition with context / perturbation semantics."""

    @property
    def context_effect(self) -> np.ndarray:
        """Cellular-context main effect (factor A)."""
        return self.factor_a_effect

    @property
    def conserved_perturbation_effect(self) -> np.ndarray:
        """Conserved perturbation main effect shared across contexts (factor B)."""
        return self.factor_b_effect

    @property
    def context_perturbation_interaction(self) -> np.ndarray:
        """Context × perturbation interaction residual."""
        return self.interaction_effect


def as_perturbation_decomposition_result(result: ResponseDecompositionResult) -> PerturbationDecompositionResult:
    """Copy a generic decomposition result into a perturbation-flavored result."""
    return PerturbationDecompositionResult(
        global_effect=result.global_effect,
        factor_a_effect=result.factor_a_effect,
        factor_b_effect=result.factor_b_effect,
        interaction_effect=result.interaction_effect,
        component_energy=result.component_energy,
        component_fraction=result.component_fraction,
        total_energy=result.total_energy,
        reconstruction_error=result.reconstruction_error,
        factor_a_names=result.factor_a_names,
        factor_b_names=result.factor_b_names,
        feature_names=result.feature_names,
        factor_names=result.factor_names,
    )


@dataclass
class SplitHalfDecompositionResult(PerturbationDecompositionResult):
    """Split-half noise-corrected perturbation decomposition."""

    signal_energy: dict[str, float]
    noise_energy: float
    n_splits: int
    naive_component_fraction: dict[str, float]
    naive_component_energy: dict[str, float]

    @property
    def noise_fraction(self) -> float:
        """Noise fraction relative to full-data total energy."""
        if self.total_energy <= 0.0:
            return 0.0
        return float(self.noise_energy / self.total_energy)

    def to_frame(self) -> pd.DataFrame:
        """Tidy table of split-half corrected energies and fractions."""
        rows = []
        for component, fraction in self.component_fraction.items():
            energy = self.noise_energy if component == "noise" else self.signal_energy.get(component)
            rows.append(
                {
                    "component": component,
                    "signal_energy": energy,
                    "fraction": fraction,
                }
            )
        return pd.DataFrame(rows)
