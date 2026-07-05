"""Synthetic AnnData fixtures for swap-task generation tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy import sparse


CELL_TYPES = ["Fibroblast", "Myeloid", "T_cell"]
CONDITIONS = ["REF", "CRC"]
NICHES = ["immune_rich", "fibrotic", "epithelial"]


def make_swap_benchmark_adata(*, seed: int = 0, cells_per_combo: int = 5) -> AnnData:
    """Build a sparse synthetic dataset for swap mask tests."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, str]] = []

    for cell_type in CELL_TYPES:
        for condition in CONDITIONS:
            for niche in NICHES:
                count = cells_per_combo
                if cell_type == "T_cell" and condition == "CRC" and niche != "fibrotic":
                    count = 2
                if cell_type == "T_cell" and condition == "CRC" and niche == "fibrotic":
                    count = 3
                for _ in range(count):
                    rows.append(
                        {
                            "cell_type": cell_type,
                            "condition": condition,
                            "niche": niche,
                        }
                    )

    obs = pd.DataFrame(rows)
    n_cells = obs.shape[0]
    n_genes = 20
    dense = rng.random((n_cells, n_genes))
    adata = AnnData(X=sparse.csr_matrix(dense), obs=obs)
    adata.var_names = [f"gene_{i}" for i in range(n_genes)]
    return adata
