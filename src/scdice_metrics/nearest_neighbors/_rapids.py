from __future__ import annotations

from typing import Literal

import numpy as np

from ._dataclass import NeighborsResults


def rapids(
    X: np.ndarray,
    n_neighbors: int,
    metric: Literal["euclidean", "l2", "cosine"] = "euclidean",
    algorithm: Literal["brute", "ivfflat", "ivfpq"] = "brute",
) -> NeighborsResults:
    """Run GPU-accelerated nearest neighbor search with RAPIDS cuML.

    Parameters
    ----------
    X
        Data matrix.
    n_neighbors
        Number of neighbors to search for.
    metric
        Distance metric passed to cuML.
    algorithm
        KNN algorithm passed to cuML.

    Returns
    -------
    NeighborsResults
        Neighbor indices and distances on host memory.
    """
    try:
        import cupy as cp
        from cuml.neighbors import NearestNeighbors
    except ImportError as exc:
        raise ImportError(
            "`rapids` nearest-neighbor backend requires optional dependencies "
            "`cupy` and `cuml`. Please install RAPIDS cuML to use this function."
        ) from exc

    X_cu = cp.asarray(X)
    nn = NearestNeighbors(
        n_neighbors=n_neighbors,
        metric=metric,
        algorithm=algorithm,
        output_type="cupy",
    )
    nn.fit(X_cu)
    distances_cu, indices_cu = nn.kneighbors(X_cu, return_distance=True)

    return NeighborsResults(
        indices=cp.asnumpy(indices_cu),
        distances=cp.asnumpy(distances_cu),
    )
