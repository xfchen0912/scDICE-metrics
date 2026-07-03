"""Shared clustering utilities for benchmark metrics."""

from __future__ import annotations

import logging
import random
import warnings
from collections.abc import Sequence
from typing import Literal

import igraph
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import spmatrix

from scdice_metrics.nearest_neighbors import NeighborsResults, pynndescent
from scdice_metrics.utils import KMeans

logger = logging.getLogger(__name__)

MIN_LEIDEN_RESOLUTION = 0.05
MAX_LEIDEN_RESOLUTION = 3.0
ClusterOrderMode = Literal["ground_truth", "cell_count", "none"]
ClusterLabelStyle = Literal["ground_truth_name", "index"]
ClusterOrder = ClusterOrderMode | Sequence[int | str] | None


def resolve_effective_cluster_order(
    order: ClusterOrder,
    ground_truth: np.ndarray | None,
) -> Literal["ground_truth", "cell_count", "none", "custom"]:
    """Resolve the cluster-order mode that will actually be applied."""
    if order is None:
        order = "ground_truth"
    if order == "ground_truth" and ground_truth is None:
        return "cell_count"
    if isinstance(order, Sequence) and not isinstance(order, (str, bytes)):
        return "custom"
    if order in ("ground_truth", "cell_count", "none"):
        return order
    raise ValueError(f"Unsupported cluster order: {order!r}")


def _format_label_counts(labels: np.ndarray) -> str:
    unique, counts = np.unique(labels, return_counts=True)
    pairs = sorted(zip(unique, counts), key=lambda item: (-int(item[1]), str(item[0])))
    return ", ".join(f"{label}={int(count)}" for label, count in pairs)


def _ground_truth_matching_details(
    pred_labels: np.ndarray,
    ground_truth: np.ndarray,
    label_style: ClusterLabelStyle = "ground_truth_name",
) -> tuple[np.ndarray, list[str], list[dict[str, object]]]:
    """Match predicted clusters to ground truth and return mapping details."""
    pred_arr = np.asarray(pred_labels)
    gt_arr = np.asarray(ground_truth).ravel()
    if pred_arr.shape[0] != gt_arr.shape[0]:
        raise ValueError("`pred_labels` and `ground_truth` must have the same length.")

    pred_unique, pred_counts = np.unique(pred_arr, return_counts=True)
    ordered_gt, gt_counts_ordered = ground_truth_categories_by_cell_count(gt_arr)

    n_pred = len(pred_unique)
    n_gt = len(ordered_gt)
    records: list[dict[str, object]] = []

    if n_pred == 0:
        return pred_arr.copy(), [], records
    if n_gt == 0:
        ordered = reorder_cluster_labels(pred_arr, order="cell_count")
        return ordered, [], records

    cost = np.abs(pred_counts[:, None].astype(float) - gt_counts_ordered[None, :].astype(float))
    max_n = max(n_pred, n_gt)
    if n_pred != n_gt:
        padded = np.full((max_n, max_n), float(cost.max() + 1.0) if cost.size else 1.0)
        padded[:n_pred, :n_gt] = cost
        row_ind, col_ind = linear_sum_assignment(padded)
        pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind) if r < n_pred and c < n_gt]
    else:
        row_ind, col_ind = linear_sum_assignment(cost)
        pairs = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]

    mapping: dict = {}
    matched_pred_indices: set[int] = set()
    for pred_idx, gt_idx in pairs:
        pred_label = pred_unique[pred_idx]
        gt_label = ordered_gt[gt_idx]
        matched_pred_indices.add(pred_idx)
        if label_style == "ground_truth_name":
            final_label = gt_label
        else:
            final_label = gt_idx
        mapping[pred_label] = final_label
        records.append(
            {
                "raw_label": pred_label,
                "raw_count": int(pred_counts[pred_idx]),
                "gt_label": gt_label,
                "gt_count": int(gt_counts_ordered[gt_idx]),
                "final_label": final_label,
                "match_type": "ground_truth_count",
            }
        )

    extra_categories: list = []
    extra_idx = 0
    unmatched = [
        (i, pred_unique[i])
        for i in range(n_pred)
        if i not in matched_pred_indices
    ]
    unmatched.sort(key=lambda item: (-pred_counts[item[0]], str(item[1])))
    for _, pred_label in unmatched:
        if label_style == "ground_truth_name":
            final_label = f"__unmatched_{extra_idx}"
            extra_categories.append(final_label)
        else:
            final_label = n_gt + extra_idx
        mapping[pred_label] = final_label
        records.append(
            {
                "raw_label": pred_label,
                "raw_count": int(pred_counts[pred_unique == pred_label][0]),
                "gt_label": None,
                "gt_count": None,
                "final_label": final_label,
                "match_type": "unmatched",
            }
        )
        extra_idx += 1

    new_labels = np.vectorize(mapping.get)(pred_arr)
    if label_style == "ground_truth_name":
        category_order = [str(x) for x in ordered_gt] + [str(x) for x in extra_categories]
    else:
        category_order = [str(i) for i in range(n_gt + extra_idx)]
    return new_labels, category_order, records


def explain_cluster_order(
    raw_labels: np.ndarray,
    ordered_labels: np.ndarray,
    order: ClusterOrder,
    ground_truth: np.ndarray | None = None,
    label_style: ClusterLabelStyle = "ground_truth_name",
) -> list[str]:
    """Build human-readable lines describing how cluster labels were relabeled."""
    effective = resolve_effective_cluster_order(order, ground_truth)
    lines = [
        f"requested cluster_order: {order!r}",
        f"effective cluster_order: {effective}",
        f"raw cluster sizes: {_format_label_counts(raw_labels)}",
        f"final cluster sizes: {_format_label_counts(ordered_labels)}",
    ]

    if order == "ground_truth" and ground_truth is None:
        lines.append("note: spatial ground-truth labels unavailable; fell back to cell_count ordering")

    if effective == "ground_truth" and ground_truth is not None:
        lines.append(f"cluster_label_style: {label_style}")
        _, _, records = _ground_truth_matching_details(raw_labels, ground_truth, label_style=label_style)
        lines.append("ground-truth count matching:")
        for record in records:
            if record["match_type"] == "ground_truth_count":
                lines.append(
                    "  "
                    f"raw {record['raw_label']} (n={record['raw_count']}) -> "
                    f"{record['final_label']} "
                    f"[GT {record['gt_label']} (n={record['gt_count']})]"
                )
            else:
                lines.append(
                    "  "
                    f"raw {record['raw_label']} (n={record['raw_count']}) -> "
                    f"{record['final_label']} [unmatched]"
                )
        return lines

    if effective == "cell_count":
        unique, counts = np.unique(raw_labels, return_counts=True)
        count_by_label = {label: int(count) for label, count in zip(unique, counts)}
        sorted_labels = sorted(unique, key=lambda label: (-count_by_label[label], str(label)))
        lines.append("cell-count relabeling:")
        for idx, raw_label in enumerate(sorted_labels):
            lines.append(f"  raw {raw_label} (n={count_by_label[raw_label]}) -> {idx}")
        return lines

    if effective == "custom":
        unique, counts = np.unique(raw_labels, return_counts=True)
        count_by_label = {label: int(count) for label, count in zip(unique, counts)}
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            return lines
        lines.append(f"custom order: {list(order)}")
        seen: set = set()
        final_idx = 0
        for raw_label in order:
            if raw_label in seen or raw_label not in count_by_label:
                continue
            lines.append(f"  raw {raw_label} (n={count_by_label[raw_label]}) -> {final_idx}")
            seen.add(raw_label)
            final_idx += 1
        remaining = [label for label in unique if label not in seen]
        remaining.sort(key=lambda label: (-count_by_label[label], str(label)))
        for raw_label in remaining:
            lines.append(
                f"  raw {raw_label} (n={count_by_label[raw_label]}) -> {final_idx} [appended by cell count]"
            )
            final_idx += 1
        return lines

    if effective == "none":
        lines.append("cluster labels kept unchanged")
    return lines


def ground_truth_categories_by_cell_count(ground_truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ground-truth labels and counts sorted by descending cell count."""
    gt_arr = np.asarray(ground_truth).ravel()
    gt_unique, gt_counts = np.unique(gt_arr, return_counts=True)
    order = sorted(range(len(gt_unique)), key=lambda i: (-gt_counts[i], str(gt_unique[i])))
    return gt_unique[order], gt_counts[order]


def match_cluster_labels_to_ground_truth(
    pred_labels: np.ndarray,
    ground_truth: np.ndarray,
    label_style: ClusterLabelStyle = "ground_truth_name",
) -> tuple[np.ndarray, list]:
    """Match predicted clusters to ground truth by similar cell counts.

    Each predicted cluster is assigned to the ground-truth region with the closest
    cell count (Hungarian matching). The returned label for a cluster is either the
    matched ground-truth name/id or its rank among ground-truth regions sorted by
    descending cell count (for color alignment).

    Returns
    -------
    labels
        Relabeled predictions.
    category_order
        Ordered category names for ``pandas.Categorical`` / scanpy palettes.
    """
    new_labels, category_order, _ = _ground_truth_matching_details(
        pred_labels,
        ground_truth,
        label_style=label_style,
    )
    return new_labels, category_order


def reorder_cluster_labels(
    labels: np.ndarray,
    order: ClusterOrder = "ground_truth",
    ground_truth: np.ndarray | None = None,
    label_style: ClusterLabelStyle = "ground_truth_name",
) -> np.ndarray:
    """Relabel clusters for stable colors and ground-truth alignment.

    Parameters
    ----------
    labels
        Raw cluster assignments.
    order
        - ``"ground_truth"`` (default): match clusters to ground-truth regions by
          similar cell counts; falls back to ``"cell_count"`` when ``ground_truth``
          is unavailable.
        - ``"cell_count"``: assign labels ``0..K-1`` by descending cluster size.
        - ``"none"``: keep raw assignments unchanged.
        - ``Sequence``: custom order of original cluster ids; index ``i`` in the
          sequence becomes new label ``i``. Unlisted ids are appended by descending
          cell count.
    ground_truth
        Ground-truth spatial domain labels used for ``order="ground_truth"``.
    label_style
        When matching to ground truth, use matched region names or numeric indices
        ranked by ground-truth cell count.
    """
    labels_arr = np.asarray(labels)
    if order is None:
        order = "ground_truth"
    if order == "ground_truth":
        if ground_truth is None:
            return reorder_cluster_labels(labels_arr, order="cell_count")
        return match_cluster_labels_to_ground_truth(labels_arr, ground_truth, label_style=label_style)[0]
    if order == "none":
        return labels_arr.copy()

    unique, counts = np.unique(labels_arr, return_counts=True)
    count_by_label = {label: int(count) for label, count in zip(unique, counts)}

    if order == "cell_count":
        sorted_labels = sorted(unique, key=lambda label: (-count_by_label[label], str(label)))
    else:
        ordered: list = []
        seen: set = set()
        for raw_label in order:
            if raw_label in seen:
                continue
            if raw_label not in count_by_label:
                raise ValueError(
                    f"Custom cluster order references unknown label `{raw_label}`. "
                    f"Available labels: {list(unique)}."
                )
            ordered.append(raw_label)
            seen.add(raw_label)
        remaining = [label for label in unique if label not in seen]
        remaining.sort(key=lambda label: (-count_by_label[label], str(label)))
        sorted_labels = ordered + remaining

    mapping = {label: idx for idx, label in enumerate(sorted_labels)}
    return np.vectorize(mapping.get)(labels_arr)


def apply_cluster_order(
    labels: np.ndarray,
    order: ClusterOrder,
    ground_truth: np.ndarray | None = None,
    label_style: ClusterLabelStyle = "ground_truth_name",
) -> tuple[np.ndarray, list[str] | None]:
    """Apply cluster relabeling and return optional categorical order for plotting."""
    labels_arr = np.asarray(labels)
    if order is None:
        order = "ground_truth"

    if order == "ground_truth" and ground_truth is not None:
        new_labels, category_order = match_cluster_labels_to_ground_truth(
            labels_arr,
            ground_truth,
            label_style=label_style,
        )
        return new_labels, category_order

    new_labels = reorder_cluster_labels(
        labels_arr,
        order=order,
        ground_truth=ground_truth,
        label_style=label_style,
    )
    if order == "cell_count":
        unique, counts = np.unique(new_labels, return_counts=True)
        order_idx = sorted(range(len(unique)), key=lambda i: (-counts[i], str(unique[i])))
        category_order = [str(unique[i]) for i in order_idx]
        return new_labels, category_order
    return new_labels, None


def compute_clustering_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    kmeans = KMeans(n_clusters, seed=seed)
    kmeans.fit(X)
    return np.asarray(kmeans.labels_)


def compute_clustering_leiden(connectivity_graph: spmatrix, resolution: float, seed: int = 42) -> np.ndarray:
    resolution = float(np.clip(resolution, MIN_LEIDEN_RESOLUTION, MAX_LEIDEN_RESOLUTION))
    rng = random.Random(seed)
    igraph.set_random_number_generator(rng)
    g = igraph.Graph.Weighted_Adjacency(connectivity_graph, mode="directed")
    g.to_undirected(mode="each")
    clustering = g.community_leiden(objective_function="modularity", weights="weight", resolution=resolution)
    return np.asarray(clustering.membership)


def _clamp_resolution(res: float) -> float:
    return float(np.clip(res, MIN_LEIDEN_RESOLUTION, MAX_LEIDEN_RESOLUTION))


def _register_candidate(
    candidates: list[tuple[float, np.ndarray, int]],
    res: float,
    labels: np.ndarray,
) -> None:
    k = int(len(np.unique(labels)))
    candidates.append((_clamp_resolution(res), labels, k))


def _pick_closest_candidate(
    candidates: list[tuple[float, np.ndarray, int]],
    target_k: int,
) -> tuple[float, np.ndarray, int]:
    if not candidates:
        raise ValueError("No Leiden resolution candidates were evaluated.")

    def sort_key(item: tuple[float, np.ndarray, int]) -> tuple[int, int, float]:
        res, _, k = item
        return (abs(k - target_k), 0 if k == 1 else 1, res)

    return min(candidates, key=sort_key)


def search_leiden_resolution(
    connectivity_graph: spmatrix,
    target_k: int,
    res_start: float = 0.1,
    res_step: float = 0.1,
    res_epochs: int = 10,
    seed: int = 42,
    min_resolution: float = MIN_LEIDEN_RESOLUTION,
    max_resolution: float = MAX_LEIDEN_RESOLUTION,
) -> tuple[float, np.ndarray]:
    """Search Leiden resolution so cluster count matches ``target_k`` (SDMBench-style).

    Resolution is clamped to ``[min_resolution, max_resolution]``. If no resolution yields
    exactly ``target_k`` clusters within the search budget, the result with cluster count
    closest to ``target_k`` is returned (avoiding negative resolutions and spurious
  single-cluster solutions when a better partition exists).
    """
    if target_k < 1:
        raise ValueError("`target_k` must be >= 1.")

    min_resolution = float(min_resolution)
    max_resolution = float(max_resolution)
    candidates: list[tuple[float, np.ndarray, int]] = []

    res = _clamp_resolution(res_start)
    labels = compute_clustering_leiden(connectivity_graph, res, seed=seed)
    _register_candidate(candidates, res, labels)
    if len(np.unique(labels)) == target_k:
        return res, labels

    old_k = len(np.unique(labels))
    run = 0
    while old_k != target_k and run <= res_epochs:
        old_sign = 1 if old_k < target_k else -1
        res = _clamp_resolution(res + res_step * old_sign)
        labels = compute_clustering_leiden(connectivity_graph, res, seed=seed)
        _register_candidate(candidates, res, labels)
        new_k = len(np.unique(labels))
        if new_k == target_k:
            return res, labels
        new_sign = 1 if new_k < target_k else -1
        if new_sign == old_sign:
            old_k = new_k
        else:
            res_step = res_step / 2
        run += 1

    # Coarse grid fallback within bounds to find closest k (covers non-monotonic k vs resolution).
    grid = np.linspace(min_resolution, max_resolution, num=20)
    for grid_res in grid:
        grid_res = float(grid_res)
        if any(abs(c[0] - grid_res) < 1e-9 for c in candidates):
            continue
        grid_labels = compute_clustering_leiden(connectivity_graph, grid_res, seed=seed)
        _register_candidate(candidates, grid_res, grid_labels)

    best_res, best_labels, best_k = _pick_closest_candidate(candidates, target_k)
    if best_k != target_k:
        logger.warning(
            "Could not find Leiden resolution with exactly %s clusters (closest: %s at resolution=%.4f).",
            target_k,
            best_k,
            best_res,
        )
    return best_res, best_labels


def cluster_labels_leiden(
    neighbors: NeighborsResults,
    target_n_clusters: int | None = None,
    optimize_resolution: bool = True,
    resolution: float = 1.0,
    seed: int = 42,
    n_jobs: int = 1,
) -> np.ndarray:
    """Cluster cells with Leiden and return label assignments."""
    conn_graph = neighbors.knn_graph_connectivities

    if target_n_clusters is not None and optimize_resolution:
        _, labels = search_leiden_resolution(conn_graph, target_k=target_n_clusters, seed=seed)
        return labels

    if optimize_resolution and target_n_clusters is None:
        resolutions = np.array([2 * x / 10 for x in range(1, 11)])
        try:
            from joblib import Parallel, delayed

            clusterings = Parallel(n_jobs=n_jobs)(
                delayed(compute_clustering_leiden)(conn_graph, float(r), seed) for r in resolutions
            )
        except ImportError:
            warnings.warn("Using for loop over clustering resolutions. `pip install joblib` for parallelization.")
            clusterings = [compute_clustering_leiden(conn_graph, float(r), seed) for r in resolutions]
        sizes = [len(np.unique(c)) for c in clusterings]
        mid = int(np.median(sizes))
        best_idx = int(np.argmin([abs(s - mid) for s in sizes]))
        return clusterings[best_idx]

    return compute_clustering_leiden(conn_graph, resolution, seed=seed)


def cluster_labels_kmeans(X: np.ndarray, n_clusters: int, seed: int = 42) -> np.ndarray:
    return compute_clustering_kmeans(X, n_clusters, seed=seed)


def neighbors_from_spatial(
    spatial: np.ndarray,
    n_neighbors: int = 15,
    n_jobs: int = 1,
) -> NeighborsResults:
    """Build a kNN graph from spatial coordinates."""
    return pynndescent(spatial, n_neighbors=n_neighbors, random_state=0, n_jobs=n_jobs)
