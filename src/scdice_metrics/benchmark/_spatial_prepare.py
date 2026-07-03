from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from anndata import AnnData

from scdice_metrics.metrics._clustering import (
    ClusterLabelStyle,
    ClusterOrder,
    apply_cluster_order,
    cluster_labels_kmeans,
    cluster_labels_leiden,
    explain_cluster_order,
    ground_truth_categories_by_cell_count,
    neighbors_from_spatial,
    resolve_effective_cluster_order,
)
from scdice_metrics.nearest_neighbors import NeighborsResults

logger = logging.getLogger("scdice_metrics")

ClusterMethod = Literal["leiden", "kmeans"]
ClusterGraph = Literal["embedding", "spatial"]


@dataclass(frozen=True)
class SpatialClusteringPrepare:
    """Configuration for per-embedding spatial-domain clustering before metrics.

    If the resolved ``obs`` key for an embedding already exists in ``adata.obs`` and
    ``overwrite`` is False, existing labels are reused.

    Parameters
    ----------
    cluster_method
        ``"leiden"`` (default) clusters on a kNN graph; ``"kmeans"`` clusters the
        embedding directly with a fixed ``target_n_clusters`` (no resolution search).
    cluster_graph
        Graph used for Leiden: ``"embedding"`` (default) or ``"spatial"``.
        Ignored when ``cluster_method="kmeans"``.
    optimize_resolution
        Search Leiden resolution to match ``target_n_clusters``. Ignored for k-means.
    target_n_clusters
        Desired number of clusters. When ``None``, inferred from ``spatial_label_key``
        if present. Required for k-means when no spatial label key is available.
    cluster_order
        How to relabel clusters after assignment:

        - ``"ground_truth"`` (default): match each predicted cluster to the
          ground-truth region with the most similar cell count (Hungarian
          matching). Requires ``spatial_label_key`` in ``adata.obs``; otherwise
          falls back to ``"cell_count"``.
        - ``"cell_count"``: labels ``0..K-1`` by descending predicted cluster size.
        - ``"none"``: keep raw algorithm labels unchanged.
        - ``Sequence`` of original cluster ids: custom mapping order; unlisted ids
          are appended by descending cell count.
    cluster_label_style
        When ``cluster_order="ground_truth"``, assign matched ground-truth region
        names (``"ground_truth_name"``, default, best for scanpy color alignment)
        or numeric indices ranked by ground-truth cell count (``"index"``).
    align_ground_truth_categories
        If True (default), reorder ``spatial_label_key`` categorical levels by
        descending cell count so ground-truth and predicted columns share the
        same palette order in scanpy.
    log_prepare
        If True (default), log the resolved prepare options and per-embedding
        cluster relabeling details at INFO level.
    build_spatial_neighbors
        If True, build a spatial-coordinate kNN graph once during prepare. The graph
        is reused for Leiden on spatial coordinates and for PAS when enabled.
    spatial_neighbors_k
        Number of spatial neighbors to precompute. Defaults to
        ``max(spatial_n_neighbors, pas_k + 1)`` when ``None``.
    reuse_spatial_neighbors_for_pas
        Passed to :func:`~scdice_metrics.metrics.pas` during benchmarking.
    chaos_use_knn
        Passed to :func:`~scdice_metrics.metrics.chaos` during benchmarking.
    """

    cluster_method: ClusterMethod = "leiden"
    cluster_graph: ClusterGraph = "embedding"
    optimize_resolution: bool = True
    resolution: float = 1.0
    target_n_clusters: int | None = None
    cluster_order: ClusterOrder = "ground_truth"
    cluster_label_style: ClusterLabelStyle = "ground_truth_name"
    align_ground_truth_categories: bool = True
    log_prepare: bool = True
    spatial_n_neighbors: int = 15
    embedding_neighbor_k: int = 15
    seed: int = 42
    n_jobs: int = 1
    overwrite: bool = False
    build_spatial_neighbors: bool = True
    spatial_neighbors_k: int | None = None
    reuse_spatial_neighbors_for_pas: bool = True
    chaos_use_knn: bool = True


def _format_cluster_order_request(order: ClusterOrder) -> str:
    if isinstance(order, Sequence) and not isinstance(order, (str, bytes)):
        return f"custom {list(order)}"
    return repr(order)


def _log_prepare_configuration(
    prepare_cfg: SpatialClusteringPrepare,
    *,
    spatial_label_key: str,
    spatial_obsm_key: str,
    embedding_obsm_keys: list[str],
    spatial_cluster_key: str | dict[str, str] | None,
    pas_k: int,
    ground_truth: np.ndarray | None,
    target_k: int | None,
) -> None:
    effective_order = resolve_effective_cluster_order(prepare_cfg.cluster_order, ground_truth)
    spatial_neighbors_k = resolve_spatial_neighbors_k(prepare_cfg, pas_k=pas_k)
    need_spatial_neighbors = prepare_cfg.cluster_graph == "spatial" or (
        prepare_cfg.build_spatial_neighbors and prepare_cfg.reuse_spatial_neighbors_for_pas
    )

    lines = [
        "[spatial clustering prepare] configuration",
        f"  embeddings: {embedding_obsm_keys}",
        f"  spatial_label_key: {spatial_label_key!r}"
        + (" (missing -> cluster_order falls back to 'cell_count')" if ground_truth is None else ""),
        f"  spatial_obsm_key: {spatial_obsm_key!r}",
        f"  spatial_cluster_key: {spatial_cluster_key!r}",
        f"  cluster_method: {prepare_cfg.cluster_method!r}",
    ]
    if prepare_cfg.cluster_method == "leiden":
        lines.extend(
            [
                f"  cluster_graph: {prepare_cfg.cluster_graph!r}",
                f"  optimize_resolution: {prepare_cfg.optimize_resolution}",
                f"  resolution: {prepare_cfg.resolution}",
                f"  embedding_neighbor_k: {prepare_cfg.embedding_neighbor_k}",
                f"  spatial_n_neighbors: {prepare_cfg.spatial_n_neighbors}",
            ]
        )
    else:
        lines.append("  cluster_graph / optimize_resolution: ignored for k-means")

    lines.extend(
        [
            f"  target_n_clusters: {prepare_cfg.target_n_clusters!r} -> resolved {target_k!r}",
            f"  cluster_order: {_format_cluster_order_request(prepare_cfg.cluster_order)} -> effective {effective_order!r}",
            f"  cluster_label_style: {prepare_cfg.cluster_label_style!r}",
            f"  align_ground_truth_categories: {prepare_cfg.align_ground_truth_categories}",
            f"  overwrite existing labels: {prepare_cfg.overwrite}",
            f"  seed: {prepare_cfg.seed}",
            f"  build_spatial_neighbors: {prepare_cfg.build_spatial_neighbors}",
            f"  spatial_neighbors_k: {prepare_cfg.spatial_neighbors_k!r} -> resolved {spatial_neighbors_k}",
            f"  need_spatial_neighbors: {need_spatial_neighbors}",
            f"  reuse_spatial_neighbors_for_pas: {prepare_cfg.reuse_spatial_neighbors_for_pas} (pas_k={pas_k})",
            f"  chaos_use_knn: {prepare_cfg.chaos_use_knn}",
        ]
    )

    if ground_truth is not None:
        ordered_gt, gt_counts = ground_truth_categories_by_cell_count(ground_truth)
        gt_summary = ", ".join(f"{label}={int(count)}" for label, count in zip(ordered_gt, gt_counts))
        lines.append(f"  ground-truth sizes (desc): {gt_summary}")

    logger.info("\n".join(lines))


def _log_embedding_cluster_result(
    prepare_cfg: SpatialClusteringPrepare,
    *,
    emb_key: str,
    obs_key: str,
    raw_labels: np.ndarray | None,
    ordered_labels: np.ndarray | None,
    category_order: list[str] | None,
    ground_truth: np.ndarray | None,
    reused: bool,
) -> None:
    if reused:
        logger.info(
            "[spatial clustering prepare] %s -> `%s`: reused existing labels (overwrite=False)",
            emb_key,
            obs_key,
        )
        return

    assert raw_labels is not None and ordered_labels is not None
    lines = [
        f"[spatial clustering prepare] {emb_key} -> `{obs_key}`",
        *explain_cluster_order(
            raw_labels,
            ordered_labels,
            prepare_cfg.cluster_order,
            ground_truth=ground_truth,
            label_style=prepare_cfg.cluster_label_style,
        ),
    ]
    if category_order is not None:
        lines.append(f"  categorical order: {category_order}")
    logger.info("\n".join(lines))


def resolve_spatial_cluster_obs_key(
    emb_key: str,
    spatial_cluster_key: str | dict[str, str] | None,
    embedding_obsm_keys: list[str],
) -> str:
    """Resolve the ``adata.obs`` column used for predicted spatial domains."""
    if isinstance(spatial_cluster_key, dict):
        if emb_key not in spatial_cluster_key:
            raise KeyError(f"Missing spatial cluster key for embedding `{emb_key}` in mapping.")
        return spatial_cluster_key[emb_key]
    if spatial_cluster_key is None:
        return f"{emb_key}_spatial_cluster"
    if len(embedding_obsm_keys) == 1:
        return spatial_cluster_key
    return f"{spatial_cluster_key}__{emb_key}"


def infer_target_n_clusters(adata: AnnData, spatial_label_key: str, target_n_clusters: int | None) -> int | None:
    if target_n_clusters is not None:
        return target_n_clusters
    if spatial_label_key not in adata.obs:
        return None
    return int(adata.obs[spatial_label_key].nunique())


def resolve_spatial_neighbors_k(prepare_cfg: SpatialClusteringPrepare, pas_k: int = 10) -> int:
    if prepare_cfg.spatial_neighbors_k is not None:
        return max(prepare_cfg.spatial_neighbors_k, pas_k + 1)
    return max(prepare_cfg.spatial_n_neighbors, pas_k + 1)


def build_spatial_neighbors(
    adata: AnnData,
    spatial_obsm_key: str,
    n_neighbors: int,
    n_jobs: int = 1,
) -> NeighborsResults:
    """Build a spatial kNN graph shared by clustering and PAS."""
    if spatial_obsm_key not in adata.obsm:
        raise ValueError(f"Spatial coordinate key `{spatial_obsm_key}` not found in `adata.obsm`.")
    spatial = np.asarray(adata.obsm[spatial_obsm_key])
    return neighbors_from_spatial(spatial, n_neighbors=n_neighbors, n_jobs=n_jobs)


def _resolve_ground_truth_labels(adata: AnnData, spatial_label_key: str) -> np.ndarray | None:
    if spatial_label_key not in adata.obs:
        return None
    return adata.obs[spatial_label_key].to_numpy()


def _align_ground_truth_obs_categories(adata: AnnData, spatial_label_key: str) -> list[str]:
    gt_labels = _resolve_ground_truth_labels(adata, spatial_label_key)
    if gt_labels is None:
        return []
    ordered_gt, _ = ground_truth_categories_by_cell_count(gt_labels)
    category_order = [str(x) for x in ordered_gt]
    adata.obs[spatial_label_key] = pd.Categorical(
        adata.obs[spatial_label_key].astype(str),
        categories=category_order,
        ordered=True,
    )
    return category_order


def _write_cluster_obs(
    adata: AnnData,
    obs_key: str,
    labels: np.ndarray,
    category_order: list[str] | None,
) -> None:
    labels_str = np.asarray(labels).astype(str)
    if category_order is not None:
        adata.obs[obs_key] = pd.Categorical(labels_str, categories=category_order, ordered=True)
    else:
        adata.obs[obs_key] = labels_str


def compute_spatial_cluster_labels(
    adata: AnnData,
    emb_key: str,
    prepare_cfg: SpatialClusteringPrepare,
    spatial_label_key: str,
    spatial_obsm_key: str,
    embedding_neighbors: NeighborsResults | None = None,
    spatial_neighbors: NeighborsResults | None = None,
) -> np.ndarray:
    """Cluster one embedding (or spatial graph) and return predicted domain labels."""
    target_k = infer_target_n_clusters(adata, spatial_label_key, prepare_cfg.target_n_clusters)

    if prepare_cfg.cluster_method == "kmeans":
        if target_k is None:
            raise ValueError(
                "`target_n_clusters` must be set for k-means spatial clustering when "
                "`spatial_label_key` is unavailable."
            )
        labels = cluster_labels_kmeans(
            np.asarray(adata.obsm[emb_key]),
            n_clusters=target_k,
            seed=prepare_cfg.seed,
        )
    else:
        if prepare_cfg.cluster_graph == "spatial":
            if spatial_neighbors is None:
                spatial_neighbors = build_spatial_neighbors(
                    adata,
                    spatial_obsm_key,
                    n_neighbors=resolve_spatial_neighbors_k(prepare_cfg),
                    n_jobs=prepare_cfg.n_jobs,
                )
            neighbors = spatial_neighbors
        else:
            if embedding_neighbors is None:
                raise ValueError(
                    "Embedding kNN graph is required for `cluster_graph='embedding'`. "
                    "Call `Benchmarker.prepare()` first or use `cluster_graph='spatial'`."
                )
            neighbors = embedding_neighbors

        labels = cluster_labels_leiden(
            neighbors,
            target_n_clusters=target_k,
            optimize_resolution=prepare_cfg.optimize_resolution,
            resolution=prepare_cfg.resolution,
            seed=prepare_cfg.seed,
            n_jobs=prepare_cfg.n_jobs,
        )

    return labels


def prepare_spatial_clusters(
    adata: AnnData,
    embedding_obsm_keys: list[str],
    spatial_label_key: str,
    spatial_obsm_key: str,
    spatial_cluster_key: str | dict[str, str] | None,
    prepare_cfg: SpatialClusteringPrepare,
    embedding_neighbors: dict[str, NeighborsResults] | None = None,
    pas_k: int = 10,
) -> tuple[dict[str, str], NeighborsResults | None]:
    """Write per-embedding spatial cluster columns to ``adata.obs`` when missing.

    Returns
    -------
    resolved
        Mapping from embedding obsm key to the ``adata.obs`` column used.
    spatial_neighbors
        Precomputed spatial kNN graph when ``build_spatial_neighbors`` is True.
    """
    if spatial_obsm_key not in adata.obsm:
        raise ValueError(f"Spatial coordinate key `{spatial_obsm_key}` not found in `adata.obsm`.")

    gt_category_order: list[str] | None = None
    if prepare_cfg.align_ground_truth_categories:
        gt_category_order = _align_ground_truth_obs_categories(adata, spatial_label_key)
    ground_truth = _resolve_ground_truth_labels(adata, spatial_label_key)
    target_k = infer_target_n_clusters(adata, spatial_label_key, prepare_cfg.target_n_clusters)

    if prepare_cfg.log_prepare:
        _log_prepare_configuration(
            prepare_cfg,
            spatial_label_key=spatial_label_key,
            spatial_obsm_key=spatial_obsm_key,
            embedding_obsm_keys=embedding_obsm_keys,
            spatial_cluster_key=spatial_cluster_key,
            pas_k=pas_k,
            ground_truth=ground_truth,
            target_k=target_k,
        )
        if gt_category_order:
            logger.info(
                "[spatial clustering prepare] aligned `%s` categories by cell count: %s",
                spatial_label_key,
                gt_category_order,
            )

    spatial_neighbors: NeighborsResults | None = None
    need_spatial_neighbors = prepare_cfg.cluster_graph == "spatial" or (
        prepare_cfg.build_spatial_neighbors and prepare_cfg.reuse_spatial_neighbors_for_pas
    )
    if need_spatial_neighbors:
        spatial_neighbors = build_spatial_neighbors(
            adata,
            spatial_obsm_key,
            n_neighbors=resolve_spatial_neighbors_k(prepare_cfg, pas_k=pas_k),
            n_jobs=prepare_cfg.n_jobs,
        )
        if prepare_cfg.log_prepare:
            logger.info(
                "[spatial clustering prepare] built spatial neighbor graph: k=%s, n_cells=%s",
                resolve_spatial_neighbors_k(prepare_cfg, pas_k=pas_k),
                adata.n_obs,
            )

    resolved: dict[str, str] = {}
    computed = 0
    reused = 0
    for emb_key in embedding_obsm_keys:
        if emb_key not in adata.obsm:
            raise ValueError(f"Embedding key `{emb_key}` not found in `adata.obsm`.")
        obs_key = resolve_spatial_cluster_obs_key(emb_key, spatial_cluster_key, embedding_obsm_keys)
        resolved[emb_key] = obs_key

        if obs_key in adata.obs and not prepare_cfg.overwrite:
            if prepare_cfg.log_prepare:
                _log_embedding_cluster_result(
                    prepare_cfg,
                    emb_key=emb_key,
                    obs_key=obs_key,
                    raw_labels=None,
                    ordered_labels=None,
                    category_order=None,
                    ground_truth=ground_truth,
                    reused=True,
                )
            reused += 1
            continue

        neigh = embedding_neighbors.get(emb_key) if embedding_neighbors else None
        raw_labels = compute_spatial_cluster_labels(
            adata,
            emb_key,
            prepare_cfg,
            spatial_label_key=spatial_label_key,
            spatial_obsm_key=spatial_obsm_key,
            embedding_neighbors=neigh,
            spatial_neighbors=spatial_neighbors,
        )
        ordered_labels, category_order = apply_cluster_order(
            raw_labels,
            order=prepare_cfg.cluster_order,
            ground_truth=ground_truth,
            label_style=prepare_cfg.cluster_label_style,
        )
        if (
            prepare_cfg.cluster_order == "ground_truth"
            and prepare_cfg.cluster_label_style == "ground_truth_name"
            and gt_category_order
        ):
            extra_categories = [c for c in (category_order or []) if c not in gt_category_order]
            category_order = gt_category_order + extra_categories
        _write_cluster_obs(adata, obs_key, ordered_labels, category_order)
        computed += 1
        if prepare_cfg.log_prepare:
            _log_embedding_cluster_result(
                prepare_cfg,
                emb_key=emb_key,
                obs_key=obs_key,
                raw_labels=raw_labels,
                ordered_labels=ordered_labels,
                category_order=category_order,
                ground_truth=ground_truth,
                reused=False,
            )

    if prepare_cfg.log_prepare:
        logger.info(
            "[spatial clustering prepare] done: computed=%s, reused=%s, obs columns=%s",
            computed,
            reused,
            resolved,
        )

    return resolved, spatial_neighbors
