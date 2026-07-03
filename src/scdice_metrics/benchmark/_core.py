import gc
import os
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from enum import Enum
from functools import partial
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import bar
from sklearn.preprocessing import MinMaxScaler

import scdice_metrics
from scdice_metrics.benchmark._progress import NestedBenchmarkProgress, iter_progress, print_status
from scdice_metrics.benchmark._spatial_prepare import (
    SpatialClusteringPrepare,
    prepare_spatial_clusters,
    resolve_spatial_cluster_obs_key,
)
from scdice_metrics.benchmark._templates import (
    BenchmarkMode,
    BenchmarkTemplate,
    infer_template,
    orient_metrics_higher_is_better,
    resolve_template,
)
from scdice_metrics.nearest_neighbors import NeighborsResults, pynndescent

Kwargs = dict[str, Any]
MetricType = bool | Kwargs

_LABELS = "labels"
_SPATIAL_LABELS = "spatial_labels"
_BATCH = "batch"
_X_PRE = "X_pre"
_DISENTANGLEMENT_FACTORS = "disentanglement_factors"
_LEAKAGE_TARGET = "leakage_target"
_SPATIAL_CLUSTER = "spatial_cluster"
_SPATIAL_COORDS = "spatial_coords"
_SPATIAL_NEIGHBORS = "spatial_neighbor_res"
_METRIC_TYPE = "Metric Type"
_AGGREGATE_SCORE = "Aggregate score"

# Mapping of metric fn names to clean DataFrame column names
metric_name_cleaner = {
    "silhouette_label": "Silhouette label",
    "silhouette_batch": "Silhouette batch",
    "isolated_labels": "Isolated labels",
    "nmi_ari_cluster_labels_leiden_nmi": "Leiden NMI",
    "nmi_ari_cluster_labels_leiden_ari": "Leiden ARI",
    "nmi_ari_cluster_labels_kmeans_nmi": "KMeans NMI",
    "nmi_ari_cluster_labels_kmeans_ari": "KMeans ARI",
    "clisi_knn": "cLISI",
    "ilisi_knn": "iLISI",
    "kbet_per_label": "KBET",
    "bras": "BRAS",
    "graph_connectivity": "Graph connectivity",
    "pcr_comparison": "PCR comparison",
    "mig_score": "MIG",
    "mig_mean_factor_mi": "Mean factor MI",
    "mig_mean_complement_mi": "Mean complement MI",
    "mixed_ksg_mig_max_mig": "Mixed-KSG maxMIG",
    "mixed_ksg_mig_concat_mig": "Mixed-KSG concatMIG",
    "mixed_ksg_mig_min_mig": "Mixed-KSG minMIG",
    "classifier_attribute_gap_concat_gap": "Classifier concat gap",
    "classifier_attribute_gap_max_gap": "Classifier max gap",
    "classifier_attribute_gap_mean_accuracy": "Classifier mean accuracy",
    "classifier_attribute_gap_mean_complement_accuracy": "Classifier complement accuracy",
    "classifier_attribute_gap_mean_competitor_accuracy": "Classifier competitor accuracy",
    "fairness_leakage_accuracy": "Leakage accuracy",
    "fairness_leakage_demographic_parity_difference": "Demographic parity difference",
    "fairness_leakage_demographic_parity_ratio": "Demographic parity ratio",
    "fairness_leakage_equalized_odds_difference": "Equalized odds difference",
    "hom": "HOM",
    "com": "COM",
    "chaos": "CHAOS",
    "pas": "PAS",
}


@dataclass(frozen=True)
class BioConservation:
    """Specification of bio conservation metrics to run in the pipeline.

    Metrics can be included using a boolean flag. Custom keyword args can be
    used by passing a dictionary here. Keyword args should not set data-related
    parameters, such as `X` or `labels`.
    """

    isolated_labels: MetricType = True
    nmi_ari_cluster_labels_leiden: MetricType = False
    nmi_ari_cluster_labels_kmeans: MetricType = True
    silhouette_label: MetricType = True
    clisi_knn: MetricType = True


@dataclass(frozen=True)
class BatchCorrection:
    """Specification of which batch correction metrics to run in the pipeline.

    Metrics can be included using a boolean flag. Custom keyword args can be
    used by passing a dictionary here. Keyword args should not set data-related
    parameters, such as `X` or `labels`.
    """

    bras: MetricType = True
    ilisi_knn: MetricType = True
    kbet_per_label: MetricType = True
    graph_connectivity: MetricType = True
    pcr_comparison: MetricType = True


@dataclass(frozen=True)
class Disentanglement:
    """Specification of which disentanglement metrics to run in the pipeline.

    Metrics can be included using a boolean flag. Custom keyword args can be
    used by passing a dictionary here. Keyword args should not set data-related
    parameters, such as `latent`, `factors`, or `target`.
    """

    mig: MetricType = True
    mixed_ksg_mig: MetricType = True
    classifier_attribute_gap: MetricType = True
    fairness_leakage: MetricType = False


@dataclass(frozen=True)
class SpatialClustering:
    """Specification of which spatial clustering metrics to run in the pipeline.

    Metrics can be included using a boolean flag. Custom keyword args can be
    used by passing a dictionary here. Keyword args should not set data-related
    parameters, such as `labels` or `spatial`.
    """

    hom: MetricType = False
    com: MetricType = False
    chaos: MetricType = True
    pas: MetricType = True


def _get_pas_k(spatial_clustering_metrics: SpatialClustering | None) -> int:
    if spatial_clustering_metrics is None:
        return 10
    pas_cfg = spatial_clustering_metrics.pas
    if isinstance(pas_cfg, dict) and "k" in pas_cfg:
        return int(pas_cfg["k"])
    return 10


def _call_metric_pas(ad: AnnData, fn: Callable[..., Any]) -> float:
    kwargs: Kwargs = {}
    if isinstance(fn, partial):
        pas_fn = fn.func
        kwargs.update(fn.keywords or {})
    else:
        pas_fn = fn
    if _SPATIAL_NEIGHBORS in ad.uns:
        kwargs.setdefault("spatial_neighbors", ad.uns[_SPATIAL_NEIGHBORS])
    return pas_fn(ad.obs[_SPATIAL_CLUSTER], ad.obsm[_SPATIAL_COORDS], **kwargs)


def _call_metric_chaos(ad: AnnData, fn: Callable[..., Any]) -> float:
    kwargs: Kwargs = {}
    if isinstance(fn, partial):
        chaos_fn = fn.func
        kwargs.update(fn.keywords or {})
    else:
        chaos_fn = fn
    return chaos_fn(ad.obs[_SPATIAL_CLUSTER], ad.obsm[_SPATIAL_COORDS], **kwargs)


class MetricAnnDataAPI(Enum):
    """Specification of the AnnData API for a metric."""

    isolated_labels = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    nmi_ari_cluster_labels_leiden = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    nmi_ari_cluster_labels_kmeans = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    silhouette_label = lambda ad, fn: fn(ad.X, ad.obs[_LABELS])
    clisi_knn = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_LABELS])
    graph_connectivity = lambda ad, fn: fn(ad.uns["15_neighbor_res"], ad.obs[_LABELS])
    bras = lambda ad, fn: fn(ad.X, ad.obs[_LABELS], ad.obs[_BATCH])
    pcr_comparison = lambda ad, fn: fn(ad.obsm[_X_PRE], ad.X, ad.obs[_BATCH], categorical=True)
    ilisi_knn = lambda ad, fn: fn(ad.uns["90_neighbor_res"], ad.obs[_BATCH])
    kbet_per_label = lambda ad, fn: fn(ad.uns["50_neighbor_res"], ad.obs[_BATCH], ad.obs[_LABELS])
    mig = lambda ad, fn: fn(ad.X, ad.uns[_DISENTANGLEMENT_FACTORS])
    mixed_ksg_mig = lambda ad, fn: fn(ad.X, ad.uns[_DISENTANGLEMENT_FACTORS])
    classifier_attribute_gap = lambda ad, fn: fn(ad.X, ad.uns[_DISENTANGLEMENT_FACTORS])
    fairness_leakage = lambda ad, fn: fn(ad.X, ad.uns[_DISENTANGLEMENT_FACTORS], ad.obs[_LEAKAGE_TARGET])
    hom = lambda ad, fn: fn(ad.obs[_SPATIAL_LABELS], ad.obs[_SPATIAL_CLUSTER])
    com = lambda ad, fn: fn(ad.obs[_SPATIAL_LABELS], ad.obs[_SPATIAL_CLUSTER])
    chaos = _call_metric_chaos
    pas = _call_metric_pas


class Benchmarker:
    """Benchmarking pipeline for the single-cell integration task.

    Parameters
    ----------
    adata
        AnnData object containing the raw count data and integrated embeddings as obsm keys.
    batch_key
        Key in `adata.obs` that contains the batch information.
    label_key
        Key in ``adata.obs`` for bio conservation / batch metrics (e.g. cell type).
    spatial_label_key
        Key in ``adata.obs`` for spatial-domain ground truth used by HOM/COM and by
        default for ``target_n_clusters`` when auto-clustering. If ``None``, falls
        back to ``label_key``.
    embedding_obsm_keys
        List of obsm keys that contain the embeddings to be benchmarked.
    bio_conservation_metrics
        Specification of which bio conservation metrics to run in the pipeline.
    batch_correction_metrics
        Specification of which batch correction metrics to run in the pipeline.
    disentanglement_metrics
        Specification of which disentanglement metrics to run in the pipeline.
    spatial_clustering_metrics
        Specification of which spatial clustering metrics to run in the pipeline.
    disentanglement_factor_keys
        Keys in `adata.obs` that contain the discrete factors used for disentanglement metrics.
    leakage_target_key
        Optional key in `adata.obs` used as the prediction target for fairness leakage metrics.
    spatial_cluster_key
        Predicted spatial-domain labels in ``adata.obs``. Can be:

        - ``None``: auto-generate ``{emb_key}_spatial_cluster`` per embedding.
        - ``str``: one column for a single embedding; for multiple embeddings use
          ``{spatial_cluster_key}__{emb_key}`` (existing columns are reused).
        - ``dict``: map each ``embedding_obsm_keys`` entry to an ``obs`` column.

        If the resolved column is missing, :meth:`prepare` clusters the embedding
        (see ``spatial_clustering_prepare``).
    spatial_obsm_key
        Key in `adata.obsm` containing spatial coordinates. Default is `"spatial"`.
    spatial_clustering_prepare
        Clustering settings used when predicted spatial labels are not already
        stored in ``adata.obs``. Ignored when all resolved columns already exist
        unless ``overwrite=True``.
    compute_neighbors
        Whether :meth:`prepare` computes embedding kNN graphs for bio/batch metrics
        and embedding-based spatial clustering.
    pre_integrated_embedding_obsm_key
        Obsm key containing a non-integrated embedding of the data. If `None`, the embedding will be computed
        in the prepare step. See the notes below for more information.
    n_jobs
        Number of jobs to use for parallelization of neighbor search.
    progress_bar
        Whether to show progress for :meth:`~scdice_metrics.benchmark.Benchmarker.prepare` and
        :meth:`~scdice_metrics.benchmark.Benchmarker.benchmark`. Style is controlled globally via
        ``scdice_metrics.settings.progress_bar_style`` (``"rich"`` or ``"tqdm"``).
    solver
        SVD solver to use during PCA. can help stability issues. Choose from: "arpack", "randomized" or "auto"
    display_template
        How to group metrics for aggregation and :meth:`plot_results_table`. Use
        ``"auto"`` (default) to infer from enabled collections, ``"scib"`` for
        Bio conservation + Batch correction, ``"sdmbench"`` for spatial
        Accuracy + Continuity, ``"full"`` for all groups, ``"legacy"`` for one
        group per metric collection, or a custom :class:`BenchmarkTemplate`.

    Notes
    -----
    `adata.X` should contain a form of the data that is not integrated, but is normalized. The `prepare` method will
    use `adata.X` for PCA via :func:`~scanpy.tl.pca`, which also only uses features masked via `adata.var['highly_variable']`.

    See further usage examples in the following tutorial:

    1. :doc:`/notebooks/lung_example`
    """

    def __init__(
        self,
        adata: AnnData,
        batch_key: str,
        label_key: str,
        embedding_obsm_keys: list[str],
        bio_conservation_metrics: BioConservation | None = BioConservation(),
        batch_correction_metrics: BatchCorrection | None = BatchCorrection(),
        disentanglement_metrics: Disentanglement | None = None,
        spatial_clustering_metrics: SpatialClustering | None = None,
        disentanglement_factor_keys: list[str] | None = None,
        leakage_target_key: str | None = None,
        spatial_cluster_key: str | dict[str, str] | None = None,
        spatial_label_key: str | None = None,
        spatial_obsm_key: str = "spatial",
        spatial_clustering_prepare: SpatialClusteringPrepare | None = None,
        pre_integrated_embedding_obsm_key: str | None = None,
        n_jobs: int = 1,
        progress_bar: bool = True,
        solver: str = "arpack",
        aggregate_metric_weights: dict[str, float] | None = None,
        display_template: BenchmarkTemplate | BenchmarkMode | None = "auto",
        compute_neighbors: bool | None = None,
    ):
        self._adata = adata
        self._embedding_obsm_keys = embedding_obsm_keys
        self._pre_integrated_embedding_obsm_key = pre_integrated_embedding_obsm_key
        self._bio_conservation_metrics = bio_conservation_metrics
        self._batch_correction_metrics = batch_correction_metrics
        self._disentanglement_metrics = disentanglement_metrics
        self._spatial_clustering_metrics = spatial_clustering_metrics
        self._disentanglement_factor_keys = disentanglement_factor_keys
        self._leakage_target_key = leakage_target_key
        self._spatial_cluster_key = spatial_cluster_key
        self._spatial_label_key = spatial_label_key
        self._spatial_obsm_key = spatial_obsm_key
        self._spatial_clustering_prepare = (
            spatial_clustering_prepare
            if spatial_clustering_prepare is not None
            else (SpatialClusteringPrepare() if spatial_clustering_metrics is not None else None)
        )
        self._spatial_cluster_obs_keys: dict[str, str] = {}
        self._spatial_neighbors: NeighborsResults | None = None
        self._results = pd.DataFrame(columns=list(self._embedding_obsm_keys) + [_METRIC_TYPE])
        self._emb_adatas = {}
        self._neighbor_values = (15, 50, 90)
        self._prepared = False
        self._benchmarked = False
        self._batch_key = batch_key
        self._label_key = label_key
        self._n_jobs = n_jobs
        self._progress_bar = progress_bar
        self._solver = solver
        self._aggregate_metric_weights = aggregate_metric_weights
        self._display_template_arg = display_template
        self._compute_neighbors = (
            compute_neighbors if compute_neighbors is not None else self._infer_compute_neighbors()
        )

        if (
            self._bio_conservation_metrics is None
            and self._batch_correction_metrics is None
            and self._disentanglement_metrics is None
            and self._spatial_clustering_metrics is None
        ):
            raise ValueError("At least one metric collection must be defined.")
        if self._disentanglement_metrics is not None and not self._disentanglement_factor_keys:
            raise ValueError("`disentanglement_factor_keys` must be provided when disentanglement metrics are enabled.")
        if self._disentanglement_factor_keys is not None:
            missing_factor_keys = [key for key in self._disentanglement_factor_keys if key not in self._adata.obs]
            if missing_factor_keys:
                raise ValueError(f"Factor keys not found in `adata.obs`: {missing_factor_keys}")
        if self._leakage_target_key is not None and self._leakage_target_key not in self._adata.obs:
            raise ValueError(f"Leakage target key `{self._leakage_target_key}` not found in `adata.obs`.")
        if (
            self._disentanglement_metrics is not None
            and self._leakage_target_key is None
            and getattr(self._disentanglement_metrics, "fairness_leakage") is not False
        ):
            raise ValueError("`leakage_target_key` must be provided when `fairness_leakage` is enabled.")
        if self._spatial_clustering_metrics is not None:
            if self._spatial_obsm_key not in self._adata.obsm:
                raise ValueError(f"Spatial coordinate key `{self._spatial_obsm_key}` not found in `adata.obsm`.")
            if self._resolved_spatial_label_key not in self._adata.obs:
                raise ValueError(
                    f"Spatial label key `{self._resolved_spatial_label_key}` not found in `adata.obs`."
                )
            if isinstance(self._spatial_cluster_key, dict):
                missing_emb = [k for k in self._embedding_obsm_keys if k not in self._spatial_cluster_key]
                if missing_emb:
                    raise ValueError(f"Missing spatial cluster mapping for embeddings: {missing_emb}")

        self._metric_collection_dict = {}
        if self._bio_conservation_metrics is not None:
            self._metric_collection_dict.update({"Bio conservation": self._bio_conservation_metrics})
        if self._batch_correction_metrics is not None:
            self._metric_collection_dict.update({"Batch correction": self._batch_correction_metrics})
        if self._disentanglement_metrics is not None:
            self._metric_collection_dict.update({"Disentanglement": self._disentanglement_metrics})
        if self._spatial_clustering_metrics is not None:
            self._metric_collection_dict.update({"Spatial clustering": self._spatial_clustering_metrics})

    @property
    def metric_collection_names(self) -> list[str]:
        """Names of enabled metric collections used when running the pipeline."""
        return list(self._metric_collection_dict.keys())

    @property
    def _resolved_spatial_label_key(self) -> str:
        if self._spatial_label_key is not None:
            return self._spatial_label_key
        return self._label_key

    @property
    def spatial_cluster_obs_keys(self) -> dict[str, str]:
        """Mapping from embedding obsm key to spatial cluster ``adata.obs`` column."""
        return dict(self._spatial_cluster_obs_keys)

    def _infer_compute_neighbors(self) -> bool:
        if self._bio_conservation_metrics is not None or self._batch_correction_metrics is not None:
            return True
        if self._spatial_clustering_metrics is None or self._spatial_clustering_prepare is None:
            return False
        return self._spatial_clustering_prepare.cluster_graph == "embedding"

    def _needs_bio_batch_neighbor_graphs(self) -> bool:
        return self._bio_conservation_metrics is not None or self._batch_correction_metrics is not None

    def resolve_spatial_cluster_obs_key(self, emb_key: str) -> str:
        """Return the ``adata.obs`` column for predicted spatial domains of one embedding."""
        if emb_key in self._spatial_cluster_obs_keys:
            return self._spatial_cluster_obs_keys[emb_key]
        return resolve_spatial_cluster_obs_key(emb_key, self._spatial_cluster_key, self._embedding_obsm_keys)

    def resolve_display_template(
        self, display_template: BenchmarkTemplate | BenchmarkMode | None = None
    ) -> BenchmarkTemplate:
        """Resolve the display template for results and plotting."""
        if display_template is None:
            display_template = self._display_template_arg
        template = resolve_template(display_template, self.metric_collection_names)
        if self._aggregate_metric_weights is None:
            return template
        template_weights = dict(template.weights or {})
        template_weights.update(self._aggregate_metric_weights)
        return BenchmarkTemplate(
            name=template.name,
            groups=template.groups,
            weights=template_weights,
            parent_groups=template.parent_groups,
            lower_is_better=template.lower_is_better,
        )

    def _metric_display_groups(self, template: BenchmarkTemplate) -> pd.Series:
        run_types = self._results[_METRIC_TYPE]
        return pd.Series(
            {
                metric: template.metric_to_group(metric, fallback=run_types[metric])
                for metric in run_types.index
            }
        )

    def _resolve_aggregate_weights(
        self, metric_types: list[str], template: BenchmarkTemplate
    ) -> dict[str, float]:
        if self._aggregate_metric_weights is not None:
            weights = {metric_type: self._aggregate_metric_weights[metric_type] for metric_type in metric_types}
            for metric_type in metric_types:
                if metric_type not in weights:
                    weights[metric_type] = 1.0
        else:
            return template.resolve_weights(metric_types)

        weight_sum = sum(weights.values())
        if weight_sum <= 0:
            raise ValueError("Aggregate metric weights must sum to a positive value.")
        return {metric_type: weight / weight_sum for metric_type, weight in weights.items()}

    def prepare_spatial_clusters(self, force: bool = False) -> dict[str, str]:
        """Ensure per-embedding spatial cluster labels exist in ``adata.obs``.

        Parameters
        ----------
        force
            Recompute labels even when the resolved ``obs`` column already exists.

        Returns
        -------
        dict
            Mapping from embedding key to ``adata.obs`` column name.
        """
        if self._spatial_clustering_metrics is None:
            raise ValueError("Spatial clustering metrics are not enabled.")
        prepare_cfg = self._spatial_clustering_prepare
        if prepare_cfg is None:
            prepare_cfg = SpatialClusteringPrepare()
        if force:
            prepare_cfg = replace(prepare_cfg, overwrite=True)

        embedding_neighbors: dict[str, NeighborsResults] = {}
        if prepare_cfg.cluster_graph == "embedding":
            k_embed = prepare_cfg.embedding_neighbor_k
            for emb_key in iter_progress(
                self._embedding_obsm_keys,
                description="[bold cyan]Building graphs for spatial clustering…[/]",
                disable=not self._progress_bar,
                total=len(self._embedding_obsm_keys),
            ):
                ad = AnnData(self._adata.obsm[emb_key])
                embedding_neighbors[emb_key] = pynndescent(
                    ad.X, n_neighbors=k_embed, random_state=0, n_jobs=self._n_jobs
                )

        pas_k = _get_pas_k(self._spatial_clustering_metrics)

        self._spatial_cluster_obs_keys, self._spatial_neighbors = prepare_spatial_clusters(
            self._adata,
            self._embedding_obsm_keys,
            self._resolved_spatial_label_key,
            self._spatial_obsm_key,
            self._spatial_cluster_key,
            prepare_cfg,
            embedding_neighbors=embedding_neighbors,
            pas_k=pas_k,
        )
        return self._spatial_cluster_obs_keys

    def prepare(self, neighbor_computer: Callable[[np.ndarray, int], NeighborsResults] | None = None) -> None:
        """Prepare the data for benchmarking.

        Parameters
        ----------
        neighbor_computer
            Function that computes the neighbors of the data. If `None`, the neighbors will be computed
            with :func:`~scdice_metrics.utils.nearest_neighbors.pynndescent`. The function should take as input
            the data and the number of neighbors to compute and return a :class:`~scdice_metrics.utils.nearest_neighbors.NeighborsResults`
            object.
        """
        gc.collect()

        # Compute PCA
        if self._pre_integrated_embedding_obsm_key is None and self._needs_bio_batch_neighbor_graphs():
            # This is how scib does it
            # https://github.com/theislab/scib/blob/896f689e5fe8c57502cb012af06bed1a9b2b61d2/scib/metrics/pcr.py#L197
            sc.tl.pca(self._adata, svd_solver=self._solver, use_highly_variable=False)
            self._pre_integrated_embedding_obsm_key = "X_pca"

        if self._spatial_clustering_metrics is not None:
            print_status(
                "[bold cyan]Clustering spatial domains…[/]",
                disable=not self._progress_bar,
            )
            self.prepare_spatial_clusters()

        for emb_key in self._embedding_obsm_keys:
            self._emb_adatas[emb_key] = AnnData(self._adata.obsm[emb_key], obs=self._adata.obs)
            self._emb_adatas[emb_key].obs[_BATCH] = np.asarray(self._adata.obs[self._batch_key].values)
            self._emb_adatas[emb_key].obs[_LABELS] = np.asarray(self._adata.obs[self._label_key].values)
            if self._pre_integrated_embedding_obsm_key is not None:
                self._emb_adatas[emb_key].obsm[_X_PRE] = self._adata.obsm[self._pre_integrated_embedding_obsm_key]
            if self._disentanglement_factor_keys is not None:
                self._emb_adatas[emb_key].uns[_DISENTANGLEMENT_FACTORS] = self._adata.obs[
                    self._disentanglement_factor_keys
                ].copy()
            if self._leakage_target_key is not None:
                self._emb_adatas[emb_key].obs[_LEAKAGE_TARGET] = np.asarray(
                    self._adata.obs[self._leakage_target_key].values
                )
            if self._spatial_clustering_metrics is not None:
                self._emb_adatas[emb_key].obs[_SPATIAL_LABELS] = np.asarray(
                    self._adata.obs[self._resolved_spatial_label_key].values
                )
                cluster_obs_key = self.resolve_spatial_cluster_obs_key(emb_key)
                self._emb_adatas[emb_key].obs[_SPATIAL_CLUSTER] = np.asarray(self._adata.obs[cluster_obs_key].values)
                self._emb_adatas[emb_key].obsm[_SPATIAL_COORDS] = np.asarray(self._adata.obsm[self._spatial_obsm_key])
                if (
                    self._spatial_neighbors is not None
                    and self._spatial_clustering_prepare is not None
                    and self._spatial_clustering_prepare.reuse_spatial_neighbors_for_pas
                ):
                    self._emb_adatas[emb_key].uns[_SPATIAL_NEIGHBORS] = self._spatial_neighbors

        # Compute neighbors
        if self._compute_neighbors:
            max_k = max(self._neighbor_values)
            for emb_key, ad in iter_progress(
                self._emb_adatas.items(),
                description="[bold cyan]Computing embedding neighbors…[/]",
                disable=not self._progress_bar,
                total=len(self._emb_adatas),
            ):
                if neighbor_computer is not None:
                    neigh_result = neighbor_computer(ad.X, max_k)
                else:
                    neigh_result = pynndescent(ad.X, n_neighbors=max_k, random_state=0, n_jobs=self._n_jobs)
                for n in self._neighbor_values:
                    ad.uns[f"{n}_neighbor_res"] = neigh_result.subset_neighbors(n=n)
        elif self._needs_bio_batch_neighbor_graphs():
            warnings.warn(
                "Bio/batch metrics require embedding neighbors but `compute_neighbors=False`.",
                UserWarning,
            )

        self._prepared = True
        print_status("[bold green]✅ Prepare complete.[/]", disable=not self._progress_bar)

    def benchmark(self) -> None:
        """Run the pipeline."""
        if self._benchmarked:
            warnings.warn(
                "The benchmark has already been run. Running it again will overwrite the previous results.",
                UserWarning,
            )

        if not self._prepared:
            self.prepare()

        num_metrics = sum(
            sum(v is not False for v in asdict(met_col).values())
            for met_col in self._metric_collection_dict.values()
        )

        print_status("[bold cyan]Running benchmark metrics…[/]", disable=not self._progress_bar)
        with NestedBenchmarkProgress(
            disable=not self._progress_bar,
            n_embeddings=len(self._emb_adatas),
            n_metrics=num_metrics,
        ) as prog:
            for emb_key, ad in self._emb_adatas.items():
                prog.start_embedding(emb_key)
                for metric_type, metric_name, use_metric_or_kwargs in self._iter_enabled_metrics():
                    gc.collect()
                    prog.set_metric(metric_type, metric_name)
                    metric_fn = getattr(scdice_metrics, metric_name)
                    metric_kwargs = (
                        dict(use_metric_or_kwargs) if isinstance(use_metric_or_kwargs, dict) else {}
                    )
                    if self._spatial_clustering_prepare is not None:
                        if metric_name == "pas":
                            metric_kwargs.setdefault(
                                "reuse_spatial_neighbors",
                                self._spatial_clustering_prepare.reuse_spatial_neighbors_for_pas,
                            )
                        elif metric_name == "chaos":
                            metric_kwargs.setdefault(
                                "use_knn",
                                self._spatial_clustering_prepare.chaos_use_knn,
                            )
                    if metric_kwargs:
                        metric_fn = partial(metric_fn, **metric_kwargs)
                    metric_value = getattr(MetricAnnDataAPI, metric_name)(ad, metric_fn)
                    if isinstance(metric_value, dict):
                        for k, v in metric_value.items():
                            self._results.loc[f"{metric_name}_{k}", emb_key] = v
                            self._results.loc[f"{metric_name}_{k}", _METRIC_TYPE] = metric_type
                    else:
                        self._results.loc[metric_name, emb_key] = metric_value
                        self._results.loc[metric_name, _METRIC_TYPE] = metric_type
                    prog.advance_metric()
                prog.finish_embedding()

        self._benchmarked = True
        print_status("[bold green]✅ Benchmark complete.[/]", disable=not self._progress_bar)

    def _iter_enabled_metrics(self):
        """Yield ``(metric_type, metric_name, config)`` for enabled metrics."""
        for metric_type, metric_collection in self._metric_collection_dict.items():
            for metric_name, use_metric_or_kwargs in asdict(metric_collection).items():
                if use_metric_or_kwargs:
                    yield metric_type, metric_name, use_metric_or_kwargs

    def get_results(
        self,
        min_max_scale: bool = False,
        clean_names: bool = True,
        display_template: BenchmarkTemplate | BenchmarkMode | None = None,
    ) -> pd.DataFrame:
        """Return the benchmarking results.

        Parameters
        ----------
        min_max_scale
            Whether to min max scale the results.
        clean_names
            Whether to clean the metric names.
        display_template
            Override the instance display template. See :class:`BenchmarkTemplate`.

        Returns
        -------
        The benchmarking results.
        """
        template = self.resolve_display_template(display_template)
        display_groups = self._metric_display_groups(template)

        raw = self._results.drop(columns=[_METRIC_TYPE], errors="ignore")
        oriented = orient_metrics_higher_is_better(raw, template.lower_is_better)

        display = raw.copy()
        if min_max_scale:
            scaled = pd.DataFrame(
                MinMaxScaler().fit_transform(display),
                index=display.index,
                columns=display.columns,
            )
            for metric in display.index:
                if metric in template.lower_is_better:
                    scaled.loc[metric] = 1.0 - scaled.loc[metric]
            display = scaled

        metric_group_labels = display_groups.loc[display.index].values
        display[_METRIC_TYPE] = metric_group_labels

        per_class_score = oriented.groupby(metric_group_labels).mean().transpose()
        metric_types = list(per_class_score.columns)
        if len(metric_types) > 1:
            aggregate_weights = self._resolve_aggregate_weights(metric_types, template)
            per_class_score["Total"] = sum(
                aggregate_weights[metric_type] * per_class_score[metric_type] for metric_type in metric_types
            )

        if clean_names:
            display = display.rename(index=metric_name_cleaner)
        df_out = pd.concat([display.transpose(), per_class_score], axis=1)
        df_out.loc[_METRIC_TYPE, per_class_score.columns] = _AGGREGATE_SCORE
        return df_out

    def plot_results_table(
        self,
        min_max_scale: bool = False,
        show: bool = True,
        save_dir: str | None = None,
        circle_cmap: str | mpl.colors.Colormap = "PRGn",
        score_cmap: str | mpl.colors.Colormap = "YlGnBu",
        circle_num_stds: float = 2.5,
        text_fontsize: int = 10,
        figsize: tuple[float, float] | None = None,
        display_template: BenchmarkTemplate | BenchmarkMode | None = None,
    ) -> Table:
        """Plot the benchmarking results.

        Parameters
        ----------
        min_max_scale
            Whether to min max scale the results.
        show
            Whether to show the plot.
        save_dir
            The directory to save the plot to. If `None`, the plot is not saved.
        circle_cmap
            Colormap used for per-metric circle cells.
        score_cmap
            Colormap used for aggregate score bars.
        circle_num_stds
            Number of standard deviations used to normalize circle colormap values.
        text_fontsize
            Base font size used in the table.
        figsize
            Optional explicit figure size. If `None`, a size is inferred from number of columns and embeddings.
        display_template
            Override the instance display template. See :class:`BenchmarkTemplate`.
        """
        num_embeds = len(self._embedding_obsm_keys)
        circle_cmap_obj = mpl.colormaps.get_cmap(circle_cmap)
        score_cmap_obj = mpl.colormaps.get_cmap(score_cmap)
        cmap_fn = lambda col_data: normed_cmap(col_data, cmap=circle_cmap_obj, num_stds=circle_num_stds)
        template = self.resolve_display_template(display_template)
        df = self.get_results(
            min_max_scale=min_max_scale,
            clean_names=True,
            display_template=display_template,
        )
        plot_df = df.drop(_METRIC_TYPE, axis=0, errors="ignore")
        score_cols = df.columns[df.loc[_METRIC_TYPE] == _AGGREGATE_SCORE]
        other_cols = [col for col in df.columns if col not in score_cols and col != _METRIC_TYPE]

        if "Total" in plot_df.columns:
            sort_col = "Total"
        else:
            sort_col = score_cols[0]
        plot_df = plot_df.sort_values(by=sort_col, ascending=False).astype(np.float64)
        plot_df["Method"] = plot_df.index

        # Circle colors: when not min-max scaled, use oriented scores so low PAS/CHAOS reads as "good"
        cmap_df = plot_df
        if not min_max_scale and template.lower_is_better:
            oriented = orient_metrics_higher_is_better(
                self._results.drop(columns=[_METRIC_TYPE], errors="ignore"),
                template.lower_is_better,
            )
            oriented = oriented.rename(index=metric_name_cleaner).transpose()
            cmap_df = plot_df.copy()
            for col in other_cols:
                if col in oriented.columns:
                    cmap_df[col] = oriented[col]

        column_definitions = [
            ColumnDefinition("Method", width=1.5, textprops={"ha": "left", "weight": "bold"}),
        ]
        # Circles for per-metric values (scIB-metrics: group = display template group)
        column_definitions += [
            ColumnDefinition(
                col,
                title=col.replace(" ", "\n", 1),
                width=1,
                textprops={
                    "ha": "center",
                    "bbox": {"boxstyle": "circle", "pad": 0.25},
                },
                cmap=cmap_fn(cmap_df[col]),
                group=df.loc[_METRIC_TYPE, col],
                formatter="{:.2f}",
            )
            for col in other_cols
        ]
        # Bars for aggregate scores (scIB-metrics: single "Aggregate score" header)
        column_definitions += [
            ColumnDefinition(
                col,
                width=1,
                title=col.replace(" ", "\n", 1),
                plot_fn=bar,
                plot_kw={
                    "cmap": score_cmap_obj,
                    "plot_bg_bar": False,
                    "annotate": True,
                    "height": 0.9,
                    "formatter": "{:.2f}",
                },
                group=df.loc[_METRIC_TYPE, col],
                border="left" if i == 0 else None,
            )
            for i, col in enumerate(score_cols)
        ]
        # Allow to manipulate text post-hoc (in illustrator)
        with mpl.rc_context({"svg.fonttype": "none"}):
            if figsize is None:
                figsize = (len(df.columns) * 1.25, 3 + 0.3 * num_embeds)
            fig, ax = plt.subplots(figsize=figsize)
            tab = Table(
                plot_df,
                cell_kw={
                    "linewidth": 0,
                    "edgecolor": "k",
                },
                column_definitions=column_definitions,
                ax=ax,
                row_dividers=True,
                footer_divider=True,
                textprops={"fontsize": text_fontsize, "ha": "center"},
                row_divider_kw={"linewidth": 1, "linestyle": (0, (1, 5))},
                col_label_divider_kw={"linewidth": 1, "linestyle": "-"},
                column_border_kw={"linewidth": 1, "linestyle": "-"},
                index_col="Method",
            ).autoset_fontcolors(colnames=plot_df.columns)
        if show:
            plt.show()
        if save_dir is not None:
            fig.savefig(os.path.join(save_dir, "scib_results.svg"), facecolor=ax.get_facecolor(), dpi=300)

        return tab
