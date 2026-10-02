# scDICE-metrics

[![Stars][badge-stars]][link-stars]
[![PyPI][badge-pypi]][link-pypi]
[![Build][badge-build]][link-build]
[![Coverage][badge-cov]][link-cov]

[badge-stars]: https://img.shields.io/github/stars/xfchen0912/scDICE-metrics?logo=GitHub&color=yellow
[link-stars]: https://github.com/xfchen0912/scDICE-metrics/stargazers
[badge-pypi]: https://img.shields.io/pypi/v/scdice-metrics.svg
[link-pypi]: https://pypi.org/project/scdice-metrics
[badge-build]: https://github.com/xfchen0912/scDICE-metrics/actions/workflows/build.yaml/badge.svg
[link-build]: https://github.com/xfchen0912/scDICE-metrics/actions/workflows/build.yaml/
[badge-cov]: https://codecov.io/gh/xfchen0912/scDICE-metrics/branch/main/graph/badge.svg
[link-cov]: https://codecov.io/gh/xfchen0912/scDICE-metrics

scDICE-metrics extends the [scib-metrics](https://github.com/YosefLab/scib-metrics) framework to address the next generation of challenges in single-cell modeling: **disentanglement and causality**. While traditional benchmarks focus on batch integration, scDICE-metrics evaluates how well models separate biological factors of variation (e.g., cell type, perturbation, spatial niche) from technical noise, and how accurately they can predict cellular responses to unseen conditions (counterfactuals).

This library provides a standardized suite of metrics derived from information theory, causal inference, and biological ground truths, enabling rigorous comparison of methods like scVI, biolord, CausCell, and scDisInFact.

Implementations use [JAX](https://jax.readthedocs.io/en/latest/) where possible for jit-compilation and hardware acceleration. Optional GPU kNN backends include [RAPIDS cuML](https://docs.rapids.ai/) (`scdice_metrics.nearest_neighbors.rapids`). All public APIs import from `scdice_metrics`.

## Key Features & Metrics

Metric families can be run individually via the direct API or composed in benchmark pipelines. There are **two main harnesses**:

| Harness | Use case | Entry point |
|---------|----------|---------------|
| **`Benchmarker`** | Batch integration, disentanglement, **spatial domain** clustering on embeddings | `SpatialClustering`, `display_template="sdmbench"` / `"saccelerator"` |
| **`CounterfactualBenchmarker`** | **OOD perturbation** prediction (cell-type-specific condition / niche swap) | `Counterfactual`, `CounterfactualTask`, optional `Counterfactual.cellsimbench()` |

See [Metric references](#metric-references) for primary citations.

### Disentanglement metrics

- **MIG (Mutual Information Gap)**: overall score plus mean matched/complement MI summaries ([Chen et al., 2018][ref-mig])
- **Mixed-KSG MIG**: `max_mig`, `concat_mig`, and `min_mig` variants using a mixed continuous–discrete KSG estimator ([Kraskov et al., 2004][ref-ksg])
- **Mixed-KSG MI**: pairwise MI via `mixed_ksg_mi`
- **Classifier Attribute Gap**: compares factor predictability from matched latent blocks against competitor and complement blocks
- **Fairness Leakage**: leakage accuracy together with demographic parity and equalized odds statistics on complement latents ([Bird et al., 2020][ref-fairlearn])
- **Helpers**: `encode_factors`, `assign_latent_blocks_by_mi` (MI-based latent-to-factor block assignment used by MIG / Mixed-KSG MIG)

### Integration benchmark metrics

Based on the scIB atlas-integration benchmark ([Luecken et al., 2022][ref-scib]):

- **Bio conservation** (defaults in `Benchmarker`): `isolated_labels`, `nmi_ari_cluster_labels_kmeans`, `silhouette_label`, `clisi_knn`; optional `nmi_ari_cluster_labels_leiden`
- **Batch correction** (defaults): `bras` ([Rautenstrauch & Ohler, 2025][ref-bras]), `ilisi_knn`, `kbet_per_label` ([Büttner et al., 2018][ref-kbet]), `graph_connectivity`, `pcr_comparison`
- **Standalone**: `silhouette_batch`, `lisi_knn`, `kbet`

### Spatial clustering metrics

Spatial-domain metrics follow the [SDMBench][ref-sdmbench-repo] protocol ([Yuan et al., 2024][ref-sdmbench]) and can be extended with optional [SACCELERATOR](https://github.com/saezlab/SACCELERATOR)-style scores (pure Python; no Snakemake/conda plugins).

**SDMBench core** (via `SpatialClustering` on `Benchmarker`):

- **Accuracy** (needs `spatial_label_key`): `hom`, `com` ([Rosenberg & Hirschberg, 2007][ref-vmeasure])
- **Continuity** (lower is better): `chaos`, `pas`; `pas_from_neighbors` reuses a precomputed spatial kNN graph
- **Auto-clustering** when predicted labels are missing: `SpatialClusteringPrepare` / `prepare_spatial_clusters()`, plus `spatial_cluster_labels_leiden`, `spatial_cluster_labels_kmeans`, `spatial_cluster_labels_from_spatial_coords`

`SpatialClustering` defaults to `chaos=True` and `pas=True`. Set `hom=True` / `com=True` when ground-truth spatial domains are in `spatial_label_key`.

**SACCELERATOR extensions** (default off; factory `SpatialClustering.saccelerator()`):

- **Supervised**: `spatial_ari`, `spatial_nmi`, `gt_mixture_entropy` (lower is better), `domain_specific_f1` (dict → `domain_specific_f1_<domain>` + `domain_specific_f1_macro`), `matched_mcc`, `matched_jaccard`
- **Unsupervised on embedding** (`ad.X` in the benchmark sub-AnnData): `calinski_harabasz`, `davies_bouldin`, `cluster_specific_silhouette` (per-cluster dict + `macro`)
- **SpatialARI (external)**: `spatial_external_ari` — placeholder returning NaN until an optional backend is wired
- **Multi-method consensus** (not part of the default `benchmark()` loop): `cross_method_ari`, `smoothness_entropy`, `cross_method_entropy`, `summarize_consensus` in `scdice_metrics.benchmark` (wide label table + coordinates)

Use `display_template="saccelerator"` for grouped **Accuracy** / **Continuity** / **Purity** (`gt_mixture_entropy`). Spatial-domain ARI/NMI use `spatial_ari` / `spatial_nmi` against `spatial_label_key`, not bio-conservation `nmi_ari_cluster_labels_*` (those compare clusters to **cell type**).

`SpatialClusteringPrepare` controls auto-clustering:

| Option | Default | Description |
|--------|---------|-------------|
| `cluster_method` | `"leiden"` | `"leiden"` or `"kmeans"` |
| `cluster_graph` | `"embedding"` | kNN graph on embedding or `"spatial"` coordinates (Leiden only) |
| `optimize_resolution` | `True` | Search Leiden resolution to match `target_n_clusters` |
| `target_n_clusters` | inferred from `spatial_label_key` | Desired cluster count |
| `cluster_order` | `"ground_truth"` | Relabel clusters by overlap Hungarian matching to ground truth, by cell count, custom order, or `"none"` |
| `build_spatial_neighbors` | `True` | Precompute spatial kNN for Leiden-on-spatial and PAS |

### Display templates

`Benchmarker.get_results()` and `plot_results_table()` support configurable grouping and aggregate weights via `display_template`. Pass `style="funkyheatmap"` to `plot_results_table()` for [funkyheatmappy](https://github.com/funkyheatmap/funkyheatmappy) rendering (`pip install scdice-metrics[funkyheatmap]` or `pip install funkyheatmappy`); default is `style="plottable"`.

| Mode | Description |
|------|-------------|
| `"auto"` (default) | Infer groups from enabled metric collections |
| `"scib"` | Bio conservation + Batch correction (0.6 / 0.4) |
| `"sdmbench"` | Spatial Accuracy (`hom`, `com`) + Continuity (`chaos`, `pas`) |
| `"saccelerator"` | Spatial Accuracy (incl. `spatial_ari`, `domain_specific_f1_macro`) + Continuity + Purity (`gt_mixture_entropy`) |
| `"full"` | Combined integration + spatial + disentanglement weights |
| `"legacy"` | One display group per enabled collection |
| `BenchmarkTemplate(...)` | Custom groups, weights, and lower-is-better metrics |

Factory helpers: `BenchmarkTemplate.scib()`, `.sdmbench()`, `.saccelerator()`, `.full()`, `.legacy_from_collections()`, and `infer_template()`. Metric collection presets: `SpatialClustering.saccelerator()`, `Counterfactual.cellsimbench()`.

### Perturbation response metrics *(direct API only)*

Systema-style evaluation utilities are available as standalone functions ([Vinas Torne et al., 2025][ref-systema]; not yet integrated into `Benchmarker`):

- `pearson_delta_reference_metrics` — PearsonΔ on delta profiles relative to a reference
- `average_of_perturbation_centroids` — reference profile for delta-based evaluation
- `calculate_centroid_accuracies`, `score_centroids`, `get_perts`

### Counterfactual OOD metrics *(direct API or `CounterfactualBenchmarker`)*

Cell-type-specific swap benchmarks evaluate how well models recover held-out condition or niche responses. Metrics operate on cell matrices or pre-aggregated gene profiles; pseudobulk aggregation happens inside each metric function.

- **Expression**: `pseudobulk_pearson`, `pseudobulk_spearman`, `pseudobulk_rmse`, `pseudobulk_mae`
- **Effect fidelity**: `systema_pearson_delta_metrics` (reuses Systema PearsonΔ), `delta_pearson`, `delta_spearman`, `delta_cosine`, `delta_rmse`, `delta_mae`, `delta_profile_metrics`
- **DE recovery**: `signed_de_recovery`
- **Distribution**: `energy_distance`, `mmd_rbf`, `mean_gene_wasserstein`

### CellSimBench-style perturbation metrics *(optional on `CounterfactualBenchmarker`)*

Aligned with perturbation-model calibration benchmarks (dual baselines, DEG-weighted scores, DRF). Enable via `Counterfactual.cellsimbench()` or per-metric flags on `Counterfactual` (defaults unchanged).

- **Task fields**: optional `dataset_mean` (fold-specific mean profile for `*deltapert*`), `deg` (`DegProfile` with GT-half weights and top-100-by-p mask), `metadata["covariate"]` for panel metrics
- **Baselines as methods**: put `dataset_mean`, `control`, `technical_duplicate`, `interpolated_duplicate` in `CounterfactualTask.predicted` for DRF
- **Dual delta**: `pearson_deltactrl` / `pearson_deltapert`, `r2_deltactrl` / `r2_deltapert`, `weighted_r2_*`, DEG top-100 variants (`*_degs`) — distinct from Systema `signed_de_recovery` top-k definitions
- **DEG-weighted expression**: `wmse`, `wmae`, `mse_degs`, `mae_degs`
- **Panel metrics** (second benchmark pass, per method): `nir`, `pds`, `knn_jaccard_deltapert` (disabled metrics omit rows; no `0.0` placeholders)
- **Calibration** (post-hoc on long DataFrame): `dynamic_range_fraction`, `score_relative_to_baselines`, `drf_all_versions`

Systema top-k true-effect genes and signed-DE top-k sets are **evaluation-only** oracle subsets derived from held-out observed-vs-source effects. Do not use them for training or model selection.

### Planned *(not yet implemented)*

- **Interventional NLL**: held-out perturbation likelihood for OOD generalization (sVAE+)
- **Latent traversal consistency** and **intrinsic vs. extrinsic separation** for spatial omics
- **Swap result plotting helper** (deferred to v0.2.1; v0.2.0 exposes tidy result tables)

## Package layout

```
scdice_metrics/
├── benchmark/          # Benchmarker, CounterfactualBenchmarker, swap tasks, consensus helpers
├── metrics/            # Metric functions (integration, spatial, counterfactual, CellSimBench, …)
├── nearest_neighbors/  # pynndescent, jax_approx_min_k, rapids
├── utils/              # PCA, k-means, LISI helpers, etc.
└── _settings.py        # ScibConfig (verbosity, progress bar style)
```

## Motivation

Understanding cellular identity requires more than just clustering; it requires decomposing the "factors of variation" that drive cell states. scDICE-metrics aims to be the standard harness for validating "Virtual Cell" and "Virtual Tissue" models.

## Installation

You need Python 3.10 or newer.

1. Install the latest release from PyPI:

```bash
pip install scdice-metrics
```

2. Install from source (editable, with test extras):

```bash
git clone https://github.com/xfchen0912/scDICE-metrics.git
cd scDICE-metrics
pip install -e ".[test]"
```

3. Install the latest development version without cloning:

```bash
pip install git+https://github.com/xfchen0912/scDICE-metrics.git@main
```

To leverage hardware acceleration (e.g., GPU), install the appropriate version of [JAX](https://github.com/google/jax#installation) separately. Conda-distributed JAX builds are often the easiest route. For RAPIDS-backed kNN, install cuML and call `scdice_metrics.nearest_neighbors.rapids`.

## Quick Start

### Direct metric API — disentanglement

```python
import scdice_metrics as sm

factors = adata.obs[["cell_type", "condition"]]

mig_scores = sm.mig(adata.obsm["X_scDICE"], factors)
ksg_scores = sm.mixed_ksg_mig(adata.obsm["X_scDICE"], factors)
gap_scores = sm.classifier_attribute_gap(adata.obsm["X_scDICE"], factors)
leakage_scores = sm.fairness_leakage(adata.obsm["X_scDICE"], factors, adata.obs["response"])

# Optional helpers
encoded = sm.encode_factors(factors)
blocks = sm.assign_latent_blocks_by_mi(adata.obsm["X_scDICE"], factors)
```

`fairness_leakage` expects a binary target or a numeric target that can be binarized by a median split.

### Direct metric API — spatial clustering

```python
import scdice_metrics as sm
from scdice_metrics.nearest_neighbors import pynndescent

# Optional: cluster an embedding to obtain predicted domain labels
neighbors = pynndescent(adata.obsm["X_scDICE"], n_neighbors=15)
pred = sm.spatial_cluster_labels_leiden(
    neighbors,
    target_n_clusters=7,
    optimize_resolution=True,
)

hom = sm.hom(adata.obs["domain_truth"], pred)
com = sm.com(adata.obs["domain_truth"], pred)
chaos = sm.chaos(pred, adata.obsm["spatial"])
pas = sm.pas(pred, adata.obsm["spatial"], k=10)

# Optional SACCELERATOR-style supervised scores (spatial GT vs predicted domains)
sm.spatial_ari(adata.obs["domain_truth"], pred)
sm.gt_mixture_entropy(adata.obs["domain_truth"], pred)  # lower is better
sm.domain_specific_f1(adata.obs["domain_truth"], pred)  # dict with "macro" and per-domain keys
```

### Direct metric API — perturbation response

```python
import scdice_metrics as sm

reference = sm.average_of_perturbation_centroids(
    adata, control_key="control", condition_key="condition"
)
scores = sm.pearson_delta_reference_metrics(X_true, X_pred, reference)
```

### Benchmarker API — integration + disentanglement

```python
from scdice_metrics.benchmark import Benchmarker, Disentanglement

bm = Benchmarker(
    adata=adata,
    batch_key="batch",
    label_key="cell_type",
    embedding_obsm_keys=["X_scDICE", "X_baseline"],
    disentanglement_metrics=Disentanglement(
        mig=True,
        mixed_ksg_mig=True,
        classifier_attribute_gap=True,
        fairness_leakage=True,
    ),
    disentanglement_factor_keys=["cell_type", "condition"],
    leakage_target_key="response",
)
bm.benchmark()
results = bm.get_results(display_template="auto")
```

### Benchmarker API — spatial clustering

```python
from scdice_metrics.benchmark import Benchmarker, SpatialClustering, SpatialClusteringPrepare

bm = Benchmarker(
    adata=adata,
    batch_key="batch",
    label_key="cell_type",
    spatial_label_key="domain_truth",  # ground truth for HOM/COM
    spatial_obsm_key="spatial",
    embedding_obsm_keys=["X_scDICE"],
    bio_conservation_metrics=None,
    batch_correction_metrics=None,
    spatial_clustering_metrics=SpatialClustering(hom=True, com=True, chaos=True, pas=True),
    spatial_clustering_prepare=SpatialClusteringPrepare(
        cluster_method="leiden",
        cluster_graph="embedding",
        optimize_resolution=True,
        cluster_order="ground_truth",
    ),
    # spatial_cluster_key=None  → auto-generate per-embedding labels via prepare()
)
bm.benchmark()
results = bm.get_results(display_template="sdmbench")
bm.plot_results_table(display_template="sdmbench", min_max_scale=False)
```

SACCELERATOR-style preset (supervised + continuity; requires `spatial_label_key`):

```python
bm = Benchmarker(
    adata=adata,
    batch_key="batch",
    label_key="cell_type",
    spatial_label_key="domain_truth",
    spatial_obsm_key="spatial",
    embedding_obsm_keys=["X_scDICE"],
    bio_conservation_metrics=None,
    batch_correction_metrics=None,
    spatial_clustering_metrics=SpatialClustering.saccelerator(),
)
bm.benchmark()
results = bm.get_results(display_template="saccelerator")
```

Multi-method label consensus (separate from `Benchmarker.benchmark()`):

```python
import pandas as pd
from scdice_metrics.benchmark import cross_method_ari, summarize_consensus

labels_wide = pd.DataFrame({"MethodA": ..., "MethodB": ...}, index=adata.obs_names)
summarize_consensus(labels_wide, spatial=adata.obsm["spatial"], k_neighbors=6)
```

`Benchmarker` organizes metrics into **Bio conservation**, **Batch correction**, **Disentanglement**, and **Spatial clustering**. When more than one family is enabled, aggregate scores are computed from the resolved display template. For disentanglement runs, `disentanglement_factor_keys` are read from `adata.obs`, and `leakage_target_key` is required only when `fairness_leakage=True`.

`bm.prepare()` runs automatically inside `benchmark()`; call it explicitly to inspect auto-generated spatial cluster columns, to run `prepare_spatial_clusters(force=True)`, or to pass a custom `neighbor_computer` (e.g. `rapids` or `jax_approx_min_k`). Set `compute_neighbors=False` when integration kNN graphs are not needed. Use `solver="randomized"` if PCA / neighbor steps fail with ARPACK.

Per-embedding predicted cluster columns are stored in `adata.obs` and exposed via `bm.spatial_cluster_obs_keys`.

### Cell-type-specific OOD swap benchmarks

Generate swap tasks from an AnnData object with `cell_type`, `condition`, and `niche` columns in `obs`:

```python
from scdice_metrics.benchmark import (
    Counterfactual,
    CounterfactualBenchmarker,
    CounterfactualTask,
    make_swap_specs,
    summarize_swap_specs,
)

# CellDISECT-compatible marginal condition transfer
condition_specs = make_swap_specs(
    adata,
    swap_type="condition",
    source_value="REF",
    target_value="CRC",
    cell_types=["Fibroblast", "Myeloid", "T_cell"],
    match_other_factor=False,
    min_source_cells=50,
    min_target_cells=50,
)

# Cellina-compatible niche transfer (matched condition required in v0.2.0)
niche_specs = make_swap_specs(
    adata,
    swap_type="niche",
    source_value="non_malignant",
    target_value="malignant",
    cell_types=["Fibroblast", "Myeloid", "T_cell"],
    match_other_factor=True,
    fixed_values=["CRC"],
)

summarize_swap_specs(condition_specs)
```

Each `SwapSpec` exposes `source_mask`, `target_mask`, `train_mask`, and `context_mask`. Target context cells from other cell types remain in training (transductive OOD). Train one model per task; `scDICE-metrics` does not train models for you.

Build tasks after running external models, then benchmark profile metrics:

```python
profile_tasks = []
for spec in condition_specs:
    predictions = {"MyMethod": run_model(spec)}  # user-side
    profile_tasks.append(
        CounterfactualTask(
            task_id=spec.task_id,
            observed=adata[spec.target_mask].X,
            predicted=predictions,
            reference=adata[spec.source_mask].X,
            gene_names=adata.var_names,
            metadata={**spec.metadata, "evaluation_space": "log1p_hvg"},
        )
    )

bm = CounterfactualBenchmarker(
    tasks=profile_tasks,
    counterfactual_metrics=Counterfactual(
        pseudobulk_pearson=True,
        systema_pearson_delta={"top_k": 20},
        delta_rmse=True,
        signed_de_recovery={"top_k": 50},
        energy_distance=False,
        mmd_rbf=False,
    ),
)
bm.benchmark()

raw = bm.get_results(long_format=True)
summary = bm.get_swap_summary()  # partitions by swap_type and match_other_factor
by_cell_type = bm.get_swap_summary(by_cell_type=True)
```

CellSimBench-style preset (pseudobulk / dual-delta; extend tasks with `dataset_mean` and `DegProfile`):

```python
from scdice_metrics.benchmark import DegProfile
from scdice_metrics.metrics import (
    deg_mask_from_pvals,
    deg_weights_from_scores,
    drf_all_versions,
)

deg = DegProfile(
    weights=deg_weights_from_scores(gene_names, scores, score_genes),
    top_mask=deg_mask_from_pvals(gene_names, pvals, pval_genes, topn=100),
    source="gt_half",
)

profile_tasks.append(
    CounterfactualTask(
        task_id=spec.task_id,
        observed=...,
        predicted={"MyMethod": ..., "dataset_mean": ..., "control": ..., "technical_duplicate": ...},
        reference=control_profile,
        dataset_mean=fold_mean_profile,
        deg=deg,
        metadata={**spec.metadata, "covariate": "donor1"},
    )
)

bm = CounterfactualBenchmarker(tasks=profile_tasks, counterfactual_metrics=Counterfactual.cellsimbench())
bm.benchmark()
drf = drf_all_versions(bm.get_results(long_format=True))
```

**Notes**

- Provide a common normalization/representation for every method before constructing tasks.
- Profile metrics (pseudobulk, delta, signed DE, Wasserstein) and distribution metrics (energy distance, MMD) may require **separate benchmark passes** when their feature spaces differ (e.g. log1p/HVG vs PCA).
- `fixed_values` creates a cell-type × fixed-value Cartesian product when `match_other_factor=True`.
- `get_swap_summary()` never pools condition-swap with niche-swap tasks, or marginal with matched condition tasks.

Direct metric API example:

```python
import scdice_metrics as sm

sm.pseudobulk_pearson(observed, predicted)
sm.systema_pearson_delta_metrics(observed, predicted, reference, top_k=20)
sm.signed_de_recovery(observed, predicted, reference, top_k=50)
sm.energy_distance(observed, predicted)
```

## Development

From the package root:

```bash
pip install -e ".[test,dev]"
pytest tests/ -v
```

In the SpaPert monorepo, the factory pixi environment can run the same suite:

```bash
cd SpaPert-model_factory
pixi run pytest ../scDICE-metrics/tests/benchmark/test_spatial_saccelerator_benchmarker.py -v
pixi run pytest ../scDICE-metrics/tests/counterfactual/ ../scDICE-metrics/tests/benchmark/test_cellsimbench_protocol.py -v
```

See [tests/README.md](tests/README.md) for layout. Configure logging and progress bars via `scdice_metrics.settings` (`ScibConfig`).

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, reach out on [scverse Discourse][link-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Metric references

| Metric / module | Primary reference |
|-----------------|-------------------|
| scIB integration benchmark (`Benchmarker`, bio/batch metrics) | [Luecken et al., *Nat. Methods* 2022][ref-scib] |
| kBET (`kbet`, `kbet_per_label`) | [Büttner et al., *Nat. Methods* 2019][ref-kbet] |
| LISI (`ilisi_knn`, `clisi_knn`) | [Korsunsky et al., *Nat. Methods* 2019][ref-harmony] |
| BRAS (`bras`) | [Rautenstrauch & Ohler, *bioRxiv* 2025][ref-bras] |
| HOM / COM | [Rosenberg & Hirschberg, *EMNLP* 2007][ref-vmeasure]; used in spatial benchmarks by [Yuan et al., 2024][ref-sdmbench] |
| CHAOS / PAS | [Yuan et al., *Nat. Methods* 2024][ref-sdmbench] ([SDMBench code][ref-sdmbench-repo]) |
| MIG | [Chen et al., ICML 2018][ref-mig] |
| Mixed-KSG MI / MIG | [Kraskov et al., *Phys. Rev. E* 2004][ref-ksg] |
| Fairness leakage | [Bird et al., 2020][ref-fairlearn] via [Fairlearn](https://fairlearn.org/) |
| Perturbation metrics | [Vinas Torne et al., *Nat. Biotechnol.* 2025][ref-systema] |
| SACCELERATOR spatial extensions | [SACCELERATOR][ref-saccelerator] (Nature Methods 2026); HOM/COM/CHAOS/PAS overlap SDMBench |
| CellSimBench-style calibration | Perturbation-model benchmark literature (dual baseline, DRF); metric names match common CellSimBench tables |

[ref-saccelerator]: https://github.com/saezlab/SACCELERATOR

## Citation

If you use scDICE-metrics in your work, please cite this repository and the underlying benchmark papers relevant to your evaluation. scDICE-metrics builds on the single-cell integration benchmarking framework:

```
@article{luecken2022benchmarking,
  title={Benchmarking atlas-level data integration in single-cell genomics},
  author={Luecken, Malte D and B{\"u}ttner, Maren and Chaichoompu, Kridsadakorn and Danese, Anna and Interlandi, Marta and M{\"u}ller, Michaela F and Strobl, Daniel C and Zappia, Luke and Dugas, Martin and Colom{\'e}-Tatch{\'e}, Maria and others},
  journal={Nature Methods},
  volume={19},
  number={1},
  pages={41--50},
  year={2022},
  publisher={Nature Publishing Group}
}
```

When reporting spatial clustering benchmarks, also cite SDMBench:

```
@article{yuan2024sdmbench,
  title={Benchmarking spatial clustering methods with spatially resolved transcriptomics data},
  author={Yuan, Zhao and Zhao, Fang and Lin, Siyuan and others},
  journal={Nature Methods},
  volume={21},
  pages={712--722},
  year={2024},
  doi={10.1038/s41592-024-02215-8}
}
```

[ref-scib]: https://doi.org/10.1038/s41592-021-01346-4
[ref-kbet]: https://doi.org/10.1038/s41592-018-0254-1
[ref-harmony]: https://doi.org/10.1038/s41592-019-0619-0
[ref-bras]: https://doi.org/10.1101/2025.01.21.634098
[ref-sdmbench]: https://doi.org/10.1038/s41592-024-02215-8
[ref-sdmbench-repo]: https://github.com/zhaofangyuan98/SDMBench
[ref-vmeasure]: https://doi.org/10.3115/1220355.1220375
[ref-mig]: https://proceedings.mlr.press/v80/chen18i.html
[ref-ksg]: https://doi.org/10.1103/PhysRevE.69.066138
[ref-fairlearn]: https://arxiv.org/abs/2005.04799
[ref-systema]: https://doi.org/10.1038/s41587-025-02777-8
[changelog]: CHANGELOG.md
[issue-tracker]: https://github.com/xfchen0912/scDICE-metrics/issues
[link-discourse]: https://discourse.scverse.org/
