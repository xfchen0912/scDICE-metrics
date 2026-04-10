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

scDICE-metrics extends the popular scib-metrics framework to address the next generation of challenges in single-cell modeling: **Disentanglement and Causality**. While traditional benchmarks focus on batch integration, scDICE-metrics evaluates how well models can separate biological factors of variation (e.g., cell type, perturbation, spatial niche) from technical noise, and how accurately they can predict cellular responses to unseen conditions (counterfactuals).

This library provides a standardized suite of metrics derived from information theory, causal inference, and biological ground truths, enabling rigorous comparison of methods like scVI, biolord, CausCell, and scDisInFact.

The implementations use [JAX](https://jax.readthedocs.io/en/latest/) when possible for jit-compilation and hardware acceleration. All implementations are in Python.

## Key Features & Metrics

### Disentanglement metrics
- **MIG (Mutual Information Gap)**: Returns the overall score together with mean matched/complement MI summaries
- **Mixed-KSG MIG**: Provides `max_mig`, `concat_mig`, and `min_mig` variants based on the Mixed-KSG mutual information estimator
- **Classifier Attribute Gap**: Compares factor predictability from matched latent blocks against competitor and complement blocks
- **Fairness Leakage**: Reports leakage accuracy together with demographic parity and equalized odds style statistics on complement latents

### Integration benchmark metrics
- **Bio conservation**: `isolated_labels`, `nmi_ari_cluster_labels_kmeans`, `nmi_ari_cluster_labels_leiden`, `silhouette_label`, `clisi_knn`
- **Batch correction**: `bras`, `ilisi_knn`, `kbet_per_label`, `graph_connectivity`, `pcr_comparison`

### Counterfactual & Perturbation Metrics
- **Interventional NLL** *(planned, not yet implemented)*: Evaluates the likelihood of held-out perturbation data to assess out-of-distribution (OOD) generalization (from sVAE+)
- **Reconstruction fidelity** *(planned, not yet implemented)*: MSE and Pearson/Spearman correlations on specific gene sets (e.g., DEGs) between predicted counterfactuals and real held-out conditions
- **Distance Metrics** *(planned, not yet implemented)*: EMD (Earth Mover's Distance) and MMD (Maximum Mean Discrepancy) to compare generated vs. real cell distributions

### Biological Interpretability
- **Latent Traversal & Consistency** *(planned, not yet implemented)*: Quantifies the preservation of biological manifolds (e.g., cell cycle, differentiation trajectories) when specific factors are manipulated
- **Intrinsic vs. Extrinsic Separation** *(planned, not yet implemented)*: Specifically designed for spatial omics to measure the decoupling of cell-intrinsic states from microenvironmental effects (inspired by MintFlow & SIMVI)

## Motivation

Understanding cellular identity requires more than just clustering; it requires decomposing the "factors of variation" that drive cell states. scDICE-metrics aims to be the standard harness for validating "Virtual Cell" and "Virtual Tissue" models.

## Installation

You need to have Python 3.10 or newer installed on your system. If you don't have
Python installed, we recommend installing [Miniconda](https://docs.conda.io/en/latest/miniconda.html).

There are several options to install scDICE-metrics:

1. Install the latest release on PyPI:

```bash
pip install scdice-metrics
```

2. Install the latest development version:

```bash
pip install git+https://github.com/xfchen0912/scDICE-metrics.git@main
```

To leverage hardware acceleration (e.g., GPU) please install the apprpriate version of [JAX](https://github.com/google/jax#installation) separately. Often this can be easier by using conda-distributed versions of JAX.

## Quick Start

### Direct metric API

```python
import scdice_metrics as sm

factors = adata.obs[["cell_type", "condition"]]

mig_scores = sm.mig(adata.obsm["X_scDICE"], factors)
ksg_scores = sm.mixed_ksg_mig(adata.obsm["X_scDICE"], factors)
gap_scores = sm.classifier_attribute_gap(adata.obsm["X_scDICE"], factors)
leakage_scores = sm.fairness_leakage(adata.obsm["X_scDICE"], factors, adata.obs["response"])
```

`fairness_leakage` expects a binary target or a numeric target that can be binarized by a median split.

### Benchmarker API

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
results = bm.get_results()
```

`Benchmarker` now separates metric families into `Bio conservation`, `Batch correction`, and `Disentanglement`. If more than one family is enabled, aggregate scores are added automatically. For disentanglement runs, `disentanglement_factor_keys` are read from `adata.obs`, and `leakage_target_key` is only required when `fairness_leakage=True`.

## Release notes

See the [changelog][changelog].

## Contact

For questions and help requests, you can reach out in the [scverse Discourse][link-discourse].
If you found a bug, please use the [issue tracker][issue-tracker].

## Citation

References for individual metrics can be found in the corresponding documentation. This package is heavily inspired by the single-cell integration benchmarking work:

```
@article{luecken2022benchmarking,
  title={Benchmarking atlas-level data integration in single-cell genomics},
  author={Luecken, Malte D and B{\"u}ttner, Maren and Chaichoompu, Kridsadakorn and Danese, Anna and Interlandi, Marta and M{\"u}ller, Michaela F and Strobl, Daniel C and Zappia, Luke and Dugas, Martin and Colom{\'e}-Tatch{\'e}, Maria and others},
  journal={Nature methods},
  volume={19},
  number={1},
  pages={41--50},
  year={2022},
  publisher={Nature Publishing Group}
}
```

[issue-tracker]: https://github.com/xfchen0912/scDICE-metrics/issues
