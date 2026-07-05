# API

## Benchmarking pipeline

Import as:

```
from scdice_metrics.benchmark import Benchmarker
```

```{eval-rst}
.. module:: scdice_metrics.benchmark
.. currentmodule:: scdice_metrics

.. autosummary::
    :toctree: generated

    benchmark.Benchmarker
    benchmark.BenchmarkTemplate
    benchmark.BioConservation
    benchmark.BatchCorrection
    benchmark.SpatialClustering
    benchmark.SpatialClusteringPrepare
    benchmark.prepare_spatial_clusters
    benchmark.infer_template
    benchmark.SwapSpec
    benchmark.make_swap_specs
    benchmark.summarize_swap_specs
    benchmark.CounterfactualTask
    benchmark.Counterfactual
    benchmark.CounterfactualBenchmarker
```

Spatial clustering helpers (also importable from ``scdice_metrics``):

```{eval-rst}
.. autosummary::
    :toctree: generated

    spatial_cluster_labels_leiden
    spatial_cluster_labels_kmeans
    spatial_cluster_labels_from_spatial_coords
```

## Metrics

Import as:

```
import scdice_metrics as sm
sm.ilisi_knn(...)
```

```{eval-rst}
.. module:: scdice_metrics
.. currentmodule:: scdice_metrics

.. autosummary::
    :toctree: generated

    isolated_labels
    nmi_ari_cluster_labels_kmeans
    nmi_ari_cluster_labels_leiden
    hom
    com
    pcr_comparison
    silhouette_label
    silhouette_batch
    bras
    ilisi_knn
    clisi_knn
    kbet
    kbet_per_label
    graph_connectivity
    chaos
    pas
    pseudobulk_pearson
    pseudobulk_spearman
    pseudobulk_rmse
    pseudobulk_mae
    systema_pearson_delta_metrics
    delta_pearson
    delta_spearman
    delta_cosine
    delta_rmse
    delta_mae
    delta_profile_metrics
    signed_de_recovery
    energy_distance
    mmd_rbf
    mean_gene_wasserstein
    pearson_delta_reference_metrics
```

## Utils

```{eval-rst}
.. module:: scdice_metrics.utils
.. currentmodule:: scdice_metrics

.. autosummary::
    :toctree: generated

    utils.cdist
    utils.pdist_squareform
    utils.silhouette_samples
    utils.KMeans
    utils.pca
    utils.principal_component_regression
    utils.one_hot
    utils.compute_simpson_index
    utils.convert_knn_graph_to_idx
    utils.check_square
    utils.diffusion_nn
```

### Nearest neighbors

```{eval-rst}
.. module:: scdice_metrics.nearest_neighbors
.. currentmodule:: scdice_metrics

.. autosummary::
    :toctree: generated

    nearest_neighbors.pynndescent
    nearest_neighbors.jax_approx_min_k
    nearest_neighbors.rapids
    nearest_neighbors.NeighborsResults
```

## Settings

An instance of the {class}`~scdice_metrics._settings.ScibConfig` is available as `scdice_metrics.settings` and allows configuring scDICE-metrics.

```{eval-rst}
.. autosummary::
   :toctree: reference/
   :nosignatures:

   _settings.ScibConfig
```
