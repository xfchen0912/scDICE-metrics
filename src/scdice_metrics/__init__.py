"""scDICE-metrics: Metrics for evaluating disentanglement and causality in single-cell models.

This package extends the scib-metrics framework to address disentanglement and causality
in single-cell modeling. It provides metrics for evaluating how well models can separate
biological factors of variation from technical noise, and how accurately they can predict
cellular responses to unseen conditions (counterfactuals).
"""

import logging
from importlib.metadata import PackageNotFoundError, version

__version__ = "0.1.0"
__author__ = "Xufeng Chen"
__email__ = "chenxufeng2022@sinh.ac.cn"

from . import nearest_neighbors, utils
from .metrics import (
    assign_latent_blocks_by_mi,
    classifier_attribute_gap,
    encode_factors,
    fairness_leakage,
    graph_connectivity,
    isolated_labels,
    kbet,
    kbet_per_label,
    clisi_knn,
    ilisi_knn,
    lisi_knn,
    mig,
    mixed_ksg_mi,
    mixed_ksg_mig,
    nmi_ari_cluster_labels_kmeans,
    nmi_ari_cluster_labels_leiden,
    pcr_comparison,
    silhouette_batch,
    silhouette_label,
    bras,
)
from ._settings import settings

__all__ = [
    "utils",
    "nearest_neighbors",
    "isolated_labels",
    "pcr_comparison",
    "silhouette_label",
    "silhouette_batch",
    "bras",
    "ilisi_knn",
    "clisi_knn",
    "lisi_knn",
    "nmi_ari_cluster_labels_kmeans",
    "nmi_ari_cluster_labels_leiden",
    "kbet",
    "kbet_per_label",
    "graph_connectivity",
    "encode_factors",
    "assign_latent_blocks_by_mi",
    "mixed_ksg_mi",
    "mig",
    "mixed_ksg_mig",
    "classifier_attribute_gap",
    "fairness_leakage",
    "settings",
]

try:
    __version__ = version("scdice-metrics")
except PackageNotFoundError:
    # Allow importing directly from a source checkout without installing the package.
    pass

settings.verbosity = logging.INFO
# Jax sets the root logger, this prevents double output.
logger = logging.getLogger("scdice_metrics")
logger.propagate = False
