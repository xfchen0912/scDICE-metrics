"""Metadata for spatial-domain benchmark metrics (SACCELERATOR-style requirements)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SpatialMetricGroup = Literal["accuracy", "continuity", "unsupervised", "consensus", "purity"]


@dataclass(frozen=True)
class SpatialMetricSpec:
    """Declares data requirements and score direction for a spatial metric."""

    requires_ground_truth: bool
    requires_embedding: bool
    requires_spatial_coords: bool
    higher_is_better: bool
    group: SpatialMetricGroup


SPATIAL_METRIC_SPECS: dict[str, SpatialMetricSpec] = {
    "hom": SpatialMetricSpec(True, False, False, True, "accuracy"),
    "com": SpatialMetricSpec(True, False, False, True, "accuracy"),
    "spatial_ari": SpatialMetricSpec(True, False, False, True, "accuracy"),
    "spatial_nmi": SpatialMetricSpec(True, False, False, True, "accuracy"),
    "gt_mixture_entropy": SpatialMetricSpec(True, False, False, False, "purity"),
    "domain_specific_f1": SpatialMetricSpec(True, False, False, True, "accuracy"),
    "matched_mcc": SpatialMetricSpec(True, False, False, True, "accuracy"),
    "matched_jaccard": SpatialMetricSpec(True, False, False, True, "accuracy"),
    "spatial_external_ari": SpatialMetricSpec(True, False, True, True, "accuracy"),
    "chaos": SpatialMetricSpec(False, False, True, False, "continuity"),
    "pas": SpatialMetricSpec(False, False, True, False, "continuity"),
    "calinski_harabasz": SpatialMetricSpec(False, True, False, True, "unsupervised"),
    "davies_bouldin": SpatialMetricSpec(False, True, False, False, "unsupervised"),
    "cluster_specific_silhouette": SpatialMetricSpec(False, True, False, True, "unsupervised"),
}
