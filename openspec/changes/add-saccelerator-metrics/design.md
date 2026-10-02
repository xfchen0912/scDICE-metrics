# Design: SACCELERATOR integration

## Mount point

All single-sample, single-method metrics run through existing `Benchmarker` + `SpatialClusteringPrepare`. Inputs are per-embedding sub-`AnnData` with `_SPATIAL_LABELS`, `_SPATIAL_CLUSTER`, `_SPATIAL_COORDS`, optional `_SPATIAL_NEIGHBORS`.

Do not register spatial domain ARI/NMI on `BioConservation` (cell type GT).

## Hungarian policy

| Stage | Purpose | Implementation |
|-------|---------|----------------|
| Prepare `cluster_order="ground_truth"` | Relabel predicted clusters for visualization / overlap | Existing confusion-matrix LSAP in `_clustering.py` |
| Supervised metrics MCC/Jaccard/domain-F1 | Score with cluster count ≠ domain count | New `match_predicted_to_ground_truth` on contingency table; domain-F1 uses dummy column padding per SACCELERATOR R |
| HOM/COM/Entropy | No matching | Direct label comparison |

## Metric metadata

Mirror SACCELERATOR `_optargs.json` as Python `SpatialMetricSpec` (requires GT, embedding, spatial coords). Used for docs and future validation, not CLI.

## Templates

- `sdmbench`: keep HOM/COM + CHAOS/PAS as today.
- `saccelerator` (new): add optional accuracy block (spatial ARI, macro domain F1, entropy) with documented lower-is-better for entropy.
- Consensus metrics excluded from template aggregates until explicitly requested.

## SpatialARI

Deferred to optional extra or separate PR. Prefer Python port with golden tests against SACCELERATOR R output; avoid hard R runtime dependency in core wheel.

## Consensus layer

Functions accept `labels_wide: pd.DataFrame` and `spatial: np.ndarray`. Output long or matrix DataFrames for downstream plotting (e.g. funkyheatmap multi-method tables). Not wired into `MetricAnnDataAPI`.
