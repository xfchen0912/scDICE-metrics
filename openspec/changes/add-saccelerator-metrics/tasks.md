# Tasks: add-saccelerator-metrics

- [x] Scaffold openspec change (this directory)
- [x] Document Hungarian matching policy and `SpatialMetricSpec`
- [x] PAS/CHAOS numerical alignment tests vs SDMBench (`compute_PAS` / `compute_CHAOS`; golden tests in `tests/metrics/test_spatial_clustering_metrics.py`)
- [x] Implement `_spatial_supervised.py` (entropy, domain F1, matched MCC/Jaccard, spatial ARI/NMI)
- [x] Wire `SpatialClustering` + `MetricAnnDataAPI` + template `saccelerator`
- [x] Add `SpatialClustering.saccelerator()` factory
- [x] P2: `_spatial_unsupervised.py` (CH, DB, cluster-specific silhouette)
- [x] P2: `smoothness_entropy` natural log vs SACCELERATOR `Smoothness_entropy.r`
- [x] P2: README DRF `*_ceiling` / `cellsimbench()` subset / `knn_jaccard_deltapert` not in original DRF table
- [x] P3: SpatialARI stub (`spatial_external_ari` → NaN)
- [x] P3: `benchmark/_consensus.py` wide-table API
- [x] README / api.md / CHANGELOG / pytest subset
