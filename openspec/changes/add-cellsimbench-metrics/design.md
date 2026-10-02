# Design: CellSimBench integration

## Layering

1. **Atomic metrics** — pure functions on pseudobulk profiles; registered in `METRIC_FUNCTIONS`.
2. **Panel metrics** — NIR, PDS, KNN-Jaccard computed after all tasks for a `(method, covariate)` group.
3. **Calibration** — post-hoc on long DataFrame; baselines are method names in `predicted`.

## Delta axes

- `deltactrl`: profile minus `task.reference` (control).
- `deltapert`: profile minus `task.dataset_mean`.

## DEG semantics

- `top_mask`: top-N genes by p-value (default 100), not required to be significant.
- `weights`: min-max(|score|)² per gene (max over duplicate gene entries).
- Distinct from `signed_de_recovery` and Systema top-k oracle subsets.

## DRF null baseline selection

- Metrics matching `*deltapert*`: null baseline method = `control`.
- Other metrics: null baseline method = `dataset_mean`.

Ceiling methods: `technical_duplicate` (drf_mean), `interpolated_duplicate` (drf_ctrl / drf_interpolated variants).
