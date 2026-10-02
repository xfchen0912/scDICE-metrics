# Change: Add CellSimBench perturbation metrics and calibration protocol

## Why

Counterfactual benchmarking in scDICE-metrics covers swap tasks and Systema-style deltas, but lacks CellSimBench-style dual-baseline evaluation (control vs dataset mean), DEG-weighted scores, panel metrics (NIR/PDS/KNN-Jaccard), and dynamic range fraction (DRF) calibration against uninformative baselines.

## What Changes

- Add optional `CounterfactualTask` fields: `dataset_mean`, `deg` (GT-half DEG weights/masks).
- Add `metrics/_cellsimbench.py` with profile and panel metric helpers aligned to CellSimBench naming.
- Add `metrics/_calibration.py` with DRF and baseline-relative scoring on long-format benchmark results.
- Extend `Counterfactual` dataclass with CellSimBench metrics (default off) and `Counterfactual.cellsimbench()` factory.
- Extend `CounterfactualBenchmarker` with a second pass for panel metrics.

## Impact

- Affected specs: counterfactual-benchmark (new capability delta)
- Affected code: `benchmark/_counterfactual.py`, `metrics/__init__.py`, tests, README, docs/api.md
- Non-breaking: default `Counterfactual()` behavior unchanged
