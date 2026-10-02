# Change: Add SACCELERATOR spatial-domain metrics and optional consensus API

## Why

scDICE-metrics already implements SDMBench-style spatial metrics (HOM, COM, CHAOS, PAS) but lacks several SACCELERATOR supervised scores (domain-specific F1, GT mixture entropy, matched MCC/Jaccard, SpatialARI) and multi-method consensus utilities. Users benchmarking spatial domain methods need parity with SACCELERATOR without adopting its Snakemake/conda plugin workflow.

## What Changes

- Add pure-Python spatial supervised metrics with documented Hungarian matching for scoring (distinct from cluster relabeling in prepare).
- Extend `SpatialClustering` and `MetricAnnDataAPI` with optional SACCELERATOR metrics (default off) and `SpatialClustering.saccelerator()` factory.
- Add optional `BenchmarkTemplate` mode / extended `sdmbench` groups for accuracy + continuity + purity.
- Optional follow-up: consensus metrics on wide label tables (not default Benchmarker path).
- Numerical regression tests aligning PAS/CHAOS with SACCELERATOR where feasible.

## Impact

- Affected specs: spatial-clustering-benchmark (new delta)
- Affected code: `metrics/`, `benchmark/_core.py`, `benchmark/_templates.py`, tests, docs
- Non-breaking: default `SpatialClustering(chaos=True, pas=True)` unchanged; new metrics default False
