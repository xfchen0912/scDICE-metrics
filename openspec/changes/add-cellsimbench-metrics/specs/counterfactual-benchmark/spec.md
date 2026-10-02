## ADDED Requirements

### Requirement: CellSimBench dual-baseline tasks

The counterfactual benchmark SHALL support optional `dataset_mean` and `deg` on each `CounterfactualTask` for CellSimBench-style evaluation without changing default metric enablement.

#### Scenario: deltapert metrics skipped without dataset mean

- **WHEN** `pearson_deltapert` is enabled and `dataset_mean` is missing on a task
- **THEN** that task is skipped for deltapert metrics or records NaN with documented behavior

#### Scenario: GT-half DEG alignment

- **WHEN** `deg` is provided with `source="gt_half"`
- **THEN** weights and top_mask lengths match `gene_names` or feature dimension

### Requirement: CellSimBench metric factory

The package SHALL provide `Counterfactual.cellsimbench()` returning a configuration with the paper metric subset enabled.

#### Scenario: Default counterfactual unchanged

- **WHEN** a user constructs `Counterfactual()` with defaults
- **THEN** CellSimBench-specific metric names are not computed

### Requirement: Dynamic range fraction calibration

The package SHALL expose `dynamic_range_fraction` and baseline-relative scoring utilities on long-format benchmark results.

#### Scenario: Higher-is-better DRF

- **WHEN** `higher_is_better=True`, `perfect=1.0`, baseline=0.2, pos=0.6
- **THEN** DRF equals `(0.6 - 0.2) / (1.0 - 0.2)` clipped to the configured range
