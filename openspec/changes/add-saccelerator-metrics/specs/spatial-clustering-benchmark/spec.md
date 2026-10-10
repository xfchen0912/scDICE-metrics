## ADDED Requirements

### Requirement: SACCELERATOR spatial supervised metrics

The package SHALL provide optional spatial-domain supervised metrics aligned with SACCELERATOR definitions, registered on `SpatialClustering` with default enablement off.

#### Scenario: Default spatial clustering unchanged

- **WHEN** a user constructs `SpatialClustering()` with defaults
- **THEN** SACCELERATOR-specific metric keys such as `domain_specific_f1` are not computed

#### Scenario: Domain-specific F1 expansion

- **WHEN** `domain_specific_f1` is enabled and ground-truth spatial labels exist
- **THEN** the benchmarker records per-domain F1 keys and a documented macro summary for templates

#### Scenario: Unmatched ground-truth domains are NA

- **WHEN** the number of predicted clusters is smaller than the number of ground-truth domains
- **THEN** every ground-truth domain still appears in `domain_specific_f1`, unmatched domains have F1 `NaN`, and `macro` is `NaN`

### Requirement: SACCELERATOR metric factory

The package SHALL provide `SpatialClustering.saccelerator()` enabling a documented subset (HOM, COM, CHAOS, PAS, and selected supervised metrics).

#### Scenario: Factory enables HOM and continuity

- **WHEN** `SpatialClustering.saccelerator()` is passed to `Benchmarker`
- **THEN** HOM, COM, CHAOS, and PAS are eligible to run when data keys are present

### Requirement: Hungarian matching for scoring

Supervised metrics that require label matching SHALL use a single documented matching helper distinct from cluster relabeling during prepare.

#### Scenario: Unequal cluster and domain counts

- **WHEN** predicted cluster count differs from ground-truth domain count
- **THEN** matched MCC/Jaccard/domain-F1 use the same contingency and assignment policy documented in design.md

### Requirement: Optional consensus metrics

The package MAY expose consensus metrics (cross-method ARI, smoothness entropy) on wide label tables without integrating them into the default single-AnnData benchmark loop.

#### Scenario: Smoothness entropy uses nats

- **WHEN** `smoothness_entropy` is computed on a wide label table and coordinates
- **THEN** neighborhood entropy uses natural log (SACCELERATOR `calc_entropy`) and k-NN excludes self
