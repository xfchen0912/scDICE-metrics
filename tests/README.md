# Test layout

Tests are grouped by the part of `scdice_metrics` they exercise. Shared helpers live in
`tests/utils/` (not collected as tests).

| Directory | Scope | Example run |
| --- | --- | --- |
| `benchmark/` | `Benchmarker`, templates, spatial cluster prep, counterfactual benchmarker | `pytest tests/benchmark -v` |
| `counterfactual/` | OOD swap tasks and counterfactual metrics | `pytest tests/counterfactual -v` |
| `decomposition/` | Perturbation / response decomposition (incl. optional Replogle smoke) | `pytest tests/decomposition -v` |
| `metrics/` | Individual metrics, clustering resolution, PCR/PCA utils | `pytest tests/metrics -v` |
| `nearest_neighbors/` | JAX / pynndescent neighbor search | `pytest tests/nearest_neighbors -v` |
| `utils/` | Fixtures and sampling helpers (`data`, `swap_fixtures`, `sampling`) | — |

Full suite (from repo root):

```bash
pytest tests/ -v
```

Optional Replogle smoke (needs local h5ad or env paths):

```bash
pytest tests/decomposition/test_smoke_replogle_decomposition.py -v
```
