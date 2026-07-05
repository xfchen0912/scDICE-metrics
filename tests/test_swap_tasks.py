import numpy as np
import pytest

from scdice_metrics.benchmark._swap import make_swap_specs, summarize_swap_specs
from tests.utils.swap_fixtures import CELL_TYPES, CONDITIONS, NICHES, make_swap_benchmark_adata


@pytest.fixture
def swap_adata():
    return make_swap_benchmark_adata(seed=0, cells_per_combo=5)


def test_condition_swap_without_matching(swap_adata):
    specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    spec = specs[0]
    obs = swap_adata.obs

    target_obs = obs.loc[spec.target_mask]
    assert (target_obs["cell_type"] == "Fibroblast").all()
    assert (target_obs["condition"] == "CRC").all()
    assert set(target_obs["niche"]) == set(NICHES)

    assert spec.source_mask.sum() >= 5
    assert np.all(spec.train_mask[spec.source_mask])
    assert not np.any(spec.train_mask[spec.target_mask])

    context_obs = obs.loc[spec.context_mask]
    assert (context_obs["condition"] == "CRC").all()
    assert (context_obs["cell_type"] != "Fibroblast").all()


def test_condition_swap_with_niche_matching(swap_adata):
    specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=True,
        fixed_values=["fibrotic"],
        min_source_cells=5,
        min_target_cells=5,
    )
    spec = specs[0]
    obs = swap_adata.obs

    for mask_name in ("source_mask", "target_mask", "context_mask"):
        subset = obs.loc[getattr(spec, mask_name)]
        if mask_name == "context_mask":
            assert (subset["niche"] == "fibrotic").all()
        else:
            assert (subset["niche"] == "fibrotic").all()


def test_niche_swap_with_condition_matching(swap_adata):
    specs = make_swap_specs(
        swap_adata,
        swap_type="niche",
        source_value="immune_rich",
        target_value="fibrotic",
        cell_types=["Myeloid"],
        match_other_factor=True,
        fixed_values=["CRC"],
        min_source_cells=5,
        min_target_cells=5,
    )
    spec = specs[0]
    obs = swap_adata.obs

    source_obs = obs.loc[spec.source_mask]
    target_obs = obs.loc[spec.target_mask]
    context_obs = obs.loc[spec.context_mask]

    assert (source_obs["condition"] == "CRC").all()
    assert (target_obs["condition"] == "CRC").all()
    assert (context_obs["condition"] == "CRC").all()
    assert (source_obs["niche"] == "immune_rich").all()
    assert (target_obs["niche"] == "fibrotic").all()
    assert (context_obs["niche"] == "fibrotic").all()


def test_cell_type_traversal(swap_adata):
    specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast", "Myeloid"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    assert {spec.cell_type for spec in specs} == {"Fibroblast", "Myeloid"}


def test_automatic_traversal(swap_adata):
    specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=None,
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    assert {spec.cell_type for spec in specs} == set(CELL_TYPES)


def test_minimum_cell_filtering(swap_adata):
    with pytest.raises(ValueError, match="No valid swap tasks"):
        make_swap_specs(
            swap_adata,
            swap_type="condition",
            source_value="REF",
            target_value="CRC",
            cell_types=["T_cell"],
            match_other_factor=False,
            min_source_cells=5,
            min_target_cells=20,
        )

    specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["T_cell"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    assert len(specs) == 1
    assert specs[0].metadata["n_target"] == 7


def test_invalid_columns(swap_adata):
    swap_adata.obs.drop(columns=["niche"], inplace=True)
    with pytest.raises(ValueError, match="Missing required obs columns"):
        make_swap_specs(
            swap_adata,
            swap_type="condition",
            source_value="REF",
            target_value="CRC",
            match_other_factor=False,
        )


def test_no_valid_tasks(swap_adata):
    with pytest.raises(ValueError, match="No valid swap tasks"):
        make_swap_specs(
            swap_adata,
            swap_type="condition",
            source_value="REF",
            target_value="CRC",
            cell_types=["MissingType"],
            match_other_factor=False,
            min_source_cells=5,
            min_target_cells=5,
        )


def test_fixed_value_cartesian_product(swap_adata):
    cell_types = ["Fibroblast", "Myeloid"]
    fixed_values = ["immune_rich", "fibrotic"]
    expected = len(cell_types) * len(fixed_values)

    with pytest.warns(UserWarning, match="match_other_factor=True"):
        specs = make_swap_specs(
            swap_adata,
            swap_type="condition",
            source_value="REF",
            target_value="CRC",
            cell_types=cell_types,
            match_other_factor=True,
            fixed_values=fixed_values,
            min_source_cells=1,
            min_target_cells=1,
        )
    assert len(specs) == expected

    auto_specs = make_swap_specs(
        swap_adata,
        swap_type="niche",
        source_value="immune_rich",
        target_value="fibrotic",
        cell_types=["Fibroblast"],
        match_other_factor=True,
        fixed_values=None,
        min_source_cells=1,
        min_target_cells=1,
    )
    assert len(auto_specs) == len(CONDITIONS)


def test_unsupported_niche_marginalization(swap_adata):
    with pytest.raises(ValueError, match="unsupported"):
        make_swap_specs(
            swap_adata,
            swap_type="niche",
            source_value="immune_rich",
            target_value="fibrotic",
            match_other_factor=False,
        )


def test_invalid_fixed_values_use(swap_adata):
    with pytest.raises(ValueError, match="fixed_values must be None"):
        make_swap_specs(
            swap_adata,
            swap_type="condition",
            source_value="REF",
            target_value="CRC",
            match_other_factor=False,
            fixed_values=["fibrotic"],
        )


def test_task_ids_are_deterministic_and_url_safe(swap_adata):
    kwargs = dict(
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=True,
        fixed_values=["fibrotic"],
        min_source_cells=5,
        min_target_cells=5,
    )
    with pytest.warns(UserWarning):
        first = make_swap_specs(swap_adata, **kwargs)
        second = make_swap_specs(swap_adata, **kwargs)
    assert [spec.task_id for spec in first] == [spec.task_id for spec in second]
    assert first[0].task_id.startswith("condition__")
    assert "cell_type=" in first[0].task_id


def test_default_matched_condition_warning(swap_adata):
    with pytest.warns(UserWarning, match="CellDISECT-compatible"):
        make_swap_specs(
            swap_adata,
            swap_type="condition",
            source_value="REF",
            target_value="CRC",
            cell_types=["Fibroblast"],
            min_source_cells=5,
            min_target_cells=5,
        )


def test_summarize_swap_specs(swap_adata):
    specs = make_swap_specs(
        swap_adata,
        swap_type="condition",
        source_value="REF",
        target_value="CRC",
        cell_types=["Fibroblast"],
        match_other_factor=False,
        min_source_cells=5,
        min_target_cells=5,
    )
    summary = summarize_swap_specs(specs)
    assert summary.shape[0] == 1
    assert summary.loc[0, "task_id"] == specs[0].task_id
    assert summary.loc[0, "n_target"] == specs[0].metadata["n_target"]
