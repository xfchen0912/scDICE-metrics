import pandas as pd
import pytest

from scdice_metrics.benchmark import BenchmarkTemplate, infer_template
from scdice_metrics.benchmark._templates import orient_metrics_higher_is_better


def test_infer_template_scib():
    template = infer_template(["Bio conservation", "Batch correction"])
    assert template.name == "scib"
    assert template.metric_to_group("silhouette_label") == "Bio conservation"
    assert template.metric_to_group("bras") == "Batch correction"


def test_infer_template_sdmbench():
    template = infer_template(["Spatial clustering"])
    assert template.name == "sdmbench"
    assert template.metric_to_group("hom") == "Accuracy"
    assert template.metric_to_group("pas") == "Continuity"


def test_orient_metrics_higher_is_better():
    raw = pd.DataFrame(
        {"emb_a": [0.9, 0.1], "emb_b": [0.8, 0.2]},
        index=["hom", "pas"],
    )
    oriented = orient_metrics_higher_is_better(raw)
    assert oriented.loc["hom", "emb_a"] > oriented.loc["hom", "emb_b"]
    assert oriented.loc["pas", "emb_a"] > oriented.loc["pas", "emb_b"]


def test_sdmbench_aggregate_ranking_accuracy_vs_continuity():
    raw = pd.DataFrame(
        {
            "good_acc": [0.95, 0.9, 0.05, 0.1],
            "good_cont": [0.5, 0.5, 0.05, 0.08],
            "bad_acc": [0.5, 0.5, 0.05, 0.08],
            "bad_cont": [0.95, 0.9, 0.05, 0.1],
        },
        index=["hom", "com", "chaos", "pas"],
    )
    template = BenchmarkTemplate.sdmbench()
    groups = pd.Series(
        {
            "hom": "Accuracy",
            "com": "Accuracy",
            "chaos": "Continuity",
            "pas": "Continuity",
        }
    )
    oriented = orient_metrics_higher_is_better(raw, template.lower_is_better)
    scores = oriented.groupby(groups).mean().T
    assert scores.loc["good_acc", "Accuracy"] > scores.loc["bad_acc", "Accuracy"]
    assert scores.loc["good_cont", "Continuity"] > scores.loc["bad_cont", "Continuity"]


def test_custom_template_weights():
    template = BenchmarkTemplate(
        name="custom",
        groups={"A": ("hom",), "B": ("pas",)},
        weights={"A": 0.7, "B": 0.3},
    )
    assert template.resolve_weights(["A", "B"]) == pytest.approx({"A": 0.7, "B": 0.3})


def test_legacy_template():
    template = BenchmarkTemplate.legacy_from_collections(["Spatial clustering"])
    assert template.metric_to_group("chaos", fallback="Spatial clustering") == "Spatial clustering"
