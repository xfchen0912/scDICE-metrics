import numpy as np
import pytest
import scanpy as sc

from scdice_metrics.benchmark import Benchmarker, SpatialClustering


def _make_spatial_adata(n=60, seed=0):
    rng = np.random.default_rng(seed)
    adata = sc.AnnData(rng.poisson(5, (n, 20)).astype(float))
    adata.obsm["spatial"] = rng.random((n, 2))
    adata.obsm["X_emb"] = rng.random((n, 5))
    adata.obs["batch"] = "b1"
    adata.obs["cell_type"] = rng.choice(["T", "B"], n)
    adata.obs["domain"] = np.array(["D0"] * 30 + ["D1"] * 30)
    sc.pp.pca(adata, n_comps=5)
    return adata


def test_default_spatial_excludes_saccelerator_metrics():
    adata = _make_spatial_adata()
    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="cell_type",
        embedding_obsm_keys=["X_pca"],
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_label_key="domain",
        spatial_clustering_metrics=SpatialClustering(hom=True, com=True),
        progress_bar=False,
        compute_neighbors=False,
    )
    bm.prepare()
    bm.benchmark()
    idx = set(bm._results.index.astype(str))
    assert "spatial_ari" not in idx
    assert "gt_mixture_entropy" not in idx


def test_saccelerator_factory_runs():
    adata = _make_spatial_adata()
    adata.obs["X_pca_spatial_cluster"] = adata.obs["domain"].astype(str).values
    bm = Benchmarker(
        adata,
        batch_key="batch",
        label_key="cell_type",
        embedding_obsm_keys=["X_pca"],
        bio_conservation_metrics=None,
        batch_correction_metrics=None,
        spatial_label_key="domain",
        spatial_cluster_key="X_pca_spatial_cluster",
        spatial_clustering_metrics=SpatialClustering.saccelerator(),
        progress_bar=False,
        compute_neighbors=False,
    )
    bm.prepare()
    bm.benchmark()
    idx = set(bm._results.index.astype(str))
    assert any("spatial_ari" == i for i in idx)
    assert any("gt_mixture_entropy" == i for i in idx)
