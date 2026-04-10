import anndata
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

import scdice_metrics
from scdice_metrics.nearest_neighbors import NeighborsResults


def dummy_x_labels(symmetric_positive=False, x_is_neighbors_results=False):
    rng = np.random.default_rng(seed=42)
    X = rng.normal(size=(100, 10))
    labels = rng.integers(0, 4, size=(100,))
    if symmetric_positive:
        X = np.abs(X @ X.T)
    if x_is_neighbors_results:
        dist_mat = csr_matrix(scdice_metrics.utils.cdist(X, X))
        nbrs = NearestNeighbors(n_neighbors=30, metric="precomputed").fit(dist_mat)
        dist, ind = nbrs.kneighbors(dist_mat)
        X = NeighborsResults(indices=ind, distances=dist)
    return X, labels


def dummy_x_labels_batch(x_is_neighbors_results=False):
    rng = np.random.default_rng(seed=43)
    X, labels = dummy_x_labels(x_is_neighbors_results=x_is_neighbors_results)
    batch = rng.integers(0, 4, size=(100,))
    return X, labels, batch


def dummy_benchmarker_adata():
    X, labels, batch = dummy_x_labels_batch(x_is_neighbors_results=False)
    adata = anndata.AnnData(X)
    labels_key = "labels"
    batch_key = "batch"
    adata.obs[labels_key] = labels
    adata.obs[batch_key] = batch
    embedding_keys = []
    for i in range(5):
        key = f"X_emb_{i}"
        adata.obsm[key] = X
        embedding_keys.append(key)
    return adata, embedding_keys, labels_key, batch_key


def dummy_disentanglement_data():
    rng = np.random.default_rng(seed=44)
    n_per_group = 30
    factor_a = np.repeat([0, 1], 2 * n_per_group)
    factor_b = np.tile(np.repeat([0, 1], n_per_group), 2)
    response = ((factor_a + factor_b) > 0).astype(int)

    latent_good = np.column_stack(
        [
            factor_a + rng.normal(scale=0.05, size=factor_a.shape[0]),
            factor_b + rng.normal(scale=0.05, size=factor_b.shape[0]),
            rng.normal(scale=0.2, size=factor_a.shape[0]),
        ]
    )
    latent_bad = rng.normal(size=latent_good.shape)
    factors = {"factor_a": factor_a, "factor_b": factor_b}
    return latent_good, latent_bad, factors, response


def dummy_disentanglement_benchmarker_adata():
    latent_good, latent_bad, factors, response = dummy_disentanglement_data()
    adata = anndata.AnnData(latent_good)
    adata.obs["labels"] = factors["factor_a"]
    adata.obs["batch"] = factors["factor_b"]
    adata.obs["factor_a"] = factors["factor_a"]
    adata.obs["factor_b"] = factors["factor_b"]
    adata.obs["response"] = response
    adata.obsm["X_good"] = latent_good
    adata.obsm["X_bad"] = latent_bad
    return adata, ["X_good", "X_bad"], "batch", "labels"
