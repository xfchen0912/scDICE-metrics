"""Perturbation response metrics adapted from the Systema benchmark.

These implementations follow the evaluation utilities described in
Vinas Torne et al., "Systema: A Framework for Evaluating Genetic Perturbation Response
Prediction Beyond Systematic Variation", *Nature Biotechnology* (2025).
https://doi.org/10.1038/s41587-025-02777-8

Original reference code: https://github.com/mlbio-epfl/systema (evaluation/).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import cdist

if TYPE_CHECKING:
    from anndata import AnnData


def pearson_delta_reference_metrics(
    X_true: np.ndarray,
    X_pred: np.ndarray,
    reference: np.ndarray,
    top20_de_idxs: np.ndarray | None = None,
) -> dict[str, float]:
    """Pearson correlation on delta profiles relative to an arbitrary reference.

    Computes PearsonΔ (all genes) and optionally PearsonΔ20 on a fixed gene subset.

    Parameters
    ----------
    X_true
        Ground-truth post-perturbation profile, shape ``(n_genes,)``.
    X_pred
        Predicted post-perturbation profile, shape ``(n_genes,)``.
    reference
        Reference profile (e.g. control mean or mean of perturbed centroids), shape ``(n_genes,)``.
    top20_de_idxs
        Integer indices (positions along the gene axis) for the top differential genes.
        If ``None``, only ``corr_all_allpert`` is returned (no Δ20 term).

    Returns
    -------
    dict
        ``corr_all_allpert``: Pearson r between ``X_true - reference`` and ``X_pred - reference``.
        If ``top20_de_idxs`` is given, also ``corr_20de_allpert``: same on the indexed subset.

    References
    ----------
    Vinas Torne et al., *Nat. Biotechnol.* (2025); Systema evaluation README.
    """
    X_true = np.asarray(X_true, dtype=float).ravel()
    X_pred = np.asarray(X_pred, dtype=float).ravel()
    reference = np.asarray(reference, dtype=float).ravel()
    if X_true.shape != X_pred.shape or X_true.shape != reference.shape:
        raise ValueError("X_true, X_pred, and reference must have the same shape.")

    delta_true = X_true - reference
    delta_pred = X_pred - reference
    out: dict[str, float] = {
        "corr_all_allpert": float(stats.pearsonr(delta_true, delta_pred)[0]),
    }
    if top20_de_idxs is not None:
        idx = np.asarray(top20_de_idxs, dtype=int).ravel()
        out["corr_20de_allpert"] = float(stats.pearsonr(delta_true[idx], delta_pred[idx])[0])
    return out


def calculate_centroid_accuracies(
    agg_post_pred_df: pd.DataFrame,
    post_gt_all_df: pd.DataFrame,
) -> pd.DataFrame:
    """Centroid accuracy: fraction of other GT centroids farther than the true one (per method).

    For each method and test perturbation, compares Euclidean distances from the predicted
    profile to every ground-truth perturbation centroid.

    Parameters
    ----------
    agg_post_pred_df
        Predicted post-perturbation profiles. Columns are genes. Rows use a MultiIndex
        ``(condition, method)`` where ``condition`` matches ``post_gt_all_df.index``.
    post_gt_all_df
        Ground-truth profiles: rows = perturbations, columns = genes.

    Returns
    -------
    pd.DataFrame
        Rows = conditions, columns = methods, values in ``[0, 1]``.

    References
    ----------
    Vinas Torne et al., *Nat. Biotechnol.* (2025); Systema ``centroid_accuracy.py``.
    """
    distances = cdist(agg_post_pred_df.values, post_gt_all_df.values, metric="euclidean")
    dist_df = pd.DataFrame(distances, index=agg_post_pred_df.index, columns=post_gt_all_df.index)

    index = [g for g in dist_df.index.get_level_values(0) if g in dist_df.columns]
    multiindex = [(g, m) for g, m in dist_df.index if g in dist_df.columns]
    col_idxs = [int(np.flatnonzero(dist_df.columns.values == g)[0]) for g in index]
    self_distances = np.diag(dist_df.iloc[np.arange(len(dist_df)), col_idxs])
    self_distances_df = pd.DataFrame(self_distances, index=pd.MultiIndex.from_tuples(multiindex))
    if not np.all(dist_df.index == self_distances_df.index):
        raise ValueError("Index alignment failed between distance matrix and self-distances.")

    scores: dict[str, pd.Series] = {}
    methods = agg_post_pred_df.index.get_level_values(1).unique()
    for method in methods:
        x_df = dist_df.xs(method, level=1).sort_index()
        y_df = self_distances_df.xs(method, level=1).sort_index()
        if not np.all(x_df.index == y_df.index):
            raise ValueError("Per-method row index mismatch.")
        scores[str(method)] = ((x_df > y_df.values).sum(axis=1)) / (x_df.shape[1] - 1)
    return pd.DataFrame(scores)


def average_of_perturbation_centroids(
    adata: AnnData,
    *,
    control_key: str = "control",
    condition_key: str = "condition",
    control_is_one: bool = True,
) -> np.ndarray:
    """Average of per-condition mean expression over perturbed cells (Systema reference).

    For each unique ``condition`` among **non-control** cells, takes the mean of ``X``,
    then averages those condition means. Used e.g. as a reference for PearsonΔ to reduce
    sensitivity to systematic perturbed-vs-control structure.

    Parameters
    ----------
    adata
        AnnData with expression in ``X``.
    control_key
        ``obs`` column; values distinguish control vs perturbed.
    condition_key
        ``obs`` column listing perturbation / condition id per cell.
    control_is_one
        If True (default), rows with ``obs[control_key] == 1`` are controls and
        ``== 0`` (or not 1) are perturbed, matching the Systema convention.
        If False, rows with ``obs[control_key] == 0`` are controls.

    Returns
    -------
    np.ndarray
        1D array of length ``n_vars``.

    References
    ----------
    Vinas Torne et al., *Nat. Biotechnol.* (2025); Systema ``eval_utils.py``.
    """
    obs = adata.obs
    ck = obs[control_key]
    if control_is_one:
        pert_mask = ck != 1
    else:
        pert_mask = ck != 0
    pert_adata = adata[pert_mask]
    pert_means: list[np.ndarray] = []
    for cond in pert_adata.obs[condition_key].unique():
        sub = pert_adata[pert_adata.obs[condition_key] == cond]
        mean_vec = np.asarray(sub.X.mean(axis=0)).ravel()
        pert_means.append(mean_vec)
    if not pert_means:
        raise ValueError("No perturbed cells found for the given control_key / control_is_one convention.")
    return np.mean(np.stack(pert_means, axis=0), axis=0)


def get_perts(
    test_perts: np.ndarray | list[str],
    phenotypes: dict[str, list[str]],
    phenotype_names: list[str],
) -> np.ndarray:
    """Intersect test perturbations with those belonging to selected phenotype groups.

    Parameters
    ----------
    test_perts
        Candidate perturbation names.
    phenotypes
        Maps phenotype label to list of perturbation names.
    phenotype_names
        Keys in ``phenotypes`` to pool.

    Returns
    -------
    np.ndarray
        Sorted intersection of ``test_perts`` with the union of listed phenotypes.

    References
    ----------
    Systema ``centroid_reference_scores.py``.
    """
    perts: list[str] = []
    for p in phenotype_names:
        perts.extend(list(phenotypes[p]))
    return np.intersect1d(test_perts, perts)


def score_centroids(
    post_gt_df_seed: pd.DataFrame,
    post_pred_df_seed: pd.DataFrame,
    perts_dict: dict[str, list[str] | np.ndarray],
    methods: list[str],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Score predictions by negative MSE distance to phenotype centroids from ground truth.

    Prototypes are means of ``post_gt_df_seed`` over perturbations listed per phenotype.

    Parameters
    ----------
    post_gt_df_seed
        Ground truth: rows = perturbations, columns = genes.
    post_pred_df_seed
        Predictions with MultiIndex ``(condition, method)`` on rows.
    perts_dict
        Maps phenotype / class name to list of perturbations used to define that centroid.
    methods
        Method names (second level of MultiIndex).

    Returns
    -------
    labels
        Shape ``(n_perts, n_prototypes)``, 1 if perturbation belongs to that class's list.
    scores_dict
        Maps method name to array of shape ``(n_perts, n_prototypes)`` (higher = closer).

    References
    ----------
    Vinas Torne et al., *Nat. Biotechnol.* (2025); Systema ``centroid_reference_scores.py``.
    """
    prototypes: dict[str, pd.Series] = {}
    for k, v in perts_dict.items():
        prototypes[k] = post_gt_df_seed.loc[v].mean(axis=0)

    perts: list[str] = []
    for _k, p in perts_dict.items():
        perts.extend([str(x) for x in p])

    labels = np.zeros((len(perts), len(prototypes)))
    scores_dict: dict[str, np.ndarray] = {}
    for method in methods:
        preds = post_pred_df_seed.xs(method, level=1).loc[perts]
        scores = np.zeros((len(preds), len(prototypes)))
        for i, (pk, proto) in enumerate(prototypes.items()):
            distances = ((preds - proto) ** 2).mean(axis=1).to_numpy()
            scores[:, i] = -distances
            labels[np.isin(perts, perts_dict[pk]), i] = 1
        scores_dict[method] = scores

    return labels, scores_dict
