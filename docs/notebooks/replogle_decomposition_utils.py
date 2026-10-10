"""Helpers for the Replogle response-decomposition tutorial notebook."""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

DEFAULT_REPLOGLE_ROOT = Path(
    "/mnt/TrueNas/project/chenxufeng/Data/Perturb_Data/single_cell/PMID35688146_Cell2022_Replogle"
)
DEFAULT_K562_FILENAME = "replogle_2022_k562_gwps.h5ad"
DEFAULT_RPE1_FILENAME = "replogle_2022_rpe1.h5ad"

REPLOGLE_K562_FIGSHARE = "https://plus.figshare.com/articles/dataset/Replogle_K562/21999546"
REPLOGLE_RPE1_FIGSHARE = "https://plus.figshare.com/articles/dataset/Replogle_RPE1/21999546"

DEFAULT_PERTDECOMP_PKL_ROOT = Path(
    os.environ.get(
        "PERTDECOMP_PKL_DIR",
        Path(__file__).resolve().parents[3] / "tmp/perturbation-decomposition/data/processed/genetic",
    )
)
DEFAULT_PERTDECOMP_CONTEXTS = ("k562", "rpe1")
DEFAULT_PERTDECOMP_CONTEXT_LABELS = {
    "k562": "K562",
    "rpe1": "RPE1",
    "hepg2": "HepG2",
    "jurkat": "Jurkat",
}

DECOMP_COLORS = {
    "global": "#3B5BA9",
    "context": "#4A90D9",
    "perturbation": "#6F5CC4",
    "context:perturbation": "#D26A3A",
    "noise": "#C9CED6",
}

DEFAULT_FIGURE_DPI = 100
DEFAULT_SAVEFIG_DPI = 300
DEFAULT_FONT_FAMILY = "Arial"


def configure_plot_style(
    *,
    figure_dpi: int = DEFAULT_FIGURE_DPI,
    savefig_dpi: int = DEFAULT_SAVEFIG_DPI,
    font_family: str = DEFAULT_FONT_FAMILY,
    font_size: float = 10.0,
) -> None:
    """Apply publication-style matplotlib defaults for tutorial figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": figure_dpi,
            "savefig.dpi": savefig_dpi,
            "savefig.bbox": "tight",
            "font.family": "sans-serif",
            "font.sans-serif": [font_family, "Helvetica", "DejaVu Sans"],
            "font.size": font_size,
            "axes.titlesize": font_size + 1,
            "axes.labelsize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def default_pertdecomp_pkl_root() -> Path:
    """Root directory for ``perturbation-decomposition`` release pseudobulk pkls."""
    return Path(os.environ.get("PERTDECOMP_PKL_DIR", DEFAULT_PERTDECOMP_PKL_ROOT))


def default_pertdecomp_pkl_paths(
    contexts: tuple[str, ...] = DEFAULT_PERTDECOMP_CONTEXTS,
) -> dict[str, Path]:
    """Resolve ``{context}.pkl`` paths under the perturbation-decomposition release tree."""
    root = default_pertdecomp_pkl_root()
    return {context: root / f"{context}.pkl" for context in contexts}


def load_pertdecomp_pkl(path: Path | str) -> dict[str, Any]:
    """Load one ``perturbation-decomposition`` genetic pseudobulk pickle."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Perturbation-decomposition pickle not found at {path}.")
    with path.open("rb") as handle:
        return pickle.load(handle)


def pertdecomp_pkl_to_anndata(
    pkl_data: dict[str, Any],
    *,
    context_name: str,
    perturbation_key: str = "gene",
    control: str = "non-targeting",
    obsm_key: str = "X_hvg",
    control_cell_count: int | None = None,
) -> ad.AnnData:
    """Expand pseudobulk ``.pkl`` data into a synthetic single-cell AnnData.

    Each perturbation is represented by ``cell_counts[pert]`` cells sharing the
    same pseudobulk expression ``ctrl_mean + delta``. This preserves group means
    consumed by :func:`scdice_metrics.prepare_perturbation_responses`.
    """
    ctrl_mean = np.asarray(pkl_data["ctrl_mean"], dtype=float)
    deltas = pkl_data["deltas"]
    cell_counts = pkl_data["cell_counts"]
    n_features = ctrl_mean.shape[0]
    hvg_names = pkl_data.get("hvg_names")
    if hvg_names is None or len(hvg_names) != n_features:
        hvg_names = [f"f{i}" for i in range(n_features)]

    if control_cell_count is None:
        positive = [int(n) for n in cell_counts.values() if int(n) > 0]
        control_cell_count = max(positive) if positive else 5

    rows: list[np.ndarray] = []
    labels: list[str] = []

    for _ in range(control_cell_count):
        rows.append(ctrl_mean.copy())
        labels.append(control)

    for pert in sorted(deltas):
        expr = ctrl_mean + np.asarray(deltas[pert], dtype=float)
        n_cells = int(cell_counts.get(pert, 0))
        if n_cells <= 0:
            continue
        for _ in range(n_cells):
            rows.append(expr.copy())
            labels.append(str(pert))

    matrix = np.vstack(rows)
    obs = pd.DataFrame({perturbation_key: labels, "context": context_name})
    var = pd.DataFrame(index=[str(name) for name in hvg_names])
    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.obsm[obsm_key] = matrix.copy()
    adata.uns[obsm_key] = [str(name) for name in hvg_names]
    return adata


def load_pertdecomp_decomposition_adatas(
    *,
    pkl_root: Path | None = None,
    contexts: tuple[str, ...] = DEFAULT_PERTDECOMP_CONTEXTS,
    context_labels: dict[str, str] | None = None,
    perturbation_key: str = "gene",
    control: str = "non-targeting",
    obsm_key: str = "X_hvg",
    control_cell_count: int | None = None,
) -> tuple[list[ad.AnnData], list[str]]:
    """Load K562/RPE1 (or other) release pkls and convert them to AnnData objects."""
    pkl_root = default_pertdecomp_pkl_root() if pkl_root is None else Path(pkl_root)
    labels = DEFAULT_PERTDECOMP_CONTEXT_LABELS if context_labels is None else context_labels

    adatas: list[ad.AnnData] = []
    context_names: list[str] = []
    for context in contexts:
        pkl_path = pkl_root / f"{context}.pkl"
        pkl_data = load_pertdecomp_pkl(pkl_path)
        context_name = labels.get(context, context.upper())
        adatas.append(
            pertdecomp_pkl_to_anndata(
                pkl_data,
                context_name=context_name,
                perturbation_key=perturbation_key,
                control=control,
                obsm_key=obsm_key,
                control_cell_count=control_cell_count,
            )
        )
        context_names.append(context_name)
    return adatas, context_names


def default_replogle_paths() -> tuple[Path, Path]:
    """Resolve K562 / RPE1 h5ad paths from environment variables."""
    data_root = Path(os.environ.get("REPLOGLE_DATA_DIR", DEFAULT_REPLOGLE_ROOT))
    k562 = Path(os.environ.get("REPLOGLE_K562_H5AD", data_root / DEFAULT_K562_FILENAME))
    rpe1 = Path(os.environ.get("REPLOGLE_RPE1_H5AD", data_root / DEFAULT_RPE1_FILENAME))
    return k562, rpe1


def _select_hvg_var_names(adata, *, n_top_genes: int = 2000) -> np.ndarray:
    if "cv" in adata.var.columns:
        top = adata.var["cv"].astype(float).nlargest(min(n_top_genes, adata.n_vars))
        return top.index.to_numpy()
    if "highly_variable" in adata.var.columns and int(adata.var["highly_variable"].sum()) >= n_top_genes:
        return adata.var_names[adata.var["highly_variable"]].to_numpy()[:n_top_genes]
    return adata.var_names.to_numpy()[: min(n_top_genes, adata.n_vars)]


def ensure_hvg_obsm(adata, *, n_top_genes: int = 2000, key: str = "X_hvg"):
    """Ensure ``obsm[key]`` exists using precomputed var metrics or scanpy HVG selection."""
    if key in adata.obsm:
        return adata

    hvg_names = _select_hvg_var_names(adata, n_top_genes=n_top_genes)
    hvg = adata[:, hvg_names].X
    if sparse.issparse(hvg):
        hvg = hvg.toarray()
    adata.obsm[key] = np.asarray(hvg, dtype=float)
    adata.uns[key] = [str(name) for name in hvg_names]
    return adata


def subsample_by_perturbation(
    adata,
    *,
    perturbation_key: str,
    control: str,
    max_cells_per_pert: int,
    random_state: int = 0,
):
    """Keep all control cells and up to ``max_cells_per_pert`` cells per perturbation."""
    if max_cells_per_pert is None:
        return adata

    rng = np.random.default_rng(random_state)
    labels = adata.obs[perturbation_key].astype(str).to_numpy()
    keep_idx: list[int] = []

    for pert in pd.Series(labels).unique():
        idx = np.flatnonzero(labels == pert)
        if pert == control:
            keep_idx.extend(idx.tolist())
            continue
        if len(idx) <= max_cells_per_pert:
            keep_idx.extend(idx.tolist())
        else:
            chosen = rng.choice(idx, size=max_cells_per_pert, replace=False)
            keep_idx.extend(chosen.tolist())

    return adata[sorted(keep_idx)].copy()


def load_replogle_decomposition_adatas(
    *,
    k562_path: Path | None = None,
    rpe1_path: Path | None = None,
    perturbation_key: str = "gene",
    control: str = "non-targeting",
    n_top_genes: int = 2000,
    k562_max_cells_per_pert: int | None = 50,
    rpe1_max_cells_per_pert: int | None = None,
    random_state: int = 0,
) -> tuple[list, list[str]]:
    """Load K562 + RPE1 objects prepared for ``scdice_metrics`` decomposition.

    The local K562 file is the genome-wide Perturb-seq screen. We restrict K562 to
    perturbations shared with the RPE1 essential screen and optionally subsample
    cells for memory/runtime. RPE1 is the essential-gene benchmark context.
    """
    if k562_path is None or rpe1_path is None:
        default_k562, default_rpe1 = default_replogle_paths()
        k562_path = default_k562 if k562_path is None else Path(k562_path)
        rpe1_path = default_rpe1 if rpe1_path is None else Path(rpe1_path)

    for path in (k562_path, rpe1_path):
        if not path.exists():
            raise FileNotFoundError(f"Replogle AnnData not found at {path}.")

    rpe1 = sc.read_h5ad(rpe1_path)
    rpe1_perts = set(rpe1.obs[perturbation_key].astype(str)) - {control}

    k562_backed = sc.read_h5ad(k562_path, backed="r")
    allowed = rpe1_perts | {control}
    k562_mask = k562_backed.obs[perturbation_key].astype(str).isin(allowed)
    k562 = k562_backed[k562_mask].to_memory()
    k562_backed.file.close()

    k562 = subsample_by_perturbation(
        k562,
        perturbation_key=perturbation_key,
        control=control,
        max_cells_per_pert=k562_max_cells_per_pert,
        random_state=random_state,
    )
    rpe1 = subsample_by_perturbation(
        rpe1,
        perturbation_key=perturbation_key,
        control=control,
        max_cells_per_pert=rpe1_max_cells_per_pert,
        random_state=random_state,
    )

    k562 = ensure_hvg_obsm(k562, n_top_genes=n_top_genes)
    rpe1 = ensure_hvg_obsm(rpe1, n_top_genes=n_top_genes)
    return [k562, rpe1], ["K562", "RPE1"]


def load_replogle_context(path: Path, *, n_top_genes: int = 2000):
    """Load one Replogle screen and ensure HVG features are available."""
    if not path.exists():
        raise FileNotFoundError(f"Replogle AnnData not found at {path}.")
    adata = sc.read_h5ad(path)
    return ensure_hvg_obsm(adata, n_top_genes=n_top_genes)


def plot_fraction_stacked_bars(
    ax,
    fraction_dicts: dict[str, dict[str, float]],
    *,
    component_order: list[str],
    title: str,
) -> None:
    """Draw horizontal stacked bars for decomposition fractions."""
    labels = list(fraction_dicts.keys())
    y = np.arange(len(labels))[::-1]
    for yi, label in zip(y, labels, strict=True):
        left = 0.0
        for component in component_order:
            value = 100.0 * fraction_dicts[label].get(component, 0.0)
            if value <= 0.0:
                continue
            ax.barh(
                yi,
                value,
                left=left,
                height=0.45,
                color=DECOMP_COLORS.get(component, "#999999"),
                edgecolor="white",
                linewidth=0.5,
            )
            left += value
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fraction of response energy (%)")
    ax.set_title(title)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.6)
