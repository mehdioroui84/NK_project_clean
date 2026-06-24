#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


DEFAULT_CELLXGENE = (
    "/rsrch5/home/genomic_med/A3D3a/Bioinformatics/sc_nk/"
    "cell-by-gene/combine/CellxGene_NK_B_T.h5ad"
)
DEFAULT_CB07 = "/rsrch5/home/genomic_med/suorouji/projects/lsf_run/seurat_manual.h5ad"
DEFAULT_OUT = "/rsrch5/home/genomic_med/suorouji/projects/lsf_run/cellxgene_nk_bt_plus_CB07_hvg2k.h5ad"
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
DEFAULT_RESCUE_MARKERS = os.path.join(
    PROJECT_ROOT,
    "nk_project",
    "annotation_agent",
    "references",
    "canonical_nk_marker_rescue_genes.txt",
)


def main() -> None:
    args = parse_args()
    ensure_dirs(os.path.dirname(args.output_h5ad), args.qc_outdir)

    print("=" * 80)
    print("Prepare CellxGene NK/B/T + CB07 input for NK_project step 1")
    print("=" * 80)
    print(f"[CELLXGENE] {args.cellxgene_h5ad}")
    print(f"[CB07]      {args.cb07_h5ad}")
    print(f"[OUTPUT]    {args.output_h5ad}")
    print(f"[N_HVG]     {args.n_hvg}")
    print(f"[RESCUE]    {args.marker_rescue_file}")

    cellxgene = load_cellxgene(args)
    cb07 = load_cb07(args)

    print("\n[ALIGN] concatenating on shared gene symbols")
    combined = ad.concat(
        [cellxgene, cb07],
        join="inner",
        label="source_panel",
        keys=["cellxgene", "CB07"],
        index_unique=None,
        merge="same",
    )
    combined.obs_names_make_unique()
    combined.var_names_make_unique()
    print(f"[COMBINED] {combined.n_obs:,} cells x {combined.n_vars:,} shared genes")

    combined.obs[cfg.ASSAY_CLEAN_KEY] = clean_assay(
        combined.obs.get(cfg.ASSAY_KEY, pd.Series("nan", index=combined.obs_names)),
        flex_fill=cfg.FLEX_ASSAY_FILL,
    )
    add_batch_composite_columns(combined.obs)

    write_qc_tables(combined, args.qc_outdir, prefix="before_hvg")

    if args.n_hvg and combined.n_vars > args.n_hvg:
        combined = subset_hvgs(combined, args)
    else:
        print("[HVG] skipping HVG subset")

    # Step 1 expects raw counts in .X.
    if "counts" in combined.layers:
        combined.X = combined.layers["counts"].copy()
    ensure_integer_like_counts(combined.X, "combined.X")

    write_qc_tables(combined, args.qc_outdir, prefix="prepared")
    combined.write(args.output_h5ad)
    print(f"\n[SAVE] {args.output_h5ad}")
    print("[DONE] Use this file as configs.default_config.MERGED_PATH for step 1.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the step-1 input AnnData from the new CellxGene NK/B/T file "
            "plus in-house CB07, with raw counts in .X and 2k HVGs."
        )
    )
    parser.add_argument("--cellxgene-h5ad", default=DEFAULT_CELLXGENE)
    parser.add_argument("--cb07-h5ad", default=DEFAULT_CB07)
    parser.add_argument("--output-h5ad", default=DEFAULT_OUT)
    parser.add_argument(
        "--qc-outdir",
        default="/rsrch5/home/genomic_med/suorouji/projects/lsf_run/prepare_cellxgene_cb07_qc",
    )
    parser.add_argument("--n-hvg", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument(
        "--cellxgene-nk-label-key",
        default="State",
        help="Column used as NK_State for CellxGene NK cells. B/T cells are forced to B/T.",
    )
    parser.add_argument(
        "--cellxgene-nk-label-fallback",
        default="Subtype",
        help="Fallback label column for CellxGene NK cells if --cellxgene-nk-label-key is missing/empty.",
    )
    parser.add_argument("--cb07-id-key", default="cb_id")
    parser.add_argument("--cb07-id-value", default=cfg.PROTECTED_DATASET)
    parser.add_argument(
        "--cb07-label-key",
        default="functional_state",
        help="CB07 obs column used as NK_State. Falls back to Unknown if missing.",
    )
    parser.add_argument("--cb07-assay", default=cfg.FLEX_ASSAY_FILL)
    parser.add_argument("--cb07-tissue", default="cord blood")
    parser.add_argument(
        "--hvg-source",
        choices=["existing", "compute"],
        default="compute",
        help=(
            "existing: use input var['highly_variable'] when available; "
            "compute: recompute HVGs from lognorm/X with Scanpy."
        ),
    )
    parser.add_argument(
        "--hvg-batch-key",
        default=cfg.ASSAY_CLEAN_KEY,
        help=(
            "Batch key used when recomputing HVGs. Default: assay_clean. "
            "Use 'none' to compute without batch-aware HVG selection."
        ),
    )
    parser.add_argument(
        "--marker-rescue-file",
        default=DEFAULT_RESCUE_MARKERS,
        help=(
            "Optional one-gene-per-line file. These genes are appended after HVG "
            "selection if present in the data and missing from the selected HVGs."
        ),
    )
    return parser.parse_args()


def load_cellxgene(args: argparse.Namespace) -> sc.AnnData:
    print("\n[LOAD] CellxGene NK/B/T")
    adata = sc.read_h5ad(args.cellxgene_h5ad)
    standardize_var_names(adata)
    set_raw_counts_x(adata)
    ensure_lognorm_layer(adata)

    obs = adata.obs
    obs[cfg.DATASET_KEY] = require_obs(obs, cfg.DATASET_KEY).astype(str)
    obs[cfg.ASSAY_KEY] = require_obs(obs, cfg.ASSAY_KEY).astype(str)
    obs["cell_type"] = require_obs(obs, "cell_type").astype(str)

    labels = make_cellxgene_labels(obs, args)
    obs[cfg.LABEL_KEY] = labels
    print(f"[CELLXGENE] {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    print("[CELLXGENE] NK_State counts:")
    print(obs[cfg.LABEL_KEY].astype(str).value_counts().head(30).to_string())
    return adata


def load_cb07(args: argparse.Namespace) -> sc.AnnData:
    print("\n[LOAD] CB07")
    inhouse = sc.read_h5ad(args.cb07_h5ad)
    standardize_var_names(inhouse)
    set_raw_counts_x(inhouse)
    ensure_lognorm_layer(inhouse)

    if args.cb07_id_key in inhouse.obs:
        mask = inhouse.obs[args.cb07_id_key].astype(str).values == str(args.cb07_id_value)
        cb07 = inhouse[mask].copy()
        print(f"[CB07] selected {mask.sum():,} cells where {args.cb07_id_key} == {args.cb07_id_value}")
    else:
        cb07 = inhouse.copy()
        print(f"[CB07] obs[{args.cb07_id_key!r}] not found; using all {cb07.n_obs:,} cells")

    cb07.obs[cfg.DATASET_KEY] = args.cb07_id_value
    cb07.obs[cfg.ASSAY_KEY] = args.cb07_assay
    cb07.obs["cell_type"] = "natural killer cell"
    cb07.obs["tissue"] = args.cb07_tissue
    cb07.obs["tissue_general"] = args.cb07_tissue

    if args.cb07_label_key in cb07.obs:
        labels = cb07.obs[args.cb07_label_key].astype(str).replace({"nan": cfg.UNLABELED_CATEGORY, "": cfg.UNLABELED_CATEGORY})
    else:
        labels = pd.Series(cfg.UNLABELED_CATEGORY, index=cb07.obs_names)
        print(f"[WARN] CB07 label key {args.cb07_label_key!r} not found; using {cfg.UNLABELED_CATEGORY}")
    cb07.obs[cfg.LABEL_KEY] = labels.astype(str).values

    print(f"[CB07] {cb07.n_obs:,} cells x {cb07.n_vars:,} genes")
    print("[CB07] NK_State counts:")
    print(cb07.obs[cfg.LABEL_KEY].astype(str).value_counts().head(30).to_string())
    return cb07


def standardize_var_names(adata: sc.AnnData) -> None:
    if "feature_name" in adata.var.columns:
        adata.var["feature_name"] = adata.var["feature_name"].astype(str)
        adata.var_names = adata.var["feature_name"].values
    elif "gene_symbol" in adata.var.columns:
        adata.var_names = adata.var["gene_symbol"].astype(str).values
    elif "gene_name" in adata.var.columns:
        adata.var_names = adata.var["gene_name"].astype(str).values
    adata.var_names_make_unique()
    adata.obs_names_make_unique()


def set_raw_counts_x(adata: sc.AnnData) -> None:
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    else:
        print("[WARN] No counts layer found; using current .X as raw counts")
        adata.layers["counts"] = adata.X.copy()


def ensure_lognorm_layer(adata: sc.AnnData) -> None:
    if "lognorm" not in adata.layers:
        print("[WARN] No lognorm layer found; creating lognorm from counts")
        tmp = adata.copy()
        sc.pp.normalize_total(tmp, target_sum=1e4)
        sc.pp.log1p(tmp)
        adata.layers["lognorm"] = tmp.X.copy()


def make_cellxgene_labels(obs: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    cell_type = obs["cell_type"].astype(str)
    labels = pd.Series(cfg.UNLABELED_CATEGORY, index=obs.index, dtype="object")
    labels[cell_type.eq("B cell")] = "B"
    labels[cell_type.eq("T cell")] = "T"

    nk_mask = ~cell_type.isin(["B cell", "T cell"])
    primary = label_column(obs, args.cellxgene_nk_label_key, nk_mask)
    fallback = label_column(obs, args.cellxgene_nk_label_fallback, nk_mask)
    nk_labels = primary.where(primary.notna() & primary.ne("") & primary.ne("nan"), fallback)
    nk_labels = nk_labels.fillna(cfg.UNLABELED_CATEGORY).replace({"nan": cfg.UNLABELED_CATEGORY, "": cfg.UNLABELED_CATEGORY})
    labels[nk_mask] = nk_labels.astype(str)
    return labels.astype(str)


def label_column(obs: pd.DataFrame, key: str, mask: pd.Series) -> pd.Series:
    if key not in obs.columns:
        return pd.Series(pd.NA, index=obs.index)
    values = obs[key].astype(str)
    values = values.where(mask, pd.NA)
    return values


def require_obs(obs: pd.DataFrame, key: str) -> pd.Series:
    if key not in obs.columns:
        raise KeyError(f"Required obs column {key!r} is missing.")
    return obs[key]


def clean_assay(values: pd.Series, *, flex_fill: str) -> pd.Series:
    out = values.astype(str).replace({"nan": flex_fill, "None": flex_fill, "": flex_fill})
    return out.astype(str)


def add_batch_composite_columns(obs: pd.DataFrame) -> None:
    """Add alternative batch keys for later batch-strategy comparisons."""
    dataset = obs[cfg.DATASET_KEY].astype(str)
    assay = obs[cfg.ASSAY_CLEAN_KEY].astype(str)
    if "tissue" in obs:
        tissue = obs["tissue"].astype(str).replace({"nan": "unknown", "None": "unknown", "": "unknown"})
    else:
        tissue = pd.Series("unknown", index=obs.index)

    obs[cfg.COMPOSITE_BATCH_KEY] = dataset + " || " + assay
    obs["batch_tissue_assay"] = tissue + " || " + assay
    obs["batch_dataset_tissue_assay"] = dataset + " || " + tissue + " || " + assay


def subset_hvgs(adata: sc.AnnData, args: argparse.Namespace) -> sc.AnnData:
    print(f"\n[HVG] selecting {args.n_hvg:,} genes")
    if args.hvg_source == "existing" and "highly_variable" in adata.var.columns:
        hvg = adata.var["highly_variable"].astype(bool).values
        n_existing = int(hvg.sum())
        print(f"[HVG] input highly_variable genes after gene intersection: {n_existing:,}")
        if n_existing >= args.n_hvg:
            score = hvg_rank_score(adata.var)
            order = np.lexsort((np.arange(adata.n_vars), -score, ~hvg))
            selected = order[: args.n_hvg]
            genes = adata.var_names[selected]
            genes = augment_hvg_genes_with_rescue_markers(adata.var_names, genes, args)
            out = adata[:, genes].copy()
            annotate_selected_var(out, selected_genes=genes, args=args)
            save_hvg_list(out.var_names, args.qc_outdir)
            print(f"[HVG] selected {out.n_vars:,} existing HVGs")
            return out
        print("[HVG] not enough existing HVGs after intersection; recomputing")

    tmp = adata.copy()
    tmp.X = tmp.layers["lognorm"].copy() if "lognorm" in tmp.layers else tmp.X
    batch_key = None if str(args.hvg_batch_key).lower() == "none" else args.hvg_batch_key
    if batch_key is not None and batch_key not in tmp.obs:
        raise KeyError(f"--hvg-batch-key {batch_key!r} not found in adata.obs")
    print(f"[HVG] recomputing from lognorm with batch_key={batch_key}")
    sc.pp.highly_variable_genes(
        tmp,
        n_top_genes=args.n_hvg,
        batch_key=batch_key,
        subset=False,
    )
    hvg = tmp.var["highly_variable"].astype(bool).values
    genes = tmp.var_names[hvg]
    genes = augment_hvg_genes_with_rescue_markers(adata.var_names, genes, args)
    out = adata[:, genes].copy()
    annotate_selected_var(out, selected_genes=genes, args=args)
    save_hvg_list(out.var_names, args.qc_outdir)
    print(f"[HVG] selected {out.n_vars:,} recomputed HVGs")
    return out


def augment_hvg_genes_with_rescue_markers(
    all_genes: pd.Index,
    hvg_genes: Iterable[str],
    args: argparse.Namespace,
) -> list[str]:
    hvg_list = list(map(str, hvg_genes))
    rescue = read_gene_list(args.marker_rescue_file)
    if not rescue:
        save_rescue_report([], [], [], args.qc_outdir)
        return hvg_list

    all_gene_set = set(map(str, all_genes))
    hvg_set = set(hvg_list)
    already_selected = [gene for gene in rescue if gene in hvg_set]
    added = [gene for gene in rescue if gene in all_gene_set and gene not in hvg_set]
    missing = [gene for gene in rescue if gene not in all_gene_set]
    final = hvg_list + added

    print(f"[RESCUE] marker genes in file: {len(rescue):,}")
    print(f"[RESCUE] already selected as HVG: {len(already_selected):,}")
    print(f"[RESCUE] appended after HVG: {len(added):,}")
    print(f"[RESCUE] missing from data: {len(missing):,}")
    if added:
        print("[RESCUE] added:", "; ".join(added))
    if missing:
        print("[RESCUE] missing:", "; ".join(missing))
    save_rescue_report(already_selected, added, missing, args.qc_outdir)
    return final


def read_gene_list(path: str | None) -> list[str]:
    if not path:
        return []
    if not os.path.exists(path):
        print(f"[WARN] marker rescue file not found: {path}")
        return []
    genes: list[str] = []
    seen: set[str] = set()
    with open(path) as handle:
        for line in handle:
            gene = line.strip()
            if not gene or gene.startswith("#"):
                continue
            gene = gene.split()[0].strip()
            if gene and gene not in seen:
                genes.append(gene)
                seen.add(gene)
    return genes


def save_rescue_report(
    already_selected: list[str],
    added: list[str],
    missing: list[str],
    outdir: str,
) -> None:
    rows = []
    for status, genes in [
        ("already_hvg", already_selected),
        ("rescued_added", added),
        ("missing_from_data", missing),
    ]:
        rows.extend({"gene": gene, "status": status} for gene in genes)
    path = os.path.join(outdir, "hvg_marker_rescue_report.csv")
    pd.DataFrame(rows, columns=["gene", "status"]).to_csv(path, index=False)
    print(f"[SAVE] {path}")


def annotate_selected_var(out: sc.AnnData, *, selected_genes: Iterable[str], args: argparse.Namespace) -> None:
    rescue = set(read_gene_list(args.marker_rescue_file))
    selected = set(map(str, selected_genes))
    out.var["selected_for_model"] = out.var_names.astype(str).isin(selected)
    out.var["canonical_nk_marker_rescue_candidate"] = out.var_names.astype(str).isin(rescue)
    out.var["selected_gene_source"] = np.where(
        out.var["canonical_nk_marker_rescue_candidate"].values,
        "hvg_or_rescue_marker",
        "hvg",
    )


def hvg_rank_score(var: pd.DataFrame) -> np.ndarray:
    score = np.zeros(var.shape[0], dtype=float)
    if "highly_variable_nbatches" in var:
        score += pd.to_numeric(var["highly_variable_nbatches"], errors="coerce").fillna(0).to_numpy() * 1e6
    if "dispersions_norm" in var:
        score += pd.to_numeric(var["dispersions_norm"], errors="coerce").fillna(0).to_numpy()
    elif "dispersions" in var:
        score += pd.to_numeric(var["dispersions"], errors="coerce").fillna(0).to_numpy()
    return score


def save_hvg_list(genes: Iterable[str], outdir: str) -> None:
    path = os.path.join(outdir, "hvg_genes.txt")
    with open(path, "w") as handle:
        for gene in genes:
            handle.write(str(gene) + "\n")
    print(f"[SAVE] {path}")


def ensure_integer_like_counts(X, name: str, n_values: int = 10000) -> None:
    if sparse.issparse(X):
        vals = X.data[:n_values]
    else:
        vals = np.asarray(X).ravel()[:n_values]
    vals = vals[np.isfinite(vals)]
    if vals.size and not np.allclose(vals, np.round(vals)):
        print(f"[WARN] {name} sample is not integer-like. Confirm this is raw counts before SCVI training.")


def write_qc_tables(adata: sc.AnnData, outdir: str, *, prefix: str) -> None:
    print(f"\n[QC TABLES] {prefix}")
    tables = {
        "label_counts": adata.obs[cfg.LABEL_KEY].astype(str).value_counts(),
        "dataset_counts": adata.obs[cfg.DATASET_KEY].astype(str).value_counts(),
        "assay_counts": adata.obs[cfg.ASSAY_KEY].astype(str).value_counts() if cfg.ASSAY_KEY in adata.obs else pd.Series(dtype=int),
        "assay_clean_counts": adata.obs[cfg.ASSAY_CLEAN_KEY].astype(str).value_counts()
        if cfg.ASSAY_CLEAN_KEY in adata.obs
        else pd.Series(dtype=int),
        "batch_composite_counts": adata.obs[cfg.COMPOSITE_BATCH_KEY].astype(str).value_counts()
        if cfg.COMPOSITE_BATCH_KEY in adata.obs
        else pd.Series(dtype=int),
        "batch_tissue_assay_counts": adata.obs["batch_tissue_assay"].astype(str).value_counts()
        if "batch_tissue_assay" in adata.obs
        else pd.Series(dtype=int),
        "batch_dataset_tissue_assay_counts": adata.obs["batch_dataset_tissue_assay"].astype(str).value_counts()
        if "batch_dataset_tissue_assay" in adata.obs
        else pd.Series(dtype=int),
        "tissue_counts": adata.obs["tissue"].astype(str).value_counts() if "tissue" in adata.obs else pd.Series(dtype=int),
        "source_panel_counts": adata.obs["source_panel"].astype(str).value_counts()
        if "source_panel" in adata.obs
        else pd.Series(dtype=int),
    }
    for name, series in tables.items():
        path = os.path.join(outdir, f"{prefix}_{name}.csv")
        series.rename("n_cells").to_csv(path)
        print(f"[SAVE] {path}")

    if cfg.DATASET_KEY in adata.obs and cfg.LABEL_KEY in adata.obs:
        path = os.path.join(outdir, f"{prefix}_label_by_dataset_counts.csv")
        pd.crosstab(adata.obs[cfg.DATASET_KEY].astype(str), adata.obs[cfg.LABEL_KEY].astype(str)).to_csv(path)
        print(f"[SAVE] {path}")


if __name__ == "__main__":
    main()
