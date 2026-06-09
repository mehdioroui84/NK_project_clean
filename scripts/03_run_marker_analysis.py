#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


DEFAULT_GROUPBY = "leiden_0_4"
N_TOP_TABLE = 50
N_TOP_PLOT_PER_CLUSTER = 3
DEFAULT_MARKER_FDR = 0.02
DEFAULT_MARKER_LOGFC = 0.50
DEFAULT_MARKER_PCT_DIFF = 0.10
DEFAULT_MARKER_MIN_PCT = 0.10
EXPRESSION_CMAP = "Reds"
DEFAULT_SCVI_DE_MODEL_NAME = "scvi_assay_clean_model"

MARKER_SETS = {
    "NK_cytotoxic": [
        "NKG7",
        "GNLY",
        "PRF1",
        "GZMB",
        "GZMH",
        "GZMA",
        "CST7",
        "FGFBP2",
        "KLRF1",
        "FCGR3A",
    ],
    "NK_regulatory_tissue": [
        "XCL1",
        "XCL2",
        "KLRC1",
        "KLRC2",
        "KLRB1",
        "CXCR6",
        "ITGAE",
        "ZNF683",
        "CCL3",
        "CCL4",
        "CCL5",
    ],
    "proliferation": [
        "MKI67",
        "TOP2A",
        "STMN1",
        "TYMS",
        "RRM2",
        "TK1",
        "PCNA",
        "PCLAF",
        "NUSAP1",
    ],
    "interferon_cytokine": [
        "ISG15",
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "IFI44L",
        "MX1",
        "STAT1",
        "IRF7",
        "IL2RA",
        "IL7R",
        "CCR7",
        "IRF4",
    ],
    "T_cell": [
        "CD3D",
        "CD3E",
        "CD3G",
        "TRAC",
        "IL7R",
        "TCF7",
        "SELL",
        "LEF1",
    ],
    "B_cell": [
        "MS4A1",
        "CD79A",
        "CD79B",
        "BANK1",
        "BLK",
        "FCRL1",
        "IGHM",
        "IGKC",
    ],
    "myeloid": [
        "LYZ",
        "LST1",
        "S100A8",
        "S100A9",
        "C5AR1",
        "CLEC7A",
        "MS4A7",
        "FCGR3A",
        "MAFB",
    ],
    "epithelial_lung": [
        "EPCAM",
        "KRT8",
        "KRT18",
        "KRT19",
        "KRT81",
        "KRT86",
        "SCGB1A1",
        "SCGB3A1",
        "SCGB3A2",
        "SFTPC",
    ],
    "erythroid": [
        "HBB",
        "HBA1",
        "HBA2",
        "HBD",
        "HBM",
        "AHSP",
    ],
    "stress_mito": [
        "HSPA1A",
        "HSPA1B",
        "HSPA6",
        "DNAJB1",
        "MT-CO1",
        "MT-CO2",
        "MT-CO3",
        "MT-ND5",
        "MT-CYB",
    ],
}


def main():
    args = parse_args()
    in_path = args.input_h5ad or os.path.join(
        cfg.BASE_OUTDIR,
        "leiden_discovery",
        "full_scvi_leiden.h5ad",
    )

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if args.groupby not in adata.obs:
        raise KeyError(f"{args.groupby!r} not found in adata.obs")

    adata.obs[args.groupby] = adata.obs[args.groupby].astype(str).astype("category")
    print(
        f"[MARKERS] full data; groupby={args.groupby}; "
        f"clusters={adata.obs[args.groupby].nunique()}"
    )

    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    if args.de_method == "both":
        parent_outdir = args.outdir or os.path.join(cfg.BASE_OUTDIR, "markers", "full")
        for de_method in ["scvi", "scanpy"]:
            method_args = copy.copy(args)
            method_args.de_method = de_method
            outdir = method_output_dir(parent_outdir, args.groupby, de_method, method_args)
            run_marker_workflow(adata, ad, method_args, outdir)
    else:
        outdir = args.outdir or os.path.join(cfg.BASE_OUTDIR, "markers", "full", args.groupby)
        run_marker_workflow(adata, ad, args, outdir)

    plt.close("all")
    print("[DONE] Full-data marker analysis complete.")


def run_marker_workflow(adata, ad, args, outdir):
    ensure_dirs(outdir)
    print(f"[OUTDIR] {outdir}")

    summary = cluster_summary(adata, args.groupby)
    summary_path = os.path.join(outdir, f"{args.groupby}_cluster_summary.csv")
    summary.to_csv(summary_path)
    print(f"[SAVE] {summary_path}")

    if not args.skip_rank_genes:
        run_rank_genes(ad, adata, args.groupby, outdir, args)
    else:
        print("[SKIP] rank_genes_groups marker analysis")

    if not args.skip_curated_markers:
        plot_curated_markers(
            ad,
            args.groupby,
            outdir,
            marker_csv=args.curated_marker_csv,
        )
    else:
        print("[SKIP] curated marker plots")

    plt.close("all")


def method_output_dir(parent_outdir, groupby, de_method, args):
    method = "scvi_de" if de_method == "scvi" else "scanpy_wilcoxon"
    if de_method == "scvi":
        method = f"{method}_{args.scvi_de_cells}"
    return os.path.join(parent_outdir, f"{groupby}_{method}_{marker_filter_tag(args)}")


def marker_filter_tag(args):
    fdr = decimal_tag(args.marker_fdr)
    logfc = decimal_tag(args.marker_logfc)
    marker_pct_diff = parse_optional_float(args.marker_pct_diff, name="--marker-pct-diff")
    pct_tag = "no_delta_pct" if marker_pct_diff is None else f"delta{decimal_tag(marker_pct_diff)}"
    min_pct = decimal_tag(args.marker_min_pct)
    return f"fdr{fdr}_logfc{logfc}_minpct{min_pct}_{pct_tag}"


def decimal_tag(value):
    text = f"{float(value):g}"
    if "." not in text:
        return text.replace("-", "m")
    whole, frac = text.split(".", 1)
    return f"{whole}{frac}".replace("-", "m")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run full-data marker analysis for refined NK annotation discovery. "
            "This combines cluster-vs-rest Wilcoxon markers and curated marker plots."
        )
    )
    parser.add_argument("--input-h5ad", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--groupby", default=DEFAULT_GROUPBY)
    parser.add_argument("--skip-rank-genes", action="store_true")
    parser.add_argument("--skip-curated-markers", action="store_true")
    parser.add_argument(
        "--curated-marker-csv",
        default=None,
        help=(
            "Optional expanded curated marker CSV with at least gene_name and "
            "marker_set or functional_state columns. If omitted, use the legacy "
            "hardcoded marker programs."
        ),
    )
    parser.add_argument(
        "--marker-fdr",
        type=float,
        default=DEFAULT_MARKER_FDR,
        help="Adjusted p-value/FDR cutoff for per-cluster marker selection.",
    )
    parser.add_argument(
        "--marker-logfc",
        type=float,
        default=DEFAULT_MARKER_LOGFC,
        help="Minimum positive log fold-change for per-cluster marker selection.",
    )
    parser.add_argument(
        "--marker-pct-diff",
        default=str(DEFAULT_MARKER_PCT_DIFF),
        help=(
            "Minimum expression specificity for per-cluster marker selection. "
            "Use 'none' to remove the delta-percent filter entirely."
        ),
    )
    parser.add_argument(
        "--marker-min-pct",
        type=float,
        default=DEFAULT_MARKER_MIN_PCT,
        help=(
            "Minimum expression fraction on the relevant side: pct in cluster "
            "for positive markers, pct in reference for negative markers."
        ),
    )
    parser.add_argument(
        "--max-markers-per-cluster",
        type=int,
        default=N_TOP_TABLE,
        help="Maximum filtered marker genes kept per cluster per direction.",
    )
    parser.add_argument(
        "--marker-direction",
        choices=["both", "up", "down"],
        default="both",
        help=(
            "Direction of markers to include. 'both' keeps genes enriched in the cluster "
            "and genes depleted in the cluster; 'up' keeps only positive markers."
        ),
    )
    parser.add_argument(
        "--top-plot-markers-per-cluster",
        type=int,
        default=N_TOP_PLOT_PER_CLUSTER,
        help="Number of selected marker genes per cluster to include in dotplot/matrixplot.",
    )
    parser.add_argument(
        "--de-method",
        choices=["scvi", "scanpy", "both"],
        default="scvi",
        help=(
            "Marker engine. Default 'scvi' uses model.differential_expression(); "
            "'scanpy' preserves the old Wilcoxon workflow; 'both' writes separate "
            "scVI and Scanpy marker folders under --outdir."
        ),
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help=(
            "Trained scVI/scANVI model directory for --de-method scvi. "
            f"Default: outputs/models/{DEFAULT_SCVI_DE_MODEL_NAME}"
        ),
    )
    parser.add_argument(
        "--model-class",
        choices=["auto", "SCVI", "SCANVI"],
        default="auto",
        help="Model class to load for model-based DE. Default auto tries SCVI then SCANVI.",
    )
    parser.add_argument(
        "--train-names",
        default=None,
        help=(
            "Optional train_obs_names.txt used to load the model with training cells first "
            "if the full AnnData has held-out batch categories."
        ),
    )
    parser.add_argument(
        "--scvi-de-cells",
        choices=["train", "all"],
        default="train",
        help=(
            "Cells used for model-based DE. Default 'train' avoids held-out/unseen "
            "batch categories that the trained model cannot decode."
        ),
    )
    parser.add_argument(
        "--scvi-de-mode",
        choices=["change", "vanilla"],
        default="change",
        help="mode argument passed to model.differential_expression().",
    )
    parser.add_argument(
        "--scvi-delta",
        type=float,
        default=0.25,
        help="delta argument for scVI-tools DE when mode='change'.",
    )
    parser.add_argument(
        "--scvi-de-batch-size",
        type=int,
        default=2048,
        help=(
            "Batch size passed to model.differential_expression(). Larger values "
            "can speed up GPU DE if memory allows."
        ),
    )
    parser.add_argument(
        "--no-scvi-de-all-stats",
        dest="scvi_de_all_stats",
        action="store_false",
        help=(
            "Use a leaner scVI differential_expression() output. This can be faster, "
            "but rerun without this flag if your scVI version omits needed columns."
        ),
    )
    parser.add_argument(
        "--no-scvi-batch-correction",
        dest="scvi_batch_correction",
        action="store_false",
        help="Turn off scVI-tools DE batch_correction. Default is on.",
    )
    parser.set_defaults(scvi_batch_correction=True)
    parser.set_defaults(scvi_de_all_stats=True)
    return parser.parse_args()


def run_rank_genes(ad, raw_adata, groupby, outdir, args):
    marker_pct_diff = parse_optional_float(args.marker_pct_diff, name="--marker-pct-diff")
    if args.de_method == "scvi":
        all_markers = run_scvi_differential_expression(raw_adata, groupby, outdir, args)
    else:
        all_markers = run_scanpy_wilcoxon(ad, groupby, outdir)

    enriched_candidates = filter_enriched_marker_candidates(
        all_markers,
        fdr=args.marker_fdr,
        min_logfc=args.marker_logfc,
        min_pct_diff=marker_pct_diff,
        min_pct=args.marker_min_pct,
        direction=args.marker_direction,
    )
    filtered_markers = cap_markers_per_cluster_per_direction(enriched_candidates, args.max_markers_per_cluster)
    top_path = os.path.join(
        outdir,
        f"{groupby}_markers_top{args.max_markers_per_cluster}_pos_top{args.max_markers_per_cluster}_neg_per_cluster.csv",
    )
    compat_top_path = os.path.join(outdir, f"{groupby}_markers_top{args.max_markers_per_cluster}_per_cluster.csv")
    filtered_markers.to_csv(top_path, index=False)
    filtered_markers.to_csv(compat_top_path, index=False)
    print(f"[SAVE] {top_path}")
    print(f"[SAVE] {compat_top_path}")
    pct_diff_text = "disabled" if marker_pct_diff is None else f"|delta_pct|>={marker_pct_diff:g}"
    print(
        "[FILTER] marker selection: "
        f"DE method={args.de_method}, FDR/probability target={args.marker_fdr:g}, "
        f"|logFC|>={args.marker_logfc:g}, relevant pct>={args.marker_min_pct:g}, "
        f"{pct_diff_text}, direction={args.marker_direction}, "
        f"cap_per_direction={args.max_markers_per_cluster}"
    )
    save_marker_filter_summary(
        filtered_markers,
        enriched_candidates,
        all_markers["group"].astype(str).unique(),
        groupby,
        outdir,
        max_markers_per_cluster=max_selected_markers_per_cluster(args),
    )

    selected = select_plot_markers(filtered_markers, n_per_cluster=args.top_plot_markers_per_cluster)
    selected_path = os.path.join(outdir, f"{groupby}_selected_plot_markers.txt")
    pd.Series(selected, name="gene").to_csv(selected_path, index=False, header=False)
    print(f"[SAVE] {selected_path}")

    if selected:
        save_dotplot(ad, selected, groupby, outdir)
        save_matrixplot(ad, selected, groupby, outdir)
    else:
        print("[WARN] No selected markers passed filtering; skipping top-marker plots.")


def parse_optional_float(value, *, name):
    text = str(value).strip().lower()
    if text in {"none", "off", "false", "no", "disabled"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number or 'none', got {value!r}") from exc


def run_scanpy_wilcoxon(ad, groupby, outdir):
    print("[DE] Running Scanpy Wilcoxon cluster-vs-rest marker analysis...")
    sc.tl.rank_genes_groups(
        ad,
        groupby=groupby,
        method="wilcoxon",
        pts=True,
        tie_correct=True,
    )

    all_markers = sc.get.rank_genes_groups_df(ad, group=None)
    all_path = os.path.join(outdir, f"{groupby}_markers_all_wilcoxon.csv")
    all_markers.to_csv(all_path, index=False)
    print(f"[SAVE] {all_path}")
    return all_markers


def run_scvi_differential_expression(adata, groupby, outdir, args):
    model_dir = args.model_dir or os.path.join(cfg.BASE_OUTDIR, "models", DEFAULT_SCVI_DE_MODEL_NAME)
    print("[DE] Running model-based scVI/scANVI cluster-vs-rest differential expression...")
    print(f"[DE_MODEL] {model_dir}")
    train_names = args.train_names or os.path.join(cfg.TABLE_OUTDIR, "train_obs_names.txt")
    de_adata = subset_de_adata(adata, train_names, mode=args.scvi_de_cells)
    model = load_scvi_de_model(model_dir, de_adata, args.model_class)
    de = model.differential_expression(
        adata=de_adata,
        groupby=groupby,
        mode=args.scvi_de_mode,
        delta=args.scvi_delta,
        batch_correction=args.scvi_batch_correction,
        batch_size=args.scvi_de_batch_size,
        fdr_target=args.marker_fdr,
        all_stats=args.scvi_de_all_stats,
        silent=False,
    )
    all_markers = standardize_scvi_de_table(de)
    all_markers = add_pct_nz_columns(de_adata, all_markers, groupby)
    all_path = os.path.join(outdir, f"{groupby}_markers_all_scvi_de.csv")
    compat_all_path = os.path.join(outdir, f"{groupby}_markers_all_wilcoxon.csv")
    all_markers.to_csv(all_path, index=False)
    all_markers.to_csv(compat_all_path, index=False)
    print(f"[SAVE] {all_path}")
    print(f"[SAVE] {compat_all_path}  # compatibility copy")
    return all_markers


def subset_de_adata(adata, train_names, mode):
    if mode == "all":
        print(f"[DE_CELLS] all cells: {adata.n_obs:,}")
        return adata
    if not train_names or not os.path.exists(train_names):
        raise FileNotFoundError(
            f"--scvi-de-cells train requires train names file, not found: {train_names}"
        )
    names = pd.read_csv(train_names, header=None)[0].astype(str)
    common = adata.obs_names.astype(str).intersection(names)
    if len(common) == 0:
        raise ValueError(f"No train_obs_names overlap with input AnnData: {train_names}")
    de_adata = adata[common].copy()
    print(f"[DE_CELLS] train cells only: {adata.n_obs:,} -> {de_adata.n_obs:,}")
    return de_adata


def load_scvi_de_model(model_dir, adata, model_class, train_names=None):
    if not os.path.exists(model_dir):
        raise FileNotFoundError(model_dir)
    import scvi
    import scarches as sca

    candidates = [model_class] if model_class != "auto" else ["SCVI", "SCANVI"]
    last_error = None
    for candidate in candidates:
        if candidate == "SCVI":
            model_classes = [scvi.model.SCVI, getattr(sca.models, "SCVI", None)]
        else:
            model_classes = [getattr(sca.models, candidate)]
        model_classes = [cls for cls in model_classes if cls is not None]
        for cls in model_classes:
            try:
                print(f"[DE_MODEL_LOAD] trying {cls.__module__}.{cls.__name__} with full AnnData")
                return cls.load(model_dir, adata=adata)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                print(f"[DE_MODEL_LOAD_WARN] {cls.__module__}.{cls.__name__} failed: {exc}")
            if train_names and os.path.exists(train_names):
                try:
                    print(f"[DE_MODEL_LOAD] retrying with train cells from {train_names}")
                    names = pd.read_csv(train_names, header=None)[0].astype(str)
                    common = adata.obs_names.astype(str).intersection(names)
                    if len(common) == 0:
                        raise ValueError("No train_obs_names overlap with input AnnData.")
                    ref_model = adata[common].copy()
                    model = cls.load(model_dir, adata=ref_model)
                    cls.prepare_query_anndata(adata, model)
                    manager = model.adata_manager.transfer_fields(adata, extend_categories=True)
                    model._register_manager_for_instance(manager)
                    return model
                except Exception as exc:  # noqa: BLE001
                    last_error = exc
                    print(f"[DE_MODEL_LOAD_WARN] train-cell fallback failed: {exc}")
    raise RuntimeError(f"Could not load model from {model_dir}") from last_error


def standardize_scvi_de_table(de):
    df = de.reset_index().copy()
    if "index" in df.columns and "names" not in df.columns:
        df = df.rename(columns={"index": "names"})
    if "gene" in df.columns and "names" not in df.columns:
        df = df.rename(columns={"gene": "names"})
    if "group1" in df.columns:
        df["group"] = df["group1"].astype(str)
    elif "comparison" in df.columns:
        df["group"] = df["comparison"].astype(str).str.split(" vs ").str[0]
    elif "group" not in df.columns:
        raise KeyError("Could not infer group column from scVI differential_expression output.")

    if "lfc_mean" in df.columns:
        df["logfoldchanges"] = pd.to_numeric(df["lfc_mean"], errors="coerce")
    elif "logfoldchanges" not in df.columns:
        raise KeyError("Could not find lfc_mean/logfoldchanges in scVI DE output.")

    if "bayes_factor" in df.columns:
        df["scores"] = pd.to_numeric(df["bayes_factor"], errors="coerce")
    elif "proba_de" in df.columns:
        df["scores"] = pd.to_numeric(df["proba_de"], errors="coerce")
    else:
        df["scores"] = np.nan

    if "proba_de" in df.columns:
        df["pvals_adj"] = 1.0 - pd.to_numeric(df["proba_de"], errors="coerce")
    elif "pvals_adj" not in df.columns:
        df["pvals_adj"] = np.nan

    fdr_cols = [col for col in df.columns if col.startswith("is_de_fdr_")]
    if fdr_cols:
        df["is_de_fdr"] = df[fdr_cols[0]].astype(bool)

    required = ["group", "names", "scores", "logfoldchanges", "pvals_adj"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"Standardized scVI DE table is missing columns: {missing}")
    return df


def add_pct_nz_columns(adata, markers, groupby):
    markers = markers.copy()
    markers["pct_nz_group"] = np.nan
    markers["pct_nz_reference"] = np.nan
    obs_groups = adata.obs[groupby].astype(str)
    var_names = pd.Index(adata.var_names.astype(str))

    for group, idx in markers.groupby("group").groups.items():
        group = str(group)
        group_mask = obs_groups.eq(group).to_numpy()
        ref_mask = ~group_mask
        genes = markers.loc[idx, "names"].astype(str).tolist()
        present = [gene for gene in genes if gene in var_names]
        if not present or not np.any(group_mask) or not np.any(ref_mask):
            continue
        gene_pos = var_names.get_indexer(present)
        x_group = adata.X[group_mask][:, gene_pos]
        x_ref = adata.X[ref_mask][:, gene_pos]
        pct_group = nonzero_fraction(x_group)
        pct_ref = nonzero_fraction(x_ref)
        pct_by_gene = {
            gene: (float(pct_group[i]), float(pct_ref[i]))
            for i, gene in enumerate(present)
        }
        row_positions = markers.index[list(idx)]
        for row_idx in row_positions:
            gene = str(markers.at[row_idx, "names"])
            if gene not in pct_by_gene:
                continue
            markers.at[row_idx, "pct_nz_group"] = pct_by_gene[gene][0]
            markers.at[row_idx, "pct_nz_reference"] = pct_by_gene[gene][1]
    return markers


def nonzero_fraction(x):
    if sparse.issparse(x):
        return np.asarray((x > 0).mean(axis=0)).ravel()
    return np.asarray((x > 0).mean(axis=0)).ravel()


def plot_curated_markers(ad, groupby, outdir, *, marker_csv=None):
    if marker_csv:
        print(f"[CURATED_MARKERS] Loading expanded marker CSV: {marker_csv}")
        marker_definitions = load_expanded_curated_markers(marker_csv)
        marker_sets = marker_sets_from_expanded_markers(marker_definitions)
    else:
        marker_definitions = None
        marker_sets = MARKER_SETS

    present_sets = {
        name: [gene for gene in genes if gene in ad.var_names]
        for name, genes in marker_sets.items()
    }
    present_sets = {name: unique_preserve_order(genes) for name, genes in present_sets.items() if genes}

    marker_list = []
    marker_rows = []
    for set_name, genes in present_sets.items():
        for gene in genes:
            if gene not in marker_list:
                marker_list.append(gene)
            marker_rows.append({"marker_set": set_name, "gene": gene})

    marker_path = os.path.join(outdir, f"{groupby}_curated_marker_genes_present.csv")
    marker_rows_df = pd.DataFrame(marker_rows)
    if marker_definitions is not None and not marker_rows_df.empty:
        marker_rows_df = marker_rows_df.merge(
            marker_definitions,
            left_on=["marker_set", "gene"],
            right_on=["marker_set", "gene_name"],
            how="left",
        )
        marker_rows_df = marker_rows_df.drop(columns=["gene_name"], errors="ignore")
    marker_rows_df.to_csv(marker_path, index=False)
    print(f"[SAVE] {marker_path}")

    print(f"[PLOT] Full-data curated dotplot with {len(marker_list)} genes")
    dot = sc.pl.dotplot(
        ad,
        var_names=present_sets,
        groupby=groupby,
        standard_scale="var",
        cmap=EXPRESSION_CMAP,
        show=False,
        return_fig=True,
    )
    dot_path = os.path.join(outdir, f"{groupby}_curated_marker_dotplot.png")
    dot.savefig(dot_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {dot_path}")
    plt.close("all")

    print("[PLOT] Full-data curated matrixplot")
    matrix = sc.pl.matrixplot(
        ad,
        var_names=present_sets,
        groupby=groupby,
        standard_scale="var",
        cmap=EXPRESSION_CMAP,
        show=False,
        return_fig=True,
    )
    matrix_path = os.path.join(outdir, f"{groupby}_curated_marker_matrixplot.png")
    matrix.savefig(matrix_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {matrix_path}")
    plt.close("all")

    print("[SUMMARY] Computing average expression by cluster for curated markers")
    expr = ad[:, marker_list].to_df()
    expr[groupby] = ad.obs[groupby].astype(str).values
    avg = expr.groupby(groupby).mean()
    avg_path = os.path.join(outdir, f"{groupby}_curated_marker_cluster_means.csv")
    avg.to_csv(avg_path)
    print(f"[SAVE] {avg_path}")


def load_expanded_curated_markers(marker_csv):
    markers = pd.read_csv(marker_csv, dtype=str).fillna("")
    if "gene_name" not in markers.columns:
        raise KeyError(f"{marker_csv} must contain a 'gene_name' column.")
    if "marker_set" not in markers.columns:
        if "functional_state" not in markers.columns:
            raise KeyError(f"{marker_csv} must contain 'marker_set' or 'functional_state'.")
        markers["marker_set"] = markers["functional_state"]
    markers["gene_name"] = markers["gene_name"].astype(str).str.strip()
    markers["marker_set"] = markers["marker_set"].astype(str).str.strip()
    markers = markers[(markers["gene_name"] != "") & (markers["marker_set"] != "")].copy()
    if markers.empty:
        raise ValueError(f"{marker_csv} did not contain any usable curated marker rows.")
    return markers


def marker_sets_from_expanded_markers(markers):
    marker_sets = {}
    for marker_set, group in markers.groupby("marker_set", sort=False):
        marker_sets[str(marker_set)] = unique_preserve_order(group["gene_name"].astype(str).tolist())
    return marker_sets


def unique_preserve_order(values):
    seen = set()
    out = []
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def cluster_summary(adata, groupby):
    out = pd.DataFrame({"n_cells": adata.obs[groupby].value_counts().sort_index()})
    for col in [cfg.LABEL_KEY, cfg.DATASET_KEY, cfg.ASSAY_CLEAN_KEY, "tissue"]:
        if col not in adata.obs:
            continue
        tab = pd.crosstab(adata.obs[groupby].astype(str), adata.obs[col].astype(str))
        total = tab.sum(axis=1)
        out[f"top_{col}"] = tab.idxmax(axis=1)
        out[f"top_{col}_frac"] = tab.max(axis=1) / total
    return out.sort_values("n_cells", ascending=False)


def filter_enriched_marker_candidates(all_markers, fdr, min_logfc, min_pct_diff, min_pct, direction):
    markers = all_markers.copy()
    markers["pct_expr_group"], markers["pct_expr_reference"] = marker_pct_columns(markers)
    markers["pct_expr_diff"] = markers["pct_expr_group"] - markers["pct_expr_reference"]

    for col in ["pvals_adj", "logfoldchanges", "pct_expr_group", "pct_expr_reference", "pct_expr_diff", "scores"]:
        if col in markers.columns:
            markers[col] = pd.to_numeric(markers[col], errors="coerce")
    markers["abs_logfoldchanges"] = markers["logfoldchanges"].abs()
    markers["abs_pct_expr_diff"] = markers["pct_expr_diff"].abs()

    required = ["group", "pvals_adj", "logfoldchanges", "pct_expr_group", "pct_expr_reference", "pct_expr_diff"]
    missing = [col for col in required if col not in markers.columns]
    if missing:
        raise KeyError(f"Missing required marker columns for filtering: {missing}")

    up = (markers["logfoldchanges"] >= min_logfc) & (markers["pct_expr_group"] >= min_pct)
    down = (markers["logfoldchanges"] <= -min_logfc) & (markers["pct_expr_reference"] >= min_pct)
    if min_pct_diff is not None:
        up = up & (markers["pct_expr_diff"] >= min_pct_diff)
        down = down & (markers["pct_expr_diff"] <= -min_pct_diff)
    if direction == "up":
        direction_mask = up
    elif direction == "down":
        direction_mask = down
    else:
        direction_mask = up | down

    significance_mask = marker_significance_mask(markers, fdr)

    candidates = markers[significance_mask & direction_mask].copy()
    candidates["marker_direction"] = np.where(candidates["logfoldchanges"] >= 0, "up", "down")

    return sort_marker_candidates(candidates).reset_index(drop=True)


def marker_significance_mask(markers, fdr):
    if "is_de_fdr" not in markers.columns:
        return markers["pvals_adj"] < fdr

    values = markers["is_de_fdr"].fillna(False)
    if pd.api.types.is_bool_dtype(values):
        return values
    if pd.api.types.is_numeric_dtype(values):
        return values.astype(float) != 0
    return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y"})


def sort_marker_candidates(markers):
    markers = markers.copy()
    if "abs_logfoldchanges" not in markers.columns and "logfoldchanges" in markers.columns:
        markers["abs_logfoldchanges"] = markers["logfoldchanges"].abs()
    if "abs_pct_expr_diff" not in markers.columns and "pct_expr_diff" in markers.columns:
        markers["abs_pct_expr_diff"] = markers["pct_expr_diff"].abs()
    if "abs_scores" not in markers.columns and "scores" in markers.columns:
        markers["abs_scores"] = markers["scores"].abs()

    sort_cols = ["group"]
    ascending = [True]
    if "marker_direction" in markers.columns:
        sort_cols.append("marker_direction")
        ascending.append(True)
    sort_cols.extend(["abs_logfoldchanges", "abs_pct_expr_diff", "pvals_adj"])
    ascending.extend([False, False, True])
    if "scores" in markers.columns:
        sort_cols.append("abs_scores")
        ascending.append(False)
    return markers.sort_values(sort_cols, ascending=ascending)


def cap_markers_per_cluster_per_direction(markers, max_markers_per_direction):
    return (
        sort_marker_candidates(markers)
        .groupby(["group", "marker_direction"], group_keys=False)
        .head(max_markers_per_direction)
        .reset_index(drop=True)
    )


def max_selected_markers_per_cluster(args):
    if args.marker_direction == "both":
        return args.max_markers_per_cluster * 2
    return args.max_markers_per_cluster


def marker_pct_columns(markers):
    pct_pairs = [
        ("pct_nz_group", "pct_nz_reference"),
        ("pct.1", "pct.2"),
        ("pts", "pts_rest"),
    ]
    for group_col, ref_col in pct_pairs:
        if group_col in markers.columns and ref_col in markers.columns:
            return (
                pd.to_numeric(markers[group_col], errors="coerce"),
                pd.to_numeric(markers[ref_col], errors="coerce"),
            )

    raise KeyError(
        "Could not find percent-expression columns. Expected one of: "
        "pct_nz_group/pct_nz_reference, pct.1/pct.2, or pts/pts_rest."
    )


def save_marker_filter_summary(
    filtered_markers,
    enriched_candidates,
    all_groups,
    groupby,
    outdir,
    max_markers_per_cluster,
):
    group_index = pd.Index(sorted(map(str, all_groups), key=cluster_sort_key), name="group")
    passing = enriched_candidates.groupby("group").size().rename("n_passing_filter")
    selected = filtered_markers.groupby("group").size().rename("n_selected_markers")
    summary = pd.concat([passing, selected], axis=1).reindex(group_index, fill_value=0).reset_index()
    up_selected = (
        filtered_markers.loc[filtered_markers["marker_direction"] == "up"]
        .groupby("group")
        .size()
        .rename("n_up_selected_markers")
    )
    down_selected = (
        filtered_markers.loc[filtered_markers["marker_direction"] == "down"]
        .groupby("group")
        .size()
        .rename("n_down_selected_markers")
    )
    summary = (
        summary.set_index("group")
        .join(up_selected)
        .join(down_selected)
        .fillna(0)
        .reset_index()
    )
    for col in ["n_up_selected_markers", "n_down_selected_markers"]:
        summary[col] = summary[col].astype(int)
    summary["capped_at_max"] = summary["n_passing_filter"] > summary["n_selected_markers"]
    summary_path = os.path.join(outdir, f"{groupby}_filtered_marker_selection_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[SAVE] {summary_path}")
    if summary.empty:
        print("[WARN] Marker filter selected zero genes for all clusters.")
        return

    max_n = int(summary["n_selected_markers"].max())
    min_n = int(summary["n_selected_markers"].min())
    print(f"[FILTER] selected markers per cluster: min={min_n}, max={max_n}")
    plot_marker_filter_summary(summary, groupby, outdir, max_markers_per_cluster)


def plot_marker_filter_summary(summary, groupby, outdir, max_markers_per_cluster):
    plot_df = summary.copy()
    plot_df["group"] = plot_df["group"].astype(str)

    fig_width = max(9, 0.38 * len(plot_df) + 3)
    fig, ax = plt.subplots(figsize=(fig_width, 4.8))
    x = np.arange(len(plot_df))
    width = 0.38
    up_bars = ax.bar(
        x - width / 2,
        plot_df["n_up_selected_markers"],
        width=width,
        color="#2166ac",
        edgecolor="white",
        linewidth=0.7,
        label="Positive markers",
    )
    down_bars = ax.bar(
        x + width / 2,
        plot_df["n_down_selected_markers"],
        width=width,
        color="#b2182b",
        edgecolor="white",
        linewidth=0.7,
        label="Negative markers",
    )
    for bars, col in [
        (up_bars, "n_up_selected_markers"),
        (down_bars, "n_down_selected_markers"),
    ]:
        for bar, capped in zip(bars, plot_df[col] >= max_markers_per_cluster):
            if capped:
                bar.set_edgecolor("#c44e52")
                bar.set_linewidth(1.4)

    ax.axhline(
        max_markers_per_cluster,
        color="#c44e52",
        linestyle="--",
        linewidth=1.1,
        label=f"Cap per direction = {max_markers_per_cluster}",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["group"], rotation=0, fontsize=8)
    ax.set_xlabel("Leiden cluster")
    ax.set_ylabel("Selected marker genes")
    ax.set_title("Filtered marker genes selected per cluster", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max_markers_per_cluster + max(2, int(max_markers_per_cluster * 0.08)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.8)

    up_patch = plt.Rectangle((0, 0), 1, 1, color="#2166ac", label="Positive markers")
    down_patch = plt.Rectangle((0, 0), 1, 1, color="#b2182b", label="Negative markers")
    capped_patch = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#c44e52", linewidth=1.4, label="Capped at max")
    ax.legend(
        [up_patch, down_patch, capped_patch],
        ["Positive markers", "Negative markers", "Capped at max"],
        frameon=False,
        loc="upper right",
        fontsize=8,
    )
    for bars, values in [
        (up_bars, plot_df["n_up_selected_markers"]),
        (down_bars, plot_df["n_down_selected_markers"]),
    ]:
        for rect, value in zip(bars, values):
            if int(value) <= 0:
                continue
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                float(value) + 0.8,
                str(int(value)),
                ha="center",
                va="bottom",
                fontsize=7,
            )

    plt.tight_layout()
    png = os.path.join(outdir, f"{groupby}_filtered_marker_selection_counts.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {png}")
    plt.close(fig)


def cluster_sort_key(value):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def select_plot_markers(top_markers, n_per_cluster):
    candidates = top_markers.copy()
    if "pvals_adj" in candidates.columns:
        candidates = candidates[candidates["pvals_adj"] < 0.05]
    if "logfoldchanges" in candidates.columns:
        candidates = candidates[candidates["logfoldchanges"] > 0]

    selected = []
    for _, df_group in candidates.groupby("group"):
        for gene in df_group["names"].head(n_per_cluster):
            gene = str(gene)
            if gene not in selected:
                selected.append(gene)
    return selected


def save_dotplot(adata, selected, groupby, outdir):
    print(f"[PLOT] Dotplot with {len(selected)} selected markers")
    dot = sc.pl.dotplot(
        adata,
        var_names=selected,
        groupby=groupby,
        standard_scale="var",
        cmap=EXPRESSION_CMAP,
        show=False,
        return_fig=True,
    )
    path = os.path.join(outdir, f"{groupby}_dotplot_top_markers.png")
    dot.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {path}")
    plt.close("all")


def save_matrixplot(adata, selected, groupby, outdir):
    print(f"[PLOT] Matrixplot with {len(selected)} selected markers")
    matrix = sc.pl.matrixplot(
        adata,
        var_names=selected,
        groupby=groupby,
        standard_scale="var",
        cmap=EXPRESSION_CMAP,
        show=False,
        return_fig=True,
    )
    path = os.path.join(outdir, f"{groupby}_matrixplot_top_markers.png")
    matrix.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {path}")
    plt.close("all")


if __name__ == "__main__":
    main()
