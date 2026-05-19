#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


DEFAULT_GROUPBY = "leiden_0_4"
N_TOP_TABLE = 50
N_TOP_PLOT_PER_CLUSTER = 3
DEFAULT_MARKER_FDR = 0.02
DEFAULT_MARKER_LOGFC = 0.25
DEFAULT_MARKER_PCT_DIFF = 0.50

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
    outdir = args.outdir or os.path.join(cfg.BASE_OUTDIR, "markers", "full", args.groupby)
    ensure_dirs(outdir)

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if args.groupby not in adata.obs:
        raise KeyError(f"{args.groupby!r} not found in adata.obs")

    adata.obs[args.groupby] = adata.obs[args.groupby].astype(str).astype("category")
    print(
        f"[MARKERS] full data; groupby={args.groupby}; "
        f"clusters={adata.obs[args.groupby].nunique()}"
    )

    summary = cluster_summary(adata, args.groupby)
    summary_path = os.path.join(outdir, f"{args.groupby}_cluster_summary.csv")
    summary.to_csv(summary_path)
    print(f"[SAVE] {summary_path}")

    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    if not args.skip_rank_genes:
        run_rank_genes(ad, args.groupby, outdir, args)
    else:
        print("[SKIP] rank_genes_groups marker analysis")

    if not args.skip_curated_markers:
        plot_curated_markers(ad, args.groupby, outdir)
    else:
        print("[SKIP] curated marker plots")

    plt.close("all")
    print("[DONE] Full-data marker analysis complete.")


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
        type=float,
        default=DEFAULT_MARKER_PCT_DIFF,
        help=(
            "Minimum expression specificity for per-cluster marker selection: "
            "pct expressed in cluster minus pct expressed outside cluster."
        ),
    )
    parser.add_argument(
        "--max-markers-per-cluster",
        type=int,
        default=N_TOP_TABLE,
        help="Maximum filtered marker genes kept per cluster.",
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
    return parser.parse_args()


def run_rank_genes(ad, groupby, outdir, args):
    print("[DE] Running Wilcoxon cluster-vs-rest marker analysis...")
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

    enriched_candidates = filter_enriched_marker_candidates(
        all_markers,
        fdr=args.marker_fdr,
        min_logfc=args.marker_logfc,
        min_pct_diff=args.marker_pct_diff,
        direction=args.marker_direction,
    )
    filtered_markers = cap_markers_per_cluster(enriched_candidates, args.max_markers_per_cluster)
    top_path = os.path.join(outdir, f"{groupby}_markers_top{args.max_markers_per_cluster}_per_cluster.csv")
    filtered_markers.to_csv(top_path, index=False)
    print(f"[SAVE] {top_path}")
    print(
        "[FILTER] marker selection: "
        f"FDR<{args.marker_fdr:g}, |logFC|>{args.marker_logfc:g}, "
        f"|pct_diff|>{args.marker_pct_diff:g}, direction={args.marker_direction}, "
        f"cap={args.max_markers_per_cluster}"
    )
    save_marker_filter_summary(
        filtered_markers,
        enriched_candidates,
        all_markers["group"].astype(str).unique(),
        groupby,
        outdir,
        max_markers_per_cluster=args.max_markers_per_cluster,
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


def plot_curated_markers(ad, groupby, outdir):
    present_sets = {
        name: [gene for gene in genes if gene in ad.var_names]
        for name, genes in MARKER_SETS.items()
    }
    present_sets = {name: genes for name, genes in present_sets.items() if genes}

    marker_list = []
    marker_rows = []
    for set_name, genes in present_sets.items():
        for gene in genes:
            if gene not in marker_list:
                marker_list.append(gene)
            marker_rows.append({"marker_set": set_name, "gene": gene})

    marker_path = os.path.join(outdir, f"{groupby}_curated_marker_genes_present.csv")
    pd.DataFrame(marker_rows).to_csv(marker_path, index=False)
    print(f"[SAVE] {marker_path}")

    print(f"[PLOT] Full-data curated dotplot with {len(marker_list)} genes")
    dot = sc.pl.dotplot(
        ad,
        var_names=present_sets,
        groupby=groupby,
        standard_scale="var",
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


def filter_enriched_marker_candidates(all_markers, fdr, min_logfc, min_pct_diff, direction):
    markers = all_markers.copy()
    markers["pct_expr_group"], markers["pct_expr_reference"] = marker_pct_columns(markers)
    markers["pct_expr_diff"] = markers["pct_expr_group"] - markers["pct_expr_reference"]

    for col in ["pvals_adj", "logfoldchanges", "pct_expr_diff", "scores"]:
        if col in markers.columns:
            markers[col] = pd.to_numeric(markers[col], errors="coerce")
    markers["abs_logfoldchanges"] = markers["logfoldchanges"].abs()
    markers["abs_pct_expr_diff"] = markers["pct_expr_diff"].abs()

    required = ["group", "pvals_adj", "logfoldchanges", "pct_expr_diff"]
    missing = [col for col in required if col not in markers.columns]
    if missing:
        raise KeyError(f"Missing required marker columns for filtering: {missing}")

    up = (markers["logfoldchanges"] > min_logfc) & (markers["pct_expr_diff"] > min_pct_diff)
    down = (markers["logfoldchanges"] < -min_logfc) & (markers["pct_expr_diff"] < -min_pct_diff)
    if direction == "up":
        direction_mask = up
    elif direction == "down":
        direction_mask = down
    else:
        direction_mask = up | down

    candidates = markers[(markers["pvals_adj"] < fdr) & direction_mask].copy()
    candidates["marker_direction"] = np.where(candidates["logfoldchanges"] >= 0, "up", "down")

    return sort_marker_candidates(candidates).reset_index(drop=True)


def sort_marker_candidates(markers):
    markers = markers.copy()
    if "abs_logfoldchanges" not in markers.columns and "logfoldchanges" in markers.columns:
        markers["abs_logfoldchanges"] = markers["logfoldchanges"].abs()
    if "abs_pct_expr_diff" not in markers.columns and "pct_expr_diff" in markers.columns:
        markers["abs_pct_expr_diff"] = markers["pct_expr_diff"].abs()
    if "abs_scores" not in markers.columns and "scores" in markers.columns:
        markers["abs_scores"] = markers["scores"].abs()

    sort_cols = ["group", "pvals_adj", "abs_logfoldchanges", "abs_pct_expr_diff"]
    ascending = [True, True, False, False]
    if "scores" in markers.columns:
        sort_cols.append("abs_scores")
        ascending.append(False)
    return markers.sort_values(sort_cols, ascending=ascending)


def cap_markers_per_cluster(markers, max_markers_per_cluster):
    return (
        sort_marker_candidates(markers)
        .groupby("group", group_keys=False)
        .head(max_markers_per_cluster)
        .reset_index(drop=True)
    )


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
    x = range(len(plot_df))
    up_bars = ax.bar(
        x,
        plot_df["n_up_selected_markers"],
        color="#4c78a8",
        edgecolor="white",
        linewidth=0.7,
    )
    down_bars = ax.bar(
        x,
        plot_df["n_down_selected_markers"],
        bottom=plot_df["n_up_selected_markers"],
        color="#f28e2b",
        edgecolor="white",
        linewidth=0.7,
    )
    for bars in [up_bars, down_bars]:
        for bar, capped in zip(bars, plot_df["capped_at_max"]):
            if capped:
                bar.set_edgecolor("#c44e52")
                bar.set_linewidth(1.4)

    ax.axhline(
        max_markers_per_cluster,
        color="#c44e52",
        linestyle="--",
        linewidth=1.1,
        label=f"Cap = {max_markers_per_cluster}",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df["group"], rotation=0, fontsize=8)
    ax.set_xlabel("Leiden cluster")
    ax.set_ylabel("Selected marker genes")
    ax.set_title("Filtered marker genes selected per cluster", fontsize=12, fontweight="bold")
    ax.set_ylim(0, max_markers_per_cluster + max(2, int(max_markers_per_cluster * 0.08)))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d8d8d8", linewidth=0.6, alpha=0.8)

    up_patch = plt.Rectangle((0, 0), 1, 1, color="#4c78a8", label="Up in cluster")
    down_patch = plt.Rectangle((0, 0), 1, 1, color="#f28e2b", label="Down in cluster")
    capped_patch = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="#c44e52", linewidth=1.4, label="Capped at max")
    ax.legend(
        [up_patch, down_patch, capped_patch],
        ["Up in cluster", "Down in cluster", "Capped at max"],
        frameon=False,
        loc="upper right",
        fontsize=8,
    )
    for rect, value in zip(up_bars, plot_df["n_selected_markers"]):
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
    pdf = os.path.join(outdir, f"{groupby}_filtered_marker_selection_counts.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")
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
        show=False,
        return_fig=True,
    )
    path = os.path.join(outdir, f"{groupby}_matrixplot_top_markers.png")
    matrix.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {path}")
    plt.close("all")


if __name__ == "__main__":
    main()
