#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


DEFAULT_GROUPBY = "leiden_0_4"
N_TOP_TABLE = 50
N_TOP_PLOT_PER_CLUSTER = 3

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
        run_rank_genes(ad, args.groupby, outdir)
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
    return parser.parse_args()


def run_rank_genes(ad, groupby, outdir):
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

    top_markers = (
        all_markers.sort_values(["group", "pvals_adj", "scores"], ascending=[True, True, False])
        .groupby("group", group_keys=False)
        .head(N_TOP_TABLE)
    )
    top_path = os.path.join(outdir, f"{groupby}_markers_top{N_TOP_TABLE}_per_cluster.csv")
    top_markers.to_csv(top_path, index=False)
    print(f"[SAVE] {top_path}")

    selected = select_plot_markers(top_markers, n_per_cluster=N_TOP_PLOT_PER_CLUSTER)
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
