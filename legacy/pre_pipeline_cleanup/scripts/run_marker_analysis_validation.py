#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


GROUPBY = "leiden_0_4"
N_TOP_TABLE = 50
N_TOP_PLOT_PER_CLUSTER = 3


def main():
    in_path = os.path.join(
        cfg.BASE_OUTDIR,
        "leiden_validation",
        "validation_scvi_leiden.h5ad",
    )
    outdir = os.path.join(cfg.BASE_OUTDIR, "markers", "validation", GROUPBY)
    ensure_dirs(outdir)

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if GROUPBY not in adata.obs:
        raise KeyError(f"{GROUPBY!r} not found in adata.obs")

    adata.obs[GROUPBY] = adata.obs[GROUPBY].astype(str).astype("category")
    print(f"[MARKERS] groupby={GROUPBY}; clusters={adata.obs[GROUPBY].nunique()}")

    summary = cluster_summary(adata, GROUPBY)
    summary_path = os.path.join(outdir, f"{GROUPBY}_cluster_summary.csv")
    summary.to_csv(summary_path)
    print(f"[SAVE] {summary_path}")

    # Marker discovery should use expression scale, not the SCVI latent space.
    # Work on a copy so the saved Leiden AnnData remains unchanged.
    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    print("[DE] Running Wilcoxon cluster-vs-rest marker analysis...")
    sc.tl.rank_genes_groups(
        ad,
        groupby=GROUPBY,
        method="wilcoxon",
        pts=True,
        tie_correct=True,
    )

    all_markers = sc.get.rank_genes_groups_df(ad, group=None)
    all_path = os.path.join(outdir, f"{GROUPBY}_markers_all_wilcoxon.csv")
    all_markers.to_csv(all_path, index=False)
    print(f"[SAVE] {all_path}")

    top_markers = (
        all_markers.sort_values(["group", "pvals_adj", "scores"], ascending=[True, True, False])
        .groupby("group", group_keys=False)
        .head(N_TOP_TABLE)
    )
    top_path = os.path.join(outdir, f"{GROUPBY}_markers_top{N_TOP_TABLE}_per_cluster.csv")
    top_markers.to_csv(top_path, index=False)
    print(f"[SAVE] {top_path}")

    selected = select_plot_markers(top_markers, n_per_cluster=N_TOP_PLOT_PER_CLUSTER)
    selected_path = os.path.join(outdir, f"{GROUPBY}_selected_plot_markers.txt")
    pd.Series(selected, name="gene").to_csv(selected_path, index=False, header=False)
    print(f"[SAVE] {selected_path}")

    if selected:
        save_dotplot(ad, selected, outdir)
        save_matrixplot(ad, selected, outdir)
    else:
        print("[WARN] No selected markers passed filtering; skipping plots.")

    print("[DONE] First-pass marker analysis complete.")


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


def save_dotplot(adata, selected, outdir):
    print(f"[PLOT] Dotplot with {len(selected)} selected markers")
    dot = sc.pl.dotplot(
        adata,
        var_names=selected,
        groupby=GROUPBY,
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    path = os.path.join(outdir, f"{GROUPBY}_dotplot_top_markers.png")
    dot.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {path}")
    plt.close("all")


def save_matrixplot(adata, selected, outdir):
    print(f"[PLOT] Matrixplot with {len(selected)} selected markers")
    matrix = sc.pl.matrixplot(
        adata,
        var_names=selected,
        groupby=GROUPBY,
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    path = os.path.join(outdir, f"{GROUPBY}_matrixplot_top_markers.png")
    matrix.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {path}")
    plt.close("all")


if __name__ == "__main__":
    main()
