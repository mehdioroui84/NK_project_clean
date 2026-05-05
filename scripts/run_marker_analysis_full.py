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
from scripts.run_marker_analysis_validation import (
    GROUPBY,
    N_TOP_PLOT_PER_CLUSTER,
    N_TOP_TABLE,
    cluster_summary,
    save_dotplot,
    save_matrixplot,
    select_plot_markers,
)


def main():
    in_path = os.path.join(cfg.BASE_OUTDIR, "leiden_discovery", "full_scvi_leiden.h5ad")
    outdir = os.path.join(cfg.BASE_OUTDIR, "markers", "full", GROUPBY)
    ensure_dirs(outdir)

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if GROUPBY not in adata.obs:
        raise KeyError(f"{GROUPBY!r} not found in adata.obs")

    adata.obs[GROUPBY] = adata.obs[GROUPBY].astype(str).astype("category")
    print(f"[MARKERS] full data; groupby={GROUPBY}; clusters={adata.obs[GROUPBY].nunique()}")

    summary = cluster_summary(adata, GROUPBY)
    summary_path = os.path.join(outdir, f"{GROUPBY}_cluster_summary.csv")
    summary.to_csv(summary_path)
    print(f"[SAVE] {summary_path}")

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

    plt.close("all")
    print("[DONE] Full-data marker analysis complete.")


if __name__ == "__main__":
    main()
