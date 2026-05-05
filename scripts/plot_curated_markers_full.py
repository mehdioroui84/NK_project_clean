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
from scripts.plot_curated_markers_validation import GROUPBY, MARKER_SETS


def main():
    in_path = os.path.join(cfg.BASE_OUTDIR, "leiden_discovery", "full_scvi_leiden.h5ad")
    outdir = os.path.join(cfg.BASE_OUTDIR, "markers", "full", GROUPBY)
    ensure_dirs(outdir)

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if GROUPBY not in adata.obs:
        raise KeyError(f"{GROUPBY!r} not found in adata.obs")
    adata.obs[GROUPBY] = adata.obs[GROUPBY].astype(str).astype("category")

    ad = adata.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

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

    marker_path = os.path.join(outdir, f"{GROUPBY}_curated_marker_genes_present.csv")
    pd.DataFrame(marker_rows).to_csv(marker_path, index=False)
    print(f"[SAVE] {marker_path}")

    print(f"[PLOT] Full-data curated dotplot with {len(marker_list)} genes")
    dot = sc.pl.dotplot(
        ad,
        var_names=present_sets,
        groupby=GROUPBY,
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    dot_path = os.path.join(outdir, f"{GROUPBY}_curated_marker_dotplot.png")
    dot.savefig(dot_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {dot_path}")
    plt.close("all")

    print("[PLOT] Full-data curated matrixplot")
    matrix = sc.pl.matrixplot(
        ad,
        var_names=present_sets,
        groupby=GROUPBY,
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    matrix_path = os.path.join(outdir, f"{GROUPBY}_curated_marker_matrixplot.png")
    matrix.savefig(matrix_path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {matrix_path}")
    plt.close("all")

    print("[SUMMARY] Computing average expression by cluster for curated markers")
    expr = ad[:, marker_list].to_df()
    expr[GROUPBY] = ad.obs[GROUPBY].astype(str).values
    avg = expr.groupby(GROUPBY).mean()
    avg_path = os.path.join(outdir, f"{GROUPBY}_curated_marker_cluster_means.csv")
    avg.to_csv(avg_path)
    print(f"[SAVE] {avg_path}")

    print("[DONE] Full-data curated marker plotting complete.")


if __name__ == "__main__":
    main()
