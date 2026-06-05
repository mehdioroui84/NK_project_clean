#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs
from nk_project.plotting import plot_composite_batch_profile
from nk_project.preprocessing import profile_batch_combinations


def main():
    ensure_dirs(cfg.FIG_OUTDIR, cfg.TABLE_OUTDIR)
    print("[LOAD] Reading merged AnnData...")
    adata = sc.read_h5ad(cfg.MERGED_PATH)
    print(f"[LOAD] {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    profile = profile_batch_combinations(
        adata.obs,
        dataset_key=cfg.DATASET_KEY,
        assay_key=cfg.ASSAY_KEY,
        low_cell_warn=500,
        min_cell_hard=100,
    )
    combo_counts = profile["combo_counts"]

    print("=" * 64)
    print("MARGINAL COUNTS")
    print("=" * 64)
    print(f"Unique dataset_id values : {profile['n_datasets']}")
    print(f"Unique assay values      : {profile['n_assays']}")
    print(f"Worst-case combinations  : {profile['worst_case']}")
    print(f"Actual observed combos   : {profile['n_combos']}")
    print()
    print(profile["assay_counts"].to_string())
    print()
    print(combo_counts.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_string())

    combo_counts.to_csv(os.path.join(cfg.TABLE_OUTDIR, "composite_batch_counts.csv"), header=["n_cells"])
    plot_path = os.path.join(cfg.FIG_OUTDIR, "composite_batch_profile.png")
    plot_composite_batch_profile(
        adata.obs,
        combo_counts,
        dataset_key=cfg.DATASET_KEY,
        assay_key=cfg.ASSAY_KEY,
        save_path=plot_path,
    )
    print(f"[SAVE] {plot_path}")


if __name__ == "__main__":
    main()
