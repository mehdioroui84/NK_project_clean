#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import argparse

import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.discovery import run_leiden_grid
from nk_project.io_utils import ensure_dirs


def main():
    args = parse_args()
    outdir = os.path.join(cfg.BASE_OUTDIR, "leiden_discovery")
    ensure_dirs(outdir)
    adata_path = os.path.join(cfg.LATENT_OUTDIR, "scvi_full_with_latent.h5ad")
    print(f"[LOAD] {adata_path}")
    adata = sc.read_h5ad(adata_path)
    resolutions = args.resolutions or cfg.LEIDEN_RESOLUTIONS
    adata, summary = run_leiden_grid(
        adata,
        latent_key="X_scVI",
        resolutions=resolutions,
        n_neighbors=cfg.DISCOVERY_N_NEIGHBORS,
        seed=cfg.SEED,
        outdir=outdir,
        label_key=cfg.LABEL_KEY,
        dataset_key=cfg.DATASET_KEY,
        assay_key=cfg.ASSAY_CLEAN_KEY,
    )
    h5ad_path = os.path.join(outdir, "full_scvi_leiden.h5ad")
    adata.write(h5ad_path)
    print(summary.to_string(index=False))
    print(f"[SAVE] {h5ad_path}")
    print(f"[SAVE] {outdir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run Leiden clustering on the full SCVI latent dataset.")
    parser.add_argument(
        "--resolutions",
        type=float,
        nargs="+",
        default=None,
        help="Leiden resolutions to run. Example: --resolutions 0.4",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
