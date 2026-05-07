#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs
from nk_project.workflows import train_scanvi


DEFAULT_OUTDIR_NAME = "refined_scanvi_v1"


def main():
    args = parse_args()
    refined_input = args.input_h5ad or os.path.join(
        cfg.BASE_OUTDIR,
        "refined_annotation_v1",
        "full_scvi_leiden_refined_v1.h5ad",
    )
    outdir = args.outdir or os.path.join(cfg.BASE_OUTDIR, DEFAULT_OUTDIR_NAME)

    if not os.path.exists(refined_input):
        raise FileNotFoundError(
            f"Refined input not found: {refined_input}\n"
            "Run scripts/04_apply_refined_v1_labels.py first."
        )

    run_cfg = make_run_config(outdir, refined_input, args.max_epochs)
    ensure_dirs(run_cfg.BASE_OUTDIR, run_cfg.FIG_OUTDIR, run_cfg.MODEL_OUTDIR, run_cfg.TABLE_OUTDIR, run_cfg.LATENT_OUTDIR)

    print(f"[INPUT] {refined_input}")
    print(f"[OUTDIR] {outdir}")
    print(f"[LABEL] {run_cfg.REFINED_LABEL_KEY}")
    print(f"[BATCH] {run_cfg.PRODUCTION_BATCH_KEY}")
    print(f"[SPLITS] Reusing split IDs from: {run_cfg.SPLIT_ID_SOURCE_DIR}")

    if args.dry_run:
        print("[DRY-RUN] Configuration is valid; skipping SCANVI training.")
        return

    train_scanvi(
        run_cfg,
        label_key=run_cfg.REFINED_LABEL_KEY,
        batch_key=run_cfg.PRODUCTION_BATCH_KEY,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train SCANVI using the full-data refined v1 annotation "
            "`NK_State_refined` generated from full-data leiden_0_4."
        )
    )
    parser.add_argument(
        "--input-h5ad",
        default=None,
        help="Default: outputs/refined_annotation_v1/full_scvi_leiden_refined_v1.h5ad",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Default: outputs/refined_scanvi_v1",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Override MAX_EPOCHS, useful for smoke tests.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/configuration without training.",
    )
    return parser.parse_args()


def make_run_config(outdir: str, refined_input: str, max_epochs: int | None):
    values = {name: getattr(cfg, name) for name in dir(cfg) if name.isupper()}
    original_table_outdir = cfg.TABLE_OUTDIR

    values["BASE_OUTDIR"] = outdir
    values["FIG_OUTDIR"] = os.path.join(outdir, "figures")
    values["MODEL_OUTDIR"] = os.path.join(outdir, "models")
    values["TABLE_OUTDIR"] = os.path.join(outdir, "tables")
    values["LATENT_OUTDIR"] = os.path.join(outdir, "latents")
    values["MERGED_PATH"] = refined_input
    values["SPLIT_ID_SOURCE_DIR"] = original_table_outdir
    if max_epochs is not None:
        values["MAX_EPOCHS"] = int(max_epochs)
    return SimpleNamespace(**values)


if __name__ == "__main__":
    main()
