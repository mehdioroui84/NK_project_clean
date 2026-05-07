#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg


def main() -> None:
    args = parse_args()

    if not args.skip_full_plots:
        run_full_refined_plots()

    if not args.skip_zeroshot_plots:
        print("[RUN] nk_project.evaluation.scanvi_zeroshot_plots", flush=True)
        from nk_project.evaluation.scanvi_zeroshot_plots import main as zeroshot_main

        zeroshot_main()

    if not args.skip_dataset_summary:
        summary_args = []
        if args.known_assays_only:
            summary_args.append("--known-assays-only")
        for assay in args.exclude_assay:
            summary_args.extend(["--exclude-assay", assay])
        for assay in args.include_assay:
            summary_args.extend(["--include-assay", assay])

        print("[RUN] nk_project.evaluation.scanvi_zeroshot_by_dataset", flush=True)
        from nk_project.evaluation.scanvi_zeroshot_by_dataset import main as summary_main

        summary_main(summary_args)

    print("[DONE] Refined-v1 SCANVI evaluation complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the refined-v1 SCANVI evaluation outputs: full-dataset plots, "
            "held-out zero-shot plots, and held-out dataset summaries."
        )
    )
    parser.add_argument("--skip-full-plots", action="store_true")
    parser.add_argument("--skip-zeroshot-plots", action="store_true")
    parser.add_argument("--skip-dataset-summary", action="store_true")
    parser.add_argument(
        "--known-assays-only",
        action="store_true",
        help="Pass through to the held-out dataset summary.",
    )
    parser.add_argument(
        "--exclude-assay",
        action="append",
        default=[],
        help="Assay_clean value to exclude from the held-out dataset summary.",
    )
    parser.add_argument(
        "--include-assay",
        action="append",
        default=[],
        help="Assay_clean value to include in the held-out dataset summary.",
    )
    return parser.parse_args()


def run_full_refined_plots() -> None:
    original = {
        "BASE_OUTDIR": cfg.BASE_OUTDIR,
        "FIG_OUTDIR": cfg.FIG_OUTDIR,
        "MODEL_OUTDIR": cfg.MODEL_OUTDIR,
        "TABLE_OUTDIR": cfg.TABLE_OUTDIR,
        "LATENT_OUTDIR": cfg.LATENT_OUTDIR,
        "LABEL_KEY": cfg.LABEL_KEY,
    }
    outdir = os.path.join(original["BASE_OUTDIR"], "refined_scanvi_v1")
    cfg.BASE_OUTDIR = outdir
    cfg.FIG_OUTDIR = os.path.join(outdir, "figures")
    cfg.MODEL_OUTDIR = os.path.join(outdir, "models")
    cfg.TABLE_OUTDIR = os.path.join(outdir, "tables")
    cfg.LATENT_OUTDIR = os.path.join(outdir, "latents")
    cfg.LABEL_KEY = cfg.REFINED_LABEL_KEY

    print("[RUN] nk_project.evaluation.scanvi_full_plots", flush=True)
    try:
        from nk_project.evaluation.scanvi_full_plots import main as full_plot_main

        full_plot_main()
    finally:
        for key, value in original.items():
            setattr(cfg, key, value)


if __name__ == "__main__":
    main()
