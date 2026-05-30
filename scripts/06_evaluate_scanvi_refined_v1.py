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
        run_full_refined_plots(args.full_plot_split, args.outdir)

    if not args.skip_zeroshot_plots:
        print("[RUN] nk_project.evaluation.scanvi_zeroshot_plots", flush=True)
        from nk_project.evaluation.scanvi_zeroshot_plots import main as zeroshot_main
        from nk_project.evaluation import scanvi_zeroshot_plots

        original_base_outdir = cfg.BASE_OUTDIR
        zero_shot_outdir = os.path.normpath(args.outdir)
        cfg.BASE_OUTDIR = os.path.dirname(zero_shot_outdir) or "."
        scanvi_zeroshot_plots.OUTDIR_NAME = os.path.basename(zero_shot_outdir)
        try:
            zeroshot_main()
        finally:
            cfg.BASE_OUTDIR = original_base_outdir

    if not args.skip_dataset_summary:
        summary_args = []
        summary_args.extend(["--ref-outdir", args.outdir])
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
        "--outdir",
        default=os.path.join(cfg.BASE_OUTDIR, "refined_scanvi_v1"),
        help="SCANVI output directory to evaluate. Default: outputs/refined_scanvi_v1.",
    )
    parser.add_argument(
        "--full-plot-split",
        choices=["all", "Train", "Val", "Held-out"],
        default="all",
        help="Cell split to show in the full SCANVI UMAP panel. Default: all.",
    )
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


def run_full_refined_plots(split: str = "all", outdir: str | None = None) -> None:
    original = {
        "BASE_OUTDIR": cfg.BASE_OUTDIR,
        "FIG_OUTDIR": cfg.FIG_OUTDIR,
        "MODEL_OUTDIR": cfg.MODEL_OUTDIR,
        "TABLE_OUTDIR": cfg.TABLE_OUTDIR,
        "LATENT_OUTDIR": cfg.LATENT_OUTDIR,
        "LABEL_KEY": cfg.LABEL_KEY,
    }
    outdir = outdir or os.path.join(original["BASE_OUTDIR"], "refined_scanvi_v1")
    cfg.BASE_OUTDIR = outdir
    cfg.FIG_OUTDIR = os.path.join(outdir, "figures")
    cfg.MODEL_OUTDIR = os.path.join(outdir, "models")
    cfg.TABLE_OUTDIR = os.path.join(outdir, "tables")
    cfg.LATENT_OUTDIR = os.path.join(outdir, "latents")
    cfg.LABEL_KEY = cfg.REFINED_LABEL_KEY

    print("[RUN] nk_project.evaluation.scanvi_full_plots", flush=True)
    try:
        from nk_project.evaluation.scanvi_full_plots import main as full_plot_main

        full_plot_main(["--split", split])
    finally:
        for key, value in original.items():
            setattr(cfg, key, value)


if __name__ == "__main__":
    main()
