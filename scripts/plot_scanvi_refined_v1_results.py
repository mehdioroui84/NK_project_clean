#!/usr/bin/env python
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg


def main():
    outdir = os.path.join(cfg.BASE_OUTDIR, "refined_scanvi_v1")
    cfg.BASE_OUTDIR = outdir
    cfg.FIG_OUTDIR = os.path.join(outdir, "figures")
    cfg.MODEL_OUTDIR = os.path.join(outdir, "models")
    cfg.TABLE_OUTDIR = os.path.join(outdir, "tables")
    cfg.LATENT_OUTDIR = os.path.join(outdir, "latents")
    cfg.LABEL_KEY = cfg.REFINED_LABEL_KEY

    from scripts.plot_scanvi_results import main as plot_main

    plot_main()


if __name__ == "__main__":
    main()
