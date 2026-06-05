"""Historical batch-key comparison.

This keeps the dataset_only / assay_only / composite_only experiment separate
from the production workflow. It trains and saves each SCANVI model immediately.
"""

from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs
from nk_project.metrics import minmax_normalize_series
from nk_project.workflows import train_scanvi


def main():
    outdir = os.path.join(cfg.BASE_OUTDIR, "batch_key_comparison")
    ensure_dirs(outdir)

    strategies = [
        ("dataset_only", cfg.DATASET_KEY),
        ("assay_only", cfg.ASSAY_CLEAN_KEY),
        ("composite_only", cfg.COMPOSITE_BATCH_KEY),
    ]
    rows = []
    for name, batch_key in strategies:
        print("\n" + "#" * 80)
        print(f"RUNNING STRATEGY: {name} | batch_key={batch_key}")
        print("#" * 80)
        _, evals = train_scanvi(cfg, label_key=cfg.LABEL_KEY, batch_key=batch_key)
        rows.append(
            {
                "strategy": name,
                "batch_key": batch_key,
                "val_macro_f1": evals["val"]["metrics"]["macro_f1"],
                "zeroshot_macro_f1": evals["heldout"]["metrics"]["macro_f1"],
                "full_macro_f1": evals["full"]["metrics"]["macro_f1"],
                "val_weighted_f1": evals["val"]["metrics"]["weighted_f1"],
                "zeroshot_weighted_f1": evals["heldout"]["metrics"]["weighted_f1"],
                "full_weighted_f1": evals["full"]["metrics"]["weighted_f1"],
            }
        )

    summary = pd.DataFrame(rows).set_index("strategy")
    for col in ["val_macro_f1", "zeroshot_macro_f1", "full_macro_f1"]:
        summary[col + "_norm"] = minmax_normalize_series(summary[col], higher_is_better=True)
    summary["overall_f1_score"] = summary[
        ["val_macro_f1_norm", "zeroshot_macro_f1_norm", "full_macro_f1_norm"]
    ].mean(axis=1)
    summary = summary.sort_values("overall_f1_score", ascending=False)
    path = os.path.join(outdir, "scanvi_batch_key_strategy_comparison.csv")
    summary.to_csv(path)
    print(summary.round(4).to_string())
    print(f"[SAVE] {path}")


if __name__ == "__main__":
    main()
