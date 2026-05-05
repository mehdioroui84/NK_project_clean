#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import argparse

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


GROUPBY = "leiden_0_4"

# First-pass labels based on the validation v1 annotation logic. These are
# intentionally conservative: they are a starting worksheet, not the final call.
VALIDATION_V1_BY_OLD_LABEL = {
    "B": "B",
    "T": "T",
    "Mature Cytotoxic": "Mature Cytotoxic",
    "Transitional Cytotoxic": "Transitional Cytotoxic",
    "Cytokine-Stimulated": "Cytokine-Stimulated review",
    "Proliferative": "Proliferative",
    "Regulatory": "Regulatory",
    "Unconventional": "Unconventional",
    "Unknown_Lung_6": "Lung Cytotoxic NK",
    "Unknown_Lung_5": "Lung GZMK+ XCL1+ NK",
    "Unknown_Lung_4": "Unknown_Lung_4",
    "Unknown_Lung_3": "Unknown_Lung_3 review",
    "Unknown_Lung_1": "Unknown_Lung_1",
    "Unknown_Kidney": "Unknown_Kidney",
    "Unknown_BM_1": "Unknown_BM_1 Erythroid-like review",
    "Unknown_BM_2": "Unknown_BM_2",
    "Developmental": "Developmental review",
}


def main():
    args = parse_args()
    groupby = f"leiden_{str(args.resolution).replace('.', '_')}"
    outdir = os.path.join(cfg.BASE_OUTDIR, "leiden_discovery")
    obs_path = os.path.join(outdir, "obs_with_leiden.csv")
    ensure_dirs(outdir)

    print(f"[LOAD] {obs_path}")
    obs = pd.read_csv(obs_path, index_col=0, low_memory=False)
    if groupby not in obs.columns:
        raise KeyError(f"{groupby!r} not found in {obs_path}")

    worksheet = build_worksheet(obs, groupby)
    out_path = os.path.join(outdir, f"full_{groupby}_annotation_worksheet.csv")
    worksheet.to_csv(out_path)
    print(f"[SAVE] {out_path}")
    print("\n[PREVIEW]")
    print(worksheet.round(3).to_string())


def parse_args():
    parser = argparse.ArgumentParser(description="Build full-data Leiden annotation worksheet.")
    parser.add_argument("--resolution", type=float, default=0.4)
    return parser.parse_args()


def build_worksheet(obs, key):
    out = pd.DataFrame({"n_cells": obs[key].astype(str).value_counts().sort_index()})

    for col in [cfg.LABEL_KEY, "tissue", cfg.DATASET_KEY, cfg.ASSAY_CLEAN_KEY]:
        if col in obs.columns:
            out = out.join(top_summary(obs, key, col))

    out["draft_refined_label"] = [
        draft_label(row) for _, row in out.iterrows()
    ]
    out["review_priority"] = [
        review_priority(row) for _, row in out.iterrows()
    ]
    out["review_notes"] = [
        review_notes(row) for _, row in out.iterrows()
    ]

    return out.sort_values("n_cells", ascending=False)


def top_summary(obs, cluster_key, col):
    tab = pd.crosstab(obs[cluster_key].astype(str), obs[col].astype(str))
    total = tab.sum(axis=1)
    return pd.DataFrame(
        {
            f"top_{col}": tab.idxmax(axis=1),
            f"top_{col}_frac": tab.max(axis=1) / total,
        }
    )


def draft_label(row):
    top_old = str(row.get(f"top_{cfg.LABEL_KEY}", "unknown"))
    base = VALIDATION_V1_BY_OLD_LABEL.get(top_old, f"{top_old} review")

    top_tissue = str(row.get("top_tissue", ""))
    top_assay = str(row.get(f"top_{cfg.ASSAY_CLEAN_KEY}", ""))

    if base == "Cytokine-Stimulated review":
        return "Cytokine-Stimulated review: split effector vs CCR7+ after markers"
    if base == "Transitional Cytotoxic" and top_tissue in {"decidua", "lung"}:
        return "Transitional Cytotoxic Tissue-Resident review"
    if base == "Mature Cytotoxic" and top_assay == "Flex Gene Expression":
        return "Mature Cytotoxic Engineered review"
    return base


def review_priority(row):
    top_frac = float(row.get(f"top_{cfg.LABEL_KEY}_frac", 0.0))
    n_cells = int(row.get("n_cells", 0))
    label = str(row.get("draft_refined_label", ""))
    notes = review_notes(row)

    if n_cells < 300 or "review" in label or "high_dataset_specificity" in notes:
        return "high"
    if top_frac < 0.70 or "high_tissue_specificity" in notes or "high_assay_specificity" in notes:
        return "medium"
    return "low"


def review_notes(row):
    notes = []
    for col, threshold, note in [
        (f"top_{cfg.LABEL_KEY}_frac", 0.70, "mixed_original_NK_State"),
        ("top_tissue_frac", 0.85, "high_tissue_specificity"),
        (f"top_{cfg.ASSAY_CLEAN_KEY}_frac", 0.85, "high_assay_specificity"),
        (f"top_{cfg.DATASET_KEY}_frac", 0.75, "high_dataset_specificity"),
    ]:
        if col in row and pd.notna(row[col]) and float(row[col]) >= threshold:
            # For label fraction, high means clean, not mixed.
            if col == f"top_{cfg.LABEL_KEY}_frac":
                continue
            notes.append(note)

    if f"top_{cfg.LABEL_KEY}_frac" in row and float(row[f"top_{cfg.LABEL_KEY}_frac"]) < 0.70:
        notes.append("mixed_original_NK_State")
    return ";".join(notes)


if __name__ == "__main__":
    main()
