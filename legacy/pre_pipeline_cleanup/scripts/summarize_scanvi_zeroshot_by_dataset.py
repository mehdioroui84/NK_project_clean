#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


DEFAULT_REF_OUTDIR_NAME = "refined_scanvi_v1"


def main():
    args = parse_args()
    ref_outdir = args.ref_outdir or os.path.join(cfg.BASE_OUTDIR, DEFAULT_REF_OUTDIR_NAME)
    table_dir = os.path.join(ref_outdir, "tables")
    fig_dir = os.path.join(ref_outdir, "figures")
    ensure_dirs(table_dir, fig_dir)

    obs_path = args.obs_csv or os.path.join(table_dir, "scanvi_full_obs_metadata.csv")
    pred_path = args.pred_csv or os.path.join(table_dir, "scanvi_full_prediction_summary.csv")
    heldout_path = args.heldout_names or os.path.join(table_dir, "heldout_obs_names.txt")
    train_path = args.train_names or os.path.join(table_dir, "train_obs_names.txt")

    print(f"[LOAD] {obs_path}")
    obs = pd.read_csv(obs_path, index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)
    print(f"[LOAD] {pred_path}")
    pred = pd.read_csv(pred_path, index_col=0, low_memory=False)
    pred.index = pred.index.astype(str)
    print(f"[LOAD] {heldout_path}")
    heldout_names = pd.read_csv(heldout_path, header=None)[0].astype(str)

    common = obs.index.intersection(pred.index).intersection(heldout_names)
    if len(common) == 0:
        raise ValueError("No overlapping held-out cells found across obs, predictions, and heldout names.")

    df = obs.loc[common].copy()
    df["pred_label"] = pred.loc[common, "pred_label"].astype(str)
    df["confidence"] = pred.loc[common, "confidence"].astype(float)
    df["certainty"] = pred.loc[common, "certainty"].astype(float)
    df["true_label"] = df[args.label_key].astype(str)
    df["correct"] = df["true_label"].eq(df["pred_label"])

    if args.known_assays_only:
        print(f"[LOAD] {train_path}")
        train_names = pd.read_csv(train_path, header=None)[0].astype(str)
        train_common = obs.index.intersection(train_names)
        train_assays = set(obs.loc[train_common, args.assay_key].astype(str).dropna())
        before = len(df)
        df = df[df[args.assay_key].astype(str).isin(train_assays)].copy()
        print(f"[FILTER] known assays only: {before:,} -> {len(df):,} cells")

    if args.exclude_assay:
        before = len(df)
        exclude = set(args.exclude_assay)
        df = df[~df[args.assay_key].astype(str).isin(exclude)].copy()
        print(f"[FILTER] exclude assay {sorted(exclude)}: {before:,} -> {len(df):,} cells")

    if args.include_assay:
        before = len(df)
        include = set(args.include_assay)
        df = df[df[args.assay_key].astype(str).isin(include)].copy()
        print(f"[FILTER] include assay {sorted(include)}: {before:,} -> {len(df):,} cells")

    if len(df) == 0:
        raise ValueError("No cells remain after filters.")

    summary_rows = []
    per_class_rows = []
    for dataset_id, sub in df.groupby(args.dataset_key, sort=False):
        metrics, kept = compute_metrics(sub, args)
        top_assay = sub[args.assay_key].astype(str).value_counts().idxmax()
        top_tissue = sub[args.tissue_key].astype(str).value_counts().idxmax() if args.tissue_key in sub else "NA"
        top_true = sub["true_label"].value_counts().idxmax()
        summary_rows.append(
            {
                "dataset_id": dataset_id,
                "n_cells": len(sub),
                "n_eval_cells": int(kept.sum()),
                "top_assay_clean": top_assay,
                "top_tissue": top_tissue,
                "top_true_label": top_true,
                "accuracy": metrics["accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "mean_confidence": sub["confidence"].mean(),
                "mean_certainty": sub["certainty"].mean(),
                "n_true_classes": sub["true_label"].nunique(),
                "eval_classes": ";".join(metrics["classes"]),
                "dropped_rare_classes": ";".join(metrics["dropped_rare"]),
            }
        )
        for label, sub_label in sub[kept].groupby("true_label", sort=False):
            y = sub_label["true_label"].astype(str)
            p = sub_label["pred_label"].astype(str)
            per_class_rows.append(
                {
                    "dataset_id": dataset_id,
                    "true_label": label,
                    "n_true": len(sub_label),
                    "accuracy_recall": accuracy_score(y, p),
                    "f1": f1_score(y, p, labels=[label], average="macro", zero_division=0),
                    "mean_confidence": sub_label["confidence"].mean(),
                }
            )

    summary = pd.DataFrame(summary_rows).sort_values(["weighted_f1", "n_cells"], ascending=[True, False])
    per_class = pd.DataFrame(per_class_rows).sort_values(["dataset_id", "f1"], ascending=[True, True])

    suffix = make_suffix(args)
    summary_path = os.path.join(table_dir, f"scanvi_zeroshot_by_dataset_summary{suffix}.csv")
    per_class_path = os.path.join(table_dir, f"scanvi_zeroshot_by_dataset_per_class{suffix}.csv")
    summary.to_csv(summary_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    print(f"[SAVE] {summary_path}")
    print(f"[SAVE] {per_class_path}")

    print("\n[SUMMARY]")
    cols = [
        "dataset_id",
        "n_cells",
        "top_assay_clean",
        "top_tissue",
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "top_true_label",
    ]
    print(summary[cols].round(4).to_string(index=False))

    plot_summary(summary, fig_dir, suffix)
    print("[DONE] Zero-shot held-out dataset summary complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize refined-v1 SCANVI zero-shot performance by held-out dataset."
    )
    parser.add_argument("--ref-outdir", default=None, help="Default: outputs/refined_scanvi_v1")
    parser.add_argument("--obs-csv", default=None)
    parser.add_argument("--pred-csv", default=None)
    parser.add_argument("--heldout-names", default=None)
    parser.add_argument("--train-names", default=None)
    parser.add_argument("--label-key", default=cfg.REFINED_LABEL_KEY)
    parser.add_argument("--dataset-key", default=cfg.DATASET_KEY)
    parser.add_argument("--assay-key", default=cfg.ASSAY_CLEAN_KEY)
    parser.add_argument("--tissue-key", default="tissue")
    parser.add_argument("--min-class-eval", type=int, default=cfg.MIN_CLASS_EVAL)
    parser.add_argument(
        "--known-assays-only",
        action="store_true",
        help="Restrict to held-out cells whose assay_clean was present in the training split.",
    )
    parser.add_argument("--exclude-assay", action="append", default=[], help="Assay_clean value to exclude.")
    parser.add_argument("--include-assay", action="append", default=[], help="Assay_clean value to include.")
    return parser.parse_args()


def compute_metrics(sub: pd.DataFrame, args):
    counts = sub["true_label"].value_counts()
    rare = sorted(counts[counts < args.min_class_eval].index.tolist())
    kept = ~sub["true_label"].isin(rare)
    y = sub.loc[kept, "true_label"].astype(str)
    p = sub.loc[kept, "pred_label"].astype(str)
    classes = sorted(y.unique())
    if len(y) == 0:
        return {
            "accuracy": np.nan,
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "classes": [],
            "dropped_rare": rare,
        }, kept
    return {
        "accuracy": accuracy_score(y, p),
        "macro_f1": f1_score(y, p, average="macro", labels=classes, zero_division=0),
        "weighted_f1": f1_score(y, p, average="weighted", labels=classes, zero_division=0),
        "classes": classes,
        "dropped_rare": rare,
    }, kept


def make_suffix(args) -> str:
    parts = []
    if args.known_assays_only:
        parts.append("known_assays_only")
    if args.exclude_assay:
        parts.append("exclude_" + "_".join(safe_name(x) for x in args.exclude_assay))
    if args.include_assay:
        parts.append("include_" + "_".join(safe_name(x) for x in args.include_assay))
    return "_" + "_".join(parts) if parts else ""


def plot_summary(summary: pd.DataFrame, fig_dir: str, suffix: str):
    if summary.empty:
        return
    plot_df = summary.sort_values("weighted_f1", ascending=True).copy()
    labels = [
        f"{row.dataset_id[:8]}\n{row.top_assay_clean}"
        for row in plot_df.itertuples(index=False)
    ]
    y = np.arange(len(plot_df))
    fig_h = max(4.5, 0.55 * len(plot_df))
    fig, ax = plt.subplots(figsize=(9.5, fig_h))
    ax.barh(y - 0.18, plot_df["macro_f1"], height=0.34, label="Macro F1", color="#4c78a8")
    ax.barh(y + 0.18, plot_df["weighted_f1"], height=0.34, label="Weighted F1", color="#f58518")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title("Refined-v1 SCANVI zero-shot performance by held-out dataset")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    png = os.path.join(fig_dir, f"scanvi_zeroshot_by_dataset_summary{suffix}.png")
    pdf = os.path.join(fig_dir, f"scanvi_zeroshot_by_dataset_summary{suffix}.pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")


def safe_name(value) -> str:
    out = str(value)
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        ("|", "_"),
        (":", "_"),
        ("'", ""),
        ('"', ""),
        ("+", "plus"),
    ]:
        out = out.replace(old, new)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")


if __name__ == "__main__":
    main()
