#!/usr/bin/env python
from __future__ import annotations

import os
import sys
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


GROUPBY = "leiden_0_4"

PAIR_KEY = "pairwise_label"
N_TOP_TABLE = 100
N_TOP_PLOT_PER_GROUP = 12

CLUSTER_LABELS = {
    "0": "Mature Cytotoxic c0",
    "1": "Transitional Cytotoxic c1",
    "2": "Mature Cytotoxic TCF7+ c2",
    "3": "T c3",
    "4": "Unknown_Kidney c4",
    "5": "Mature Cytotoxic Engineered c5",
    "6": "Mature Cytotoxic c6",
    "7": "Proliferative c7",
    "8": "Unknown_Lung_6 c8",
    "9": "Cytokine-Stimulated Effector c9",
    "10": "Transitional Cytotoxic Tissue-Resident c10",
    "11": "Transitional Cytotoxic Tissue-Resident c11",
    "12": "Cytokine-Stimulated CCR7+ c12",
    "13": "Unknown_Lung_5 c13",
    "14": "Unconventional c14",
    "15": "B c15",
    "16": "Unknown_BM_1 c16",
    "17": "Unknown_Lung_4 c17",
    "18": "Regulatory c18",
    "19": "Unknown_Lung_1 c19",
    "20": "Myeloid-like c20",
}

PRESET_PAIRS = {
    "cytokine": [("9", "12")],
    "priority_pairs": [
        ("10", "11"),
        ("10", "1"),
        ("18", "10"),
        ("8", "13"),
        ("14", "1"),
    ],
    "review_pairs": [
        ("10", "11"),
        ("10", "1"),
        ("11", "1"),
        ("8", "13"),
        ("8", "17"),
        ("13", "17"),
        ("18", "10"),
        ("18", "11"),
        ("14", "1"),
        ("14", "10"),
        ("4", "8"),
        ("4", "13"),
    ],
}


def main():
    args = parse_args()
    in_path = os.path.join(
        cfg.BASE_OUTDIR,
        "leiden_validation",
        "validation_scvi_leiden.h5ad",
    )
    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if GROUPBY not in adata.obs:
        raise KeyError(f"{GROUPBY!r} not found in adata.obs")

    adata.obs[GROUPBY] = adata.obs[GROUPBY].astype(str)

    pairs = get_pairs(args)
    print(f"[PAIRWISE] Running {len(pairs)} comparison(s)")
    for cluster_a, cluster_b, label_a, label_b in pairs:
        run_pairwise(adata, cluster_a, cluster_b, label_a, label_b)

    print("[DONE] Pairwise DE complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run validation-set pairwise DE between leiden_0_4 clusters."
    )
    parser.add_argument("--groupby", default=GROUPBY)
    parser.add_argument("--cluster-a")
    parser.add_argument("--cluster-b")
    parser.add_argument("--label-a")
    parser.add_argument("--label-b")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_PAIRS),
        help="Run a predefined set of cluster comparisons.",
    )
    return parser.parse_args()


def get_pairs(args):
    if args.groupby != GROUPBY:
        raise ValueError(
            f"This script currently expects --groupby {GROUPBY!r}; got {args.groupby!r}"
        )

    if args.preset:
        return [
            (
                cluster_a,
                cluster_b,
                CLUSTER_LABELS.get(cluster_a, f"cluster_{cluster_a}"),
                CLUSTER_LABELS.get(cluster_b, f"cluster_{cluster_b}"),
            )
            for cluster_a, cluster_b in PRESET_PAIRS[args.preset]
        ]

    if not args.cluster_a or not args.cluster_b:
        raise ValueError(
            "Provide --cluster-a and --cluster-b, or use --preset review_pairs"
        )

    label_a = args.label_a or CLUSTER_LABELS.get(args.cluster_a, f"cluster_{args.cluster_a}")
    label_b = args.label_b or CLUSTER_LABELS.get(args.cluster_b, f"cluster_{args.cluster_b}")
    return [(str(args.cluster_a), str(args.cluster_b), label_a, label_b)]


def run_pairwise(adata, cluster_a, cluster_b, label_a, label_b):
    comparison_name = f"{GROUPBY}_{cluster_a}_vs_{cluster_b}"
    outdir = os.path.join(
        cfg.BASE_OUTDIR,
        "markers",
        "validation",
        GROUPBY,
        "pairwise",
        comparison_name,
    )
    ensure_dirs(outdir)

    mask = adata.obs[GROUPBY].isin([cluster_a, cluster_b]).values
    adata_pair = adata[mask].copy()
    if adata_pair.n_obs == 0:
        raise ValueError(f"No cells found for clusters {cluster_a} and {cluster_b}")

    label_map = {cluster_a: label_a, cluster_b: label_b}
    adata_pair.obs[PAIR_KEY] = (
        adata_pair.obs[GROUPBY].astype(str).map(label_map).astype("category")
    )

    print(
        f"\n[PAIRWISE] {label_a} vs {label_b}: "
        f"{adata_pair.n_obs:,} cells"
    )
    print(adata_pair.obs[PAIR_KEY].value_counts().to_string())

    summary = summarize_pair_metadata(adata_pair)
    summary_path = os.path.join(outdir, f"{comparison_name}_metadata_summary.csv")
    summary.to_csv(summary_path)
    print(f"[SAVE] {summary_path}")

    # DE should use expression, not latent space. Normalize/log on a copy.
    ad = adata_pair.copy()
    sc.pp.normalize_total(ad, target_sum=1e4)
    sc.pp.log1p(ad)

    print("[DE] Running direct Wilcoxon pairwise comparison...")
    sc.tl.rank_genes_groups(
        ad,
        groupby=PAIR_KEY,
        method="wilcoxon",
        pts=True,
        tie_correct=True,
    )

    all_markers = sc.get.rank_genes_groups_df(ad, group=None)
    all_path = os.path.join(outdir, f"{comparison_name}_all_markers_wilcoxon.csv")
    all_markers.to_csv(all_path, index=False)
    print(f"[SAVE] {all_path}")

    top_markers = select_top_markers(all_markers, n_top=N_TOP_TABLE)
    top_path = os.path.join(outdir, f"{comparison_name}_top{N_TOP_TABLE}_per_group.csv")
    top_markers.to_csv(top_path, index=False)
    print(f"[SAVE] {top_path}")

    selected = select_plot_markers(top_markers, n_per_group=N_TOP_PLOT_PER_GROUP)
    selected_path = os.path.join(outdir, f"{comparison_name}_selected_plot_markers.txt")
    pd.Series(selected, name="gene").to_csv(selected_path, index=False, header=False)
    print(f"[SAVE] {selected_path}")

    if selected:
        save_dotplot(ad, selected, outdir, comparison_name)
        save_matrixplot(ad, selected, outdir, comparison_name)
    else:
        print("[WARN] No selected markers passed filtering; skipping plots.")


def summarize_pair_metadata(adata):
    rows = []
    for label, obs in adata.obs.groupby(PAIR_KEY, observed=True):
        row = {"pairwise_label": label, "n_cells": int(obs.shape[0])}
        for col in [cfg.LABEL_KEY, "tissue", cfg.DATASET_KEY, cfg.ASSAY_CLEAN_KEY]:
            if col not in obs:
                continue
            vc = obs[col].astype(str).value_counts()
            row[f"top_{col}"] = vc.index[0]
            row[f"top_{col}_frac"] = float(vc.iloc[0] / vc.sum())
            row[f"{col}_composition"] = "; ".join(
                f"{idx}:{cnt}" for idx, cnt in vc.head(10).items()
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("pairwise_label")


def select_top_markers(markers, n_top):
    df = markers.copy()
    if "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"] > 0].copy()
    sort_cols = [c for c in ["group", "pvals_adj", "scores"] if c in df.columns]
    ascending = [True, True, False][: len(sort_cols)]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)
    return df.groupby("group", group_keys=False).head(n_top)


def select_plot_markers(top_markers, n_per_group):
    selected = []
    for _, df_group in top_markers.groupby("group"):
        for gene in df_group["names"].astype(str).head(n_per_group):
            if gene not in selected:
                selected.append(gene)
    return selected


def save_dotplot(adata, selected, outdir, comparison_name):
    print(f"[PLOT] Dotplot with {len(selected)} selected markers")
    dot = sc.pl.dotplot(
        adata,
        var_names=selected,
        groupby=PAIR_KEY,
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    path = os.path.join(outdir, f"{comparison_name}_dotplot.png")
    dot.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {path}")
    plt.close("all")


def save_matrixplot(adata, selected, outdir, comparison_name):
    print(f"[PLOT] Matrixplot with {len(selected)} selected markers")
    matrix = sc.pl.matrixplot(
        adata,
        var_names=selected,
        groupby=PAIR_KEY,
        standard_scale="var",
        show=False,
        return_fig=True,
    )
    path = os.path.join(outdir, f"{comparison_name}_matrixplot.png")
    matrix.savefig(path, dpi=300, bbox_inches="tight")
    print(f"[SAVE] {path}")
    plt.close("all")


if __name__ == "__main__":
    main()
