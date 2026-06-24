#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
for cache_dir in [os.environ["MPLCONFIGDIR"], os.environ["NUMBA_CACHE_DIR"]]:
    os.makedirs(cache_dir, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs
from nk_project.plot_style import (
    LEGEND_FONT_SIZE,
    SMALL_TICK_LABEL_SIZE,
    set_presentation_style,
    style_axis,
    style_figure,
)

set_presentation_style()


DEFAULT_INPUT = "outputs/scvi_batch_strategy_comparison/runs/dataset_assay/latents/scvi_full_with_latent.h5ad"
DEFAULT_RESOLUTIONS = [0.2, 0.3, 0.4, 0.5, 0.6]


def main():
    args = parse_args()
    ensure_dirs(args.outdir)

    print(f"[LOAD] {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad)
    if args.latent_key not in adata.obsm:
        raise KeyError(f"{args.latent_key!r} not found in adata.obsm")

    full_assignments = {}
    full_counts = {}
    rows = []
    jaccard_rows = []

    for resolution in args.resolutions:
        key = leiden_key(resolution)
        print("\n" + "=" * 90)
        print(f"[RESOLUTION] {resolution:g} ({key})")
        print("=" * 90)

        if key in adata.obs and not args.recompute_full:
            print(f"[FULL] using existing {key}")
            full_labels = adata.obs[key].astype(str).to_numpy()
        else:
            full_labels = run_leiden_labels(
                adata,
                resolution=resolution,
                latent_key=args.latent_key,
                n_neighbors=args.n_neighbors,
                seed=args.seed,
            )
            adata.obs[key] = pd.Categorical(full_labels)

        full_assignments[resolution] = full_labels
        full_counts[resolution] = pd.Series(full_labels).value_counts().sort_index()
        print(
            f"[FULL] n_clusters={full_counts[resolution].size}; "
            f"min={full_counts[resolution].min():,}; max={full_counts[resolution].max():,}"
        )

        for repeat in range(args.n_repeats):
            repeat_seed = args.seed + repeat + int(round(resolution * 1000))
            rng = np.random.default_rng(repeat_seed)
            n_sub = int(round(args.subsample_frac * adata.n_obs))
            sub_idx = np.sort(rng.choice(np.arange(adata.n_obs), size=n_sub, replace=False))
            sub = adata[sub_idx].copy()

            sub_labels = run_leiden_labels(
                sub,
                resolution=resolution,
                latent_key=args.latent_key,
                n_neighbors=args.n_neighbors,
                seed=repeat_seed,
            )
            full_sub_labels = full_labels[sub_idx]
            ari = adjusted_rand_score(full_sub_labels, sub_labels)
            nmi = normalized_mutual_info_score(full_sub_labels, sub_labels)
            jac = cluster_jaccard_recovery(full_sub_labels, sub_labels)

            print(
                f"[REPEAT {repeat + 1:02d}/{args.n_repeats}] "
                f"n={n_sub:,}; clusters={len(set(sub_labels))}; ARI={ari:.3f}; NMI={nmi:.3f}; "
                f"median_jaccard={jac['best_jaccard'].median():.3f}"
            )

            rows.append(
                {
                    "resolution": resolution,
                    "repeat": repeat + 1,
                    "seed": repeat_seed,
                    "subsample_frac": args.subsample_frac,
                    "n_subsample_cells": int(n_sub),
                    "n_full_clusters": int(full_counts[resolution].size),
                    "n_subsample_clusters": int(len(set(sub_labels))),
                    "ari_vs_full": float(ari),
                    "nmi_vs_full": float(nmi),
                    "median_cluster_jaccard": float(jac["best_jaccard"].median()),
                    "mean_cluster_jaccard": float(jac["best_jaccard"].mean()),
                    "pct_clusters_jaccard_ge_0_50": float((jac["best_jaccard"] >= 0.50).mean() * 100.0),
                    "pct_clusters_jaccard_ge_0_60": float((jac["best_jaccard"] >= 0.60).mean() * 100.0),
                    "pct_clusters_jaccard_ge_0_75": float((jac["best_jaccard"] >= 0.75).mean() * 100.0),
                }
            )
            jac.insert(0, "repeat", repeat + 1)
            jac.insert(0, "resolution", resolution)
            jaccard_rows.append(jac)

    metrics = pd.DataFrame(rows)
    jaccard = pd.concat(jaccard_rows, ignore_index=True) if jaccard_rows else pd.DataFrame()
    summary = summarize_stability(metrics, full_counts, args)

    metrics_path = os.path.join(args.outdir, "leiden_stability_repeats.csv")
    jaccard_path = os.path.join(args.outdir, "leiden_stability_cluster_jaccard.csv")
    summary_path = os.path.join(args.outdir, "leiden_stability_summary.csv")
    metrics.to_csv(metrics_path, index=False)
    jaccard.to_csv(jaccard_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[SAVE] {metrics_path}")
    print(f"[SAVE] {jaccard_path}")
    print(f"[SAVE] {summary_path}")

    full_h5ad_path = os.path.join(args.outdir, "full_leiden_with_stability_resolutions.h5ad")
    adata.write(full_h5ad_path)
    print(f"[SAVE] {full_h5ad_path}")

    plot_stability_summary(summary, args.outdir)
    plot_jaccard_boxplot(jaccard, args.outdir)
    plot_cluster_size_summary(summary, args.outdir)
    print("[DONE] Leiden cluster stability analysis complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Assess Leiden cluster stability by repeated subsampling and comparison "
            "to the full-data solution."
        )
    )
    parser.add_argument("--input-h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--outdir", default=os.path.join(cfg.BASE_OUTDIR, "leiden_discovery", "stability"))
    parser.add_argument("--latent-key", default="X_scVI")
    parser.add_argument("--resolutions", type=float, nargs="+", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--subsample-frac", type=float, default=0.50)
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-neighbors", type=int, default=cfg.DISCOVERY_N_NEIGHBORS)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--min-cluster-size", type=int, default=500)
    parser.add_argument("--recompute-full", action="store_true")
    return parser.parse_args()


def run_leiden_labels(adata, *, resolution, latent_key, n_neighbors, seed):
    work = adata.copy()
    sc.pp.neighbors(work, use_rep=latent_key, n_neighbors=n_neighbors, random_state=seed)
    key = "leiden_tmp"
    sc.tl.leiden(
        work,
        resolution=resolution,
        key_added=key,
        random_state=seed,
        flavor="igraph",
        n_iterations=2,
        directed=False,
    )
    return work.obs[key].astype(str).to_numpy()


def cluster_jaccard_recovery(full_labels, sub_labels):
    full_labels = np.asarray(full_labels).astype(str)
    sub_labels = np.asarray(sub_labels).astype(str)
    sub_clusters = sorted(pd.unique(sub_labels), key=cluster_sort_key)
    rows = []
    for full_cluster in sorted(pd.unique(full_labels), key=cluster_sort_key):
        full_mask = full_labels == full_cluster
        best_jaccard = 0.0
        best_sub_cluster = None
        best_intersection = 0
        best_union = int(full_mask.sum())
        for sub_cluster in sub_clusters:
            sub_mask = sub_labels == sub_cluster
            intersection = int(np.logical_and(full_mask, sub_mask).sum())
            if intersection == 0:
                continue
            union = int(np.logical_or(full_mask, sub_mask).sum())
            score = intersection / union if union else 0.0
            if score > best_jaccard:
                best_jaccard = score
                best_sub_cluster = sub_cluster
                best_intersection = intersection
                best_union = union
        rows.append(
            {
                "full_cluster": str(full_cluster),
                "full_cluster_subsample_n": int(full_mask.sum()),
                "best_subsample_cluster": best_sub_cluster,
                "best_jaccard": float(best_jaccard),
                "best_intersection": int(best_intersection),
                "best_union": int(best_union),
            }
        )
    return pd.DataFrame(rows)


def summarize_stability(metrics, full_counts, args):
    rows = []
    for resolution, sub in metrics.groupby("resolution", sort=True):
        counts = full_counts[resolution]
        n_cells = int(counts.sum())
        tiny_cells = int(counts[counts < args.min_cluster_size].sum())
        rows.append(
            {
                "resolution": resolution,
                "n_clusters": int(counts.size),
                "min_cluster_size": int(counts.min()),
                "median_cluster_size": float(counts.median()),
                "max_cluster_size": int(counts.max()),
                "n_tiny_clusters": int((counts < args.min_cluster_size).sum()),
                "pct_cells_in_tiny_clusters": 100.0 * tiny_cells / max(n_cells, 1),
                "mean_ari_vs_full": float(sub["ari_vs_full"].mean()),
                "sd_ari_vs_full": float(sub["ari_vs_full"].std(ddof=1)),
                "mean_nmi_vs_full": float(sub["nmi_vs_full"].mean()),
                "sd_nmi_vs_full": float(sub["nmi_vs_full"].std(ddof=1)),
                "mean_median_cluster_jaccard": float(sub["median_cluster_jaccard"].mean()),
                "mean_pct_clusters_jaccard_ge_0_60": float(sub["pct_clusters_jaccard_ge_0_60"].mean()),
                "mean_pct_clusters_jaccard_ge_0_75": float(sub["pct_clusters_jaccard_ge_0_75"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("resolution")


def plot_stability_summary(summary, outdir):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = summary["resolution"].astype(float).to_numpy()
    ax.errorbar(
        x,
        summary["mean_ari_vs_full"],
        yerr=summary["sd_ari_vs_full"].fillna(0),
        marker="o",
        linewidth=2.0,
        capsize=3,
        label="ARI vs full",
    )
    ax.errorbar(
        x,
        summary["mean_nmi_vs_full"],
        yerr=summary["sd_nmi_vs_full"].fillna(0),
        marker="o",
        linewidth=2.0,
        capsize=3,
        label="NMI vs full",
    )
    ax.plot(
        x,
        summary["mean_median_cluster_jaccard"],
        marker="o",
        linewidth=2.0,
        label="median cluster Jaccard",
    )
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("Stability score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Leiden stability across 50% subsampling")
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    style_axis(ax)
    fig.tight_layout()
    path = os.path.join(outdir, "leiden_stability_ari_nmi_jaccard.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {path}")


def plot_jaccard_boxplot(jaccard, outdir):
    if jaccard.empty:
        return
    resolutions = sorted(jaccard["resolution"].unique())
    values = [jaccard.loc[jaccard["resolution"] == r, "best_jaccard"].dropna().values for r in resolutions]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bp = ax.boxplot(values, labels=[f"{r:g}" for r in resolutions], patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#80b1d3")
        patch.set_alpha(0.75)
    ax.axhline(0.60, color="#b23a48", linestyle="--", linewidth=1.5, label="Jaccard 0.60")
    ax.axhline(0.75, color="#4d9221", linestyle="--", linewidth=1.5, label="Jaccard 0.75")
    ax.set_xlabel("Leiden resolution")
    ax.set_ylabel("Best cluster-wise Jaccard")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cluster-wise recovery across subsamples")
    ax.legend(frameon=False, fontsize=LEGEND_FONT_SIZE)
    style_axis(ax)
    fig.tight_layout()
    path = os.path.join(outdir, "leiden_cluster_jaccard_by_resolution.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {path}")


def plot_cluster_size_summary(summary, outdir):
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    x = np.arange(summary.shape[0])
    labels = [f"{r:g}" for r in summary["resolution"]]
    ax1.bar(x, summary["n_clusters"], color="#4c78a8", alpha=0.85, label="clusters")
    ax1.set_ylabel("Number of clusters")
    ax1.set_xlabel("Leiden resolution")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        summary["pct_cells_in_tiny_clusters"],
        color="#b23a48",
        marker="o",
        linewidth=2.0,
        label=f"cells in clusters < {int(summary['min_cluster_size'].min())}",
    )
    ax2.set_ylabel("Cells in tiny clusters (%)")
    ax1.set_title("Resolution granularity and tiny-cluster burden")
    style_axis(ax1)
    style_axis(ax2)
    fig.tight_layout()
    path = os.path.join(outdir, "leiden_resolution_cluster_size_summary.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {path}")


def leiden_key(resolution):
    return f"leiden_{str(resolution).replace('.', '_')}"


def cluster_sort_key(value):
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


if __name__ == "__main__":
    main()
