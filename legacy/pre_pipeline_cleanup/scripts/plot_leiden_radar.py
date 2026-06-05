#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


SELECTED_RESOLUTIONS = [0.3, 0.4, 0.6, 0.8, 1.2]
SILHOUETTE_MAX_CELLS = 10000


def main():
    in_path = os.path.join(cfg.BASE_OUTDIR, "leiden_validation", "validation_scvi_leiden.h5ad")
    outdir = os.path.join(cfg.BASE_OUTDIR, "leiden_validation")
    ensure_dirs(outdir)

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if "X_scVI" not in adata.obsm:
        raise KeyError("X_scVI not found in adata.obsm")

    rows = []
    for resolution in cfg.LEIDEN_RESOLUTIONS:
        key = f"leiden_{str(resolution).replace('.', '_')}"
        if key not in adata.obs:
            print(f"[SKIP] {key}: not found")
            continue

        labels = adata.obs[cfg.LABEL_KEY].astype(str).values
        clusters = adata.obs[key].astype(str).values
        dataset = adata.obs[cfg.DATASET_KEY].astype(str).values
        assay = adata.obs[cfg.ASSAY_CLEAN_KEY].astype(str).values
        cluster_sizes = pd.Series(clusters).value_counts()

        rows.append(
            {
                "resolution": resolution,
                "leiden_key": key,
                "n_clusters": int(cluster_sizes.size),
                "min_cluster_size": int(cluster_sizes.min()),
                "median_cluster_size": float(cluster_sizes.median()),
                "cluster_size_cv": float(cluster_sizes.std() / cluster_sizes.mean()),
                "ARI": adjusted_rand_score(labels, clusters),
                "NMI": normalized_mutual_info_score(labels, clusters),
                "silhouette": compute_silhouette(adata.obsm["X_scVI"], clusters, cfg.SEED),
                "dataset_purity": weighted_cluster_purity(clusters, dataset),
                "assay_purity": weighted_cluster_purity(clusters, assay),
            }
        )

    metrics = pd.DataFrame(rows)
    metrics = add_radar_scores(metrics)

    csv_path = os.path.join(outdir, "leiden_radar_metrics.csv")
    metrics.to_csv(csv_path, index=False)
    print("[METRICS]")
    print(metrics.round(4).to_string(index=False))
    print(f"[SAVE] {csv_path}")

    plot_radar(metrics, outdir)


def compute_silhouette(X, clusters, seed):
    clusters = np.asarray(clusters).astype(str)
    if len(np.unique(clusters)) < 2:
        return np.nan

    rng = np.random.default_rng(seed)
    n = X.shape[0]
    if n > SILHOUETTE_MAX_CELLS:
        idx = rng.choice(np.arange(n), size=SILHOUETTE_MAX_CELLS, replace=False)
        X = X[idx]
        clusters = clusters[idx]

    if len(np.unique(clusters)) < 2:
        return np.nan
    return float(silhouette_score(X, clusters, metric="euclidean"))


def weighted_cluster_purity(clusters, values):
    tab = pd.crosstab(pd.Series(clusters, name="cluster"), pd.Series(values, name="value"))
    cluster_sizes = tab.sum(axis=1)
    dominant_frac = tab.max(axis=1) / cluster_sizes
    return float(np.average(dominant_frac, weights=cluster_sizes))


def minmax_to_good(values, *, higher_is_better=True):
    s = pd.Series(values, dtype=float)
    if s.notna().sum() <= 1:
        return pd.Series(np.ones(len(s)), index=s.index)
    lo = s.min(skipna=True)
    hi = s.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        out = pd.Series(np.ones(len(s)), index=s.index)
    else:
        out = (s - lo) / (hi - lo)
    if not higher_is_better:
        out = 1.0 - out
    return out.clip(0, 1)


def add_radar_scores(metrics):
    metrics = metrics.copy()
    metrics["ARI_score"] = metrics["ARI"].clip(lower=0, upper=1)
    metrics["NMI_score"] = metrics["NMI"].clip(lower=0, upper=1)
    metrics["silhouette_score"] = ((metrics["silhouette"] + 1.0) / 2.0).clip(0, 1)
    metrics["cluster_size_balance"] = minmax_to_good(
        metrics["cluster_size_cv"], higher_is_better=False
    )
    metrics["dataset_mixing"] = (1.0 - metrics["dataset_purity"]).clip(0, 1)
    metrics["assay_mixing"] = (1.0 - metrics["assay_purity"]).clip(0, 1)
    return metrics


def plot_radar(metrics, outdir):
    selected = metrics[metrics["resolution"].isin(SELECTED_RESOLUTIONS)].copy()
    if selected.empty:
        selected = metrics.copy()

    axes_labels = [
        "ARI",
        "NMI",
        "Silhouette",
        "Size balance",
    ]
    value_cols = [
        "ARI_score",
        "NMI_score",
        "silhouette_score",
        "cluster_size_balance",
    ]

    angles = np.linspace(0, 2 * np.pi, len(value_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    cmap = plt.get_cmap("tab10")

    for i, row in selected.sort_values("resolution").reset_index(drop=True).iterrows():
        values = [float(row[col]) for col in value_cols]
        values += values[:1]
        label = f"{row['resolution']} ({int(row['n_clusters'])} clusters)"
        color = cmap(i % cmap.N)
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8)
    ax.set_title(
        "Validation Leiden resolution tradeoffs\n"
        "Agreement with NK_State + clustering quality",
        fontsize=13,
        pad=18,
    )
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.10, 1.08), fontsize=9)

    png = os.path.join(outdir, "leiden_resolution_radar.png")
    pdf = os.path.join(outdir, "leiden_resolution_radar.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")


if __name__ == "__main__":
    main()
