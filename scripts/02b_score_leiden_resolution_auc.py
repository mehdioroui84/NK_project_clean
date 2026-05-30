#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from itertools import chain

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")
for cache_dir in [os.environ["MPLCONFIGDIR"], os.environ["NUMBA_CACHE_DIR"]]:
    os.makedirs(cache_dir, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import rankdata
from sklearn.cluster import MiniBatchKMeans

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


DEFAULT_RESOLUTIONS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
DEFAULT_TOP_N = 50
DEFAULT_MAX_CELLS_PER_SIDE = 3000
DEFAULT_MIN_CLUSTER_SIZE = 500
DEFAULT_MIN_GOOD_MARKERS = 20
DEFAULT_POSITIVE_AUC = 0.80
DEFAULT_NEGATIVE_AUC = 0.20
DEFAULT_COMPARISON_CLUSTER_COUNTS = [10, 15, 20, 25, 30]
POINT_SIZE = 0.08
POINT_ALPHA = 0.75


def main():
    args = parse_args()
    input_h5ad = args.input_h5ad or os.path.join(
        cfg.BASE_OUTDIR,
        "leiden_discovery",
        "full_scvi_leiden.h5ad",
    )
    outdir = args.outdir or os.path.join(
        cfg.BASE_OUTDIR,
        "leiden_discovery",
        "resolution_auc_qc",
    )
    ensure_dirs(outdir)

    print(f"[LOAD] {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    if args.latent_key not in adata.obsm:
        available = ", ".join(map(str, adata.obsm.keys()))
        raise KeyError(f"{args.latent_key!r} not found in adata.obsm. Available: {available}")

    if not args.skip_leiden:
        ensure_neighbors(adata, args)
        add_leiden_resolutions(adata, args.resolutions, args)

    print("[NORMALIZE] copy + normalize_total + log1p for AUC ranking")
    expr = adata.copy()
    sc.pp.normalize_total(expr, target_sum=1e4)
    sc.pp.log1p(expr)

    if not args.skip_leiden:
        cluster_tables = []
        summary_rows = []
        for resolution in args.resolutions:
            groupby = leiden_key(resolution)
            print(f"[AUC] scoring {groupby}")
            cluster_table = score_resolution_auc(expr, groupby, args)
            cluster_table.insert(0, "resolution", resolution)
            cluster_tables.append(cluster_table)
            summary_rows.append(summarize_resolution(cluster_table, resolution, args))

        by_cluster = pd.concat(cluster_tables, ignore_index=True)
        summary = pd.DataFrame(summary_rows).sort_values("resolution").reset_index(drop=True)
        recommended_resolution = choose_recommended_resolution(summary)
        summary["recommended"] = summary["resolution"].eq(recommended_resolution)

        by_cluster_path = os.path.join(outdir, "leiden_resolution_auc_by_cluster.csv")
        summary_path = os.path.join(outdir, "leiden_resolution_auc_summary.csv")
        by_cluster.to_csv(by_cluster_path, index=False)
        summary.to_csv(summary_path, index=False)
        print(f"[SAVE] {by_cluster_path}")
        print(f"[SAVE] {summary_path}")

        plot_auc_summary(summary, recommended_resolution, outdir)
        plot_recommended_umap(adata, recommended_resolution, outdir)
        save_recommended_assignments(adata, recommended_resolution, outdir)
    if args.include_kmeans:
        run_kmeans_cluster_count_benchmark(expr, adata, args, outdir)

    if not args.skip_leiden:
        rec = summary.loc[summary["recommended"]].iloc[0]
        print("\n[RECOMMENDED]")
        print(
            f"resolution={recommended_resolution:g}; "
            f"n_clusters={int(rec['n_clusters'])}; "
            f"median_top50_combined_auc_strength={rec['median_top50_combined_auc_strength']:.3f}; "
            f"pct_clusters_with_at_least_{args.min_good_markers}_good_markers={rec['pct_clusters_with_min_good_markers']:.1f}%; "
            f"composite_quality_score={rec['composite_quality_score']:.3f}"
        )
    print("[DONE] Exploratory Leiden resolution AUC QC complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Exploratory Leiden-resolution QC using one-vs-rest ROC-AUC marker separability. "
            "This does not modify the main annotation pipeline."
        )
    )
    parser.add_argument("--input-h5ad", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--latent-key", default="X_scVI")
    parser.add_argument("--resolutions", type=float, nargs="+", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--n-neighbors", type=int, default=cfg.DISCOVERY_N_NEIGHBORS)
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--max-cells-per-side", type=int, default=DEFAULT_MAX_CELLS_PER_SIDE)
    parser.add_argument("--block-size", type=int, default=400)
    parser.add_argument("--min-cluster-size", type=int, default=DEFAULT_MIN_CLUSTER_SIZE)
    parser.add_argument("--min-good-markers", type=int, default=DEFAULT_MIN_GOOD_MARKERS)
    parser.add_argument("--positive-auc-cutoff", type=float, default=DEFAULT_POSITIVE_AUC)
    parser.add_argument("--negative-auc-cutoff", type=float, default=DEFAULT_NEGATIVE_AUC)
    parser.add_argument("--recompute-neighbors", action="store_true")
    parser.add_argument(
        "--skip-leiden",
        action="store_true",
        help="Skip Leiden resolution scoring and only run requested optional benchmarks.",
    )
    parser.add_argument(
        "--include-kmeans",
        action="store_true",
        help="Also benchmark MiniBatchKMeans cluster counts on the scVI latent space.",
    )
    parser.add_argument(
        "--cluster-counts",
        type=int,
        nargs="+",
        default=DEFAULT_COMPARISON_CLUSTER_COUNTS,
        help="Cluster counts for optional KMeans benchmark. Default: 10 through 30.",
    )
    parser.add_argument("--kmeans-batch-size", type=int, default=8192)
    return parser.parse_args()


def ensure_neighbors(adata, args):
    if args.recompute_neighbors or "neighbors" not in adata.uns:
        print(f"[NEIGHBORS] computing on {args.latent_key}, n_neighbors={args.n_neighbors}")
        sc.pp.neighbors(
            adata,
            use_rep=args.latent_key,
            n_neighbors=args.n_neighbors,
            random_state=args.seed,
        )
    else:
        print("[NEIGHBORS] using existing neighbor graph")


def add_leiden_resolutions(adata, resolutions, args):
    for resolution in resolutions:
        key = leiden_key(resolution)
        if key in adata.obs:
            print(f"[LEIDEN] using existing {key}")
            adata.obs[key] = adata.obs[key].astype(str).astype("category")
            continue
        print(f"[LEIDEN] computing {key}")
        sc.tl.leiden(
            adata,
            resolution=resolution,
            key_added=key,
            random_state=args.seed,
        )
        adata.obs[key] = adata.obs[key].astype(str).astype("category")


def score_resolution_auc(adata, groupby, args):
    labels = adata.obs[groupby].astype(str).to_numpy()
    clusters = sorted(pd.unique(labels), key=cluster_sort_key)
    rng = np.random.default_rng(args.seed)
    rows = []
    for cluster_id in clusters:
        pos_idx = np.flatnonzero(labels == cluster_id)
        neg_idx = np.flatnonzero(labels != cluster_id)
        if len(pos_idx) < 2 or len(neg_idx) < 2:
            continue
        pos_sample = sample_indices(pos_idx, args.max_cells_per_side, rng)
        neg_sample = sample_indices(neg_idx, args.max_cells_per_side, rng)
        auc = marker_auc_for_cluster(
            adata.X,
            pos_sample,
            neg_sample,
            block_size=args.block_size,
        )
        rows.append(cluster_auc_summary(cluster_id, len(pos_idx), auc, adata.var_names, args))
    return pd.DataFrame(rows)


def marker_auc_for_cluster(x, pos_idx, neg_idx, *, block_size):
    sample_idx = np.concatenate([pos_idx, neg_idx])
    n_pos = len(pos_idx)
    n_neg = len(neg_idx)
    n_genes = x.shape[1]
    auc = np.empty(n_genes, dtype=np.float32)
    base = n_pos * (n_pos + 1) / 2.0
    denom = float(n_pos * n_neg)

    for start in range(0, n_genes, block_size):
        end = min(start + block_size, n_genes)
        block = x[sample_idx, start:end]
        if sparse.issparse(block):
            block = block.toarray()
        else:
            block = np.asarray(block)
        ranks = rankdata(block, axis=0, method="average")
        pos_rank_sum = ranks[:n_pos, :].sum(axis=0)
        auc[start:end] = (pos_rank_sum - base) / denom
    return auc


def cluster_auc_summary(cluster_id, n_cells, auc, var_names, args):
    auc = np.asarray(auc, dtype=float)
    pos_order = np.argsort(-auc)
    neg_order = np.argsort(auc)
    top_pos = auc[pos_order[: args.top_n]]
    top_neg = 1.0 - auc[neg_order[: args.top_n]]
    combined = np.concatenate([top_pos, top_neg])
    n_pos_good = int(np.sum(auc >= args.positive_auc_cutoff))
    n_neg_good = int(np.sum(auc <= args.negative_auc_cutoff))
    return {
        "cluster_id": str(cluster_id),
        "n_cells": int(n_cells),
        "median_top50_positive_auc": float(np.median(top_pos)),
        "median_top50_negative_auc_strength": float(np.median(top_neg)),
        "median_top50_combined_auc_strength": float(np.median(combined)),
        "n_positive_auc_ge_cutoff": n_pos_good,
        "n_negative_auc_le_cutoff": n_neg_good,
        "n_good_markers": n_pos_good + n_neg_good,
        "top_positive_genes": ";".join(map(str, np.asarray(var_names)[pos_order[: args.top_n]])),
        "top_negative_genes": ";".join(map(str, np.asarray(var_names)[neg_order[: args.top_n]])),
    }


def summarize_resolution(cluster_table, resolution, args):
    n_clusters = int(cluster_table.shape[0])
    n_tiny = int((cluster_table["n_cells"] < args.min_cluster_size).sum()) if n_clusters else 0
    pct_good = (
        100.0 * float((cluster_table["n_good_markers"] >= args.min_good_markers).mean())
        if n_clusters
        else 0.0
    )
    median_combined = float(cluster_table["median_top50_combined_auc_strength"].median()) if n_clusters else np.nan
    tiny_fraction = n_tiny / n_clusters if n_clusters else 1.0
    composite = median_combined * (pct_good / 100.0) * (1.0 - tiny_fraction)
    return {
        "resolution": float(resolution),
        "n_clusters": n_clusters,
        "median_cluster_size": float(cluster_table["n_cells"].median()) if n_clusters else np.nan,
        "min_cluster_size": int(cluster_table["n_cells"].min()) if n_clusters else 0,
        "n_clusters_lt_min_size": n_tiny,
        "median_top50_positive_auc": float(cluster_table["median_top50_positive_auc"].median()) if n_clusters else np.nan,
        "median_top50_negative_auc_strength": float(cluster_table["median_top50_negative_auc_strength"].median()) if n_clusters else np.nan,
        "median_top50_combined_auc_strength": median_combined,
        "median_n_good_markers": float(cluster_table["n_good_markers"].median()) if n_clusters else np.nan,
        "pct_clusters_with_min_good_markers": pct_good,
        "composite_quality_score": float(composite),
    }


def choose_recommended_resolution(summary):
    if summary.empty:
        raise ValueError("No resolution summary rows were produced.")
    best = summary.sort_values(
        ["composite_quality_score", "median_top50_combined_auc_strength", "resolution"],
        ascending=[False, False, False],
    ).iloc[0]
    return float(best["resolution"])


def plot_auc_summary(summary, recommended_resolution, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    x = summary["resolution"].astype(float)

    axes[0].plot(x, summary["n_clusters"], marker="o", color="#0072B2")
    axes[0].set_title("Number of Leiden clusters")
    axes[0].set_xlabel("Resolution")
    axes[0].set_ylabel("n clusters")

    axes[1].plot(x, summary["median_top50_positive_auc"], marker="o", label="top50 positive AUC", color="#D55E00")
    axes[1].plot(x, summary["median_top50_negative_auc_strength"], marker="o", label="top50 negative strength", color="#0072B2")
    axes[1].plot(x, summary["median_top50_combined_auc_strength"], marker="o", label="combined", color="#009E73")
    axes[1].set_title("Marker separability")
    axes[1].set_xlabel("Resolution")
    axes[1].set_ylabel("Median cluster AUC strength")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(x, summary["pct_clusters_with_min_good_markers"], marker="o", color="#CC79A7")
    axes[2].set_title("Clusters with enough high-AUC markers")
    axes[2].set_xlabel("Resolution")
    axes[2].set_ylabel("% clusters")
    axes[2].set_ylim(0, 105)

    axes[3].plot(x, summary["composite_quality_score"], marker="o", color="#E7298A")
    axes[3].set_title("Composite quantitative score")
    axes[3].set_xlabel("Resolution")
    axes[3].set_ylabel("score")

    for ax in axes:
        ax.axvline(recommended_resolution, color="#333333", linestyle="--", linewidth=1)
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)

    fig.suptitle(
        f"Exploratory Leiden resolution AUC QC: recommended {recommended_resolution:g}",
        fontsize=14,
    )
    plt.tight_layout()
    path = os.path.join(outdir, "leiden_resolution_auc_summary.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {path}")
    plt.close(fig)


def plot_recommended_umap(adata, resolution, outdir):
    if "X_umap" not in adata.obsm:
        print("[SKIP] X_umap not found; cannot save recommended-resolution UMAP.")
        return
    key = leiden_key(resolution)
    values = adata.obs[key].astype(str).to_numpy()
    xy = adata.obsm["X_umap"]
    fig, ax = plt.subplots(figsize=(9, 8))
    scatter_categorical(ax, xy, values, f"Recommended Leiden resolution {resolution:g}: {key}")
    path = os.path.join(outdir, f"{key}_recommended_resolution_umap.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {path}")
    plt.close(fig)


def save_recommended_assignments(adata, resolution, outdir):
    key = leiden_key(resolution)
    path = os.path.join(outdir, f"{key}_recommended_assignments.csv")
    adata.obs[[key]].to_csv(path)
    print(f"[SAVE] {path}")


def run_kmeans_cluster_count_benchmark(expr, adata, args, outdir):
    latent = np.asarray(adata.obsm[args.latent_key])
    cluster_counts = sorted({int(k) for k in args.cluster_counts if int(k) >= 2})
    if not cluster_counts:
        print("[SKIP] No valid cluster counts for KMeans benchmark.")
        return

    print(f"[KMEANS] benchmarking cluster counts: {cluster_counts[0]}..{cluster_counts[-1]}")
    cluster_tables = []
    summary_rows = []
    for n_clusters in cluster_counts:
        key = f"kmeans_{n_clusters}"
        print(f"[KMEANS] fitting {key}")
        model = MiniBatchKMeans(
            n_clusters=n_clusters,
            random_state=args.seed,
            batch_size=args.kmeans_batch_size,
            n_init=10,
            reassignment_ratio=0.01,
        )
        adata.obs[key] = model.fit_predict(latent).astype(str)
        expr.obs[key] = adata.obs[key].values
        cluster_table = score_resolution_auc(expr, key, args)
        cluster_table.insert(0, "method", "MiniBatchKMeans")
        cluster_table.insert(1, "n_requested_clusters", n_clusters)
        cluster_tables.append(cluster_table)
        summary_rows.append(summarize_cluster_count(cluster_table, "MiniBatchKMeans", n_clusters, args))

    by_cluster = pd.concat(cluster_tables, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values("n_requested_clusters").reset_index(drop=True)

    by_cluster_path = os.path.join(outdir, "kmeans_cluster_count_auc_by_cluster.csv")
    summary_path = os.path.join(outdir, "kmeans_cluster_count_auc_summary.csv")
    by_cluster.to_csv(by_cluster_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[SAVE] {by_cluster_path}")
    print(f"[SAVE] {summary_path}")

    plot_cluster_count_benchmark(summary, outdir)


def summarize_cluster_count(cluster_table, method, n_requested_clusters, args):
    row = summarize_resolution(cluster_table, float(n_requested_clusters), args)
    row.pop("resolution", None)
    row["method"] = method
    row["n_requested_clusters"] = int(n_requested_clusters)
    return row


def plot_cluster_count_benchmark(summary, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    x = summary["n_requested_clusters"].astype(int)

    axes[0].plot(x, summary["n_clusters"], marker="o", color="#0072B2")
    axes[0].plot(x, x, linestyle="--", linewidth=1, color="#777777", label="requested k")
    axes[0].set_title("Recovered KMeans clusters")
    axes[0].set_xlabel("Requested cluster count")
    axes[0].set_ylabel("n non-empty clusters")
    axes[0].legend(frameon=False, fontsize=8)

    axes[1].plot(x, summary["median_top50_positive_auc"], marker="o", label="top50 positive AUC", color="#D55E00")
    axes[1].plot(x, summary["median_top50_negative_auc_strength"], marker="o", label="top50 negative strength", color="#0072B2")
    axes[1].plot(x, summary["median_top50_combined_auc_strength"], marker="o", label="combined", color="#009E73")
    axes[1].set_title("Marker separability")
    axes[1].set_xlabel("Requested cluster count")
    axes[1].set_ylabel("Median cluster AUC strength")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(x, summary["pct_clusters_with_min_good_markers"], marker="o", color="#CC79A7")
    axes[2].set_title("Clusters with enough high-AUC markers")
    axes[2].set_xlabel("Requested cluster count")
    axes[2].set_ylabel("% clusters")
    axes[2].set_ylim(0, 105)

    axes[3].plot(x, summary["composite_quality_score"], marker="o", color="#E7298A")
    axes[3].set_title("Composite quantitative score")
    axes[3].set_xlabel("Requested cluster count")
    axes[3].set_ylabel("score")

    for ax in axes:
        ax.axvline(25, color="#333333", linestyle="--", linewidth=1)
        ax.grid(True, color="#dddddd", linewidth=0.6, alpha=0.8)

    fig.suptitle("Exploratory MiniBatchKMeans cluster-count AUC QC: k=10..30", fontsize=14)
    plt.tight_layout()
    path = os.path.join(outdir, "kmeans_cluster_count_auc_summary.png")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {path}")
    plt.close(fig)


def scatter_categorical(ax, xy, values, title):
    categories = sorted(pd.unique(values), key=cluster_sort_key)
    colors = category_colors(categories)
    for category in categories:
        mask = values == category
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=0,
            c=colors[category],
            rasterized=True,
        )
        center = np.median(xy[mask], axis=0)
        ax.text(
            center[0],
            center[1],
            str(category),
            ha="center",
            va="center",
            fontsize=8,
            weight="bold",
            color="#222222",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.55, "pad": 1.0},
        )
    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=colors[c], markeredgecolor="none", markersize=5, label=str(c))
        for c in categories
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="datalim")


def category_colors(categories):
    palette = list(
        chain(
            plt.get_cmap("tab20").colors,
            plt.get_cmap("tab20b").colors,
            plt.get_cmap("tab20c").colors,
            plt.get_cmap("Set3").colors,
            plt.get_cmap("Dark2").colors,
        )
    )
    return {category: palette[i % len(palette)] for i, category in enumerate(categories)}


def sample_indices(indices, max_n, rng):
    indices = np.asarray(indices)
    if len(indices) <= max_n:
        return indices
    return np.sort(rng.choice(indices, size=max_n, replace=False))


def leiden_key(resolution):
    text = f"{resolution:g}".replace(".", "_")
    return f"leiden_{text}"


def cluster_sort_key(value):
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


if __name__ == "__main__":
    main()
