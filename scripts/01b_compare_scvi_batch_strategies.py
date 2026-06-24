#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs
from nk_project.evaluation.scanvi_full_plots import (
    PREFERRED_TISSUE_COLORS,
    distinct_color_map,
    scatter_by_category,
)
from nk_project.metrics import (
    compute_batch_asw_label_aware,
    compute_graph_connectivity,
    compute_knn_batch_accuracy,
    compute_knn_label_accuracy,
    compute_label_asw,
    subsample_for_metrics,
)
from nk_project.plot_style import (
    LEGEND_FONT_SIZE,
    SMALL_TICK_LABEL_SIZE,
    set_presentation_style,
    style_all_legends,
    style_axis,
    style_figure,
)
from nk_project.workflows import train_scvi

set_presentation_style()


DEFAULT_INPUT = (
    "/rsrch5/home/genomic_med/suorouji/projects/lsf_run/"
    "cellxgene_nk_bt_plus_CB07_hvg2k_nkmarker_rescue.h5ad"
)

STRATEGIES = {
    "assay_only": cfg.ASSAY_CLEAN_KEY,
    "dataset_only": cfg.DATASET_KEY,
    "dataset_assay": cfg.COMPOSITE_BATCH_KEY,
    "tissue_assay": "batch_tissue_assay",
    "dataset_tissue_assay": "batch_dataset_tissue_assay",
}

PREFERRED_ASSAY_COLORS = {
    "10x 3' transcription profiling": "#1F77B4",
    "10x 3' v1": "#FF7F0E",
    "10x 3' v2": "#2CA02C",
    "10x 3' v3": "#D62728",
    "10x 5' transcription profiling": "#9467BD",
    "10x 5' v1": "#8C564B",
    "10x 5' v2": "#E377C2",
    "BD Rhapsody Whole Transcriptome Analysis": "#BCBD22",
    "Flex Gene Expression": "#17BECF",
    "GEXSCOPE technology": "#AEC7E8",
    "ScaleBio single cell RNA sequencing": "#FFBB78",
    "Seq-Well": "#98DF8A",
    "inDrop": "#FF9896",
    "Drop-seq": "#00A6D6",
    "Smart-seq2": "#C5B0D5",
    "CEL-seq2": "#C49C94",
    "microwell-seq": "#7F7F7F",
    "MARS-seq": "#DBDB8D",
}

PREFERRED_SOURCE_COLORS = {
    "CB07": "#1F77B4",
    "cellxgene": "#FF7F0E",
}

BATCH_MIXING_METRICS = [
    "dataset_asw_mixing",
    "assay_asw_mixing",
    "tissue_asw_mixing",
    "dataset_knn_mixing",
    "assay_knn_mixing",
    "tissue_knn_mixing",
]

BIOLOGY_PRESERVATION_METRICS = [
    "nk_state_asw",
    "knn_label_acc",
    "graph_connectivity",
]

DEFAULT_PLOT_METRICS = [
    "dataset_asw_mixing",
    "assay_asw_mixing",
    "tissue_asw_mixing",
    "knn_label_acc",
    "graph_connectivity",
]

PLOT_METRIC_LABELS = {
    "dataset_asw_mixing": "dataset\nASW\nmixing",
    "assay_asw_mixing": "assay\nASW\nmixing",
    "tissue_asw_mixing": "tissue\nASW\nmixing",
    "global_dataset_asw_mixing": "global\ndataset\nASW",
    "global_assay_asw_mixing": "global\nassay\nASW",
    "global_tissue_asw_mixing": "global\ntissue\nASW",
    "dataset_knn_mixing": "dataset\nkNN\nmixing",
    "assay_knn_mixing": "assay\nkNN\nmixing",
    "tissue_knn_mixing": "tissue\nkNN\nmixing",
    "nk_state_asw": "NK_State\nASW",
    "knn_label_acc": "NK_State\nkNN",
    "graph_connectivity": "graph\nconnectivity",
}


def main():
    args = parse_args()
    ensure_dirs(args.outdir)

    if args.umap_only:
        for strategy in args.strategies:
            plot_strategy_umap_panels(strategy, args)
        return

    rows = []
    for strategy in args.strategies:
        row = run_scvi_strategy(strategy, args)
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("strategy")
    summary = add_normalized_scores(summary)

    csv_path = os.path.join(args.outdir, "scvi_batch_strategy_metrics.csv")
    summary.to_csv(csv_path)
    print("\n" + "=" * 90)
    print("SCVI BATCH STRATEGY COMPARISON")
    print("=" * 90)
    print(summary.round(4).to_string())
    print(f"[SAVE] {csv_path}")

    plot_path = os.path.join(args.outdir, "scvi_batch_strategy_scores.png")
    plot_scores(summary, plot_path, args.plot_metrics)
    print(f"[SAVE] {plot_path}")

    if args.plot_umaps:
        for strategy in args.strategies:
            plot_strategy_umap_panels(strategy, args)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train and compare SCVI models with alternative batch keys before "
            "Leiden discovery/annotation."
        )
    )
    parser.add_argument("--input-h5ad", default=DEFAULT_INPUT)
    parser.add_argument("--outdir", default=os.path.join(cfg.BASE_OUTDIR, "scvi_batch_strategy_comparison"))
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["assay_only", "dataset_only", "dataset_assay", "tissue_assay", "dataset_tissue_assay"],
        choices=list(STRATEGIES),
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--metric-max-cells", type=int, default=50000)
    parser.add_argument(
        "--include-protected-in-metrics",
        action="store_true",
        help=(
            "Include the protected dataset, usually CB07, in metric calculations. "
            "By default it is excluded so treated cord-blood cells are not penalized "
            "for being biologically distinct."
        ),
    )
    parser.add_argument(
        "--plot-metrics",
        nargs="+",
        default=DEFAULT_PLOT_METRICS,
        choices=list(PLOT_METRIC_LABELS),
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain even if the strategy metric cache already exists.",
    )
    parser.add_argument(
        "--plot-umaps",
        action="store_true",
        help="After training/metric refresh, generate UMAP panels for each strategy.",
    )
    parser.add_argument(
        "--umap-only",
        action="store_true",
        help="Only generate UMAP panels from cached scvi_full_with_latent.h5ad files; do not train or refresh metrics.",
    )
    parser.add_argument(
        "--umap-max-cells",
        type=int,
        default=None,
        help="Optional maximum number of cells to show per UMAP panel. Defaults to all cells.",
    )
    return parser.parse_args()


def run_scvi_strategy(strategy: str, args):
    batch_key = STRATEGIES[strategy]
    run_cfg = make_strategy_cfg(strategy, batch_key, args)
    cache_path = os.path.join(run_cfg.BASE_OUTDIR, f"{strategy}_scvi_metrics.csv")

    print("\n" + "#" * 90)
    print(f"[RUN] {strategy} | batch_key={batch_key}")
    print("#" * 90)

    if os.path.exists(cache_path) and not args.force_retrain:
        print(f"[CACHE] Reusing metrics for {strategy}: {cache_path}")
        row = pd.read_csv(cache_path).iloc[0].to_dict()
        if cache_needs_metric_refresh(row):
            print(f"[CACHE] Refreshing metrics for {strategy}; no retraining")
            full_path = os.path.join(run_cfg.LATENT_OUTDIR, "scvi_full_with_latent.h5ad")
            full = sc.read_h5ad(full_path)
            z = np.asarray(full.obsm["X_scVI"], dtype=np.float32)
            row.update(compute_latent_comparison_metrics(z, full.obs, strategy, args))
            pd.DataFrame([row]).to_csv(cache_path, index=False)
            print(f"[SAVE] {cache_path}")
        return row

    _model, full = train_scvi(run_cfg, batch_key=batch_key)
    z = np.asarray(full.obsm["X_scVI"], dtype=np.float32)

    row = {
        "strategy": strategy,
        "batch_key": batch_key,
        "model_type": "SCVI",
        "input_h5ad": args.input_h5ad,
        "n_cells_full": int(full.n_obs),
        "n_genes_full": int(full.n_vars),
    }
    row.update(compute_latent_comparison_metrics(z, full.obs, strategy, args))
    pd.DataFrame([row]).to_csv(cache_path, index=False)
    print(f"[SAVE] {cache_path}")
    return row


def make_strategy_cfg(strategy: str, batch_key: str, args):
    run_cfg = SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    run_cfg.MERGED_PATH = args.input_h5ad
    run_cfg.PRODUCTION_BATCH_KEY = batch_key
    run_cfg.BASE_OUTDIR = os.path.join(args.outdir, "runs", strategy)
    run_cfg.FIG_OUTDIR = os.path.join(run_cfg.BASE_OUTDIR, "figures")
    run_cfg.MODEL_OUTDIR = os.path.join(run_cfg.BASE_OUTDIR, "models")
    run_cfg.TABLE_OUTDIR = os.path.join(run_cfg.BASE_OUTDIR, "tables")
    run_cfg.LATENT_OUTDIR = os.path.join(run_cfg.BASE_OUTDIR, "latents")
    if args.max_epochs is not None:
        run_cfg.MAX_EPOCHS = args.max_epochs
    return run_cfg


def compute_latent_comparison_metrics(z, obs, strategy, args):
    labels = obs[cfg.LABEL_KEY].astype(str).values
    dataset = obs[cfg.DATASET_KEY].astype(str).values
    assay = obs[cfg.ASSAY_CLEAN_KEY].astype(str).values
    tissue = obs["tissue"].astype(str).values

    metric_mask = labels != cfg.UNLABELED_CATEGORY
    if not args.include_protected_in_metrics and cfg.PROTECTED_DATASET in set(dataset):
        metric_mask &= dataset != cfg.PROTECTED_DATASET
        print(f"[{strategy}] Metrics exclude {cfg.PROTECTED_DATASET} by default")

    z0 = z[metric_mask]
    labels0 = labels[metric_mask]
    dataset0 = dataset[metric_mask]
    assay0 = assay[metric_mask]
    tissue0 = tissue[metric_mask]

    z_lab, labels_lab, dataset_lab = subsample_for_metrics(
        z0, labels0, dataset0, max_cells=args.metric_max_cells, seed=cfg.SEED
    )
    _, _, assay_lab = subsample_for_metrics(
        z0, labels0, assay0, max_cells=args.metric_max_cells, seed=cfg.SEED
    )
    _, _, tissue_lab = subsample_for_metrics(
        z0, labels0, tissue0, max_cells=args.metric_max_cells, seed=cfg.SEED
    )

    print(f"[{strategy}] Metric subsample: {z_lab.shape[0]:,} cells")
    global_dataset_asw = compute_label_asw(z_lab, dataset_lab)
    global_assay_asw = compute_label_asw(z_lab, assay_lab)
    global_tissue_asw = compute_label_asw(z_lab, tissue_lab)
    return {
        "n_cells_metric": int(z_lab.shape[0]),
        "n_datasets_metric": int(pd.Series(dataset_lab).nunique()),
        "n_assays_metric": int(pd.Series(assay_lab).nunique()),
        "n_tissues_metric": int(pd.Series(tissue_lab).nunique()),
        "nk_state_asw": compute_label_asw(z_lab, labels_lab),
        "global_dataset_asw_separation": global_dataset_asw,
        "global_assay_asw_separation": global_assay_asw,
        "global_tissue_asw_separation": global_tissue_asw,
        "global_dataset_asw_mixing": global_asw_mixing_score(global_dataset_asw),
        "global_assay_asw_mixing": global_asw_mixing_score(global_assay_asw),
        "global_tissue_asw_mixing": global_asw_mixing_score(global_tissue_asw),
        "graph_connectivity": compute_graph_connectivity(z_lab, labels_lab, n_neighbors=cfg.METRIC_KNN_K),
        "knn_label_acc": compute_knn_label_accuracy(z_lab, labels_lab, k=cfg.METRIC_KNN_K),
        "dataset_asw_mixing": compute_batch_asw_label_aware(z_lab, dataset_lab, labels_lab),
        "assay_asw_mixing": compute_batch_asw_label_aware(z_lab, assay_lab, labels_lab),
        "tissue_asw_mixing": compute_batch_asw_label_aware(z_lab, tissue_lab, labels_lab),
        "dataset_knn_batch_acc": compute_knn_batch_accuracy(
            z_lab, dataset_lab, within_labels=labels_lab, k=cfg.METRIC_KNN_K
        ),
        "assay_knn_batch_acc": compute_knn_batch_accuracy(
            z_lab, assay_lab, within_labels=labels_lab, k=cfg.METRIC_KNN_K
        ),
        "tissue_knn_batch_acc": compute_knn_batch_accuracy(
            z_lab, tissue_lab, within_labels=labels_lab, k=cfg.METRIC_KNN_K
        ),
        "dataset_knn_baseline_acc": weighted_within_label_majority_baseline(dataset_lab, labels_lab),
        "assay_knn_baseline_acc": weighted_within_label_majority_baseline(assay_lab, labels_lab),
        "tissue_knn_baseline_acc": weighted_within_label_majority_baseline(tissue_lab, labels_lab),
    }


def cache_needs_metric_refresh(row):
    required = [
        "global_dataset_asw_separation",
        "global_assay_asw_separation",
        "global_tissue_asw_separation",
        "global_dataset_asw_mixing",
        "global_assay_asw_mixing",
        "global_tissue_asw_mixing",
    ]
    return any(col not in row or pd.isna(row.get(col)) for col in required)


def global_asw_mixing_score(asw_separation):
    """Convert shifted ASW separation to a 0-1 global mixing score.

    compute_label_asw returns (silhouette + 1) / 2. A value near 0.5 means
    labels are not globally separated, so that is best mixing. Values close
    to 0 or 1 indicate strong global structure by the tested label.
    """
    if pd.isna(asw_separation):
        return np.nan
    return float(np.clip(1.0 - abs(float(asw_separation) - 0.5) * 2.0, 0.0, 1.0))


def weighted_within_label_majority_baseline(batch_labels, labels):
    batch_labels = np.asarray(batch_labels).astype(str)
    labels = np.asarray(labels).astype(str)
    baselines = []
    weights = []
    for label in sorted(pd.unique(labels)):
        mask = labels == label
        if mask.sum() == 0:
            continue
        counts = pd.Series(batch_labels[mask]).value_counts()
        baselines.append(float(counts.iloc[0] / counts.sum()))
        weights.append(int(mask.sum()))
    return float(np.average(baselines, weights=weights)) if weights else np.nan


def batch_knn_mixing_score(knn_acc, baseline_acc):
    if pd.isna(knn_acc) or pd.isna(baseline_acc):
        return np.nan
    if baseline_acc >= 1.0:
        return 1.0 if knn_acc <= baseline_acc else 0.0
    excess = max(0.0, float(knn_acc) - float(baseline_acc))
    return float(np.clip(1.0 - excess / (1.0 - float(baseline_acc)), 0.0, 1.0))


def add_normalized_scores(summary):
    out = summary.copy()
    for batch_name in ["dataset", "assay", "tissue"]:
        acc_col = f"{batch_name}_knn_batch_acc"
        base_col = f"{batch_name}_knn_baseline_acc"
        mix_col = f"{batch_name}_knn_mixing"
        if acc_col in out and base_col in out:
            out[acc_col] = pd.to_numeric(out[acc_col], errors="coerce")
            out[base_col] = pd.to_numeric(out[base_col], errors="coerce")
            out[mix_col] = [
                batch_knn_mixing_score(acc, base)
                for acc, base in zip(out[acc_col], out[base_col])
            ]

    for col in BATCH_MIXING_METRICS + BIOLOGY_PRESERVATION_METRICS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(0.0, 1.0)

    return out.sort_index()


def plot_scores(summary, path, plot_cols):
    missing = [col for col in plot_cols if col not in summary.columns]
    if missing:
        raise KeyError(f"Requested plot metrics are missing from summary: {missing}")

    plot_df = summary[plot_cols].copy()
    labels = [PLOT_METRIC_LABELS[col] for col in plot_cols]

    fig_width = max(10, 1.7 * len(plot_cols) + 3.5)
    fig_height = max(4.8, 0.75 * len(plot_df) + 2.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(plot_df.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=SMALL_TICK_LABEL_SIZE, fontweight="bold")
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=SMALL_TICK_LABEL_SIZE, fontweight="bold")
    ax.set_title("SCVI batch strategy comparison", fontsize=18, fontweight="bold")

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            val = plot_df.iloc[i, j]
            ax.text(
                j,
                i,
                "" if pd.isna(val) else f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val < 0.55 else "black",
                fontsize=12,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("score (0=bad, 1=good)", fontsize=14, fontweight="bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontweight("bold")
    style_axis(ax, tick_size=SMALL_TICK_LABEL_SIZE)
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_strategy_umap_panels(strategy: str, args):
    batch_key = STRATEGIES[strategy]
    run_cfg = make_strategy_cfg(strategy, batch_key, args)
    h5ad_path = os.path.join(run_cfg.LATENT_OUTDIR, "scvi_full_with_latent.h5ad")
    if not os.path.exists(h5ad_path):
        raise FileNotFoundError(f"Cached SCVI latent AnnData not found for {strategy}: {h5ad_path}")

    outdir = os.path.join(args.outdir, "umap_mixing_panels")
    ensure_dirs(outdir)

    print(f"[UMAP] {strategy}: loading {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    xy = load_or_build_umap(adata, run_cfg, h5ad_path)

    plot_idx = np.arange(adata.n_obs)
    if args.umap_max_cells is not None and adata.n_obs > args.umap_max_cells:
        rng = np.random.default_rng(cfg.SEED)
        plot_idx = np.sort(rng.choice(plot_idx, size=args.umap_max_cells, replace=False))

    xy_p = xy[plot_idx]
    obs = adata.obs.iloc[plot_idx].copy()
    state = obs[cfg.LABEL_KEY].astype(str).values if cfg.LABEL_KEY in obs else np.array(["NA"] * len(obs))
    assay = obs[cfg.ASSAY_CLEAN_KEY].astype(str).values if cfg.ASSAY_CLEAN_KEY in obs else np.array(["NA"] * len(obs))
    tissue = obs["tissue"].astype(str).values if "tissue" in obs else np.array(["NA"] * len(obs))
    source = obs["source_panel"].astype(str).values if "source_panel" in obs else np.array(["NA"] * len(obs))

    state_colors = leiden_discovery_style_colors(state)
    assay_colors = distinct_color_map(assay, preferred=PREFERRED_ASSAY_COLORS)
    tissue_colors = distinct_color_map(tissue, preferred=PREFERRED_TISSUE_COLORS)
    source_colors = distinct_color_map(source, preferred=PREFERRED_SOURCE_COLORS)

    fig, axes = plt.subplots(2, 2, figsize=(24, 16))
    fig.subplots_adjust(left=0.03, right=0.78, top=0.90, bottom=0.05, wspace=0.42, hspace=0.22)
    fig.suptitle(strategy, fontsize=24, fontweight="bold")

    scatter_by_category(axes[0, 0], xy_p, state, state_colors, legend=True, title="NK_State")
    scatter_by_category(axes[0, 1], xy_p, assay, assay_colors, legend=True, title="assay_clean")
    scatter_by_category(axes[1, 0], xy_p, tissue, tissue_colors, legend=True, title="tissue")
    scatter_by_category(axes[1, 1], xy_p, source, source_colors, legend=True, title="source_panel")

    style_figure(fig, tick_size=SMALL_TICK_LABEL_SIZE, legend_size=LEGEND_FONT_SIZE)
    style_all_legends(fig)

    out_png = os.path.join(outdir, f"{strategy}_umap_mixing.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {out_png}")


def load_or_build_umap(adata, run_cfg, h5ad_path):
    umap_npy = os.path.join(run_cfg.LATENT_OUTDIR, "scvi_full_umap.npy")
    umap_csv = os.path.join(run_cfg.TABLE_OUTDIR, "scvi_full_umap.csv")
    if os.path.exists(umap_npy) and os.path.exists(umap_csv) and os.path.getmtime(umap_npy) >= os.path.getmtime(h5ad_path):
        try:
            xy = np.load(umap_npy)
            cached_index = pd.read_csv(umap_csv, index_col=0).index.astype(str).values
            if xy.shape == (adata.n_obs, 2) and np.array_equal(cached_index, adata.obs_names.astype(str)):
                print(f"[UMAP] using cached {umap_csv}")
                return xy
        except Exception as exc:
            print(f"[WARN] Could not reuse cached UMAP; rebuilding. Reason: {exc}")

    print("[UMAP] building from X_scVI")
    sc.pp.neighbors(adata, use_rep="X_scVI", n_neighbors=cfg.UMAP_N_NEIGHBORS, random_state=cfg.UMAP_SEED)
    sc.tl.umap(adata, min_dist=cfg.UMAP_MIN_DIST, random_state=cfg.UMAP_SEED)
    xy = np.asarray(adata.obsm["X_umap"], dtype=np.float32)
    ensure_dirs(run_cfg.TABLE_OUTDIR, run_cfg.LATENT_OUTDIR)
    np.save(umap_npy, xy)
    pd.DataFrame(xy, index=adata.obs_names.astype(str), columns=["UMAP1", "UMAP2"]).to_csv(umap_csv)
    print(f"[SAVE] {umap_npy}")
    print(f"[SAVE] {umap_csv}")
    return xy


def leiden_discovery_style_colors(values):
    categories = sorted(set(np.asarray(values).astype(str)), key=category_sort_key)
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3", "Paired"):
        cmap = plt.get_cmap(cmap_name)
        palette.extend([cmap(i) for i in range(cmap.N)])
    return {category: palette[i % len(palette)] for i, category in enumerate(categories)}


def category_sort_key(value):
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


if __name__ == "__main__":
    main()
