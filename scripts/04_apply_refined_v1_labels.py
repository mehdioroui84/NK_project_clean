#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.evaluation.scanvi_full_plots import PREFERRED_STATE_COLORS
from nk_project.io_utils import ensure_dirs
from nk_project.plot_style import (
    LEGEND_FONT_SIZE,
    SMALL_TICK_LABEL_SIZE,
    set_presentation_style,
    style_all_legends,
    style_axis,
    style_figure,
    style_legend,
)

set_presentation_style()


GROUPBY = "leiden_0_4"
OUTDIR_NAME = "refined_annotation_v1"
POINT_SIZE = 0.06
POINT_ALPHA = 0.52
QC_POINT_SIZE = 0.08
QC_POINT_ALPHA = 0.95
MAX_LEGEND_LABEL_CHARS = 38

PAN_NK_SCORE_MARKERS = [
    "NCR1",
    "KLRD1",
    "KLRF1",
    "NCAM1",
    "EOMES",
    "TBX21",
    "NKG7",
    "GNLY",
    "PRF1",
    "GZMB",
    "GZMA",
    "CST7",
]

NK_EXCLUDED_SCORE_MARKERS = [
    "CD3D",
    "CD3E",
    "CD3G",
    "TRAC",
    "TRBC1",
    "TRBC2",
    "MS4A1",
    "CD79A",
    "CD79B",
    "CD19",
    "BANK1",
    "LYZ",
    "LST1",
    "S100A8",
    "S100A9",
    "CST3",
    "HBB",
    "HBA1",
    "HBA2",
    "AHSP",
    "EPCAM",
    "KRT8",
    "KRT18",
    "KRT19",
    "COL1A1",
    "DCN",
    "LUM",
]


def main():
    args = parse_args()
    in_path = args.input_h5ad or os.path.join(cfg.BASE_OUTDIR, "leiden_discovery", "full_scvi_leiden.h5ad")
    outdir = args.outdir or os.path.join(cfg.BASE_OUTDIR, OUTDIR_NAME)
    figdir = os.path.join(outdir, "figures")
    ensure_dirs(outdir, figdir)

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if GROUPBY not in adata.obs:
        raise KeyError(f"{GROUPBY!r} not found in adata.obs.")
    if "X_umap" not in adata.obsm:
        raise KeyError("X_umap not found in full-data SCVI Leiden AnnData.")

    expected_clusters = sorted(adata.obs[GROUPBY].astype(str).unique(), key=cluster_sort_key)
    label_mapping, label_source, free_label_mapping = load_label_mapping(
        args.mapping_csv,
        label_column=args.label_column,
        expected_clusters=expected_clusters,
    )
    apply_labels(adata, label_mapping, label_source=label_source, free_label_mapping=free_label_mapping)
    write_outputs(adata, outdir, label_mapping)
    plot_refined_umap(adata, figdir)
    plot_annotation_qc_umap(adata, figdir, panel9_min_cells=args.panel9_min_cells)
    if args.make_3d_umap:
        xyz = get_umap3d(adata, args, outdir)
        plot_refined_umap_3d(adata, figdir, xyz)
        plot_annotation_qc_umap_3d(adata, figdir, xyz, panel9_min_cells=args.panel9_min_cells)
    print("[DONE] Full-data refined v1 label application complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply annotation-agent labels to full-data Leiden clusters."
        )
    )
    parser.add_argument(
        "--input-h5ad",
        default=None,
        help=(
            "Input AnnData with Leiden clusters and X_umap. Default: "
            "outputs/leiden_discovery/full_scvi_leiden.h5ad."
        ),
    )
    parser.add_argument(
        "--mapping-csv",
        required=True,
        help=(
            "Annotation mapping CSV. Expected columns: leiden_0_4 and final_structured_label "
            "or another label column selected by --label-column."
        ),
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help=(
            "Column from --mapping-csv to use as the final SCANVI training label. "
            "Default: final_structured_label. Use free_label only if you explicitly "
            "want the model trained on the free biological names."
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help=(
            "Optional output directory. Default: outputs/refined_annotation_v1. "
            "Use this to write a separate annotation set without overwriting the default."
        ),
    )
    parser.add_argument(
        "--make-3d-umap",
        action="store_true",
        help=(
            "Also save optional 3D UMAP PNGs. Uses --umap-3d-key if present; "
            "otherwise computes a 3D UMAP from --umap-3d-latent-key."
        ),
    )
    parser.add_argument(
        "--umap-3d-key",
        default="X_umap_3d",
        help="adata.obsm key for an existing 3D UMAP. Default: X_umap_3d.",
    )
    parser.add_argument(
        "--umap-3d-latent-key",
        default="X_scVI",
        help="adata.obsm latent key used to compute 3D UMAP if --umap-3d-key is absent. Default: X_scVI.",
    )
    parser.add_argument(
        "--recompute-3d-umap",
        action="store_true",
        help="Ignore cached 3D UMAP coordinates and recompute them.",
    )
    parser.add_argument(
        "--panel9-min-cells",
        type=int,
        default=0,
        help=(
            "Hide clusters with fewer than this many cells from panel 9 "
            "(cluster-level NK vs exclusion marker score scatter). Default 0 shows all clusters."
        ),
    )
    return parser.parse_args()


def load_label_mapping(mapping_csv, *, label_column=None, expected_clusters=None):
    print(f"[MAPPING] Loading reviewed mapping CSV: {mapping_csv}")
    mapping = pd.read_csv(mapping_csv, dtype=str)
    if GROUPBY not in mapping.columns:
        raise KeyError(f"{mapping_csv} must contain {GROUPBY!r}.")

    if label_column:
        if label_column not in mapping.columns:
            raise KeyError(f"{mapping_csv} does not contain requested --label-column {label_column!r}.")
        label_col = label_column
    else:
        label_col = None
        for candidate in [
            "final_structured_label",
            "free_label",
            cfg.REFINED_LABEL_KEY,
            "refined_label",
        ]:
            if candidate in mapping.columns:
                label_col = candidate
                break
    if label_col is None:
        raise KeyError(
            f"{mapping_csv} must contain one of: {cfg.REFINED_LABEL_KEY!r}, "
            "'final_structured_label', 'free_label', or 'refined_label'."
        )
    print(f"[MAPPING_LABEL_COLUMN] {label_col}")

    labels = mapping[label_col].fillna("").astype(str).str.strip()
    if labels.eq("").any():
        bad = mapping.loc[labels.eq(""), GROUPBY].astype(str).tolist()
        raise ValueError(f"Mapping CSV has empty labels in {label_col!r} for clusters: {bad}")
    out = dict(zip(mapping[GROUPBY].astype(str), labels))
    expected = set(expected_clusters or [])
    missing = sorted(expected - set(out), key=cluster_sort_key)
    if missing:
        raise ValueError(f"Mapping CSV is missing {GROUPBY} clusters: {missing}")
    free_label_mapping = {}
    if "free_label" in mapping.columns:
        free_labels = mapping["free_label"].fillna("").astype(str).str.strip()
        free_label_mapping = dict(zip(mapping[GROUPBY].astype(str), free_labels))
    return out, label_col, free_label_mapping


def apply_labels(adata, label_mapping, *, label_source, free_label_mapping=None):
    clusters = adata.obs[GROUPBY].astype(str)
    labels = clusters.map(label_mapping)
    if labels.isna().any():
        missing = sorted(clusters[labels.isna()].unique(), key=cluster_sort_key)
        raise ValueError(f"Missing refined labels for {GROUPBY} clusters: {missing}")

    adata.obs[cfg.REFINED_LABEL_KEY] = labels.astype("category")
    adata.obs["NK_State_refined_v1_source"] = f"full_data_leiden_0_4_mapping:{label_source}"
    if free_label_mapping:
        free_labels = clusters.map(free_label_mapping)
        if not free_labels.isna().any():
            adata.obs["NK_State_free_label"] = free_labels.astype("category")

    print("\n[REFINED LABEL COUNTS]")
    print(adata.obs[cfg.REFINED_LABEL_KEY].astype(str).value_counts().to_string())


def write_outputs(adata, outdir, label_mapping):
    h5ad_path = os.path.join(outdir, "full_scvi_leiden_refined_v1.h5ad")
    obs_path = os.path.join(outdir, "full_refined_v1_obs_metadata.csv")
    mapping_path = os.path.join(outdir, "full_leiden_0_4_to_refined_v1_mapping.csv")
    counts_path = os.path.join(outdir, "full_refined_v1_label_counts.csv")

    mapping = pd.DataFrame(
        {
            GROUPBY: list(label_mapping.keys()),
            cfg.REFINED_LABEL_KEY: list(label_mapping.values()),
        }
    )
    mapping[GROUPBY] = pd.Categorical(
        mapping[GROUPBY],
        categories=sorted(label_mapping, key=cluster_sort_key),
        ordered=True,
    )
    mapping = mapping.sort_values(GROUPBY)
    mapping.to_csv(mapping_path, index=False)

    counts = (
        adata.obs[cfg.REFINED_LABEL_KEY]
        .astype(str)
        .value_counts()
        .rename_axis(cfg.REFINED_LABEL_KEY)
        .reset_index(name="n_cells")
    )
    counts.to_csv(counts_path, index=False)

    adata.obs.to_csv(obs_path)
    adata.write(h5ad_path)
    print(f"[SAVE] {h5ad_path}")
    print(f"[SAVE] {obs_path}")
    print(f"[SAVE] {mapping_path}")
    print(f"[SAVE] {counts_path}")


def plot_refined_umap(adata, figdir):
    xy = adata.obsm["X_umap"]
    panels = [
        (GROUPBY, f"1. Leiden {GROUPBY}", False, True),
        (cfg.REFINED_LABEL_KEY, f"2. final annotation: {cfg.REFINED_LABEL_KEY}", True, False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle("Full-data SCVI latent space: annotation labels", fontsize=20, fontweight="bold")
    for ax, (obs_key, title, show_legend, annotate) in zip(axes, panels):
        scatter_categorical(
            ax,
            xy,
            adata.obs[obs_key].astype(str).values,
            title,
            show_legend=show_legend,
            annotate_clusters=annotate,
        )

    plt.tight_layout()
    png = os.path.join(figdir, "full_refined_v1_umap.png")
    style_figure(fig, tick_size=SMALL_TICK_LABEL_SIZE, legend_size=LEGEND_FONT_SIZE)
    style_all_legends(fig)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    plt.close(fig)


def get_umap3d(adata, args, outdir):
    cache_path = os.path.join(outdir, "X_umap_3d.npy")
    if os.path.exists(cache_path) and not args.recompute_3d_umap:
        xyz = np.load(cache_path)
        if xyz.shape[0] == adata.n_obs and xyz.shape[1] >= 3:
            print(f"[UMAP3D] Using cached coordinates: {cache_path}")
            return xyz[:, :3]
        print(f"[UMAP3D] Ignoring stale cache with shape {xyz.shape}: {cache_path}")

    if args.umap_3d_key in adata.obsm:
        xyz = np.asarray(adata.obsm[args.umap_3d_key])
        if xyz.shape[1] < 3:
            raise ValueError(f"{args.umap_3d_key!r} exists but has only {xyz.shape[1]} dimensions.")
        print(f"[UMAP3D] Using existing {args.umap_3d_key!r}.")
        xyz = xyz[:, :3]
        np.save(cache_path, xyz)
        print(f"[SAVE] {cache_path}")
        return xyz

    if args.umap_3d_latent_key not in adata.obsm:
        raise KeyError(
            f"Cannot compute 3D UMAP: {args.umap_3d_key!r} is absent and "
            f"{args.umap_3d_latent_key!r} is not in adata.obsm."
        )

    print(f"[UMAP3D] Computing 3D UMAP from {args.umap_3d_latent_key!r}...")
    ad_umap = sc.AnnData(X=np.zeros((adata.n_obs, 1), dtype=np.float32))
    ad_umap.obsm[args.umap_3d_latent_key] = np.asarray(
        adata.obsm[args.umap_3d_latent_key],
        dtype=np.float32,
    )
    sc.pp.neighbors(
        ad_umap,
        use_rep=args.umap_3d_latent_key,
        n_neighbors=cfg.UMAP_N_NEIGHBORS,
        random_state=cfg.UMAP_SEED,
    )
    sc.tl.umap(
        ad_umap,
        min_dist=cfg.UMAP_MIN_DIST,
        random_state=cfg.UMAP_SEED,
        n_components=3,
    )
    xyz = np.asarray(ad_umap.obsm["X_umap"])[:, :3]
    np.save(cache_path, xyz)
    print(f"[SAVE] {cache_path}")
    return xyz


def plot_refined_umap_3d(adata, figdir, xyz):
    panels = [
        (GROUPBY, f"1. 3D {GROUPBY}", False, True),
        (cfg.REFINED_LABEL_KEY, f"2. final annotation: {cfg.REFINED_LABEL_KEY}", True, False),
    ]

    fig = plt.figure(figsize=(20, 8))
    fig.suptitle("Full-data SCVI latent space: annotation labels, 3D UMAP", fontsize=20, fontweight="bold")
    axes = [fig.add_subplot(1, 2, idx + 1, projection="3d") for idx in range(2)]
    for ax, (obs_key, title, show_legend, annotate) in zip(axes, panels):
        scatter_categorical_3d(
            ax,
            xyz,
            adata.obs[obs_key].astype(str).values,
            title,
            show_legend=show_legend,
            annotate_clusters=annotate,
        )

    plt.tight_layout()
    png = os.path.join(figdir, "full_refined_v1_umap_3d.png")
    style_figure(fig, tick_size=SMALL_TICK_LABEL_SIZE, legend_size=LEGEND_FONT_SIZE)
    style_all_legends(fig)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    plt.close(fig)


def plot_annotation_qc_umap(adata, figdir, *, panel9_min_cells=0):
    xy = adata.obsm["X_umap"]
    positive_score, positive_used, positive_missing = marker_mean_score(adata, PAN_NK_SCORE_MARKERS)
    excluded_score, excluded_used, excluded_missing = marker_mean_score(adata, NK_EXCLUDED_SCORE_MARKERS)
    tissue = obs_values(adata, "tissue")
    dataset = obs_values(adata, cfg.DATASET_KEY)
    assay = obs_values(adata, cfg.ASSAY_CLEAN_KEY)

    availability = pd.DataFrame(
        [
            {
                "score": "positive_NK_score",
                "n_requested": len(PAN_NK_SCORE_MARKERS),
                "n_used": len(positive_used),
                "used_genes": ";".join(positive_used),
                "missing_genes": ";".join(positive_missing),
            },
            {
                "score": "NK_excluded_score",
                "n_requested": len(NK_EXCLUDED_SCORE_MARKERS),
                "n_used": len(excluded_used),
                "used_genes": ";".join(excluded_used),
                "missing_genes": ";".join(excluded_missing),
            },
        ]
    )
    availability_path = os.path.join(figdir, "annotation_qc_marker_availability.csv")
    availability.to_csv(availability_path, index=False)
    print(f"[SAVE] {availability_path}")

    signed_score = zscore(positive_score) - zscore(excluded_score)
    refined_values = adata.obs[cfg.REFINED_LABEL_KEY].astype(str).values
    leiden_values = adata.obs[GROUPBY].astype(str).values
    refined_label_order = label_order_by_cluster(refined_values, leiden_values)
    refined_label_colors = label_colors_from_cluster_colors(
        refined_label_order,
        refined_values,
        leiden_values,
        preferred=PREFERRED_STATE_COLORS,
    )
    fig, axes = plt.subplots(3, 3, figsize=(30, 26))
    axes = axes.ravel()
    fig.suptitle("Annotation QC: cluster labels, metadata, and NK marker scores", fontsize=20, fontweight="bold")

    scatter_categorical(
        axes[0],
        xy,
        adata.obs[GROUPBY].astype(str).values,
        f"1. Leiden {GROUPBY}",
        show_legend=False,
        annotate_clusters=True,
    )
    scatter_categorical(
        axes[1],
        xy,
        refined_values,
        f"2. final annotation: {cfg.REFINED_LABEL_KEY}",
        show_legend=True,
        annotate_clusters=False,
        category_order=refined_label_order,
        colors_override=refined_label_colors,
    )
    scatter_categorical(
        axes[2],
        xy,
        tissue,
        "3. Tissue",
        show_legend=True,
        annotate_clusters=False,
    )
    scatter_categorical(
        axes[3],
        xy,
        dataset,
        "4. Dataset ID",
        show_legend=False,
        annotate_clusters=False,
    )
    scatter_categorical(
        axes[4],
        xy,
        assay,
        "5. Assay clean",
        show_legend=True,
        annotate_clusters=False,
    )
    scatter_continuous(
        axes[5],
        xy,
        signed_score,
        "6. signed NK identity score (red=NK-like, blue=NK-excluded)",
        cmap="RdBu_r",
        symmetric=True,
        robust=True,
    )
    scatter_continuous(
        axes[6],
        xy,
        positive_score,
        "7. positive NK score (standardized; Reds)",
        cmap="Reds",
        robust=True,
    )
    scatter_continuous(
        axes[7],
        xy,
        excluded_score,
        "8. NK-excluded score (standardized; Blues)",
        cmap="Blues",
        robust=True,
    )
    plot_cluster_marker_agreement(
        axes[8],
        adata,
        positive_score,
        excluded_score,
        min_cells=panel9_min_cells,
    )

    plt.tight_layout()
    png = os.path.join(figdir, "annotation_umap_review_panels.png")
    style_figure(fig, tick_size=SMALL_TICK_LABEL_SIZE, legend_size=LEGEND_FONT_SIZE)
    style_all_legends(fig)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    plt.close(fig)


def plot_annotation_qc_umap_3d(adata, figdir, xyz, *, panel9_min_cells=0):
    positive_score, positive_used, _ = marker_mean_score(adata, PAN_NK_SCORE_MARKERS)
    excluded_score, excluded_used, _ = marker_mean_score(adata, NK_EXCLUDED_SCORE_MARKERS)
    tissue = obs_values(adata, "tissue")
    dataset = obs_values(adata, cfg.DATASET_KEY)
    assay = obs_values(adata, cfg.ASSAY_CLEAN_KEY)
    signed_score = zscore(positive_score) - zscore(excluded_score)
    refined_values = adata.obs[cfg.REFINED_LABEL_KEY].astype(str).values
    leiden_values = adata.obs[GROUPBY].astype(str).values
    refined_label_order = label_order_by_cluster(refined_values, leiden_values)
    refined_label_colors = label_colors_from_cluster_colors(
        refined_label_order,
        refined_values,
        leiden_values,
        preferred=PREFERRED_STATE_COLORS,
    )

    fig = plt.figure(figsize=(30, 26))
    fig.suptitle("Annotation QC: 3D UMAP labels, metadata, and NK marker scores", fontsize=20, fontweight="bold")
    axes = [fig.add_subplot(3, 3, idx + 1, projection="3d") for idx in range(8)]
    axes.append(fig.add_subplot(3, 3, 9))

    scatter_categorical_3d(
        axes[0],
        xyz,
        adata.obs[GROUPBY].astype(str).values,
        f"1. Leiden {GROUPBY}",
        show_legend=False,
        annotate_clusters=True,
    )
    scatter_categorical_3d(
        axes[1],
        xyz,
        refined_values,
        f"2. final annotation: {cfg.REFINED_LABEL_KEY}",
        show_legend=True,
        annotate_clusters=False,
        category_order=refined_label_order,
        colors_override=refined_label_colors,
    )
    scatter_categorical_3d(axes[2], xyz, tissue, "3. Tissue", show_legend=True, annotate_clusters=False)
    scatter_categorical_3d(axes[3], xyz, dataset, "4. Dataset ID", show_legend=False, annotate_clusters=False)
    scatter_categorical_3d(axes[4], xyz, assay, "5. Assay clean", show_legend=True, annotate_clusters=False)
    scatter_continuous_3d(
        axes[5],
        xyz,
        signed_score,
        "6. signed NK identity score (red=NK-like, blue=NK-excluded)",
        cmap="RdBu_r",
        symmetric=True,
        robust=True,
    )
    scatter_continuous_3d(
        axes[6],
        xyz,
        positive_score,
        "7. positive NK score (standardized; Reds)",
        cmap="Reds",
        robust=True,
    )
    scatter_continuous_3d(
        axes[7],
        xyz,
        excluded_score,
        "8. NK-excluded score (standardized; Blues)",
        cmap="Blues",
        robust=True,
    )
    plot_cluster_marker_agreement(
        axes[8],
        adata,
        positive_score,
        excluded_score,
        min_cells=panel9_min_cells,
    )

    plt.tight_layout()
    png = os.path.join(figdir, "annotation_umap_review_panels_3d.png")
    style_figure(fig, tick_size=SMALL_TICK_LABEL_SIZE, legend_size=LEGEND_FONT_SIZE)
    style_all_legends(fig)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    plt.close(fig)


def marker_mean_score(adata, markers):
    available = [gene for gene in markers if gene in adata.var_names]
    missing = [gene for gene in markers if gene not in adata.var_names]
    if not available:
        print(f"[WARN] No requested markers found for score: {markers}")
        return np.zeros(adata.n_obs, dtype=float), available, missing
    matrix = adata[:, available].X
    if hasattr(matrix, "toarray"):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.log1p(np.clip(matrix, a_min=0, a_max=None))
    gene_mean = np.nanmean(matrix, axis=0)
    gene_std = np.nanstd(matrix, axis=0)
    gene_std[~np.isfinite(gene_std) | (gene_std == 0)] = 1.0
    score = ((matrix - gene_mean) / gene_std).mean(axis=1)
    return score, available, missing


def obs_values(adata, key):
    if key in adata.obs:
        return adata.obs[key].astype(str).fillna("NA").values
    return np.array(["NA"] * adata.n_obs)


def zscore(values):
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values)
    std = np.nanstd(values)
    if not np.isfinite(std) or std == 0:
        return np.zeros_like(values, dtype=float)
    return (values - mean) / std


def plot_cluster_marker_agreement(ax, adata, positive_score, excluded_score, *, min_cells=0):
    obs = pd.DataFrame(
        {
            GROUPBY: adata.obs[GROUPBY].astype(str).values,
            "label": adata.obs[cfg.REFINED_LABEL_KEY].astype(str).values,
            "positive_score": np.asarray(positive_score, dtype=float),
            "excluded_score": np.asarray(excluded_score, dtype=float),
        }
    )
    cluster = (
        obs.groupby(GROUPBY, sort=False)
        .agg(
            label=("label", lambda x: x.value_counts().idxmax()),
            positive_score=("positive_score", "mean"),
            excluded_score=("excluded_score", "mean"),
            n_cells=("label", "size"),
        )
        .reset_index()
    )
    cluster = cluster.sort_values(GROUPBY, key=lambda s: s.map(cluster_sort_key))
    n_before = len(cluster)
    if min_cells and min_cells > 0:
        hidden = cluster.loc[cluster["n_cells"] < min_cells, GROUPBY].astype(str).tolist()
        if hidden:
            print(f"[PANEL9] Hiding clusters with n_cells < {min_cells}: {', '.join(hidden)}")
        cluster = cluster.loc[cluster["n_cells"] >= min_cells].copy()
        if cluster.empty:
            raise ValueError(
                f"Panel 9 filter removed all clusters; lower --panel9-min-cells below {min_cells}."
            )
    colors = category_colors(cluster["label"].tolist())
    sizes = 28 + 120 * np.sqrt(cluster["n_cells"] / cluster["n_cells"].max())

    for _, row in cluster.iterrows():
        ax.scatter(
            row["positive_score"],
            row["excluded_score"],
            s=float(sizes.loc[row.name]) if hasattr(sizes, "loc") else 80,
            color=colors[row["label"]],
            alpha=0.82,
            edgecolors="white",
            linewidths=0.35,
        )
        ax.text(
            row["positive_score"],
            row["excluded_score"],
            str(row[GROUPBY]),
            ha="center",
            va="center",
            fontsize=9,
            color="#222222",
            weight="bold",
        )

    ax.axvline(0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.axhline(0, color="#bbbbbb", linewidth=0.8, zorder=0)
    ax.grid(True, color="#e2e2e2", linewidth=0.6, alpha=0.8)
    title = "9. Cluster-level NK vs exclusion marker scores"
    if min_cells and min_cells > 0 and len(cluster) < n_before:
        title += f" (n>={min_cells})"
    ax.set_title(title)
    ax.set_xlabel("Mean standardized positive NK score")
    ax.set_ylabel("Mean standardized NK-excluded score")
    set_padded_limits(ax, cluster["positive_score"], axis="x")
    set_padded_limits(ax, cluster["excluded_score"], axis="y")
    style_axis(ax, tick_size=SMALL_TICK_LABEL_SIZE)


def set_padded_limits(ax, values, *, axis: str):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        limits = (-1.0, 1.0)
    else:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
        span = max(vmax - vmin, 0.05)
        pad = 0.12 * span
        limits = (vmin - pad, vmax + pad)
        if limits[0] == limits[1]:
            limits = (limits[0] - 0.05, limits[1] + 0.05)
    if axis == "x":
        ax.set_xlim(*limits)
    else:
        ax.set_ylim(*limits)


def scatter_continuous(
    ax,
    xy,
    values,
    title,
    *,
    cmap,
    robust=False,
    symmetric=False,
):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    plot_values = values.copy()
    plot_values[~finite] = np.nan
    if finite.any() and robust:
        if symmetric:
            limit = np.nanpercentile(np.abs(plot_values[finite]), 99)
            vmin, vmax = -limit, limit
        else:
            vmin = np.nanpercentile(plot_values[finite], 1)
            vmax = np.nanpercentile(plot_values[finite], 99)
    elif finite.any():
        vmin, vmax = np.nanmin(plot_values[finite]), np.nanmax(plot_values[finite])
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    order = np.argsort(np.nan_to_num(plot_values, nan=-np.inf))
    sc = ax.scatter(
        xy[order, 0],
        xy[order, 1],
        c=plot_values[order],
        s=QC_POINT_SIZE,
        alpha=QC_POINT_ALPHA,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
        linewidths=0,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)


def scatter_continuous_3d(
    ax,
    xyz,
    values,
    title,
    *,
    cmap,
    robust=False,
    symmetric=False,
):
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    plot_values = values.copy()
    plot_values[~finite] = np.nan
    if finite.any() and robust:
        if symmetric:
            limit = np.nanpercentile(np.abs(plot_values[finite]), 99)
            vmin, vmax = -limit, limit
        else:
            vmin = np.nanpercentile(plot_values[finite], 1)
            vmax = np.nanpercentile(plot_values[finite], 99)
    elif finite.any():
        vmin, vmax = np.nanmin(plot_values[finite]), np.nanmax(plot_values[finite])
    else:
        vmin, vmax = 0.0, 1.0
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = 0.0, 1.0

    order = np.argsort(np.nan_to_num(plot_values, nan=-np.inf))
    sc = ax.scatter(
        xyz[order, 0],
        xyz[order, 1],
        xyz[order, 2],
        c=plot_values[order],
        s=QC_POINT_SIZE,
        alpha=QC_POINT_ALPHA,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        depthshade=True,
        rasterized=True,
        linewidths=0,
    )
    format_3d_axis(ax, title, xyz)
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)


def scatter_categorical(
    ax,
    xy,
    values,
    title,
    *,
    show_legend=True,
    annotate_clusters=False,
    category_order=None,
    colors_override=None,
):
    values = np.asarray(values).astype(str)
    categories = ordered_categories(values, category_order)
    colors = colors_override or category_colors(categories)

    for category in categories:
        mask = values == category
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=colors[category],
            rasterized=True,
            label=category,
        )

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)

    if annotate_clusters:
        annotate_category_centers(ax, xy, values)

    if not show_legend:
        return

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=10,
            markerfacecolor=colors[category],
            markeredgecolor="none",
            alpha=1.0,
            label=short_legend_label(category),
        )
        for category in categories
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        handletextpad=0.4,
    )
    style_legend(ax.get_legend())


def scatter_categorical_3d(
    ax,
    xyz,
    values,
    title,
    *,
    show_legend=True,
    annotate_clusters=False,
    category_order=None,
    colors_override=None,
):
    values = np.asarray(values).astype(str)
    categories = ordered_categories(values, category_order)
    colors = colors_override or category_colors(categories)

    for category in categories:
        mask = values == category
        ax.scatter(
            xyz[mask, 0],
            xyz[mask, 1],
            xyz[mask, 2],
            s=POINT_SIZE * 1.8,
            alpha=min(0.9, POINT_ALPHA + 0.15),
            color=colors[category],
            depthshade=True,
            rasterized=True,
            label=category,
            linewidths=0,
        )

    format_3d_axis(ax, title, xyz)
    if annotate_clusters:
        annotate_category_centers_3d(ax, xyz, values)
    if not show_legend:
        return
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=10,
            markerfacecolor=colors[category],
            markeredgecolor="none",
            alpha=1.0,
            label=short_legend_label(category),
        )
        for category in categories
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        handletextpad=0.4,
    )
    style_legend(ax.get_legend())


def short_legend_label(label: str, *, max_chars: int = MAX_LEGEND_LABEL_CHARS) -> str:
    text = str(label)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip("_- ") + "..."


def ordered_categories(values, category_order=None):
    seen = set(np.asarray(values).astype(str))
    if category_order:
        ordered = [str(value) for value in category_order if str(value) in seen]
        ordered.extend(sorted(seen - set(ordered), key=category_sort_key))
        return ordered
    return sorted(seen, key=category_sort_key)


def label_order_by_cluster(labels, clusters):
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels).astype(str),
            "cluster": np.asarray(clusters).astype(str),
        }
    )
    rows = []
    for label, group in frame.groupby("label", sort=False):
        cluster_counts = group["cluster"].value_counts()
        dominant_cluster = cluster_counts.index[0]
        rows.append((cluster_sort_key(dominant_cluster), str(label)))
    rows.sort(key=lambda item: item[0])
    return [label for _, label in rows]


def label_colors_from_cluster_colors(label_order, labels, clusters, *, preferred=None):
    preferred = preferred or {}
    label_to_cluster = {}
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels).astype(str),
            "cluster": np.asarray(clusters).astype(str),
        }
    )
    for label in label_order:
        sub = frame.loc[frame["label"] == str(label), "cluster"]
        if sub.empty:
            continue
        label_to_cluster[str(label)] = sub.value_counts().index[0]
    cluster_colors = category_colors(sorted(set(clusters), key=category_sort_key))
    return {
        label: preferred.get(label, cluster_colors.get(cluster, "#999999"))
        for label, cluster in label_to_cluster.items()
    }


def format_3d_axis(ax, title, xyz):
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("UMAP1", labelpad=2)
    ax.set_ylabel("UMAP2", labelpad=2)
    ax.set_zlabel("UMAP3", labelpad=2)
    ax.tick_params(axis="both", which="major", labelsize=9, length=2, pad=-2, colors="#666666")
    ax.set_proj_type("persp", focal_length=0.9)
    ax.view_init(elev=24, azim=-42)
    set_equal_3d_limits(ax, xyz)
    ax.grid(True, color="#cfcfcf", linewidth=0.45, alpha=0.65)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((0.96, 0.96, 0.96, 0.18))
        axis.pane.set_edgecolor("#c7c7c7")
        axis.pane.set_alpha(0.18)
    try:
        ax.zaxis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.3)
        ax.xaxis._axinfo["grid"]["color"] = (0.82, 0.82, 0.82, 0.3)
        ax.yaxis._axinfo["grid"]["color"] = (0.68, 0.68, 0.68, 0.8)
    except Exception:
        pass


def set_equal_3d_limits(ax, xyz):
    mins = np.nanpercentile(xyz, 0.2, axis=0)
    maxs = np.nanpercentile(xyz, 99.8, axis=0)
    ranges = maxs - mins
    ranges[~np.isfinite(ranges) | (ranges == 0)] = 1.0
    pad = 0.015 * ranges
    ax.set_xlim(mins[0] - pad[0], maxs[0] + pad[0])
    ax.set_ylim(mins[1] - pad[1], maxs[1] + pad[1])
    ax.set_zlim(mins[2] - pad[2], maxs[2] + pad[2])
    try:
        ax.set_box_aspect(tuple(ranges / np.nanmax(ranges)), zoom=1.18)
    except AttributeError:
        pass


def annotate_category_centers(ax, xy, values):
    values = np.asarray(values).astype(str)
    for category in sorted(set(values), key=category_sort_key):
        mask = values == category
        if mask.sum() == 0:
            continue
        center = np.median(xy[mask], axis=0)
        ax.text(
            center[0],
            center[1],
            category,
            ha="center",
            va="center",
            fontsize=10,
            color="#2b2b2b",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.35,
            },
        )


def annotate_category_centers_3d(ax, xyz, values):
    values = np.asarray(values).astype(str)
    for category in sorted(set(values), key=category_sort_key):
        mask = values == category
        if mask.sum() == 0:
            continue
        center = np.median(xyz[mask], axis=0)
        ax.text(
            center[0],
            center[1],
            center[2],
            category,
            ha="center",
            va="center",
            fontsize=10,
            color="#2b2b2b",
            weight="bold",
        )


def category_sort_key(value):
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def cluster_sort_key(value):
    return int(value)


def category_colors(categories):
    preferred = {
        "blood": "#D62728",
        "cord blood": "#FF7F00",
        "bone marrow": "#9467BD",
        "lung": "#56B4E9",
        "liver": "#8C564B",
        "kidney": "#2CA02C",
        "spleen": "#D33682",
        "lymph node": "#F0E442",
        "thymus": "#BCBD22",
        "decidua": "#E377C2",
        "B": "#1f77b4",
        "T": "#d62728",
        "Cytokine-Stimulated": "#E7298A",
        "Developmental": "#D55E00",
        "Mature Cytotoxic": "#FDB462",
        "Mature Cytotoxic TCF7+": "#8DD3C7",
        "Transitional Cytotoxic": "#FB8072",
        "Transitional Cytotoxic Tissue-Resident": "#BC80BD",
        "Cytokine-Stimulated CCR7+": "#aec7e8",
        "Cytokine-Stimulated Cycling": "#17becf",
        "Proliferative": "#33A02C",
        "Regulatory": "#B2DF8A",
        "Unconventional": "#7570B3",
        "Lung Cytotoxic NK": "#BCBD22",
        "Lung DOCK4+ SLC8A1+ NK": "#8c564b",
        "Unknown_Kidney": "#80CDC1",
        "Unknown_BM_1": "#CAB2D6",
        "Unknown_BM_2": "#B15928",
        "Unknown_BM_1 Erythroid-like": "#c5b0d5",
        "Unknown_Lung_1": "#BC80BD",
        "Unknown_Lung_3": "#8C6D31",
        "Unknown_Lung_4": "#F0E442",
        "Unknown_Lung_5": "#FCCDE5",
        "Unknown_Lung_6": "#00A6D6",
        "Myeloid-like": "#7f7f7f",
        "L6_Developmental_immature_Proliferating": "#E7298A",
        "L6_Developmental_immature": "#E7298A",
        "NK1_Chemokine_inflammatory": "#0072B2",
        "NK1_Cytotoxic_activated": "#D55E00",
        "NK1_Checkpoint_exhausted": "#7570B3",
        "NK1_Proliferating": "#80B1D3",
        "NK2_Chemokine_inflammatory": "#33A02C",
        "NK2_CIMP_cytokine_primed_memory_like": "#F0E442",
        "NK2_Checkpoint_exhausted": "#CC79A7",
        "NK2_Cytotoxic_activated": "#009E73",
        "NK2_ER_stress_UPR": "#8DD3C7",
        "NK2_Homeostatic_quiescent": "#B2DF8A",
        "NK2_Proliferating": "#56B4E9",
        "cNK_Cytotoxic_activated": "#8C2D04",
        "cNK_Metabolic_stress_hypoxia": "#00A6D6",
        "cNK_Homeostatic_quiescent": "#A65628",
        "cNK_Proliferating": "#CAB2D6",
        "cNK_ER_stress_UPR": "#7B3294",
        "trNK_Chemokine_inflammatory": "#A6761D",
        "trNK_Homeostatic_quiescent": "#FF7F00",
        "Non-NK": "#8A8A8A",
        "Unsure_Chemokine_inflammatory": "#E6AB02",
        "Unsure_Homeostatic_quiescent": "#8DA0CB",
        "Unsure_Proliferating": "#FB8072",
        "0": "#0072B2",
        "1": "#D55E00",
        "2": "#7570B3",
        "3": "#E7298A",
        "4": "#33A02C",
        "5": "#E6AB02",
        "6": "#A6761D",
        "7": "#8DD3C7",
        "8": "#56B4E9",
        "9": "#B2DF8A",
        "10": "#FB8072",
        "11": "#FDB462",
        "12": "#CAB2D6",
        "13": "#FFFF33",
        "14": "#B15928",
        "15": "#80CDC1",
        "16": "#F0E442",
        "17": "#BEBADA",
        "18": "#FF6F61",
        "19": "#80B1D3",
        "20": "#FF7F00",
        "21": "#B3DE69",
        "22": "#FCCDE5",
        "23": "#BC80BD",
        "24": "#8C6D31",
    }
    # Keep refined-label colors synchronized with SCANVI and annotation-flow
    # plots. Numeric Leiden colors above are intentionally preserved.
    preferred.update(PREFERRED_STATE_COLORS)
    palette = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#F0E442",
        "#56B4E9",
        "#7B3294",
        "#A6761D",
        "#E7298A",
        "#8C6D31",
        "#4E79A7",
        "#FF7F00",
        "#1B9E77",
        "#B15928",
        "#BC80BD",
        "#00A6D6",
        "#FF6F61",
        "#8DD3C7",
        "#E6AB02",
        "#CAB2D6",
        "#33A02C",
        "#FB8072",
        "#80B1D3",
        "#B3DE69",
        "#FDB462",
        "#FCCDE5",
        "#BEBADA",
        "#FFFF33",
    ]

    colors = {}
    fallback_i = 0
    for category in categories:
        if category in preferred:
            colors[category] = preferred[category]
        else:
            colors[category] = palette[fallback_i % len(palette)]
            fallback_i += 1
    return colors


if __name__ == "__main__":
    main()
