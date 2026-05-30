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
from nk_project.io_utils import ensure_dirs


GROUPBY = "leiden_0_4"
OUTDIR_NAME = "refined_annotation_v1"
POINT_SIZE = 0.06
POINT_ALPHA = 0.65
QC_POINT_SIZE = 0.08
QC_POINT_ALPHA = 0.95

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


# Collapsed full-data v1 labels. Leiden clusters are used as evidence; the
# final labels intentionally merge related clusters to avoid over-fragmenting
# the training target.
REFINED_LABEL_BY_CLUSTER = {
    "0": "Mature Cytotoxic",
    "1": "Proliferative",
    "2": "Mature Cytotoxic",
    "3": "T",
    "4": "Transitional Cytotoxic Tissue-Resident",
    "5": "Cytokine-Stimulated CCR7+",
    "6": "Lung Cytotoxic NK",
    "7": "Unknown_Kidney",
    "8": "T",
    "9": "Transitional Cytotoxic",
    "10": "Lung Cytotoxic NK",
    "11": "Mature Cytotoxic",
    "12": "Cytokine-Stimulated Cycling",
    "13": "Transitional Cytotoxic Tissue-Resident",
    "14": "Mature Cytotoxic TCF7+",
    "15": "T",
    "16": "B",
    "17": "Mature Cytotoxic",
    "18": "Regulatory",
    "19": "B",
    "20": "B",
    "21": "Unknown_BM_1 Erythroid-like",
    "22": "Myeloid-like",
    "23": "Lung Cytotoxic NK",
    "24": "Lung DOCK4+ SLC8A1+ NK",
}


def main():
    args = parse_args()
    in_path = os.path.join(cfg.BASE_OUTDIR, "leiden_discovery", "full_scvi_leiden.h5ad")
    outdir = args.outdir or os.path.join(cfg.BASE_OUTDIR, OUTDIR_NAME)
    figdir = os.path.join(outdir, "figures")
    ensure_dirs(outdir, figdir)

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if GROUPBY not in adata.obs:
        raise KeyError(f"{GROUPBY!r} not found in adata.obs.")
    if "X_umap" not in adata.obsm:
        raise KeyError("X_umap not found in full-data SCVI Leiden AnnData.")

    label_mapping, label_source = load_label_mapping(args.mapping_csv, label_column=args.label_column)
    apply_labels(adata, label_mapping, label_source=label_source)
    write_outputs(adata, outdir, label_mapping)
    plot_refined_umap(adata, figdir)
    plot_annotation_qc_umap(adata, figdir)
    if args.make_3d_umap:
        xyz = get_umap3d(adata, args, outdir)
        plot_refined_umap_3d(adata, figdir, xyz)
        plot_annotation_qc_umap_3d(adata, figdir, xyz)
    print("[DONE] Full-data refined v1 label application complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply reviewed refined labels to full-data Leiden clusters. "
            "By default uses the curated hardcoded v1 mapping; optionally accepts "
            "a reviewed mapping CSV from the annotation agent."
        )
    )
    parser.add_argument(
        "--mapping-csv",
        default=None,
        help=(
            "Optional reviewed mapping CSV. Expected columns: leiden_0_4 and either "
            "candidate_refined_label or NK_State_refined."
        ),
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help=(
            "Column from --mapping-csv to use as the final SCANVI training label. "
            "Useful choices for annotation-agent outputs: candidate_refined_label, "
            "current_final_label, agent_preferred_label, or approved_label. "
            "Default: auto-detect approved_label, NK_State_refined, candidate_refined_label, then refined_label."
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
    return parser.parse_args()


def load_label_mapping(mapping_csv=None, *, label_column=None):
    if mapping_csv is None:
        print("[MAPPING] Using curated hardcoded refined-v1 mapping.")
        return dict(REFINED_LABEL_BY_CLUSTER), "hardcoded_curated_mapping"

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
            "approved_label",
            cfg.REFINED_LABEL_KEY,
            "candidate_refined_label",
            "current_final_label",
            "refined_label",
        ]:
            if candidate in mapping.columns:
                label_col = candidate
                break
    if label_col is None:
        raise KeyError(
            f"{mapping_csv} must contain one of: {cfg.REFINED_LABEL_KEY!r}, "
            "'approved_label', 'candidate_refined_label', 'current_final_label', or 'refined_label'."
        )
    print(f"[MAPPING_LABEL_COLUMN] {label_col}")

    labels = mapping[label_col].fillna("").astype(str).str.strip()
    if labels.eq("").any():
        bad = mapping.loc[labels.eq(""), GROUPBY].astype(str).tolist()
        raise ValueError(f"Mapping CSV has empty labels in {label_col!r} for clusters: {bad}")
    out = dict(zip(mapping[GROUPBY].astype(str), labels))
    missing = sorted(set(REFINED_LABEL_BY_CLUSTER) - set(out), key=cluster_sort_key)
    if missing:
        raise ValueError(f"Mapping CSV is missing {GROUPBY} clusters: {missing}")
    return out, label_col


def apply_labels(adata, label_mapping, *, label_source):
    clusters = adata.obs[GROUPBY].astype(str)
    labels = clusters.map(label_mapping)
    if labels.isna().any():
        missing = sorted(clusters[labels.isna()].unique(), key=cluster_sort_key)
        raise ValueError(f"Missing refined labels for {GROUPBY} clusters: {missing}")

    adata.obs["NK_State_original"] = adata.obs[cfg.LABEL_KEY].astype(str)
    adata.obs[cfg.REFINED_LABEL_KEY] = labels.astype("category")
    adata.obs["NK_State_refined_v1_source"] = f"full_data_leiden_0_4_mapping:{label_source}"

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
        (GROUPBY, f"1.1 full {GROUPBY} clusters", False, True),
        (cfg.LABEL_KEY, f"1.2 original annotation: {cfg.LABEL_KEY}", True, False),
        (cfg.REFINED_LABEL_KEY, f"1.3 refined v1 annotation: {cfg.REFINED_LABEL_KEY}", True, False),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    fig.suptitle("Full-data SCVI latent space: refined annotation v1", fontsize=15)
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
        (GROUPBY, f"1.1 3D {GROUPBY} clusters", False, True),
        (cfg.LABEL_KEY, f"1.2 original annotation: {cfg.LABEL_KEY}", True, False),
        (cfg.REFINED_LABEL_KEY, f"1.3 refined v1 annotation: {cfg.REFINED_LABEL_KEY}", True, False),
    ]

    fig = plt.figure(figsize=(22, 7))
    fig.suptitle("Full-data SCVI latent space: refined annotation v1, 3D UMAP", fontsize=15)
    axes = [fig.add_subplot(1, 3, idx + 1, projection="3d") for idx in range(3)]
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
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    plt.close(fig)


def plot_annotation_qc_umap(adata, figdir):
    xy = adata.obsm["X_umap"]
    positive_score, positive_used, positive_missing = marker_mean_score(adata, PAN_NK_SCORE_MARKERS)
    excluded_score, excluded_used, excluded_missing = marker_mean_score(adata, NK_EXCLUDED_SCORE_MARKERS)
    original_label_key = "NK_State_original" if "NK_State_original" in adata.obs else cfg.LABEL_KEY
    original_label = obs_values(adata, original_label_key)
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
    fig, axes = plt.subplots(3, 3, figsize=(24, 22))
    axes = axes.ravel()
    fig.suptitle("Annotation QC: cluster labels, metadata, and NK marker scores", fontsize=15)

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
        original_label,
        f"2. Yuntao/original annotation: {original_label_key}",
        show_legend=True,
        annotate_clusters=False,
    )
    scatter_categorical(
        axes[2],
        xy,
        adata.obs[cfg.REFINED_LABEL_KEY].astype(str).values,
        f"3. final annotation: {cfg.REFINED_LABEL_KEY}",
        show_legend=True,
        annotate_clusters=False,
    )
    scatter_categorical(
        axes[3],
        xy,
        tissue,
        "4. Tissue",
        show_legend=True,
        annotate_clusters=False,
    )
    scatter_categorical(
        axes[4],
        xy,
        dataset,
        "5. Dataset ID",
        show_legend=False,
        annotate_clusters=False,
    )
    scatter_categorical(
        axes[5],
        xy,
        assay,
        "6. Assay clean",
        show_legend=True,
        annotate_clusters=False,
    )
    scatter_continuous(
        axes[6],
        xy,
        positive_score,
        f"7. positive NK score (Reds; {len(positive_used)}/{len(PAN_NK_SCORE_MARKERS)} genes)",
        cmap="Reds",
        robust=True,
    )
    scatter_continuous(
        axes[7],
        xy,
        signed_score,
        "8. signed NK identity score (red=NK-like, blue=NK-excluded)",
        cmap="RdBu_r",
        symmetric=True,
        robust=True,
    )
    scatter_continuous(
        axes[8],
        xy,
        excluded_score,
        f"9. NK-excluded score (Blues; {len(excluded_used)}/{len(NK_EXCLUDED_SCORE_MARKERS)} genes)",
        cmap="Blues",
        robust=True,
    )

    plt.tight_layout()
    png = os.path.join(figdir, "annotation_umap_review_panels.png")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    plt.close(fig)


def plot_annotation_qc_umap_3d(adata, figdir, xyz):
    positive_score, positive_used, _ = marker_mean_score(adata, PAN_NK_SCORE_MARKERS)
    excluded_score, excluded_used, _ = marker_mean_score(adata, NK_EXCLUDED_SCORE_MARKERS)
    original_label_key = "NK_State_original" if "NK_State_original" in adata.obs else cfg.LABEL_KEY
    original_label = obs_values(adata, original_label_key)
    tissue = obs_values(adata, "tissue")
    dataset = obs_values(adata, cfg.DATASET_KEY)
    assay = obs_values(adata, cfg.ASSAY_CLEAN_KEY)
    signed_score = zscore(positive_score) - zscore(excluded_score)

    fig = plt.figure(figsize=(24, 22))
    fig.suptitle("Annotation QC: 3D UMAP labels, metadata, and NK marker scores", fontsize=15)
    axes = [fig.add_subplot(3, 3, idx + 1, projection="3d") for idx in range(9)]

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
        original_label,
        f"2. Yuntao/original annotation: {original_label_key}",
        show_legend=True,
        annotate_clusters=False,
    )
    scatter_categorical_3d(
        axes[2],
        xyz,
        adata.obs[cfg.REFINED_LABEL_KEY].astype(str).values,
        f"3. final annotation: {cfg.REFINED_LABEL_KEY}",
        show_legend=True,
        annotate_clusters=False,
    )
    scatter_categorical_3d(axes[3], xyz, tissue, "4. Tissue", show_legend=True, annotate_clusters=False)
    scatter_categorical_3d(axes[4], xyz, dataset, "5. Dataset ID", show_legend=False, annotate_clusters=False)
    scatter_categorical_3d(axes[5], xyz, assay, "6. Assay clean", show_legend=True, annotate_clusters=False)
    scatter_continuous_3d(
        axes[6],
        xyz,
        positive_score,
        f"7. positive NK score (Reds; {len(positive_used)}/{len(PAN_NK_SCORE_MARKERS)} genes)",
        cmap="Reds",
        robust=True,
    )
    scatter_continuous_3d(
        axes[7],
        xyz,
        signed_score,
        "8. signed NK identity score (red=NK-like, blue=NK-excluded)",
        cmap="RdBu_r",
        symmetric=True,
        robust=True,
    )
    scatter_continuous_3d(
        axes[8],
        xyz,
        excluded_score,
        f"9. NK-excluded score (Blues; {len(excluded_used)}/{len(NK_EXCLUDED_SCORE_MARKERS)} genes)",
        cmap="Blues",
        robust=True,
    )

    plt.tight_layout()
    png = os.path.join(figdir, "annotation_umap_review_panels_3d.png")
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
    ax.set_title(title)
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
):
    values = np.asarray(values).astype(str)
    categories = sorted(set(values), key=category_sort_key)
    colors = category_colors(categories)

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

    ax.set_title(title)
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
            markersize=8,
            markerfacecolor=colors[category],
            markeredgecolor="none",
            alpha=1.0,
            label=category,
        )
        for category in categories
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        handletextpad=0.4,
    )


def scatter_categorical_3d(
    ax,
    xyz,
    values,
    title,
    *,
    show_legend=True,
    annotate_clusters=False,
):
    values = np.asarray(values).astype(str)
    categories = sorted(set(values), key=category_sort_key)
    colors = category_colors(categories)

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
            markersize=8,
            markerfacecolor=colors[category],
            markeredgecolor="none",
            alpha=1.0,
            label=category,
        )
        for category in categories
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        handletextpad=0.4,
    )


def format_3d_axis(ax, title, xyz):
    ax.set_title(title)
    ax.set_xlabel("UMAP1", labelpad=2)
    ax.set_ylabel("UMAP2", labelpad=2)
    ax.set_zlabel("UMAP3", labelpad=2)
    ax.tick_params(axis="both", which="major", labelsize=6, length=2, pad=-2, colors="#666666")
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
            fontsize=7,
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
            fontsize=7,
            color="#2b2b2b",
            weight="bold",
        )


def category_sort_key(value):
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def cluster_sort_key(value):
    return int(value)


def category_colors(categories):
    preferred = {
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
        "NK2_Chemokine_inflammatory": "#CC79A7",
        "NK2_CIMP_cytokine_primed_memory_like": "#F0E442",
        "NK2_Checkpoint_exhausted": "#33A02C",
        "NK2_Cytotoxic_activated": "#009E73",
        "NK2_Proliferating": "#56B4E9",
        "cNK_Cytotoxic_activated": "#8C2D04",
        "cNK_Metabolic_stress_hypoxia": "#80CDC1",
        "cNK_Homeostatic_quiescent": "#A65628",
        "cNK_Proliferating": "#00A6D6",
        "cNK_ER_stress_UPR": "#7B3294",
        "trNK_Chemokine_inflammatory": "#A6761D",
        "trNK_Homeostatic_quiescent": "#FF7F00",
        "Non-NK": "#6A3D9A",
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
