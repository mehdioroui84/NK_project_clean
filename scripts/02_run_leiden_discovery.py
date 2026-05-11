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
from nk_project.discovery import run_leiden_grid
from nk_project.io_utils import ensure_dirs


DEFAULT_RESOLUTIONS = [0.2, 0.3, 0.4]
DEFAULT_WORKSHEET_RESOLUTION = 0.4
POINT_SIZE_RESOLUTION = 0.035
POINT_ALPHA_RESOLUTION = 0.35
POINT_SIZE_OVERVIEW = 0.06
POINT_ALPHA_OVERVIEW = 0.38

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
    "Unknown_Lung_5": "Lung DOCK4+ SLC8A1+ NK",
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
    outdir = os.path.join(cfg.BASE_OUTDIR, "leiden_discovery")
    figdir = os.path.join(outdir, "figures")
    ensure_dirs(outdir, figdir)

    h5ad_path = os.path.join(outdir, "full_scvi_leiden.h5ad")

    if not args.skip_clustering:
        adata_path = args.input_h5ad or os.path.join(cfg.LATENT_OUTDIR, "scvi_full_with_latent.h5ad")
        print(f"[LOAD] {adata_path}")
        adata = sc.read_h5ad(adata_path)
        adata, summary = run_leiden_grid(
            adata,
            latent_key=args.latent_key,
            resolutions=args.resolutions,
            n_neighbors=cfg.DISCOVERY_N_NEIGHBORS,
            seed=cfg.SEED,
            outdir=outdir,
            label_key=cfg.LABEL_KEY,
            dataset_key=cfg.DATASET_KEY,
            assay_key=cfg.ASSAY_CLEAN_KEY,
        )
        adata.write(h5ad_path)
        print(summary.to_string(index=False))
        print(f"[SAVE] {h5ad_path}")
        print(f"[SAVE] {outdir}")
    else:
        print("[SKIP] Leiden clustering")
        print(f"[LOAD] {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)

    if not args.skip_plots:
        plot_resolution_overview(adata, args.resolutions, figdir)
        plot_single_resolution_overview(adata, args.worksheet_resolution, figdir)
    else:
        print("[SKIP] Leiden overview plots")

    if not args.skip_worksheets:
        obs_path = os.path.join(outdir, "obs_with_leiden.csv")
        if os.path.exists(obs_path):
            print(f"[LOAD] {obs_path}")
            obs = pd.read_csv(obs_path, index_col=0, low_memory=False)
        else:
            print("[WARN] obs_with_leiden.csv not found; using AnnData obs.")
            obs = adata.obs.copy()
        for resolution in args.worksheet_resolutions:
            build_and_save_worksheet(obs, resolution, outdir)
    else:
        print("[SKIP] annotation worksheets")

    print("[DONE] Full-data Leiden discovery complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run full-data SCVI Leiden discovery, resolution overview plots, "
            "and first-pass annotation worksheets."
        )
    )
    parser.add_argument("--input-h5ad", default=None)
    parser.add_argument("--latent-key", default="X_scVI")
    parser.add_argument("--resolutions", type=float, nargs="+", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--worksheet-resolution", type=float, default=DEFAULT_WORKSHEET_RESOLUTION)
    parser.add_argument("--worksheet-resolutions", type=float, nargs="+", default=DEFAULT_RESOLUTIONS)
    parser.add_argument("--skip-clustering", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-worksheets", action="store_true")
    return parser.parse_args()


def plot_resolution_overview(adata, resolutions, outdir):
    if "X_umap" not in adata.obsm:
        raise KeyError("X_umap not found in full Leiden AnnData.")
    nrows = len(resolutions)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 6.0 * nrows))
    if nrows == 1:
        axes = np.asarray([axes])

    xy = adata.obsm["X_umap"]
    for row_idx, resolution in enumerate(resolutions):
        key = leiden_key(resolution)
        if key not in adata.obs:
            raise KeyError(f"{key!r} not found. Rerun with --resolutions {resolution}.")

        for col_idx, ax in enumerate(axes[row_idx]):
            if col_idx == 0:
                values = adata.obs[key].astype(str).values
                scatter_categorical(
                    ax,
                    xy,
                    values,
                    f"{key} ({len(set(values))} clusters)",
                    point_size=POINT_SIZE_RESOLUTION,
                    point_alpha=POINT_ALPHA_RESOLUTION,
                    show_legend=False,
                    annotate_clusters=True,
                )
            elif col_idx == 1:
                scatter_categorical(
                    ax,
                    xy,
                    adata.obs[cfg.LABEL_KEY].astype(str).values,
                    f"{key} background: {cfg.LABEL_KEY}",
                    point_size=POINT_SIZE_RESOLUTION,
                    point_alpha=POINT_ALPHA_RESOLUTION,
                    show_legend=True,
                )
            else:
                scatter_categorical(
                    ax,
                    xy,
                    adata.obs[cfg.ASSAY_CLEAN_KEY].astype(str).values,
                    f"{key} background: {cfg.ASSAY_CLEAN_KEY}",
                    point_size=POINT_SIZE_RESOLUTION,
                    point_alpha=POINT_ALPHA_RESOLUTION,
                    show_legend=True,
                )

    fig.suptitle("Full-data SCVI latent space: Leiden resolution comparison", fontsize=15)
    plt.tight_layout()
    png = os.path.join(outdir, "full_leiden_resolution_overview.png")
    pdf = os.path.join(outdir, "full_leiden_resolution_overview.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")
    plt.close(fig)


def plot_single_resolution_overview(adata, resolution, outdir):
    groupby = leiden_key(resolution)
    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} not found in adata.obs. Rerun with --resolutions {resolution}.")
    if "X_umap" not in adata.obsm:
        raise KeyError("X_umap not found in full Leiden AnnData.")

    xy = adata.obsm["X_umap"]
    panels = [
        (groupby, f"1.1 full {groupby} cluster IDs", False, True),
        (cfg.LABEL_KEY, f"1.2 original annotation: {cfg.LABEL_KEY}", True, False),
        ("tissue", "2.1 tissue", True, False),
        (cfg.DATASET_KEY, f"2.2 {cfg.DATASET_KEY}", False, False),
        (cfg.ASSAY_CLEAN_KEY, f"3.1 {cfg.ASSAY_CLEAN_KEY}", True, False),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(20, 22))
    fig.suptitle(f"Full-data SCVI latent space: {groupby}", fontsize=14)
    axes = axes.ravel()

    for ax, (obs_key, title, show_legend, annotate) in zip(axes, panels):
        if obs_key in adata.obs:
            values = adata.obs[obs_key].astype(str).values
        else:
            values = np.array(["unknown"] * adata.n_obs)
        scatter_categorical(
            ax,
            xy,
            values,
            title,
            point_size=POINT_SIZE_OVERVIEW,
            point_alpha=POINT_ALPHA_OVERVIEW,
            show_legend=show_legend,
            annotate_clusters=annotate,
        )

    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.95,
        "Next step:\nmap full Leiden clusters\n"
        "to NK_State_refined labels using\n"
        "cluster composition + markers + DE.",
        ha="left",
        va="top",
        fontsize=16,
        color="#17202a",
    )

    plt.tight_layout()
    png = os.path.join(outdir, f"full_{groupby}_overview.png")
    pdf = os.path.join(outdir, f"full_{groupby}_overview.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")
    plt.close(fig)


def build_and_save_worksheet(obs, resolution, outdir):
    groupby = leiden_key(resolution)
    if groupby not in obs.columns:
        raise KeyError(f"{groupby!r} not found in obs.")
    worksheet = build_worksheet(obs, groupby)
    out_path = os.path.join(outdir, f"full_{groupby}_annotation_worksheet.csv")
    worksheet.to_csv(out_path)
    print(f"[SAVE] {out_path}")
    print(f"\n[PREVIEW {groupby}]")
    print(worksheet.round(3).to_string())


def build_worksheet(obs, key):
    out = pd.DataFrame({"n_cells": obs[key].astype(str).value_counts().sort_index()})
    for col in [cfg.LABEL_KEY, "tissue", cfg.DATASET_KEY, cfg.ASSAY_CLEAN_KEY]:
        if col in obs.columns:
            out = out.join(top_summary(obs, key, col))

    out["draft_refined_label"] = [draft_label(row) for _, row in out.iterrows()]
    out["review_priority"] = [review_priority(row) for _, row in out.iterrows()]
    out["review_notes"] = [review_notes(row) for _, row in out.iterrows()]
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
        ("top_tissue_frac", 0.85, "high_tissue_specificity"),
        (f"top_{cfg.ASSAY_CLEAN_KEY}_frac", 0.85, "high_assay_specificity"),
        (f"top_{cfg.DATASET_KEY}_frac", 0.75, "high_dataset_specificity"),
    ]:
        if col in row and pd.notna(row[col]) and float(row[col]) >= threshold:
            notes.append(note)

    frac_col = f"top_{cfg.LABEL_KEY}_frac"
    if frac_col in row and float(row[frac_col]) < 0.70:
        notes.append("mixed_original_NK_State")
    return ";".join(notes)


def scatter_categorical(
    ax,
    xy,
    values,
    title,
    *,
    point_size,
    point_alpha,
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
            s=point_size,
            alpha=point_alpha,
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

    if not show_legend or len(categories) > 30:
        return

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
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
            color="black",
            weight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.35,
            },
        )


def category_sort_key(value):
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def category_colors(categories):
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3", "Paired"):
        cmap = plt.get_cmap(cmap_name)
        palette.extend([cmap(i) for i in range(cmap.N)])
    return {category: palette[i % len(palette)] for i, category in enumerate(categories)}


def leiden_key(resolution):
    return f"leiden_{str(resolution).replace('.', '_')}"


if __name__ == "__main__":
    main()
