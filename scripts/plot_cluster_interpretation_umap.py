#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


GROUPBY = "leiden_0_4"
POINT_SIZE = 0.08
POINT_ALPHA = 0.45

DRAFT_PARENT_LABELS = {
    "Mature Cytotoxic TCF7+": "Mature Cytotoxic",
    "Mature Cytotoxic Engineered": "Mature Cytotoxic",
    "Cytokine-Stimulated Effector": "Cytokine-Stimulated",
    "Cytokine-Stimulated CCR7+": "Cytokine-Stimulated",
    "Transitional Cytotoxic Tissue-Resident": "Transitional Cytotoxic",
    "Lung Cytotoxic NK": "Unknown_Lung_6",
    "Lung GZMK+ XCL1+ NK": "Unknown_Lung_5",
    "Unknown_BM_1 Erythroid-like": "Unknown_BM_1",
}

DRAFT_COLOR_VARIANTS = {
    "Mature Cytotoxic TCF7+": ("lighten", 0.25),
    "Mature Cytotoxic Engineered": ("darken", 0.82),
    "Cytokine-Stimulated Effector": ("darken", 0.82),
    "Cytokine-Stimulated CCR7+": ("lighten", 0.25),
    "Transitional Cytotoxic Tissue-Resident": ("lighten", 0.20),
    "Lung Cytotoxic NK": ("darken", 0.90),
    "Lung GZMK+ XCL1+ NK": ("darken", 0.90),
    "Unknown_BM_1 Erythroid-like": ("darken", 0.85),
}

EXTRA_DRAFT_COLORS = {
    "Myeloid-like": "#4d4d4d",
}


def main():
    marker_dir = os.path.join(cfg.BASE_OUTDIR, "markers", "validation", GROUPBY)
    leiden_dir = os.path.join(cfg.BASE_OUTDIR, "leiden_validation")
    outdir = os.path.join(marker_dir, "figures")
    ensure_dirs(outdir)

    adata_path = os.path.join(leiden_dir, "validation_scvi_leiden.h5ad")
    worksheet_path = os.path.join(marker_dir, f"{GROUPBY}_interpretation_worksheet.csv")

    print(f"[LOAD] {adata_path}")
    adata = sc.read_h5ad(adata_path)
    if GROUPBY not in adata.obs:
        raise KeyError(f"{GROUPBY!r} not found in adata.obs")
    if "X_umap" not in adata.obsm:
        raise KeyError(
            "X_umap not found in validation AnnData. Rerun "
            "scripts/run_leiden_validation.py first."
        )

    print(f"[LOAD] {worksheet_path}")
    worksheet = pd.read_csv(worksheet_path, index_col=0)
    worksheet.index = worksheet.index.astype(str)

    adata.obs[GROUPBY] = adata.obs[GROUPBY].astype(str)
    attach_worksheet_columns(adata, worksheet)

    save_main_panel(adata, outdir)
    save_mapping_table(adata, outdir)

    print("[DONE] Interpretation UMAP plotting complete.")


def attach_worksheet_columns(adata, worksheet):
    required = [
        "broad_compartment_draft",
        "refined_NK_state_draft",
        "review_priority",
    ]
    missing = [col for col in required if col not in worksheet.columns]
    if missing:
        raise KeyError(f"Missing columns in worksheet: {missing}")

    for col in required:
        mapper = worksheet[col].astype(str).to_dict()
        adata.obs[col] = adata.obs[GROUPBY].map(mapper).fillna("unknown")


def save_main_panel(adata, outdir):
    xy = adata.obsm["X_umap"]
    manual_values = adata.obs[cfg.LABEL_KEY].astype(str).values
    draft_values = adata.obs["refined_NK_state_draft"].astype(str).values
    manual_categories = sorted(set(manual_values))
    manual_colors = category_colors(manual_categories)
    draft_colors = build_draft_colors(sorted(set(draft_values)), manual_colors)

    fig, axes = plt.subplots(3, 2, figsize=(18, 20))
    fig.suptitle(f"Validation {GROUPBY} interpretation draft", fontsize=14)
    axes = axes.ravel()

    scatter_categorical(
        axes[0],
        xy,
        adata.obs[GROUPBY].astype(str).values,
        f"1.1 {GROUPBY} cluster IDs",
        show_legend=False,
        annotate_clusters=True,
    )
    scatter_categorical(
        axes[1],
        xy,
        manual_values,
        f"1.2 Manual annotation: {cfg.LABEL_KEY}",
        fixed_colors=manual_colors,
    )
    scatter_categorical(
        axes[2],
        xy,
        draft_values,
        "2.1 Refined NK/reference state draft",
        fixed_colors=draft_colors,
    )
    scatter_categorical(
        axes[3],
        xy,
        adata.obs["tissue"].astype(str).values if "tissue" in adata.obs else np.array(["unknown"] * adata.n_obs),
        "2.2 Tissue",
    )
    scatter_categorical(
        axes[4],
        xy,
        adata.obs[cfg.DATASET_KEY].astype(str).values
        if cfg.DATASET_KEY in adata.obs
        else np.array(["unknown"] * adata.n_obs),
        f"3.1 {cfg.DATASET_KEY}",
        show_legend=False,
    )
    scatter_categorical(
        axes[5],
        xy,
        adata.obs[cfg.ASSAY_CLEAN_KEY].astype(str).values
        if cfg.ASSAY_CLEAN_KEY in adata.obs
        else np.array(["unknown"] * adata.n_obs),
        f"3.2 {cfg.ASSAY_CLEAN_KEY}",
    )

    plt.tight_layout()
    png = os.path.join(outdir, f"{GROUPBY}_interpretation_umap.png")
    pdf = os.path.join(outdir, f"{GROUPBY}_interpretation_umap.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")
    plt.close(fig)


def save_mapping_table(adata, outdir):
    cols = [
        GROUPBY,
        "broad_compartment_draft",
        "refined_NK_state_draft",
        "review_priority",
    ]
    table = (
        adata.obs[cols]
        .drop_duplicates()
        .sort_values(GROUPBY, key=lambda s: s.astype(int))
        .reset_index(drop=True)
    )
    path = os.path.join(outdir, f"{GROUPBY}_interpretation_cluster_mapping.csv")
    table.to_csv(path, index=False)
    print(f"[SAVE] {path}")


def scatter_categorical(
    ax,
    xy,
    values,
    title,
    *,
    fixed_colors=None,
    show_legend=True,
    annotate_clusters=False,
):
    values = np.asarray(values).astype(str)
    categories = sorted(set(values))
    colors = category_colors(categories, fixed_colors=fixed_colors)

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
        fontsize=8,
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


def category_colors(categories, *, fixed_colors=None):
    if fixed_colors is None:
        fixed_colors = {}

    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3", "Paired"):
        cmap = plt.get_cmap(cmap_name)
        palette.extend([cmap(i) for i in range(cmap.N)])

    colors = {}
    palette_i = 0
    for category in categories:
        if category in fixed_colors:
            colors[category] = fixed_colors[category]
        else:
            colors[category] = palette[palette_i % len(palette)]
            palette_i += 1
    return colors


def build_draft_colors(draft_categories, manual_colors):
    draft_colors = {}
    for category in draft_categories:
        if category in manual_colors:
            draft_colors[category] = manual_colors[category]
            continue
        if category in EXTRA_DRAFT_COLORS:
            draft_colors[category] = EXTRA_DRAFT_COLORS[category]
            continue

        parent = DRAFT_PARENT_LABELS.get(category)
        if parent in manual_colors:
            base_color = manual_colors[parent]
            mode, amount = DRAFT_COLOR_VARIANTS.get(category, ("lighten", 0.15))
            draft_colors[category] = adjust_color(base_color, mode=mode, amount=amount)

    return draft_colors


def adjust_color(color, *, mode, amount):
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    if mode == "darken":
        rgb = rgb * float(amount)
    elif mode == "lighten":
        rgb = rgb + (1.0 - rgb) * float(amount)
    else:
        raise ValueError(f"Unknown color adjustment mode: {mode}")
    return tuple(np.clip(rgb, 0.0, 1.0))


if __name__ == "__main__":
    main()
