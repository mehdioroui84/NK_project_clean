#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


GROUPBY = "leiden_0_4"
POINT_SIZE = 0.06
POINT_ALPHA = 0.38


def main():
    in_path = os.path.join(cfg.BASE_OUTDIR, "leiden_discovery", "full_scvi_leiden.h5ad")
    outdir = os.path.join(cfg.BASE_OUTDIR, "leiden_discovery", "figures")
    ensure_dirs(outdir)

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if GROUPBY not in adata.obs:
        raise KeyError(f"{GROUPBY!r} not found in adata.obs. Rerun run_leiden_discovery.py --resolutions 0.4")
    if "X_umap" not in adata.obsm:
        raise KeyError("X_umap not found in full Leiden AnnData.")

    plot_overview(adata, outdir)


def plot_overview(adata, outdir):
    xy = adata.obsm["X_umap"]

    panels = [
        (GROUPBY, f"1.1 full {GROUPBY} cluster IDs", False, True),
        (cfg.LABEL_KEY, f"1.2 original annotation: {cfg.LABEL_KEY}", True, False),
        ("tissue", "2.1 tissue", True, False),
        (cfg.DATASET_KEY, f"2.2 {cfg.DATASET_KEY}", False, False),
        (cfg.ASSAY_CLEAN_KEY, f"3.1 {cfg.ASSAY_CLEAN_KEY}", True, False),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(20, 22))
    fig.suptitle(f"Full-data SCVI latent space: {GROUPBY}", fontsize=14)
    axes = axes.ravel()

    for ax, (obs_key, title, show_legend, annotate) in zip(axes, panels):
        if obs_key not in adata.obs:
            values = np.array(["unknown"] * adata.n_obs)
        else:
            values = adata.obs[obs_key].astype(str).values
        scatter_categorical(
            ax,
            xy,
            values,
            title,
            show_legend=show_legend,
            annotate_clusters=annotate,
        )

    axes[-1].axis("off")
    axes[-1].text(
        0.02,
        0.95,
        "Next step:\nmap full leiden_0_4 clusters\n"
        "to NK_State_refined labels using\n"
        "cluster composition + markers + DE.",
        ha="left",
        va="top",
        fontsize=16,
        color="#17202a",
    )

    plt.tight_layout()
    png = os.path.join(outdir, f"full_{GROUPBY}_overview.png")
    pdf = os.path.join(outdir, f"full_{GROUPBY}_overview.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")
    plt.close(fig)


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


if __name__ == "__main__":
    main()
