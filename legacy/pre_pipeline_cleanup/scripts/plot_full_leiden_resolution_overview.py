#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import scanpy as sc

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


POINT_SIZE = 0.035
POINT_ALPHA = 0.35


def main():
    args = parse_args()
    in_path = os.path.join(cfg.BASE_OUTDIR, "leiden_discovery", "full_scvi_leiden.h5ad")
    outdir = os.path.join(cfg.BASE_OUTDIR, "leiden_discovery", "figures")
    ensure_dirs(outdir)

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if "X_umap" not in adata.obsm:
        raise KeyError("X_umap not found in full Leiden AnnData.")

    plot_resolution_overview(adata, args.resolutions, outdir)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot full-data Leiden resolution comparison from full_scvi_leiden.h5ad."
    )
    parser.add_argument("--resolutions", type=float, nargs="+", default=[0.2, 0.3, 0.4])
    return parser.parse_args()


def plot_resolution_overview(adata, resolutions, outdir):
    nrows = len(resolutions)
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 6.0 * nrows))
    if nrows == 1:
        axes = np.asarray([axes])

    xy = adata.obsm["X_umap"]

    for row_idx, resolution in enumerate(resolutions):
        key = f"leiden_{str(resolution).replace('.', '_')}"
        if key not in adata.obs:
            raise KeyError(f"{key!r} not found. Rerun run_leiden_discovery.py with --resolutions {resolution}.")

        for col_idx, ax in enumerate(axes[row_idx]):
            if col_idx == 0:
                values = adata.obs[key].astype(str).values
                title = f"{key} ({len(set(values))} clusters)"
                scatter_categorical(
                    ax,
                    xy,
                    values,
                    title,
                    show_legend=False,
                    annotate_clusters=True,
                )
            elif col_idx == 1:
                values = adata.obs[cfg.LABEL_KEY].astype(str).values
                scatter_categorical(
                    ax,
                    xy,
                    values,
                    f"{key} background: {cfg.LABEL_KEY}",
                    show_legend=True,
                )
            else:
                values = adata.obs[cfg.ASSAY_CLEAN_KEY].astype(str).values
                scatter_categorical(
                    ax,
                    xy,
                    values,
                    f"{key} background: {cfg.ASSAY_CLEAN_KEY}",
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


if __name__ == "__main__":
    main()
