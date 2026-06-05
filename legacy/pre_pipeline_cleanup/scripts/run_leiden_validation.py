#!/usr/bin/env python
from __future__ import annotations

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


def main():
    outdir = os.path.join(cfg.BASE_OUTDIR, "leiden_validation")
    ensure_dirs(outdir)

    adata_path = os.path.join(cfg.LATENT_OUTDIR, "scvi_full_with_latent.h5ad")
    val_ids_path = os.path.join(cfg.TABLE_OUTDIR, "val_obs_names.txt")

    print(f"[LOAD] {adata_path}")
    adata = sc.read_h5ad(adata_path)

    print(f"[LOAD] {val_ids_path}")
    val_ids = pd.read_csv(val_ids_path, header=None)[0].astype(str).tolist()
    val_ids = [x for x in val_ids if x in adata.obs_names]
    if not val_ids:
        raise ValueError("No validation obs_names were found in scvi_full_with_latent.h5ad")

    adata_val = adata[val_ids].copy()
    print(f"[VALIDATION] {adata_val.n_obs:,} cells x {adata_val.n_vars:,} genes")

    adata_val, summary = run_leiden_grid(
        adata_val,
        latent_key="X_scVI",
        resolutions=cfg.LEIDEN_RESOLUTIONS,
        n_neighbors=cfg.DISCOVERY_N_NEIGHBORS,
        seed=cfg.SEED,
        outdir=outdir,
        label_key=cfg.LABEL_KEY,
        dataset_key=cfg.DATASET_KEY,
        assay_key=cfg.ASSAY_CLEAN_KEY,
    )

    print("\n[LEIDEN SUMMARY]")
    print(summary.to_string(index=False))

    adata_val.write(os.path.join(outdir, "validation_scvi_leiden.h5ad"))
    plot_validation_overview(adata_val, outdir)
    print(f"[SAVE] {outdir}")


def plot_validation_overview(adata, outdir):
    resolutions = cfg.LEIDEN_RESOLUTIONS
    chosen = [0.3, 0.4, 0.6, 0.8]
    chosen = [r for r in chosen if r in resolutions]
    ncols = 3
    nrows = len(chosen)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4.6 * nrows))
    if nrows == 1:
        axes = np.asarray([axes])

    xy = adata.obsm["X_umap"]
    point_size = 0.06
    point_alpha = 0.40

    for row, resolution in enumerate(chosen):
        leiden_key = f"leiden_{str(resolution).replace('.', '_')}"
        for col_idx, ax in enumerate(axes[row]):
            if col_idx == 0:
                values = adata.obs[leiden_key].astype(str).values
                n_clusters = len(set(values))
                title = f"{leiden_key} ({n_clusters} clusters)"
                scatter_categorical(
                    ax,
                    xy,
                    values,
                    title,
                    point_size,
                    point_alpha,
                    show_legend=False,
                    annotate_clusters=True,
                )
                continue
            elif col_idx == 1:
                values = adata.obs[cfg.LABEL_KEY].astype(str).values
                title = f"{leiden_key} background: NK_State"
            else:
                values = adata.obs[cfg.ASSAY_CLEAN_KEY].astype(str).values
                title = f"{leiden_key} background: assay_clean"
            scatter_categorical(ax, xy, values, title, point_size, point_alpha)

    plt.tight_layout()
    png = os.path.join(outdir, "validation_leiden_overview.png")
    pdf = os.path.join(outdir, "validation_leiden_overview.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")


def scatter_categorical(
    ax,
    xy,
    values,
    title,
    point_size,
    point_alpha,
    *,
    show_legend=True,
    annotate_clusters=False,
):
    values = np.asarray(values).astype(str)
    colors = category_colors(values)
    categories = sorted(set(values))
    for value in categories:
        mask = values == value
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=point_size,
            alpha=point_alpha,
            color=colors[value],
            rasterized=True,
            label=value,
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

    if show_legend and len(categories) <= 30:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=7,
                markerfacecolor=colors[value],
                markeredgecolor="none",
                alpha=1.0,
                label=value,
            )
            for value in categories
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
    for value in sorted(set(values), key=lambda x: int(x) if x.isdigit() else x):
        mask = values == value
        if mask.sum() == 0:
            continue
        center = np.median(xy[mask], axis=0)
        ax.text(
            center[0],
            center[1],
            value,
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


def category_colors(values):
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3", "Paired"):
        cmap = plt.get_cmap(cmap_name)
        palette.extend([cmap(i) for i in range(cmap.N)])
    return {value: palette[i % len(palette)] for i, value in enumerate(sorted(set(values)))}


if __name__ == "__main__":
    main()
