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
POINT_SIZE = 0.035
POINT_ALPHA = 0.35


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
    "12": "Cytokine-Stimulated Proliferative",
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
    outdir = os.path.join(cfg.BASE_OUTDIR, OUTDIR_NAME)
    figdir = os.path.join(outdir, "figures")
    ensure_dirs(outdir, figdir)

    print(f"[LOAD] {in_path}")
    adata = sc.read_h5ad(in_path)
    if GROUPBY not in adata.obs:
        raise KeyError(f"{GROUPBY!r} not found in adata.obs.")
    if "X_umap" not in adata.obsm:
        raise KeyError("X_umap not found in full-data SCVI Leiden AnnData.")

    label_mapping = load_label_mapping(args.mapping_csv)
    apply_labels(adata, label_mapping)
    write_outputs(adata, outdir, label_mapping)
    plot_refined_umap(adata, figdir)
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
    return parser.parse_args()


def load_label_mapping(mapping_csv=None):
    if mapping_csv is None:
        print("[MAPPING] Using curated hardcoded refined-v1 mapping.")
        return dict(REFINED_LABEL_BY_CLUSTER)

    print(f"[MAPPING] Loading reviewed mapping CSV: {mapping_csv}")
    mapping = pd.read_csv(mapping_csv, dtype=str)
    if GROUPBY not in mapping.columns:
        raise KeyError(f"{mapping_csv} must contain {GROUPBY!r}.")

    label_col = None
    for candidate in [cfg.REFINED_LABEL_KEY, "candidate_refined_label", "refined_label"]:
        if candidate in mapping.columns:
            label_col = candidate
            break
    if label_col is None:
        raise KeyError(
            f"{mapping_csv} must contain one of: {cfg.REFINED_LABEL_KEY!r}, "
            "'candidate_refined_label', or 'refined_label'."
        )

    out = dict(zip(mapping[GROUPBY].astype(str), mapping[label_col].astype(str)))
    missing = sorted(set(REFINED_LABEL_BY_CLUSTER) - set(out), key=cluster_sort_key)
    if missing:
        raise ValueError(f"Mapping CSV is missing {GROUPBY} clusters: {missing}")
    return out


def apply_labels(adata, label_mapping):
    clusters = adata.obs[GROUPBY].astype(str)
    labels = clusters.map(label_mapping)
    if labels.isna().any():
        missing = sorted(clusters[labels.isna()].unique(), key=cluster_sort_key)
        raise ValueError(f"Missing refined labels for {GROUPBY} clusters: {missing}")

    adata.obs["NK_State_original"] = adata.obs[cfg.LABEL_KEY].astype(str)
    adata.obs[cfg.REFINED_LABEL_KEY] = labels.astype("category")
    adata.obs["NK_State_refined_v1_source"] = "full_data_leiden_0_4_manual_mapping"

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
    pdf = os.path.join(figdir, "full_refined_v1_umap.pdf")
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


def cluster_sort_key(value):
    return int(value)


def category_colors(categories):
    preferred = {
        "B": "#1f77b4",
        "T": "#d62728",
        "Mature Cytotoxic": "#ffbb78",
        "Mature Cytotoxic TCF7+": "#f7b6d2",
        "Transitional Cytotoxic": "#ff9896",
        "Transitional Cytotoxic Tissue-Resident": "#e377c2",
        "Cytokine-Stimulated CCR7+": "#aec7e8",
        "Cytokine-Stimulated Proliferative": "#17becf",
        "Proliferative": "#2ca02c",
        "Regulatory": "#98df8a",
        "Unconventional": "#9467bd",
        "Lung Cytotoxic NK": "#bcbd22",
        "Lung DOCK4+ SLC8A1+ NK": "#8c564b",
        "Unknown_Kidney": "#c49c94",
        "Unknown_BM_1 Erythroid-like": "#c5b0d5",
        "Myeloid-like": "#7f7f7f",
    }
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3", "Paired"):
        cmap = plt.get_cmap(cmap_name)
        palette.extend([cmap(i) for i in range(cmap.N)])

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
