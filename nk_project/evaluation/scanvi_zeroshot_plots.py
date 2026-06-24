#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs
from nk_project.plot_style import (
    LEGEND_FONT_SIZE,
    SMALL_TICK_LABEL_SIZE,
    set_presentation_style,
    style_all_legends,
    style_figure,
    style_legend,
)

set_presentation_style()


OUTDIR_NAME = "refined_scanvi_v1"
SPLIT_VALUE = "Held-out"
POINT_SIZE = 0.20
POINT_ALPHA = 0.55
CONTINUOUS_POINT_SIZE = 0.20
LEGEND_MARKER_SIZE = 10


PREFERRED_COLORS = {
    # Keep historical colors for existing labels so taxonomy-preferred runs
    # can be compared directly against earlier refined-v1 figures.
    "B": "#1f77b4",
    "T": "#d62728",
    "Mature Cytotoxic": "#ffbb78",
    "Mature Cytotoxic TCF7+": "#f7b6d2",
    "Transitional Cytotoxic": "#ff9896",
    "Transitional Cytotoxic Tissue-Resident": "#e377c2",
    "Cytokine-Stimulated CCR7+": "#aec7e8",
    "Cytokine-Stimulated Cycling": "#17becf",
    "Cytokine-Stimulated Proliferative": "#17becf",
    "Proliferative": "#2ca02c",
    "Regulatory": "#98df8a",
    "Lung Cytotoxic NK": "#bcbd22",
    "Lung DOCK4+ SLC8A1+ NK": "#8c564b",
    "Unknown_Kidney": "#c49c94",
    "Unknown_BM_1 Erythroid-like": "#c5b0d5",
    "Myeloid-like": "#7f7f7f",
    # New taxonomy-preferred labels get distinct colors from the old labels.
    "Chemokine-Inflammatory T": "#54278f",
    "NK1-like Mature Cytotoxic": "#7570b3",
    "NK1-like Lung Cytotoxic": "#1b9e77",
    "NK2-like Transitional Cytotoxic": "#e7298a",
    "Unknown Lung Stromal-like": "#d95f02",
    # Current subtype/state annotation colors. Keep these synchronized with
    # scripts/04_apply_refined_v1_labels.py and scanvi_full_plots.py.
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
}

PREFERRED_TISSUE_COLORS = {
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
}


def main():
    base = os.path.join(cfg.BASE_OUTDIR, OUTDIR_NAME)
    figdir = os.path.join(base, "figures")
    tabledir = os.path.join(base, "tables")
    latentdir = os.path.join(base, "latents")
    ensure_dirs(figdir, tabledir)

    latent_path = os.path.join(latentdir, "scanvi_latents.npz")
    obs_path = os.path.join(tabledir, "scanvi_full_obs_metadata.csv")
    pred_path = os.path.join(tabledir, "scanvi_full_prediction_summary.csv")

    print(f"[LOAD] {latent_path}")
    latent = np.load(latent_path, allow_pickle=True)
    z = latent["X_SCANVI"].astype(np.float32)
    obs_names = latent["obs_names"].astype(str)

    print(f"[LOAD] {obs_path}")
    obs = read_aligned_csv(obs_path, obs_names)
    print(f"[LOAD] {pred_path}")
    pred = read_aligned_csv(pred_path, obs_names)

    if "_split" not in obs.columns:
        raise KeyError("'_split' column not found in SCANVI obs metadata.")
    if cfg.REFINED_LABEL_KEY not in obs.columns:
        raise KeyError(f"{cfg.REFINED_LABEL_KEY!r} column not found in SCANVI obs metadata.")

    split_mask = obs["_split"].astype(str).values == SPLIT_VALUE
    if not split_mask.any():
        raise ValueError(f"No rows found with _split == {SPLIT_VALUE!r}.")

    z = z[split_mask]
    obs = obs.loc[split_mask].copy()
    pred = pred.loc[split_mask].copy()
    obs_names = obs.index.astype(str).values
    print(f"[ZERO-SHOT] {len(obs_names):,} held-out cells")

    true = obs[cfg.REFINED_LABEL_KEY].astype(str).values
    pred_label = pred["pred_label"].astype(str).values
    confidence = pred["confidence"].astype(float).values
    certainty = pred["certainty"].astype(float).values
    correct = true == pred_label

    print("[UMAP] Building held-out-only UMAP from SCANVI latent space...")
    ad_umap = sc.AnnData(X=np.zeros((z.shape[0], 1), dtype=np.float32))
    ad_umap.obsm["X_SCANVI"] = z
    sc.pp.neighbors(
        ad_umap,
        use_rep="X_SCANVI",
        n_neighbors=cfg.UMAP_N_NEIGHBORS,
        random_state=cfg.UMAP_SEED,
    )
    sc.tl.umap(ad_umap, min_dist=cfg.UMAP_MIN_DIST, random_state=cfg.UMAP_SEED)
    xy = ad_umap.obsm["X_umap"]

    np.save(os.path.join(latentdir, "scanvi_zeroshot_umap.npy"), xy)
    pd.DataFrame(xy, index=obs_names, columns=["UMAP1", "UMAP2"]).to_csv(
        os.path.join(tabledir, "scanvi_zeroshot_umap.csv")
    )

    make_panel_plot(obs, xy, true, pred_label, confidence, certainty, correct, figdir, tabledir)
    print("[DONE] Zero-shot plotting complete.")


def read_aligned_csv(path, obs_names):
    df = pd.read_csv(path, index_col=0, low_memory=False)
    if set(obs_names).issubset(set(df.index.astype(str))):
        df.index = df.index.astype(str)
        return df.loc[obs_names].copy()
    if len(df) == len(obs_names):
        print(f"[WARN] {os.path.basename(path)} index mismatch; aligning by row order.")
        df.index = obs_names
        return df.copy()
    raise ValueError(
        f"Cannot align {path}: rows={len(df):,}, expected={len(obs_names):,}."
    )


def make_panel_plot(obs, xy, true, pred_label, confidence, certainty, correct, figdir, tabledir):
    tissue = obs["tissue"].astype(str).values if "tissue" in obs else np.array(["NA"] * len(obs))
    dataset = obs[cfg.DATASET_KEY].astype(str).values if cfg.DATASET_KEY in obs else np.array(["NA"] * len(obs))
    assay = obs[cfg.ASSAY_CLEAN_KEY].astype(str).values if cfg.ASSAY_CLEAN_KEY in obs else np.array(["NA"] * len(obs))
    state_colors = distinct_color_map(np.r_[true, pred_label], preferred=PREFERRED_COLORS)

    fig, axes = plt.subplots(3, 3, figsize=(30, 21))
    fig.subplots_adjust(left=0.04, right=0.76, top=0.94, bottom=0.07, wspace=0.4, hspace=0.38)
    err_rate = 1.0 - float(np.mean(correct))
    fig.suptitle(f"Refined SCANVI zero-shot held-out UMAP (error={err_rate:.1%})", fontsize=20, fontweight="bold")

    scatter_by_category(axes[0, 0], xy, true, state_colors, legend=True, title=f"1.1 TRUE {cfg.REFINED_LABEL_KEY}")
    scatter_by_category(axes[0, 1], xy, pred_label, state_colors, legend=False, title=f"1.2 PRED {cfg.REFINED_LABEL_KEY}")

    ax = axes[0, 2]
    ax.scatter(xy[correct, 0], xy[correct, 1], s=POINT_SIZE, alpha=POINT_ALPHA, color="#2166ac", rasterized=True)
    ax.scatter(xy[~correct, 0], xy[~correct, 1], s=POINT_SIZE, alpha=POINT_ALPHA, color="#d62728", rasterized=True)
    clean_ax(ax)
    ax.set_title(f"1.3 Correct vs Incorrect\nerror={err_rate:.1%}")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markersize=LEGEND_MARKER_SIZE, markerfacecolor="#2166ac", markeredgecolor="none", label="Correct"),
            Line2D([0], [0], marker="o", linestyle="", markersize=LEGEND_MARKER_SIZE, markerfacecolor="#d62728", markeredgecolor="none", label="Incorrect"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
    )

    scatter_continuous(axes[1, 0], xy, confidence, fig, "2.1 Confidence")
    scatter_continuous(axes[1, 1], xy, certainty, fig, "2.2 Certainty")
    plot_per_class_metrics(axes[1, 2], true, pred_label, state_colors, tabledir)

    scatter_by_category(
        axes[2, 0],
        xy,
        tissue,
        distinct_color_map(tissue, preferred=PREFERRED_TISSUE_COLORS),
        legend=True,
        title="3.1 Tissue",
    )
    scatter_by_category(axes[2, 1], xy, dataset, distinct_color_map(dataset), legend=False, title="3.2 Dataset ID")
    scatter_by_category(axes[2, 2], xy, assay, distinct_color_map(assay), legend=True, title="3.3 Assay clean")

    png = os.path.join(figdir, "scanvi_zeroshot_umap_panels.png")
    style_figure(fig, tick_size=SMALL_TICK_LABEL_SIZE, legend_size=LEGEND_FONT_SIZE)
    style_all_legends(fig)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    plt.close(fig)


def plot_per_class_metrics(ax, true, pred_label, color_map, tabledir):
    classes = sorted(set(true))
    rep = classification_report(true, pred_label, labels=classes, output_dict=True, zero_division=0)
    metrics = pd.DataFrame(
        {
            "accuracy": pd.DataFrame({"true": true, "correct": true == pred_label}).groupby("true")["correct"].mean(),
            "f1": pd.Series({cls: rep[cls]["f1-score"] for cls in classes}),
            "n_true": pd.Series({cls: rep[cls]["support"] for cls in classes}),
        }
    ).sort_values("f1", ascending=False)

    x = np.arange(len(metrics))
    width = 0.38
    colors = [color_map.get(c, (0.5, 0.5, 0.5)) for c in metrics.index]
    ax.bar(x - width / 2, metrics["accuracy"].values, width=width, color=colors, alpha=0.9, label="Accuracy")
    ax.bar(x + width / 2, metrics["f1"].values, width=width, color=colors, alpha=0.45, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics.index, rotation=60, ha="right", fontsize=SMALL_TICK_LABEL_SIZE)
    ax.set_ylim(0, 1.05)
    ax.set_title("2.3 Zero-shot per-class Accuracy & F1")
    ax.legend(frameon=False, loc="upper right", fontsize=LEGEND_FONT_SIZE)
    style_legend(ax.get_legend())
    out = os.path.join(tabledir, "scanvi_zeroshot_per_class_accuracy_f1.csv")
    metrics.to_csv(out)
    print(f"[SAVE] {out}")


def scatter_continuous(ax, xy, values, fig, title):
    sc_plot = ax.scatter(
        xy[:, 0],
        xy[:, 1],
        c=values,
        cmap="RdBu",
        vmin=0,
        vmax=1,
        s=CONTINUOUS_POINT_SIZE,
        alpha=1.0,
        rasterized=True,
    )
    clean_ax(ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    fig.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.02)


def scatter_by_category(ax, xy, values, color_map, *, legend=False, title=""):
    values = np.asarray(values).astype(str)
    for value in sorted(set(values)):
        mask = values == value
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=color_map.get(value, (0.5, 0.5, 0.5)),
            label=value,
            rasterized=True,
        )
    clean_ax(ax)
    ax.set_title(title, fontsize=14, fontweight="bold")
    if legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=LEGEND_MARKER_SIZE,
                markerfacecolor=color_map.get(value, (0.5, 0.5, 0.5)),
                markeredgecolor="none",
                alpha=1.0,
                label=value,
            )
            for value in sorted(set(values))
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.02, 1.0),
            fontsize=LEGEND_FONT_SIZE,
            handletextpad=0.4,
        )
        style_legend(ax.get_legend())


def clean_ax(ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def distinct_color_map(values, preferred=None):
    preferred = preferred or {}
    values = sorted(set(map(str, values)))
    colors = {}
    used = set()
    for value, color in preferred.items():
        if value in values:
            colors[value] = color
            used.add(color)

    palette = []
    for name in ("tab20", "tab20b", "tab20c", "Set3", "Paired"):
        cmap = plt.get_cmap(name)
        palette.extend([cmap(i) for i in range(cmap.N)])

    i = 0
    for value in values:
        if value in colors:
            continue
        while i < len(palette) and palette[i] in used:
            i += 1
        colors[value] = palette[i % len(palette)]
        i += 1
    return colors


if __name__ == "__main__":
    main()
