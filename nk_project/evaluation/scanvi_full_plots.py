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


PREFERRED_STATE_COLORS = {
    # Keep historical colors for existing labels so new runs remain comparable
    # with earlier SCANVI/refined-annotation figures.
    "B": "#1f77b4",
    "T": "#d62728",
    "Developmental": "#9ecae1",
    "Mature Cytotoxic": "#ffbb78",
    "Mature Cytotoxic TCF7+": "#f7b6d2",
    "Transitional Cytotoxic": "#ff9896",
    "Transitional Cytotoxic Tissue-Resident": "#e377c2",
    "Cytokine-Stimulated": "#d62728",
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
    # scripts/04_apply_refined_v1_labels.py so annotation QC and SCANVI plots
    # tell the same visual story.
    "L6_Developmental_immature_Proliferating": "#E7298A",
    "L6_Developmental_immature_Metabolic_stress_hypoxia": "#7F3C8D",
    "L6_Developmental_immature": "#E7298A",
    "NK1_Chemokine_inflammatory": "#0072B2",
    "NK1_Cytotoxic_activated": "#D55E00",
    "NK1_Checkpoint_exhausted": "#7570B3",
    "NK1_Metabolic_stress_hypoxia": "#11A579",
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

UMAP_POINT_SIZE = 0.08
UMAP_POINT_ALPHA = 0.40
CONTINUOUS_POINT_SIZE = 0.08
CONTINUOUS_POINT_ALPHA = 1.00
LEGEND_MARKER_SIZE = 10


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


def scatter_by_category(ax, xy, values, color_map, *, size=UMAP_POINT_SIZE, alpha=UMAP_POINT_ALPHA, legend=False, title=""):
    values = np.asarray(values).astype(str)
    for value in sorted(set(values)):
        mask = values == value
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=size,
            alpha=alpha,
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


def safe_name(value: str) -> str:
    return (
        str(value)
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("'", "")
        .replace("+", "pos")
        .replace("-", "_")
        .lower()
    )


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    ensure_dirs(cfg.FIG_OUTDIR)

    latent_path = os.path.join(cfg.LATENT_OUTDIR, "scanvi_latents.npz")
    obs_path = os.path.join(cfg.TABLE_OUTDIR, "scanvi_full_obs_metadata.csv")
    pred_path = os.path.join(cfg.TABLE_OUTDIR, "scanvi_full_prediction_summary.csv")

    print(f"[LOAD] {latent_path}")
    latent = np.load(latent_path, allow_pickle=True)
    z = latent["X_SCANVI"]
    obs_names = latent["obs_names"].astype(str)

    print(f"[LOAD] {obs_path}")
    obs = pd.read_csv(obs_path, index_col=0)
    if set(obs_names).issubset(set(obs.index.astype(str))):
        obs.index = obs.index.astype(str)
        obs = obs.loc[obs_names].copy()
    elif len(obs) == len(obs_names):
        print("[WARN] obs metadata index does not match latent obs_names; aligning by row order.")
        obs.index = obs_names
    else:
        raise ValueError(
            "Cannot align obs metadata with latent obs_names: "
            f"obs rows={len(obs):,}, latent rows={len(obs_names):,}"
        )

    print(f"[LOAD] {pred_path}")
    pred = pd.read_csv(pred_path, index_col=0)
    if set(obs_names).issubset(set(pred.index.astype(str))):
        pred.index = pred.index.astype(str)
        pred = pred.loc[obs_names].copy()
    elif len(pred) == len(obs_names):
        print("[WARN] prediction index does not match latent obs_names; aligning by row order.")
        pred.index = obs_names
    else:
        raise ValueError(
            "Cannot align predictions with latent obs_names: "
            f"prediction rows={len(pred):,}, latent rows={len(obs_names):,}. "
            "This usually means predictions were saved for a filtered subset."
        )

    true = obs[cfg.LABEL_KEY].astype(str).values
    pred_label = pred["pred_label"].astype(str).values
    confidence = pred["confidence"].astype(float).values
    certainty = pred["certainty"].astype(float).values
    correct = true == pred_label
    if args.split != "all":
        if "_split" not in obs.columns:
            raise KeyError("Requested --split, but `_split` is missing from scanvi_full_obs_metadata.csv")
        eval_mask = obs["_split"].astype(str).values == args.split
        if not np.any(eval_mask):
            raise ValueError(f"No cells found for --split {args.split!r}.")
    else:
        eval_mask = np.ones(len(obs), dtype=bool)
    panel_label = "full-dataset" if args.split == "all" else f"{args.split} split"
    file_suffix = "full" if args.split == "all" else safe_name(args.split)

    umap_npy = os.path.join(cfg.LATENT_OUTDIR, "scanvi_full_umap.npy")
    umap_csv = os.path.join(cfg.TABLE_OUTDIR, "scanvi_full_umap.csv")
    xy = load_cached_umap(umap_npy, umap_csv, latent_path, obs_names)
    if xy is None:
        print("[UMAP] Building UMAP from SCANVI latent space...")
        ad_umap = sc.AnnData(X=np.zeros((z.shape[0], 1), dtype=np.float32))
        ad_umap.obsm["X_SCANVI"] = z.astype(np.float32)
        sc.pp.neighbors(ad_umap, use_rep="X_SCANVI", n_neighbors=cfg.UMAP_N_NEIGHBORS, random_state=cfg.UMAP_SEED)
        sc.tl.umap(ad_umap, min_dist=cfg.UMAP_MIN_DIST, random_state=cfg.UMAP_SEED)
        xy = ad_umap.obsm["X_umap"]

        np.save(umap_npy, xy)
        pd.DataFrame(xy, index=obs_names, columns=["UMAP1", "UMAP2"]).to_csv(umap_csv)
    else:
        print(f"[UMAP] Using cached UMAP: {umap_csv}")

    eval_idx = np.flatnonzero(eval_mask)
    rng = np.random.default_rng(cfg.SEED)
    if cfg.PLOT_MAX_POINTS and len(eval_idx) > cfg.PLOT_MAX_POINTS:
        plot_idx = np.sort(rng.choice(eval_idx, size=cfg.PLOT_MAX_POINTS, replace=False))
    else:
        plot_idx = eval_idx

    xy_p = xy[plot_idx]
    true_p = true[plot_idx]
    pred_p = pred_label[plot_idx]
    correct_p = correct[plot_idx]
    confidence_p = confidence[plot_idx]
    certainty_p = certainty[plot_idx]

    class_colors = distinct_color_map(
        np.concatenate([true.astype(str), pred_label.astype(str)]),
        preferred=PREFERRED_STATE_COLORS,
    )
    tissue = obs["tissue"].astype(str).values if "tissue" in obs else np.array(["NA"] * len(obs))
    dataset = obs[cfg.DATASET_KEY].astype(str).values if cfg.DATASET_KEY in obs else np.array(["NA"] * len(obs))
    assay = obs[cfg.ASSAY_CLEAN_KEY].astype(str).values if cfg.ASSAY_CLEAN_KEY in obs else np.array(["NA"] * len(obs))

    tissue_p = tissue[plot_idx]
    dataset_p = dataset[plot_idx]
    assay_p = assay[plot_idx]

    fig, axes = plt.subplots(3, 3, figsize=(30, 21))
    fig.subplots_adjust(left=0.04, right=0.76, top=0.94, bottom=0.07, wspace=0.4, hspace=0.38)
    fig.suptitle(f"SCANVI assay_clean model: {panel_label} UMAP", fontsize=20, fontweight="bold")

    scatter_by_category(axes[0, 0], xy_p, true_p, class_colors, legend=True, title="1.1 TRUE NK_State")
    scatter_by_category(axes[0, 1], xy_p, pred_p, class_colors, legend=False, title="1.2 PRED NK_State")

    ax = axes[0, 2]
    correctness_size = UMAP_POINT_SIZE
    correctness_alpha = UMAP_POINT_ALPHA
    ax.scatter(
        xy_p[correct_p, 0],
        xy_p[correct_p, 1],
        s=correctness_size,
        alpha=correctness_alpha,
        color="#2166ac",
        label="Correct",
        rasterized=True,
    )
    ax.scatter(
        xy_p[~correct_p, 0],
        xy_p[~correct_p, 1],
        s=correctness_size,
        alpha=correctness_alpha,
        color="#d62728",
        label="Incorrect",
        rasterized=True,
    )
    clean_ax(ax)
    err_rate = 1.0 - float(np.mean(correct[eval_mask]))
    ax.set_title(f"1.3 Correct vs Incorrect\nsame alpha/size; error={err_rate:.1%}")
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=LEGEND_MARKER_SIZE,
                markerfacecolor="#2166ac",
                markeredgecolor="none",
                label="Correct",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=LEGEND_MARKER_SIZE,
                markerfacecolor="#d62728",
                markeredgecolor="none",
                label="Incorrect",
            ),
        ],
        frameon=False,
        loc="upper left",
        fontsize=LEGEND_FONT_SIZE,
    )

    ax = axes[1, 0]
    sc1 = ax.scatter(
        xy_p[:, 0],
        xy_p[:, 1],
        c=confidence_p,
        cmap="RdBu",
        vmin=0,
        vmax=1,
        s=CONTINUOUS_POINT_SIZE,
        alpha=CONTINUOUS_POINT_ALPHA,
        rasterized=True,
    )
    clean_ax(ax)
    ax.set_title("2.1 Confidence")
    fig.colorbar(sc1, ax=ax, fraction=0.046, pad=0.02)

    ax = axes[1, 1]
    sc2 = ax.scatter(
        xy_p[:, 0],
        xy_p[:, 1],
        c=certainty_p,
        cmap="RdBu",
        vmin=0,
        vmax=1,
        s=CONTINUOUS_POINT_SIZE,
        alpha=CONTINUOUS_POINT_ALPHA,
        rasterized=True,
    )
    clean_ax(ax)
    ax.set_title("2.2 Certainty")
    fig.colorbar(sc2, ax=ax, fraction=0.046, pad=0.02)

    ax = axes[1, 2]
    true_eval = true[eval_mask]
    pred_eval = pred_label[eval_mask]
    correct_eval = correct[eval_mask]
    classes = sorted(set(true_eval))
    rep = classification_report(true_eval, pred_eval, labels=classes, output_dict=True, zero_division=0)
    class_metrics = pd.DataFrame(
        {
            "accuracy": pd.DataFrame({"true": true_eval, "correct": correct_eval}).groupby("true")["correct"].mean(),
            "f1": pd.Series({cls: rep[cls]["f1-score"] for cls in classes}),
            "n_true": pd.Series({cls: rep[cls]["support"] for cls in classes}),
        }
    ).sort_values("f1", ascending=False)
    x = np.arange(len(class_metrics))
    width = 0.38
    bar_colors = [class_colors.get(c, (0.5, 0.5, 0.5)) for c in class_metrics.index]
    ax.bar(x - width / 2, class_metrics["accuracy"].values, width=width, color=bar_colors, alpha=0.9, label="Accuracy")
    ax.bar(x + width / 2, class_metrics["f1"].values, width=width, color=bar_colors, alpha=0.45, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_metrics.index, rotation=60, ha="right", fontsize=SMALL_TICK_LABEL_SIZE)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"2.3 Per-class {panel_label} Accuracy & F1")
    ax.legend(frameon=False, loc="upper right", fontsize=LEGEND_FONT_SIZE)
    class_metrics.to_csv(os.path.join(cfg.TABLE_OUTDIR, f"scanvi_{file_suffix}_per_class_accuracy_f1.csv"))

    scatter_by_category(
        axes[2, 0],
        xy_p,
        tissue_p,
        distinct_color_map(tissue_p, preferred=PREFERRED_TISSUE_COLORS),
        legend=True,
        title="3.1 Tissue",
    )
    scatter_by_category(
        axes[2, 1],
        xy_p,
        dataset_p,
        distinct_color_map(dataset_p),
        legend=False,
        title="3.2 Dataset ID",
    )
    scatter_by_category(
        axes[2, 2],
        xy_p,
        assay_p,
        distinct_color_map(assay_p),
        legend=True,
        title="3.3 Assay clean",
    )

    png = os.path.join(cfg.FIG_OUTDIR, f"scanvi_{file_suffix}_umap_panels.png")
    style_figure(fig, tick_size=SMALL_TICK_LABEL_SIZE, legend_size=LEGEND_FONT_SIZE)
    style_all_legends(fig)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    plt.close(fig)


def load_cached_umap(umap_npy: str, umap_csv: str, latent_path: str, obs_names: np.ndarray):
    if not (os.path.exists(umap_npy) and os.path.exists(umap_csv)):
        return None
    if os.path.getmtime(umap_npy) < os.path.getmtime(latent_path):
        return None
    try:
        cached = np.load(umap_npy)
        cached_index = pd.read_csv(umap_csv, index_col=0).index.astype(str).values
    except Exception as exc:
        print(f"[WARN] Could not read cached UMAP; rebuilding. Reason: {exc}")
        return None
    if cached.shape != (len(obs_names), 2):
        return None
    if not np.array_equal(cached_index, obs_names.astype(str)):
        return None
    return cached


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot SCANVI prediction panels for all cells or one saved split."
    )
    parser.add_argument(
        "--split",
        choices=["all", "Train", "Val", "Held-out"],
        default="all",
        help="Cells to show/evaluate. Uses full-data UMAP coordinates, filtered to this split.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
