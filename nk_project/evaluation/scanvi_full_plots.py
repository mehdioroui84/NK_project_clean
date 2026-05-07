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


PREFERRED_STATE_COLORS = {
    "Developmental": "#9ecae1",
    "Proliferative": "#a1d99b",
    "Regulatory": "#c7b9ff",
    "Transitional Cytotoxic": "#f2d64b",
    "Mature Cytotoxic": "#f39c34",
    "Cytokine-Stimulated": "#d62728",
}

UMAP_POINT_SIZE = 0.08
UMAP_POINT_ALPHA = 0.40
CONTINUOUS_POINT_SIZE = 0.08
CONTINUOUS_POINT_ALPHA = 1.00
LEGEND_MARKER_SIZE = 7


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
    ax.set_title(title)
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
            fontsize=8,
            handletextpad=0.4,
        )


def clean_ax(ax):
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
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

    print("[UMAP] Building UMAP from SCANVI latent space...")
    ad_umap = sc.AnnData(X=np.zeros((z.shape[0], 1), dtype=np.float32))
    ad_umap.obsm["X_SCANVI"] = z.astype(np.float32)
    sc.pp.neighbors(ad_umap, use_rep="X_SCANVI", n_neighbors=cfg.UMAP_N_NEIGHBORS, random_state=cfg.UMAP_SEED)
    sc.tl.umap(ad_umap, min_dist=cfg.UMAP_MIN_DIST, random_state=cfg.UMAP_SEED)
    xy = ad_umap.obsm["X_umap"]

    np.save(os.path.join(cfg.LATENT_OUTDIR, "scanvi_full_umap.npy"), xy)
    pd.DataFrame(xy, index=obs_names, columns=["UMAP1", "UMAP2"]).to_csv(
        os.path.join(cfg.TABLE_OUTDIR, "scanvi_full_umap.csv")
    )

    rng = np.random.default_rng(cfg.SEED)
    if cfg.PLOT_MAX_POINTS and xy.shape[0] > cfg.PLOT_MAX_POINTS:
        plot_idx = np.sort(rng.choice(np.arange(xy.shape[0]), size=cfg.PLOT_MAX_POINTS, replace=False))
    else:
        plot_idx = np.arange(xy.shape[0])

    xy_p = xy[plot_idx]
    true_p = true[plot_idx]
    pred_p = pred_label[plot_idx]
    correct_p = correct[plot_idx]
    confidence_p = confidence[plot_idx]
    certainty_p = certainty[plot_idx]

    class_colors = distinct_color_map(true, preferred=PREFERRED_STATE_COLORS)
    tissue = obs["tissue"].astype(str).values if "tissue" in obs else np.array(["NA"] * len(obs))
    dataset = obs[cfg.DATASET_KEY].astype(str).values if cfg.DATASET_KEY in obs else np.array(["NA"] * len(obs))
    assay = obs[cfg.ASSAY_CLEAN_KEY].astype(str).values if cfg.ASSAY_CLEAN_KEY in obs else np.array(["NA"] * len(obs))

    tissue_p = tissue[plot_idx]
    dataset_p = dataset[plot_idx]
    assay_p = assay[plot_idx]

    fig, axes = plt.subplots(3, 3, figsize=(26, 18))
    fig.subplots_adjust(left=0.04, right=0.78, top=0.94, bottom=0.06, wspace=0.35, hspace=0.35)
    fig.suptitle("SCANVI assay_clean model: full-dataset UMAP", fontsize=14)

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
    err_rate = 1.0 - float(np.mean(correct))
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
        fontsize=8,
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
    classes = sorted(set(true))
    rep = classification_report(true, pred_label, labels=classes, output_dict=True, zero_division=0)
    class_metrics = pd.DataFrame(
        {
            "accuracy": pd.DataFrame({"true": true, "correct": correct}).groupby("true")["correct"].mean(),
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
    ax.set_xticklabels(class_metrics.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("2.3 Per-class full-dataset Accuracy & F1")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    class_metrics.to_csv(os.path.join(cfg.TABLE_OUTDIR, "scanvi_full_per_class_accuracy_f1.csv"))

    scatter_by_category(
        axes[2, 0],
        xy_p,
        tissue_p,
        distinct_color_map(tissue_p),
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

    png = os.path.join(cfg.FIG_OUTDIR, "scanvi_full_umap_panels.png")
    pdf = os.path.join(cfg.FIG_OUTDIR, "scanvi_full_umap_panels.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")

    fig_err, ax_err = plt.subplots(1, 1, figsize=(14, 12))
    ax_err.scatter(
        xy[correct, 0],
        xy[correct, 1],
        s=0.035,
        alpha=0.24,
        color="#2166ac",
        label="Correct",
        rasterized=True,
    )
    ax_err.scatter(
        xy[~correct, 0],
        xy[~correct, 1],
        s=0.035,
        alpha=0.24,
        color="#d62728",
        label="Incorrect",
        rasterized=True,
    )
    clean_ax(ax_err)
    ax_err.set_title(f"SCANVI correct vs incorrect, full UMAP (error={err_rate:.1%})", fontsize=13)
    ax_err.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor="#2166ac", markeredgecolor="none", label="Correct"),
            Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor="#d62728", markeredgecolor="none", label="Incorrect"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=10,
    )
    err_png = os.path.join(cfg.FIG_OUTDIR, "scanvi_incorrect_predictions_large.png")
    err_pdf = os.path.join(cfg.FIG_OUTDIR, "scanvi_incorrect_predictions_large.pdf")
    fig_err.savefig(err_png, dpi=450, bbox_inches="tight", facecolor="white")
    fig_err.savefig(err_pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {err_png}")
    print(f"[SAVE] {err_pdf}")

    fig_local, ax_local = plt.subplots(1, 1, figsize=(14, 12))
    ax_local.scatter(
        xy[:, 0],
        xy[:, 1],
        s=0.01,
        alpha=0.02,
        color="0.45",
        rasterized=True,
    )
    hb = ax_local.hexbin(
        xy[:, 0],
        xy[:, 1],
        C=(~correct).astype(float),
        reduce_C_function=np.mean,
        gridsize=85,
        mincnt=20,
        cmap="Reds",
        vmin=0,
        vmax=1,
        linewidths=0,
        alpha=0.95,
    )
    clean_ax(ax_local)
    ax_local.set_title(
        f"SCANVI local error rate, full UMAP (global error={err_rate:.1%})",
        fontsize=13,
    )
    cbar = fig_local.colorbar(hb, ax=ax_local, fraction=0.035, pad=0.02)
    cbar.set_label("fraction incorrect in local bin", fontsize=10)
    local_png = os.path.join(cfg.FIG_OUTDIR, "scanvi_local_error_rate.png")
    local_pdf = os.path.join(cfg.FIG_OUTDIR, "scanvi_local_error_rate.pdf")
    fig_local.savefig(local_png, dpi=450, bbox_inches="tight", facecolor="white")
    fig_local.savefig(local_pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {local_png}")
    print(f"[SAVE] {local_pdf}")


if __name__ == "__main__":
    main()
