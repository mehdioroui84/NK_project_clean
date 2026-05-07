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


OUTDIR_NAME = "refined_scanvi_v1"
SPLIT_VALUE = "Held-out"
POINT_SIZE = 0.20
POINT_ALPHA = 0.55
CONTINUOUS_POINT_SIZE = 0.20
LEGEND_MARKER_SIZE = 7


PREFERRED_COLORS = {
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
    "Lung Cytotoxic NK": "#bcbd22",
    "Lung GZMK+ XCL1+ NK": "#8c564b",
    "Unknown_Kidney": "#c49c94",
    "Unknown_BM_1 Erythroid-like": "#c5b0d5",
    "Myeloid-like": "#7f7f7f",
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
    make_correct_incorrect_plot(xy, correct, figdir)
    make_local_error_plot(xy, correct, figdir)
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

    fig, axes = plt.subplots(3, 3, figsize=(26, 18))
    fig.subplots_adjust(left=0.04, right=0.78, top=0.94, bottom=0.06, wspace=0.35, hspace=0.35)
    err_rate = 1.0 - float(np.mean(correct))
    fig.suptitle(f"Refined SCANVI zero-shot held-out UMAP (error={err_rate:.1%})", fontsize=14)

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
        fontsize=8,
    )

    scatter_continuous(axes[1, 0], xy, confidence, fig, "2.1 Confidence")
    scatter_continuous(axes[1, 1], xy, certainty, fig, "2.2 Certainty")
    plot_per_class_metrics(axes[1, 2], true, pred_label, state_colors, tabledir)

    scatter_by_category(axes[2, 0], xy, tissue, distinct_color_map(tissue), legend=True, title="3.1 Tissue")
    scatter_by_category(axes[2, 1], xy, dataset, distinct_color_map(dataset), legend=False, title="3.2 Dataset ID")
    scatter_by_category(axes[2, 2], xy, assay, distinct_color_map(assay), legend=True, title="3.3 Assay clean")

    png = os.path.join(figdir, "scanvi_zeroshot_umap_panels.png")
    pdf = os.path.join(figdir, "scanvi_zeroshot_umap_panels.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")
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
    ax.set_xticklabels(metrics.index, rotation=60, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title("2.3 Zero-shot per-class Accuracy & F1")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    out = os.path.join(tabledir, "scanvi_zeroshot_per_class_accuracy_f1.csv")
    metrics.to_csv(out)
    print(f"[SAVE] {out}")


def make_correct_incorrect_plot(xy, correct, figdir):
    err_rate = 1.0 - float(np.mean(correct))
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    ax.scatter(xy[correct, 0], xy[correct, 1], s=0.12, alpha=0.35, color="#2166ac", rasterized=True)
    ax.scatter(xy[~correct, 0], xy[~correct, 1], s=0.12, alpha=0.35, color="#d62728", rasterized=True)
    clean_ax(ax)
    ax.set_title(f"Refined SCANVI zero-shot correct vs incorrect (error={err_rate:.1%})", fontsize=13)
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor="#2166ac", markeredgecolor="none", label="Correct"),
            Line2D([0], [0], marker="o", linestyle="", markersize=9, markerfacecolor="#d62728", markeredgecolor="none", label="Incorrect"),
        ],
        frameon=False,
        loc="upper left",
        fontsize=10,
    )
    png = os.path.join(figdir, "scanvi_zeroshot_correct_incorrect.png")
    pdf = os.path.join(figdir, "scanvi_zeroshot_correct_incorrect.pdf")
    fig.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")
    plt.close(fig)


def make_local_error_plot(xy, correct, figdir):
    err_rate = 1.0 - float(np.mean(correct))
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    hb = ax.hexbin(
        xy[:, 0],
        xy[:, 1],
        C=(~correct).astype(float),
        reduce_C_function=np.mean,
        gridsize=70,
        mincnt=10,
        cmap="Reds",
        vmin=0,
        vmax=1,
        linewidths=0,
        alpha=0.95,
    )
    clean_ax(ax)
    ax.set_title(f"Refined SCANVI zero-shot local error rate (global error={err_rate:.1%})", fontsize=13)
    cbar = fig.colorbar(hb, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("fraction incorrect in local bin", fontsize=10)
    png = os.path.join(figdir, "scanvi_zeroshot_local_error_rate.png")
    pdf = os.path.join(figdir, "scanvi_zeroshot_local_error_rate.pdf")
    fig.savefig(png, dpi=450, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")
    plt.close(fig)


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
    ax.set_title(title)
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
