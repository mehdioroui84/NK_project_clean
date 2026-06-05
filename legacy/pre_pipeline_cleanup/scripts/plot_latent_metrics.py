#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs
from nk_project.metrics import compute_integration_metrics_from_latent, minmax_normalize_series


def main():
    ensure_dirs(cfg.FIG_OUTDIR, cfg.TABLE_OUTDIR)

    latent_specs = [
        {
            "name": "SCANVI",
            "path": os.path.join(cfg.LATENT_OUTDIR, "scanvi_latents.npz"),
            "array_key": "X_SCANVI",
            "obs_path": os.path.join(cfg.TABLE_OUTDIR, "scanvi_full_obs_metadata.csv"),
        },
        {
            "name": "SCVI",
            "path": os.path.join(cfg.LATENT_OUTDIR, "scvi_latents.npz"),
            "array_key": "X_scVI",
            "obs_path": os.path.join(cfg.TABLE_OUTDIR, "scvi_obs_metadata.csv"),
        },
    ]

    rows = []
    for spec in latent_specs:
        if not os.path.exists(spec["path"]) or not os.path.exists(spec["obs_path"]):
            print(f"[SKIP] {spec['name']}: missing latent or obs file")
            continue

        print(f"[LOAD] {spec['name']} latent: {spec['path']}")
        latent = np.load(spec["path"], allow_pickle=True)
        z = latent[spec["array_key"]]
        obs_names = latent["obs_names"].astype(str)

        print(f"[LOAD] {spec['name']} obs: {spec['obs_path']}")
        obs = pd.read_csv(spec["obs_path"], index_col=0, low_memory=False)
        obs = align_obs(obs, obs_names, spec["name"])

        labels = obs[cfg.LABEL_KEY].astype(str).values
        batch = obs[cfg.PRODUCTION_BATCH_KEY].astype(str).values

        metrics = compute_integration_metrics_from_latent(
            z=z,
            labels=labels,
            batch_labels=batch,
            strategy_name=spec["name"],
            unlabeled=cfg.UNLABELED_CATEGORY,
            n_neighbors=cfg.DISCOVERY_N_NEIGHBORS,
            leiden_resolution=cfg.LEIDEN_RES,
            max_metric_cells=cfg.METRIC_MAX_CELLS,
            seed=cfg.SEED,
            knn_k=cfg.METRIC_KNN_K,
        )
        row = {k: v for k, v in metrics.items() if k != "cluster_sizes"}
        rows.append(row)

    if not rows:
        raise SystemExit("No latent files found. Train SCANVI and/or SCVI first.")

    summary = pd.DataFrame(rows).set_index("strategy")
    score_cols = [
        "asw_batch",
        "asw_nk_state",
        "graph_connectivity",
        "knn_label_acc",
        "nmi",
        "ari",
    ]
    for col in score_cols:
        summary[col] = pd.to_numeric(summary[col], errors="coerce")

    batch_cols = ["asw_batch"]
    bio_cols = ["asw_nk_state", "graph_connectivity", "knn_label_acc", "nmi", "ari"]
    norm = pd.DataFrame(index=summary.index)
    for col in batch_cols + bio_cols:
        norm[col + "_norm"] = minmax_normalize_series(summary[col], higher_is_better=True)
    summary["batch_score"] = norm[[c + "_norm" for c in batch_cols]].mean(axis=1)
    summary["biology_score"] = norm[[c + "_norm" for c in bio_cols]].mean(axis=1)
    summary["overall_score"] = 0.4 * summary["batch_score"] + 0.6 * summary["biology_score"]

    csv_path = os.path.join(cfg.TABLE_OUTDIR, "latent_batch_biology_metrics.csv")
    summary.to_csv(csv_path)
    print("\n[METRICS]")
    print(summary.round(4).to_string())
    print(f"[SAVE] {csv_path}")

    plot_metrics(summary)


def align_obs(obs: pd.DataFrame, obs_names: np.ndarray, name: str) -> pd.DataFrame:
    obs.index = obs.index.astype(str)
    if set(obs_names).issubset(set(obs.index)):
        return obs.loc[obs_names].copy()
    if len(obs) == len(obs_names):
        print(f"[WARN] {name}: obs index does not match latent obs_names; aligning by row order.")
        obs = obs.copy()
        obs.index = obs_names
        return obs
    raise ValueError(
        f"Cannot align {name} obs metadata with latent obs_names: "
        f"obs rows={len(obs):,}, latent rows={len(obs_names):,}"
    )


def plot_metrics(summary: pd.DataFrame) -> None:
    raw_metrics = {
        "Batch mixing": ["asw_batch"],
        "Biology preservation": ["asw_nk_state", "graph_connectivity", "knn_label_acc", "nmi", "ari"],
        "Summary scores": ["batch_score", "biology_score", "overall_score"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Latent-space batch mixing and biology preservation", fontsize=14)

    for ax, (title, cols) in zip(axes, raw_metrics.items()):
        x = np.arange(len(summary.index))
        width = min(0.8 / len(cols), 0.25)
        for i, col in enumerate(cols):
            offset = (i - (len(cols) - 1) / 2) * width
            ax.bar(x + offset, summary[col].values, width=width, label=col)
        ax.set_xticks(x)
        ax.set_xticklabels(summary.index, rotation=20, ha="right")
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    plt.tight_layout()
    png = os.path.join(cfg.FIG_OUTDIR, "latent_batch_biology_metrics.png")
    pdf = os.path.join(cfg.FIG_OUTDIR, "latent_batch_biology_metrics.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")


if __name__ == "__main__":
    main()
