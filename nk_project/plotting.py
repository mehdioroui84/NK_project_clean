from __future__ import annotations

import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from .preprocessing import log10_pivot_counts


def plot_composite_batch_profile(
    obs,
    combo_counts,
    *,
    dataset_key: str = "dataset_id",
    assay_key: str = "assay",
    low_cell_warn: int = 500,
    min_cell_hard: int = 100,
    save_path: str | None = None,
):
    n_combos = len(combo_counts)
    n_datasets = obs[dataset_key].astype(str).nunique()
    n_assays = obs[assay_key].astype(str).nunique()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle(
        f"Composite batch profile ({n_combos} observed combos from "
        f"{n_datasets} datasets x {n_assays} assays)",
        fontsize=13,
    )

    ax = axes[0]
    colors = [
        "#d62728" if c < min_cell_hard else "#ff7f0e" if c < low_cell_warn else "#2ca02c"
        for c in combo_counts.values
    ]
    ax.bar(np.arange(n_combos), combo_counts.values, color=colors, width=1.0)
    ax.axhline(low_cell_warn, color="#ff7f0e", lw=1.5, ls="--", label=f"Risky ({low_cell_warn:,})")
    ax.axhline(min_cell_hard, color="#d62728", lw=1.5, ls="--", label=f"Hard ({min_cell_hard:,})")
    ax.set_xlabel("Composite batch")
    ax.set_ylabel("Cell count")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_title("Cells per composite batch")
    ax.legend(frameon=False, fontsize=9)

    ax = axes[1]
    log_pivot = log10_pivot_counts(obs, dataset_key=dataset_key, assay_key=assay_key)
    im = ax.imshow(log_pivot.values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=np.nanmax(log_pivot.values))
    ax.set_xticks(np.arange(len(log_pivot.columns)))
    ax.set_xticklabels(log_pivot.columns, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(log_pivot.index)))
    ax.set_yticklabels(log_pivot.index, fontsize=6)
    ax.set_title("Dataset x assay heatmap")
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("log10 cells", fontsize=8)

    ax = axes[2]
    bins = [0, 100, 500, 1000, 5000, 10000, 50000, int(combo_counts.max()) + 1]
    counts_hist, edges = np.histogram(combo_counts.values, bins=bins)
    labels = [f"{int(edges[i]):,}-{int(edges[i + 1]) - 1:,}" for i in range(len(edges) - 1)]
    bar_colors = ["#d62728", "#ff7f0e", "#ff7f0e", "#2ca02c", "#2ca02c", "#2ca02c", "#2ca02c"]
    bars = ax.bar(np.arange(len(counts_hist)), counts_hist, color=bar_colors[: len(counts_hist)], edgecolor="white")
    ax.set_xticks(np.arange(len(counts_hist)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Number of composite batches")
    ax.set_title("Distribution of composite batch sizes")
    for bar, count in zip(bars, counts_hist):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2, str(count), ha="center", fontsize=9)

    plt.tight_layout()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    return fig
