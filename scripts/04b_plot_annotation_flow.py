#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scanpy as sc
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.evaluation.scanvi_full_plots import PREFERRED_STATE_COLORS
from nk_project.io_utils import ensure_dirs
from nk_project.plot_style import (
    LEGEND_FONT_SIZE,
    set_presentation_style,
    style_all_legends,
    style_figure,
    style_legend,
)

set_presentation_style()


DEFAULT_GREY_PATTERNS = [
    r"^B$",
    r"^T$",
    r"\bB[ _-]*cell",
    r"\bT[ _-]*cell",
    r"\bCD3\b",
    r"\bNon[ _-]*NK\b",
]

POINT_SIZE = 0.06
POINT_ALPHA = 0.38
GREY = "#8a8a8a"
DEFAULT_MAX_LEGEND_LABEL_CHARS = 34

LEGEND_LABEL_REPLACEMENTS = {
    "Developmental": "Dev",
    "developmental": "dev",
    "immature": "imm.",
    "Metabolic_stress_hypoxia": "Metabolic_hypoxia",
    "Chemokine_inflammatory": "Chemokine_infl.",
    "Cytotoxic_activated": "Cytotoxic_act.",
    "cytokine_primed_memory_like": "CIMP_memory",
    "Homeostatic_quiescent": "Homeostatic",
    "Proliferating": "Prolif.",
}

MATRIX_LABEL_REPLACEMENTS = {
    "Developmental": "Devel",
    "developmental": "devel",
    "immature": "immat",
    "Metabolic": "Metab",
    "metabolic": "metab",
    "hypoxia": "hypox",
    "Chemokine": "Chem",
    "chemokine": "chem",
    "inflammatory": "inflam",
    "Cytotoxic": "Cyto",
    "cytotoxic": "cyto",
    "activated": "act",
    "Proliferating": "Prolif",
    "proliferating": "prolif",
    "Homeostatic": "Homeo",
    "homeostatic": "homeo",
    "quiescent": "quies",
    "adaptive": "adapt",
    "Development": "Devel",
    "stress": "stress",
}


def main() -> None:
    args = parse_args()
    ensure_dirs(args.outdir)

    print(f"[LOAD] {args.input_h5ad}")
    adata = sc.read_h5ad(args.input_h5ad)
    required = [args.left_annotation, args.middle_key, args.right_annotation]
    missing = [key for key in required if key not in adata.obs]
    if missing:
        raise KeyError(f"Missing obs columns: {missing}")
    if not args.matrix_only and "X_umap" not in adata.obsm:
        raise KeyError("X_umap not found in adata.obsm")

    df = (
        adata.obs[required]
        .copy()
        .astype(str)
        .replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    )
    df.columns = ["left", "middle", "right"]
    df = add_agent_confidence(df, args)
    xy = np.asarray(adata.obsm["X_umap"]) if "X_umap" in adata.obsm else None

    grey_patterns = [re.compile(pattern, flags=re.IGNORECASE) for pattern in args.grey_pattern]
    colors = build_shared_colors(df, grey_patterns)

    if args.matrix_only:
        plot_matrix_heatmaps(
            df,
            args.outdir,
            args.prefix,
            pair=args.matrix_pair,
            left_title=args.left_title,
            middle_title=args.middle_title,
            right_title=args.right_title,
            label_mode=args.matrix_label_mode,
            color_scale=args.matrix_color_scale,
            min_count_label=args.matrix_min_count_label,
            min_row_pct_label=args.matrix_min_row_pct_label,
            confidence_col="agent_confidence" if "agent_confidence" in df.columns else None,
        )
        print("[DONE] Annotation matrix heatmap complete.")
        return

    umap_path = os.path.join(args.outdir, f"{args.prefix}_umap_side_by_side.png")
    html_path = os.path.join(args.outdir, f"{args.prefix}_alluvial.html")
    static_alluvial_path = os.path.join(args.outdir, f"{args.prefix}_static_alluvial.png")

    plot_umap_panels(
        xy,
        df,
        colors,
        umap_path,
        left_title=args.left_title,
        middle_title=args.middle_title,
        right_title=args.right_title,
        max_legend_label_chars=args.max_legend_label_chars,
    )
    plot_alluvial(
        df,
        colors,
        html_path,
        left_title=args.left_title,
        middle_title=args.middle_title,
        right_title=args.right_title,
    )
    plot_static_alluvial(
        df,
        colors,
        static_alluvial_path,
        left_title=args.left_title,
        middle_title=args.middle_title,
        right_title=args.right_title,
    )
    if args.include_matrix:
        plot_matrix_heatmaps(
            df,
            args.outdir,
            args.prefix,
            pair=args.matrix_pair,
            left_title=args.left_title,
            middle_title=args.middle_title,
            right_title=args.right_title,
            label_mode=args.matrix_label_mode,
            color_scale=args.matrix_color_scale,
            min_count_label=args.matrix_min_count_label,
            min_row_pct_label=args.matrix_min_row_pct_label,
            confidence_col="agent_confidence" if "agent_confidence" in df.columns else None,
        )

    print("[DONE] Annotation flow plots complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot two annotation columns around a Leiden/cluster column as "
            "UMAP panels and an interactive alluvial/Sankey flow."
        )
    )
    parser.add_argument(
        "--input-h5ad",
        default="outputs/refined_annotation_v1_agent_preferred_gpt5mini_sanity_v1/full_scvi_leiden_refined_v1.h5ad",
    )
    parser.add_argument("--left-annotation", default=cfg.LABEL_KEY)
    parser.add_argument("--middle-key", default="leiden_0_4")
    parser.add_argument("--right-annotation", default=cfg.REFINED_LABEL_KEY)
    parser.add_argument(
        "--outdir",
        default="outputs/refined_annotation_v1_agent_preferred_gpt5mini_sanity_v1/figures",
    )
    parser.add_argument("--prefix", default="manual_annotation_to_leiden_to_agent_annotation")
    parser.add_argument("--left-title", default="Manual annotation")
    parser.add_argument("--middle-title", default="Leiden clusters")
    parser.add_argument("--right-title", default="Agent annotation")
    parser.add_argument(
        "--matrix-only",
        action="store_true",
        help="Only generate annotation count-matrix heatmap(s); skip UMAP and alluvial plots.",
    )
    parser.add_argument(
        "--include-matrix",
        action="store_true",
        help="Also generate annotation count-matrix heatmap(s) alongside the UMAP/alluvial plots.",
    )
    parser.add_argument(
        "--matrix-pair",
        choices=["left-right", "left-middle", "middle-right", "all"],
        default="left-right",
        help=(
            "Which mapping to summarize as a matrix. Default left-right is "
            "manual annotation to agent annotation."
        ),
    )
    parser.add_argument(
        "--matrix-label-mode",
        choices=["count", "row_pct", "both"],
        default="both",
        help="Cell labels inside the heatmap.",
    )
    parser.add_argument(
        "--matrix-color-scale",
        choices=["none", "log", "linear", "row_pct"],
        default="none",
        help=(
            "Color scale for matrix cells. Default none makes a clean table-style "
            "matrix with no color encoding. Use row_pct to color by row percentage "
            "while keeping raw cell counts in labels."
        ),
    )
    parser.add_argument(
        "--matrix-min-count-label",
        type=int,
        default=1,
        help="Only annotate matrix cells with at least this many cells.",
    )
    parser.add_argument(
        "--matrix-min-row-pct-label",
        type=float,
        default=0.02,
        help="Only annotate matrix cells with at least this row fraction.",
    )
    parser.add_argument(
        "--agent-mapping-csv",
        default=None,
        help=(
            "Optional annotation-agent cluster_annotation_mapping.csv. When provided, "
            "matrix cells involving the agent/right annotation include mean confidence."
        ),
    )
    parser.add_argument(
        "--agent-confidence-column",
        default="confidence_score_0_5",
        help="Confidence column in --agent-mapping-csv. Default: confidence_score_0_5.",
    )
    parser.add_argument(
        "--agent-cluster-column",
        default=None,
        help=(
            "Cluster column in --agent-mapping-csv. Default uses --middle-key, "
            "usually leiden_0_4."
        ),
    )
    parser.add_argument(
        "--grey-pattern",
        action="append",
        default=DEFAULT_GREY_PATTERNS.copy(),
        help=(
            "Regex for labels that should be colored grey. Can be repeated. "
            "Defaults include B, T, Non-NK, myeloid, epithelial, stromal, erythroid."
        ),
    )
    parser.add_argument(
        "--max-legend-label-chars",
        type=int,
        default=DEFAULT_MAX_LEGEND_LABEL_CHARS,
        help=(
            "Maximum displayed legend label length after abbreviation. "
            "Use 0 to disable truncation. Data labels are not changed."
        ),
    )
    return parser.parse_args()


def add_agent_confidence(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if not args.agent_mapping_csv:
        return df
    print(f"[LOAD] {args.agent_mapping_csv}")
    mapping = pd.read_csv(args.agent_mapping_csv)
    cluster_col = args.agent_cluster_column or args.middle_key
    confidence_col = args.agent_confidence_column
    missing = [col for col in [cluster_col, confidence_col] if col not in mapping.columns]
    if missing:
        raise KeyError(f"Missing columns in --agent-mapping-csv: {missing}")

    confidence = (
        mapping[[cluster_col, confidence_col]]
        .dropna(subset=[cluster_col])
        .assign(
            **{
                cluster_col: lambda x: x[cluster_col].astype(str),
                confidence_col: lambda x: pd.to_numeric(x[confidence_col], errors="coerce"),
            }
        )
        .dropna(subset=[confidence_col])
        .drop_duplicates(subset=[cluster_col], keep="first")
        .set_index(cluster_col)[confidence_col]
    )
    out = df.copy()
    out["agent_confidence"] = out["middle"].astype(str).map(confidence)
    n_mapped = int(out["agent_confidence"].notna().sum())
    print(f"[CONFIDENCE] mapped confidence for {n_mapped:,}/{len(out):,} cells")
    return out


def build_shared_colors(df: pd.DataFrame, grey_patterns: list[re.Pattern]) -> dict[str, dict[str, object]]:
    middle_categories = sorted(df["middle"].unique(), key=category_sort_key)
    middle_colors = palette_colors(middle_categories)
    left_colors = annotation_colors_from_middle(df, "left", middle_colors, grey_patterns)
    right_colors = annotation_colors_from_middle(df, "right", middle_colors, grey_patterns)

    return {"left": left_colors, "middle": middle_colors, "right": right_colors}


def annotation_colors_from_middle(
    df: pd.DataFrame,
    col: str,
    middle_colors: dict[str, object],
    grey_patterns: list[re.Pattern],
) -> dict[str, object]:
    out = {}
    used_non_grey = []
    fallback_palette = distinct_non_grey_palette()
    for label in sorted(df[col].unique(), key=category_sort_key):
        if is_grey_label(label, grey_patterns):
            out[label] = GREY
            continue
        if label in PREFERRED_STATE_COLORS:
            out[label] = PREFERRED_STATE_COLORS[label]
            if not is_greyish_color(out[label]):
                used_non_grey.append(out[label])
            continue
        sub = df.loc[df[col].eq(label)]
        dominant_cluster = sub["middle"].value_counts().idxmax()
        inherited = middle_colors.get(str(dominant_cluster), GREY)
        if is_greyish_color(inherited) or color_too_close(inherited, used_non_grey):
            inherited = next_distinct_color(fallback_palette, used_non_grey, str(label))
        out[label] = inherited
        used_non_grey.append(inherited)
    return out


def plot_umap_panels(
    xy: np.ndarray,
    df: pd.DataFrame,
    colors: dict[str, dict[str, object]],
    path: str,
    *,
    left_title: str,
    middle_title: str,
    right_title: str,
    max_legend_label_chars: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    fig.suptitle("Annotation labels and Leiden clusters", fontsize=20, fontweight="bold")
    scatter_panel(
        axes[0],
        xy,
        df["left"].values,
        left_title,
        colors["left"],
        legend=True,
        max_legend_label_chars=max_legend_label_chars,
    )
    scatter_panel(axes[1], xy, df["middle"].values, middle_title, colors["middle"], legend=False, annotate=True)
    scatter_panel(
        axes[2],
        xy,
        df["right"].values,
        right_title,
        colors["right"],
        legend=True,
        max_legend_label_chars=max_legend_label_chars,
    )
    style_figure(fig, tick_size=10, legend_size=LEGEND_FONT_SIZE)
    style_all_legends(fig)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {path}")


def scatter_panel(
    ax,
    xy: np.ndarray,
    values,
    title: str,
    colors: dict[str, object],
    *,
    legend: bool = False,
    annotate: bool = False,
    max_legend_label_chars: int = DEFAULT_MAX_LEGEND_LABEL_CHARS,
) -> None:
    values = np.asarray(values).astype(str)
    categories = sorted(set(values), key=category_sort_key)
    for category in categories:
        mask = values == category
        ax.scatter(
            xy[mask, 0],
            xy[mask, 1],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            color=colors.get(category, GREY),
            rasterized=True,
            label=category,
        )
    if annotate:
        annotate_centers(ax, xy, values)
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=10,
                markerfacecolor=colors.get(category, GREY),
                markeredgecolor="none",
                label=shorten_legend_label(category, max_chars=max_legend_label_chars),
            )
            for category in categories
        ]
        ax.legend(handles=handles, frameon=False, fontsize=LEGEND_FONT_SIZE, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        style_legend(ax.get_legend())


def shorten_legend_label(label: str, *, max_chars: int = DEFAULT_MAX_LEGEND_LABEL_CHARS) -> str:
    display = str(label)
    for old, new in LEGEND_LABEL_REPLACEMENTS.items():
        display = display.replace(old, new)
    if max_chars and max_chars > 3 and len(display) > max_chars:
        return display[: max_chars - 3].rstrip("_- ") + "..."
    return display


def shorten_matrix_label(label: str) -> str:
    display = str(label)
    for old, new in MATRIX_LABEL_REPLACEMENTS.items():
        display = display.replace(old, new)
    display = display.replace("CIMP_cytokine_primed_memory_like", "CIMP_memory")
    display = display.replace("ER_stress_UPR", "ER_stress")
    return display


def annotate_centers(ax, xy: np.ndarray, values) -> None:
    values = np.asarray(values).astype(str)
    for category in sorted(set(values), key=category_sort_key):
        mask = values == category
        if not mask.any():
            continue
        center = np.median(xy[mask], axis=0)
        ax.text(
            center[0],
            center[1],
            category,
            ha="center",
            va="center",
            fontsize=10,
            weight="bold",
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "none", "alpha": 0.35},
        )


def plot_matrix_heatmaps(
    df: pd.DataFrame,
    outdir: str,
    prefix: str,
    *,
    pair: str,
    left_title: str,
    middle_title: str,
    right_title: str,
    label_mode: str,
    color_scale: str,
    min_count_label: int,
    min_row_pct_label: float,
    confidence_col: str | None = None,
) -> None:
    left_order, middle_order, right_order = ordered_layers(df)
    pair_specs = {
        "left-right": ("left", "right", left_order, right_order, left_title, right_title),
        "left-middle": ("left", "middle", left_order, middle_order, left_title, middle_title),
        "middle-right": ("middle", "right", middle_order, right_order, middle_title, right_title),
    }
    selected = list(pair_specs) if pair == "all" else [pair]
    for pair_name in selected:
        row_col, col_col, row_order, col_order, row_title, col_title = pair_specs[pair_name]
        counts = count_matrix(df, row_col, col_col, row_order, col_order)
        row_pct = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
        confidence = None
        if confidence_col and confidence_col in df.columns and "right" in {row_col, col_col}:
            confidence = mean_matrix(df, row_col, col_col, confidence_col, row_order, col_order)
            confidence_path = os.path.join(outdir, f"{prefix}_matrix_{pair_name}_confidence.csv")
            confidence.to_csv(confidence_path)
            print(f"[SAVE] {confidence_path}")
        count_path = os.path.join(outdir, f"{prefix}_matrix_{pair_name}_counts.csv")
        row_pct_path = os.path.join(outdir, f"{prefix}_matrix_{pair_name}_row_pct.csv")
        counts.to_csv(count_path)
        row_pct.to_csv(row_pct_path)
        print(f"[SAVE] {count_path}")
        print(f"[SAVE] {row_pct_path}")

        png = os.path.join(outdir, f"{prefix}_matrix_{pair_name}.png")
        plot_count_matrix(
            counts,
            row_pct,
            png,
            row_title=row_title,
            col_title=col_title,
            label_mode=label_mode,
            color_scale=color_scale,
            min_count_label=min_count_label,
            min_row_pct_label=min_row_pct_label,
            confidence=confidence,
        )


def count_matrix(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    row_order: list[str],
    col_order: list[str],
) -> pd.DataFrame:
    counts = pd.crosstab(df[row_col].astype(str), df[col_col].astype(str))
    return counts.reindex(index=row_order, columns=col_order, fill_value=0).astype(int)


def mean_matrix(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    row_order: list[str],
    col_order: list[str],
) -> pd.DataFrame:
    values = df.copy()
    values[value_col] = pd.to_numeric(values[value_col], errors="coerce")
    means = values.pivot_table(
        index=row_col,
        columns=col_col,
        values=value_col,
        aggfunc="mean",
        observed=False,
    )
    return means.reindex(index=row_order, columns=col_order)


def plot_count_matrix(
    counts: pd.DataFrame,
    row_pct: pd.DataFrame,
    path: str,
    *,
    row_title: str,
    col_title: str,
    label_mode: str,
    color_scale: str,
    min_count_label: int,
    min_row_pct_label: float,
    confidence: pd.DataFrame | None = None,
) -> None:
    if counts.empty:
        print(f"[WARN] Empty matrix for {row_title} -> {col_title}; skipping.")
        return
    values = counts.to_numpy(dtype=float)
    use_color = color_scale != "none"
    if color_scale == "log":
        color_values = np.log10(values + 1.0)
        cbar_label = "log10(cell count + 1)"
        cmap = "Blues"
        vmin = 0.0
        vmax = max(float(np.nanmax(color_values)), 1.0)
    elif color_scale == "linear":
        color_values = values
        cbar_label = "Cell count"
        cmap = "Blues"
        vmin = 0.0
        vmax = max(float(np.nanmax(color_values)), 1.0)
    elif color_scale == "row_pct":
        color_values = row_pct.to_numpy(dtype=float)
        cbar_label = f"Percent of {row_title} cells"
        cmap = "Reds"
        vmin = 0.0
        vmax = 1.0
    else:
        color_values = np.zeros_like(values)
        cbar_label = ""
        cmap = "Blues"
        vmin = 0.0
        vmax = 1.0

    n_rows, n_cols = counts.shape
    has_confidence = confidence is not None
    fig_w = max(12, 0.82 * n_cols + 4)
    fig_h = max(7, 0.52 * n_rows + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(color_values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if not use_color:
        ax.set_facecolor("white")
        im.set_alpha(0.0)
    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels([shorten_matrix_label(label) for label in counts.columns], rotation=45, ha="right", fontsize=10)
    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels([shorten_matrix_label(label) for label in counts.index], fontsize=10)
    ax.set_xlabel(col_title)
    ax.set_ylabel(row_title)
    if color_scale == "row_pct":
        title = f"{row_title} -> {col_title}: row-percentage heatmap"
    else:
        title = f"{row_title} -> {col_title}: cell-count matrix"
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xticks(np.arange(n_cols + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#d8d8d8", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)

    vmax = float(np.nanmax(color_values)) if np.isfinite(color_values).any() else 0.0
    for i, row_name in enumerate(counts.index):
        for j, col_name in enumerate(counts.columns):
            count = int(counts.loc[row_name, col_name])
            pct = float(row_pct.loc[row_name, col_name])
            if count < min_count_label or pct < min_row_pct_label:
                continue
            conf = None
            if confidence is not None:
                conf_value = confidence.loc[row_name, col_name]
                if pd.notna(conf_value):
                    conf = float(conf_value)
            text = matrix_cell_label(count, pct, label_mode, confidence=conf)
            color_value = float(color_values[i, j])
            text_color = "white" if use_color and vmax > 0 and color_value > 0.50 * vmax else "#17202a"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8.0 if has_confidence else 8.5,
                fontweight="bold",
                color=text_color,
            )

    if use_color:
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        cbar.set_label(cbar_label, fontweight="bold")
    style_figure(fig, tick_size=10, legend_size=LEGEND_FONT_SIZE)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {path}")


def matrix_cell_label(
    count: int,
    row_pct: float,
    label_mode: str,
    *,
    confidence: float | None = None,
) -> str:
    if label_mode == "count":
        label = f"{count:,}"
    elif label_mode == "row_pct":
        label = f"{row_pct:.0%}"
    else:
        label = f"{count:,}\n{row_pct:.0%}"
    if confidence is not None:
        label = f"{label}\nC={confidence:.1f}"
    return label


def plot_alluvial(
    df: pd.DataFrame,
    colors: dict[str, dict[str, object]],
    path: str,
    *,
    left_title: str,
    middle_title: str,
    right_title: str,
) -> None:
    left_order, middle_order, right_order = ordered_layers(df)
    labels = left_order + middle_order + right_order
    layer_for_node = ["left"] * len(left_order) + ["middle"] * len(middle_order) + ["right"] * len(right_order)
    node_index = {(layer, label): i for i, (layer, label) in enumerate(zip(layer_for_node, labels))}

    node_colors = [to_rgb(colors[layer].get(label, GREY)) for layer, label in zip(layer_for_node, labels)]
    xs = [0.01] * len(left_order) + [0.50] * len(middle_order) + [0.99] * len(right_order)
    left_weights = node_weights(df, "left")
    middle_weights = node_weights(df, "middle")
    right_weights = node_weights(df, "right")
    ys = (
        layer_y_positions(left_order, left_weights)
        + layer_y_positions(middle_order, middle_weights)
        + layer_y_positions(right_order, right_weights)
    )

    sources, targets, values, link_colors, hover = [], [], [], [], []
    for row in df.groupby(["left", "middle"], observed=True).size().reset_index(name="n").itertuples(index=False):
        left, middle, n = str(row.left), str(row.middle), int(row.n)
        sources.append(node_index[("left", left)])
        targets.append(node_index[("middle", middle)])
        values.append(n)
        link_colors.append(to_rgba(colors["middle"].get(middle, GREY), 0.28))
        hover.append(f"{left_title}: {left}<br>{middle_title}: {middle}<br>Cells: {n:,}")

    for row in df.groupby(["middle", "right"], observed=True).size().reset_index(name="n").itertuples(index=False):
        middle, right, n = str(row.middle), str(row.right), int(row.n)
        sources.append(node_index[("middle", middle)])
        targets.append(node_index[("right", right)])
        values.append(n)
        link_colors.append(to_rgba(colors["middle"].get(middle, GREY), 0.35))
        hover.append(f"{middle_title}: {middle}<br>{right_title}: {right}<br>Cells: {n:,}")

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    pad=18,
                    thickness=16,
                    line=dict(color="rgba(60,60,60,0.35)", width=0.5),
                    label=labels,
                    color=node_colors,
                    x=xs,
                    y=ys,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    color=link_colors,
                    customdata=hover,
                    hovertemplate="%{customdata}<extra></extra>",
                ),
            )
        ]
    )
    fig.update_layout(
        title=f"{left_title} -> {middle_title} -> {right_title}",
        font=dict(size=12),
        width=1650,
        height=max(950, 42 * max(len(left_order), len(middle_order), len(right_order))),
        margin=dict(l=20, r=20, t=90, b=20),
        annotations=[
            dict(x=0.01, y=1.05, text=left_title, showarrow=False, xref="paper", yref="paper", font=dict(size=14)),
            dict(x=0.50, y=1.05, text=middle_title, showarrow=False, xref="paper", yref="paper", font=dict(size=14)),
            dict(x=0.99, y=1.05, text=right_title, showarrow=False, xref="paper", yref="paper", font=dict(size=14)),
        ],
    )
    fig.write_html(path, include_plotlyjs="cdn")
    print(f"[SAVE] {path}")


def plot_static_alluvial(
    df: pd.DataFrame,
    colors: dict[str, dict[str, object]],
    path: str,
    *,
    left_title: str,
    middle_title: str,
    right_title: str,
) -> None:
    left_order, middle_order, right_order = ordered_layers(df)
    left_weights = node_weights(df, "left")
    middle_weights = node_weights(df, "middle")
    right_weights = node_weights(df, "right")

    left_spans = static_layer_spans(left_order, left_weights, gap=0.012)
    middle_spans = static_layer_spans(middle_order, middle_weights, gap=0.016)
    right_spans = static_layer_spans(right_order, right_weights, gap=0.012)

    lm = df.groupby(["left", "middle"], observed=True).size().reset_index(name="n")
    mr = df.groupby(["middle", "right"], observed=True).size().reset_index(name="n")

    left_link_spans = allocate_link_spans(
        lm,
        node_col="left",
        other_col="middle",
        key_cols=("left", "middle"),
        node_spans=left_spans,
        other_order=middle_order,
    )
    middle_left_link_spans = allocate_link_spans(
        lm,
        node_col="middle",
        other_col="left",
        key_cols=("left", "middle"),
        node_spans=middle_spans,
        other_order=left_order,
    )
    middle_right_link_spans = allocate_link_spans(
        mr,
        node_col="middle",
        other_col="right",
        key_cols=("middle", "right"),
        node_spans=middle_spans,
        other_order=right_order,
    )
    right_link_spans = allocate_link_spans(
        mr,
        node_col="right",
        other_col="middle",
        key_cols=("middle", "right"),
        node_spans=right_spans,
        other_order=middle_order,
    )

    fig_h = max(10, 0.48 * max(len(left_order), len(middle_order), len(right_order)))
    fig, ax = plt.subplots(figsize=(18, fig_h))
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.axis("off")

    x_left, x_middle, x_right = 0.025, 0.50, 0.975
    block_w = 0.010
    left_flow_x = x_left + block_w
    middle_left_x = x_middle - block_w / 2
    middle_right_x = x_middle + block_w / 2
    right_flow_x = x_right - block_w

    for row in lm.itertuples(index=False):
        left = str(row.left)
        middle = str(row.middle)
        key = (left, middle)
        draw_ribbon(
            ax,
            left_flow_x,
            left_link_spans[key],
            middle_left_x,
            middle_left_link_spans[key],
            colors["middle"].get(middle, GREY),
            alpha=0.23,
        )

    for row in mr.itertuples(index=False):
        middle = str(row.middle)
        right = str(row.right)
        key = (middle, right)
        draw_ribbon(
            ax,
            middle_right_x,
            middle_right_link_spans[key],
            right_flow_x,
            right_link_spans[key],
            colors["middle"].get(middle, GREY),
            alpha=0.28,
        )

    draw_blocks(ax, left_spans, colors["left"], x_left, block_w, align="left")
    draw_blocks(ax, middle_spans, colors["middle"], x_middle - block_w / 2, block_w, align="middle")
    draw_blocks(ax, right_spans, colors["right"], x_right - block_w, block_w, align="right")

    ax.text(x_left, 0.012, left_title, ha="left", va="bottom", fontsize=13)
    ax.text(x_middle, 0.012, middle_title, ha="center", va="bottom", fontsize=13)
    ax.text(x_right, 0.012, right_title, ha="right", va="bottom", fontsize=13)
    ax.set_title(f"{left_title} -> {middle_title} -> {right_title}", fontsize=15, pad=18)

    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {path}")


def draw_blocks(ax, spans: dict[str, tuple[float, float]], colors: dict[str, object], x: float, width: float, *, align: str) -> None:
    for label, (y0, y1) in spans.items():
        color = colors.get(label, GREY)
        ax.add_patch(Rectangle((x, y0), width, y1 - y0, facecolor=color, edgecolor="white", linewidth=0.4, zorder=5))
        y = (y0 + y1) / 2.0
        if align == "left":
            ax.text(x + width + 0.004, y, label, ha="left", va="center", fontsize=8, zorder=6)
        elif align == "right":
            ax.text(x - 0.004, y, label, ha="right", va="center", fontsize=8, zorder=6)
        else:
            ax.text(x + width + 0.004, y, label, ha="left", va="center", fontsize=8, zorder=6)


def draw_ribbon(ax, x0: float, span0: tuple[float, float], x1: float, span1: tuple[float, float], color, *, alpha: float) -> None:
    y0a, y0b = span0
    y1a, y1b = span1
    dx = x1 - x0
    c0 = x0 + dx * 0.48
    c1 = x1 - dx * 0.48
    verts = [
        (x0, y0a),
        (c0, y0a),
        (c1, y1a),
        (x1, y1a),
        (x1, y1b),
        (c1, y1b),
        (c0, y0b),
        (x0, y0b),
        (x0, y0a),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    patch = PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", alpha=alpha, zorder=2)
    ax.add_patch(patch)


def static_layer_spans(order: list[str], weights: dict[str, float], *, gap: float) -> dict[str, tuple[float, float]]:
    if not order:
        return {}
    top = 0.045
    bottom = 0.965
    values = np.array([max(float(weights.get(label, 1.0)), 1.0) for label in order], dtype=float)
    usable_gap = gap * max(len(order) - 1, 0)
    usable_height = max(bottom - top - usable_gap, 0.2)
    heights = usable_height * values / values.sum()
    spans = {}
    y = top
    min_height = 0.004
    for label, height in zip(order, heights):
        height = max(float(height), min_height)
        spans[str(label)] = (y, min(y + height, bottom))
        y += height + gap
    return spans


def allocate_link_spans(
    links: pd.DataFrame,
    *,
    node_col: str,
    other_col: str,
    key_cols: tuple[str, str],
    node_spans: dict[str, tuple[float, float]],
    other_order: list[str],
) -> dict[tuple[str, str], tuple[float, float]]:
    other_rank = {str(value): idx for idx, value in enumerate(other_order)}
    out = {}
    for node, sub in links.groupby(node_col, observed=True):
        node = str(node)
        if node not in node_spans:
            continue
        y0, y1 = node_spans[node]
        total = float(sub["n"].sum())
        if total <= 0:
            continue
        sub = sub.copy()
        sub["_rank"] = sub[other_col].astype(str).map(other_rank).fillna(len(other_rank))
        sub = sub.sort_values(["_rank", other_col])
        cursor = y0
        for row in sub.itertuples(index=False):
            height = (y1 - y0) * float(row.n) / total
            key = tuple(str(getattr(row, col)) for col in key_cols)
            out[key] = (cursor, cursor + height)
            cursor += height
    return out


def ordered_layers(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    left = sorted(df["left"].unique(), key=category_sort_key)
    middle = sorted(df["middle"].unique(), key=category_sort_key)
    right = sorted(df["right"].unique(), key=category_sort_key)

    lm = df.groupby(["left", "middle"], observed=True).size().reset_index(name="n")
    mr = df.groupby(["middle", "right"], observed=True).size().reset_index(name="n")

    for _ in range(12):
        middle_pos = position_map(middle)
        left = sort_by_barycenter(left, lm, "left", "middle", middle_pos)
        right = sort_by_barycenter(right, mr, "right", "middle", middle_pos)
        left_pos = position_map(left)
        right_pos = position_map(right)
        middle = sort_middle_by_two_sides(middle, lm, mr, left_pos, right_pos)

    return left, middle, right


def sort_by_barycenter(categories, links: pd.DataFrame, source_col: str, target_col: str, target_pos: dict[str, float]):
    weights = defaultdict(float)
    totals = defaultdict(float)
    for row in links.itertuples(index=False):
        source = str(getattr(row, source_col))
        target = str(getattr(row, target_col))
        weight = float(row.n)
        weights[source] += weight * target_pos.get(target, 0.0)
        totals[source] += weight
    return sorted(
        map(str, categories),
        key=lambda cat: (weights.get(cat, 0.0) / totals.get(cat, 1.0), category_sort_key(cat)),
    )


def sort_middle_by_two_sides(middle, lm: pd.DataFrame, mr: pd.DataFrame, left_pos, right_pos):
    sums = defaultdict(float)
    totals = defaultdict(float)
    for row in lm.itertuples(index=False):
        left = str(row.left)
        mid = str(row.middle)
        weight = float(row.n)
        sums[mid] += weight * left_pos.get(left, 0.0)
        totals[mid] += weight
    for row in mr.itertuples(index=False):
        mid = str(row.middle)
        right = str(row.right)
        weight = float(row.n)
        sums[mid] += weight * right_pos.get(right, 0.0)
        totals[mid] += weight
    return sorted(
        map(str, middle),
        key=lambda cat: (sums.get(cat, 0.0) / totals.get(cat, 1.0), category_sort_key(cat)),
    )


def position_map(values) -> dict[str, float]:
    values = list(map(str, values))
    if len(values) <= 1:
        return {values[0]: 0.5} if values else {}
    return {value: idx / (len(values) - 1) for idx, value in enumerate(values)}


def node_weights(df: pd.DataFrame, col: str) -> dict[str, float]:
    return df[col].value_counts().astype(float).to_dict()


def layer_y_positions(order: list[str], weights: dict[str, float]) -> list[float]:
    if len(order) <= 1:
        return [0.5] * len(order)
    values = np.array([max(float(weights.get(label, 1.0)), 1.0) for label in order], dtype=float)
    gap = min(0.012, 0.18 / max(len(order) - 1, 1))
    usable = 0.96 - gap * (len(order) - 1)
    heights = usable * values / values.sum()
    y = []
    cursor = 0.02
    for height in heights:
        y.append(float(cursor))
        cursor += float(height) + gap
    return y


def is_grey_label(label: str, patterns: list[re.Pattern]) -> bool:
    text = str(label).replace("_", " ").replace("-", " ")
    return any(pattern.search(text) for pattern in patterns)


def is_greyish_color(color) -> bool:
    if isinstance(color, str) and color.startswith("#"):
        h = color.lstrip("#")
        if len(h) >= 6:
            r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
            return max(r, g, b) - min(r, g, b) < 18
    if isinstance(color, str):
        return color.lower() in {"grey", "gray", GREY}
    r, g, b, *_ = color
    return max(r, g, b) - min(r, g, b) < 0.07


def fallback_non_grey_color(label: str):
    usable = distinct_non_grey_palette()
    idx = sum(ord(char) for char in str(label)) % len(usable)
    return usable[idx]


def distinct_non_grey_palette():
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3", "Paired", "Dark2", "Accent"):
        cmap = plt.get_cmap(cmap_name)
        palette.extend([cmap(i) for i in range(cmap.N)])
    return [color for color in palette if not is_greyish_color(color)]


def next_distinct_color(palette, used, label: str):
    if not palette:
        return fallback_non_grey_color(label)
    start = sum(ord(char) for char in label) % len(palette)
    for offset in range(len(palette)):
        candidate = palette[(start + offset) % len(palette)]
        if not color_too_close(candidate, used):
            return candidate
    return palette[start]


def color_too_close(color, used, *, threshold: float = 0.28) -> bool:
    if not used:
        return False
    c = np.array(color_rgb01(color))
    for other in used:
        o = np.array(color_rgb01(other))
        if np.linalg.norm(c - o) < threshold:
            return True
    return False


def color_rgb01(color) -> tuple[float, float, float]:
    if isinstance(color, str) and color.startswith("#"):
        h = color.lstrip("#")
        return int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0
    if isinstance(color, str) and color.startswith("rgb("):
        nums = [float(x) for x in color.removeprefix("rgb(").removesuffix(")").split(",")[:3]]
        return nums[0] / 255.0, nums[1] / 255.0, nums[2] / 255.0
    r, g, b, *_ = color
    return float(r), float(g), float(b)


def category_sort_key(value):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def palette_colors(categories) -> dict[str, object]:
    palette = []
    for cmap_name in ("tab20", "tab20b", "tab20c", "Set3", "Paired"):
        cmap = plt.get_cmap(cmap_name)
        palette.extend([cmap(i) for i in range(cmap.N)])
    return {str(category): palette[i % len(palette)] for i, category in enumerate(categories)}


def to_rgb(color) -> str:
    if isinstance(color, str):
        if color.startswith("#"):
            return color
        return color
    r, g, b, *_ = color
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"


def to_rgba(color, alpha: float) -> str:
    if isinstance(color, str) and color.startswith("#"):
        h = color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"
    if isinstance(color, str) and color.startswith("rgb("):
        nums = color.removeprefix("rgb(").removesuffix(")").split(",")
        return f"rgba({nums[0]},{nums[1]},{nums[2]},{alpha})"
    r, g, b, *_ = color
    return f"rgba({int(r * 255)},{int(g * 255)},{int(b * 255)},{alpha})"


if __name__ == "__main__":
    main()
