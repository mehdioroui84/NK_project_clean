#!/usr/bin/env python3
"""Plot the canonical Leiden 0.1-to-0.5 cluster-overlap alluvial diagram.

The diagram treats cluster identifiers as strings, keeps a connection only when
it represents at least 1% of both its parent and child by default, colors each
ribbon by its Leiden 0.1 parent, and groups Leiden 0.5 children with their main
parents to reduce crossings.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch, Rectangle


DEFAULT_INPUT = Path(
    "outputs/leiden_discovery_assay_only/"
    "full_scvi_leiden_all_resolutions_with_umap.h5ad"
)
DEFAULT_OUTPUT = Path(
    "outputs/leiden_discovery_assay_only/"
    "leiden_0_1_to_leiden_0_5_alluvial.png"
)

# Deliberately arranged to group children with their dominant parent and reduce
# ribbon crossings. Cluster identifiers remain strings throughout.
PREFERRED_PARENT_ORDER = ["0", "3", "2", "1", "10", "8", "6", "5", "7", "9", "4"]
PREFERRED_CHILD_ORDER = [
    "1", "0", "2", "4",             # principally parent 0
    "10", "7", "5", "6",           # principally parent 3
    "9",                              # principally parent 2
    "8", "11", "13", "14", "3",   # principally parent 1
    "12",                             # parent 10
    "15",                             # parent 8
    "16", "17", "25",               # principally parent 6
    "22", "19", "20",               # principally parent 5/6
    "18",                             # parent 7
    "21", "23",                      # parent 9
    "24",                             # parent 4
]

PARENT_COLORS = {
    "0": "#4C78A8",
    "1": "#F58518",
    "2": "#E45756",
    "3": "#72B7B2",
    "4": "#54A24B",
    "5": "#EECA3B",
    "6": "#B279A2",
    "7": "#FF9DA6",
    "8": "#9D755D",
    "9": "#BAB0AC",
    "10": "#6F4E7C",
}


def natural_key(value: str) -> tuple:
    return tuple(int(x) if x.isdigit() else x for x in re.split(r"(\d+)", str(value)))


def preferred_order(values: set[str], preferred: list[str]) -> list[str]:
    kept = [x for x in preferred if x in values]
    return kept + sorted(values.difference(kept), key=natural_key)


def stacked_intervals(
    order: list[str], totals: pd.Series, lower: float = 0.055, upper: float = 0.925
) -> dict[str, tuple[float, float]]:
    """Return top-to-bottom proportional node intervals with fixed visual gaps."""
    n = len(order)
    gap = min(0.011, (upper - lower) / max(4 * n, 1))
    usable = (upper - lower) - gap * max(n - 1, 0)
    total = float(sum(float(totals.get(x, 0)) for x in order))
    if total <= 0:
        raise ValueError("No visible flow remains after filtering.")

    intervals: dict[str, tuple[float, float]] = {}
    cursor = upper
    for item in order:
        height = usable * float(totals.get(item, 0)) / total
        intervals[item] = (cursor - height, cursor)
        cursor -= height + gap
    return intervals


def ribbon_path(
    x0: float,
    x1: float,
    source_bottom: float,
    source_top: float,
    target_bottom: float,
    target_top: float,
) -> MplPath:
    bend = 0.46 * (x1 - x0)
    vertices = [
        (x0, source_bottom),
        (x0 + bend, source_bottom),
        (x1 - bend, target_bottom),
        (x1, target_bottom),
        (x1, target_top),
        (x1 - bend, target_top),
        (x0 + bend, source_top),
        (x0, source_top),
        (x0, source_bottom),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(vertices, codes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-h5ad", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-png", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--parent-key", default="leiden_0_1")
    parser.add_argument("--child-key", default="leiden_0_5")
    parser.add_argument("--minimum-parent-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-child-fraction", type=float, default=0.01)
    parser.add_argument("--dpi", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("[LOAD]", args.input_h5ad)
    adata = ad.read_h5ad(args.input_h5ad, backed="r")
    missing = [x for x in (args.parent_key, args.child_key) if x not in adata.obs]
    if missing:
        raise KeyError(f"Missing obs column(s): {missing}")

    obs = adata.obs[[args.parent_key, args.child_key]].dropna().copy()
    obs[args.parent_key] = obs[args.parent_key].astype(str)
    obs[args.child_key] = obs[args.child_key].astype(str)

    counts = (
        obs.groupby([args.parent_key, args.child_key], observed=True)
        .size()
        .rename("cells")
        .reset_index()
    )
    parent_totals_all = obs[args.parent_key].value_counts()
    child_totals_all = obs[args.child_key].value_counts()
    counts["parent_fraction"] = counts.apply(
        lambda row: row["cells"] / parent_totals_all.loc[row[args.parent_key]], axis=1
    )
    counts["child_fraction"] = counts.apply(
        lambda row: row["cells"] / child_totals_all.loc[row[args.child_key]], axis=1
    )
    visible = counts.loc[
        (counts["parent_fraction"] >= args.minimum_parent_fraction)
        & (counts["child_fraction"] >= args.minimum_child_fraction)
    ].copy()
    if visible.empty:
        raise ValueError("No connections passed both percentage thresholds.")

    parents = preferred_order(set(visible[args.parent_key]), PREFERRED_PARENT_ORDER)
    children = preferred_order(set(visible[args.child_key]), PREFERRED_CHILD_ORDER)
    parent_rank = {x: i for i, x in enumerate(parents)}
    child_rank = {x: i for i, x in enumerate(children)}

    parent_visible_totals = visible.groupby(args.parent_key)["cells"].sum()
    child_visible_totals = visible.groupby(args.child_key)["cells"].sum()
    parent_nodes = stacked_intervals(parents, parent_visible_totals)
    child_nodes = stacked_intervals(children, child_visible_totals)

    x_parent_left, x_parent_right = 0.115, 0.13
    x_child_left, x_child_right = 0.87, 0.885
    parent_cursor = {x: parent_nodes[x][1] for x in parents}
    child_cursor = {x: child_nodes[x][1] for x in children}

    # Assign matching vertical segments at each side before drawing.
    segments = []
    parent_scale = {
        p: (parent_nodes[p][1] - parent_nodes[p][0]) / parent_visible_totals.loc[p]
        for p in parents
    }
    child_scale = {
        c: (child_nodes[c][1] - child_nodes[c][0]) / child_visible_totals.loc[c]
        for c in children
    }
    for p in parents:
        subset = visible.loc[visible[args.parent_key] == p].sort_values(
            args.child_key, key=lambda s: s.map(child_rank)
        )
        for _, row in subset.iterrows():
            c = str(row[args.child_key])
            height = float(row["cells"]) * parent_scale[p]
            source_top = parent_cursor[p]
            source_bottom = source_top - height
            parent_cursor[p] = source_bottom
            segments.append(
                {
                    **row.to_dict(),
                    "parent": p,
                    "child": c,
                    "source_bottom": source_bottom,
                    "source_top": source_top,
                }
            )

    # At each child, stack incoming ribbons by the displayed parent order.
    for c in children:
        incoming = sorted(
            (x for x in segments if x["child"] == c),
            key=lambda x: parent_rank[x["parent"]],
        )
        for segment in incoming:
            height = float(segment["cells"]) * child_scale[c]
            target_top = child_cursor[c]
            target_bottom = target_top - height
            child_cursor[c] = target_bottom
            segment["target_bottom"] = target_bottom
            segment["target_top"] = target_top

    fig, ax = plt.subplots(figsize=(18, 20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Draw broad ribbons first so narrow connections remain visible.
    for segment in sorted(segments, key=lambda x: x["cells"], reverse=True):
        patch = PathPatch(
            ribbon_path(
                x_parent_right,
                x_child_left,
                segment["source_bottom"],
                segment["source_top"],
                segment["target_bottom"],
                segment["target_top"],
            ),
            facecolor=PARENT_COLORS.get(segment["parent"], "#808080"),
            edgecolor="none",
            alpha=0.42,
            zorder=1,
        )
        ax.add_patch(patch)

    for p in parents:
        bottom, top = parent_nodes[p]
        ax.add_patch(
            Rectangle(
                (x_parent_left, bottom),
                x_parent_right - x_parent_left,
                top - bottom,
                facecolor=PARENT_COLORS.get(p, "#808080"),
                edgecolor="#555555",
                linewidth=0.8,
                zorder=3,
            )
        )
        ax.text(
            x_parent_left - 0.009,
            (bottom + top) / 2,
            f"{p}\n(n={int(parent_totals_all.loc[p]):,})",
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    for c in children:
        bottom, top = child_nodes[c]
        ax.add_patch(
            Rectangle(
                (x_child_left, bottom),
                x_child_right - x_child_left,
                top - bottom,
                facecolor="#D9D9D9",
                edgecolor="#666666",
                linewidth=0.7,
                zorder=3,
            )
        )
        ax.text(
            x_child_right + 0.009,
            (bottom + top) / 2,
            f"{c}\n(n={int(child_totals_all.loc[c]):,})",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
        )

    # Plain black percentages (no white boxes), positioned near each endpoint.
    for segment in segments:
        sy = (segment["source_bottom"] + segment["source_top"]) / 2
        ty = (segment["target_bottom"] + segment["target_top"]) / 2
        ax.text(
            x_parent_right + 0.027,
            sy,
            f"{100 * segment['parent_fraction']:.1f}%",
            ha="left",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="black",
            zorder=5,
        )
        ax.text(
            x_child_left - 0.027,
            ty,
            f"{100 * segment['child_fraction']:.1f}%",
            ha="right",
            va="center",
            fontsize=13,
            fontweight="bold",
            color="black",
            zorder=5,
        )

    fig.text(
        0.5,
        0.977,
        "Leiden cluster flow: resolution 0.1 → resolution 0.5",
        ha="center",
        va="top",
        fontsize=20,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.948,
        "Children grouped with their parent clusters; only connections ≥1% of both parent and child are shown",
        ha="center",
        va="top",
        fontsize=16,
        fontweight="bold",
    )
    ax.text(0.115, 0.957, "Leiden 0.1 parent clusters", ha="center", fontsize=15, fontweight="bold")
    ax.text(0.885, 0.957, "Leiden 0.5 child clusters", ha="center", fontsize=15, fontweight="bold")
    ax.text(0.158, 0.938, "% of parent", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.842, 0.938, "% of child", ha="center", fontsize=13, fontweight="bold")

    visible_cells = int(visible["cells"].sum())
    total_cells = int(len(obs))
    fig.text(
        0.5,
        0.015,
        f"Total cells: {total_cells:,}  |  Visible connections: {len(visible)}/{len(counts)}  |  "
        f"Cells in visible ribbons: {visible_cells:,}  |  Hidden cells: {total_cells - visible_cells:,}",
        ha="center",
        fontsize=10,
    )

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("[SAVE]", args.output_png)
    print("[PARENTS]", len(parents), parents)
    print("[CHILDREN]", len(children), children)
    print("[VISIBLE CONNECTIONS]", len(visible), "/", len(counts))
    print("[VISIBLE CELLS]", visible_cells, "/", total_cells)


if __name__ == "__main__":
    main()
