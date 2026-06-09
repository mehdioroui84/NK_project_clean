#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nk_project.annotation_agent.taxonomy_reference import (
    TIER_WEIGHTS,
    TaxonomyEntry,
    load_taxonomy_entries,
    max_taxonomy_score,
    support_level_rank,
    taxonomy_support_level,
)
from nk_project.io_utils import ensure_dirs


SUBTYPE_LABELS = {
    "NK1",
    "NK2",
    "adaptive_NK_CMV",
    "adaptive_NK_nonCMV",
    "trNK",
    "cNK",
    "L6_Developmental_immature",
}

STATE_LABELS = {
    "Chemokine_inflammatory",
    "Checkpoint_exhausted",
    "ER_stress_UPR",
    "Metabolic_stress_hypoxia",
    "Proliferating",
    "IFN_stimulated",
    "Cytotoxic_activated",
    "Homeostatic_quiescent",
    "CIML_cytokine_preactivated",
    "CIMP_cytokine_primed_memory_like",
}


def main() -> None:
    args = parse_args()
    table_dir = os.path.join(args.outdir, "tables")
    fig_dir = os.path.join(args.outdir, "figures")
    ensure_dirs(args.outdir, table_dir, fig_dir)

    print(f"[ATTRIBUTION] {args.attribution_table}")
    print(f"[OUTDIR] {args.outdir}")
    attr = load_attribution_table(args.attribution_table, args.cluster_col)
    entries = load_taxonomy_entries(args.taxonomy_reference)
    if not entries:
        raise ValueError("No taxonomy marker entries were loaded.")
    print(f"[TAXONOMY] {len(entries)} marker programs")

    selected = select_top_attribution_genes(attr, args.top_n)
    selected_path = os.path.join(table_dir, f"top{args.top_n}_attribution_genes_used_for_concordance.csv")
    selected.to_csv(selected_path, index=False)
    print(f"[SAVE] {selected_path}")

    long = build_concordance_table(selected, entries, run_name=args.run_name)
    long_path = os.path.join(table_dir, "attribution_taxonomy_concordance_long.csv")
    long.to_csv(long_path, index=False)
    print(f"[SAVE] {long_path}")

    score_wide = make_wide(long, "weighted_support_score")
    pct_wide = make_wide(long, "weighted_support_pct")
    hit_wide = make_wide(long, "n_supporting_hits")
    for name, df in [
        ("attribution_taxonomy_concordance_weighted_score.csv", score_wide),
        ("attribution_taxonomy_concordance_weighted_pct.csv", pct_wide),
        ("attribution_taxonomy_concordance_hit_count.csv", hit_wide),
    ]:
        path = os.path.join(table_dir, name)
        df.to_csv(path)
        print(f"[SAVE] {path}")

    top_matches = top_matches_per_cluster(long, args.top_matches_per_cluster)
    top_path = os.path.join(table_dir, "top_taxonomy_matches_per_cluster.csv")
    top_matches.to_csv(top_path, index=False)
    print(f"[SAVE] {top_path}")

    gene_overlap = build_gene_overlap_table(selected, entries, run_name=args.run_name)
    overlap_path = os.path.join(table_dir, "attribution_gene_taxonomy_overlap.csv")
    gene_overlap.to_csv(overlap_path, index=False)
    print(f"[SAVE] {overlap_path}")

    recurrent = build_recurrent_gene_table(gene_overlap)
    recurrent_path = os.path.join(table_dir, "recurrent_top_attribution_genes.csv")
    recurrent.to_csv(recurrent_path, index=False)
    print(f"[SAVE] {recurrent_path}")

    plot_heatmap(score_wide, fig_dir, "concordance_weighted_score_heatmap", "Weighted concordance score")
    plot_heatmap(hit_wide, fig_dir, "concordance_hit_count_heatmap", "Number of supporting taxonomy hits")
    plot_layer_heatmaps(long, fig_dir)
    plot_top_match_heatmap(long, fig_dir)
    plot_recurrent_genes(
        recurrent,
        fig_dir,
        args.recurrent_gene_plot_n,
        min_recurrent_clusters=args.min_recurrent_clusters,
    )
    plot_taxonomy_tree(long, entries, fig_dir)

    print("[DONE] Attribution-taxonomy concordance complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post hoc concordance between data-driven gene attribution results and "
            "curated NK taxonomy marker programs. The taxonomy markers are not "
            "used to train or rank the classifier."
        )
    )
    parser.add_argument("--attribution-table", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--run-name", default="attribution")
    parser.add_argument("--taxonomy-reference", default=None)
    parser.add_argument(
        "--cluster-col",
        default="auto",
        help="Cluster/state column in attribution table. Default auto detects NK_State_refined or NK_state.",
    )
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--top-matches-per-cluster", type=int, default=5)
    parser.add_argument("--recurrent-gene-plot-n", type=int, default=35)
    parser.add_argument(
        "--min-recurrent-clusters",
        type=int,
        default=1,
        help=(
            "Minimum number of data-driven clusters where a gene must appear in the "
            "top attribution genes to be shown in the recurrent-gene figure."
        ),
    )
    return parser.parse_args()


def load_attribution_table(path: str, cluster_col: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if cluster_col == "auto":
        candidates = ["NK_State_refined", "NK_state", "state", "pred_label", "label"]
        cluster_col = next((col for col in candidates if col in df.columns), "")
        if not cluster_col:
            raise KeyError(f"Could not auto-detect cluster column. Columns: {list(df.columns)}")
    required = {cluster_col, "gene", "mean_attr", "mean_abs_attr"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing required attribution columns: {sorted(missing)}")
    out = df.rename(columns={cluster_col: "cluster"}).copy()
    out["cluster"] = out["cluster"].astype(str)
    out["gene"] = out["gene"].astype(str).str.upper()
    out["mean_attr"] = pd.to_numeric(out["mean_attr"], errors="coerce")
    out["mean_abs_attr"] = pd.to_numeric(out["mean_abs_attr"], errors="coerce")
    out = out.dropna(subset=["mean_attr", "mean_abs_attr", "gene", "cluster"])
    return out


def select_top_attribution_genes(attr: pd.DataFrame, top_n: int) -> pd.DataFrame:
    selected = (
        attr.sort_values(["cluster", "mean_abs_attr", "mean_attr", "gene"], ascending=[True, False, False, True])
        .groupby("cluster", group_keys=False, sort=False)
        .head(top_n)
        .copy()
    )
    selected["rank_for_cluster"] = selected.groupby("cluster")["mean_abs_attr"].rank(
        method="first",
        ascending=False,
    ).astype(int)
    selected["attribution_direction"] = np.where(selected["mean_attr"] >= 0, "positive", "negative")
    return selected


def build_concordance_table(selected: pd.DataFrame, entries: list[TaxonomyEntry], *, run_name: str) -> pd.DataFrame:
    rows = []
    for cluster, sub in selected.groupby("cluster", sort=False):
        positive = set(sub.loc[sub["mean_attr"] > 0, "gene"])
        negative = set(sub.loc[sub["mean_attr"] < 0, "gene"])
        all_top = set(sub["gene"])
        for entry in entries:
            layer = normalized_taxonomy_layer(entry)
            core_hits = intersect(entry.core, positive)
            support_hits = intersect(entry.support, positive)
            context_hits = intersect(entry.context, positive)
            negative_expected_low_hits = intersect(entry.negative, negative)
            negative_contradictions = intersect(entry.negative, positive)
            all_hits = intersect(
                unique(entry.core + entry.support + entry.context + entry.negative),
                all_top,
            )
            positive_hit_score = (
                TIER_WEIGHTS["core"] * len(core_hits)
                + TIER_WEIGHTS["support"] * len(support_hits)
                + TIER_WEIGHTS["context"] * len(context_hits)
            )
            expected_low_score = TIER_WEIGHTS["support"] * len(negative_expected_low_hits)
            contradiction_penalty = TIER_WEIGHTS["core"] * len(negative_contradictions)
            weighted_support_score = positive_hit_score + expected_low_score - contradiction_penalty
            possible = max_taxonomy_score(entry)
            weighted_support_pct = 100.0 * max(weighted_support_score, 0) / possible if possible else 0.0
            support_level = taxonomy_support_level(
                core_hits=core_hits,
                support_hits=support_hits,
                context_hits=context_hits,
                negative_expected_low_hits=negative_expected_low_hits,
                negative_contradictions=negative_contradictions,
            )
            rows.append(
                {
                    "run_name": run_name,
                    "cluster": str(cluster),
                    "taxonomy_layer": layer,
                    "taxonomy_name": entry.name,
                    "taxonomy_label": entry.canonical_label,
                    "program_id": program_id(entry),
                    "n_top_genes": len(sub),
                    "n_positive_top_genes": len(positive),
                    "n_negative_top_genes": len(negative),
                    "n_core_positive_hits": len(core_hits),
                    "n_support_positive_hits": len(support_hits),
                    "n_context_positive_hits": len(context_hits),
                    "n_negative_expected_low_hits": len(negative_expected_low_hits),
                    "n_negative_contradiction_hits": len(negative_contradictions),
                    "n_supporting_hits": len(core_hits) + len(support_hits) + len(context_hits) + len(negative_expected_low_hits),
                    "n_all_top_gene_hits": len(all_hits),
                    "weighted_support_score": weighted_support_score,
                    "weighted_support_pct": weighted_support_pct,
                    "support_level": support_level,
                    "core_positive_hit_genes": join_genes(core_hits),
                    "support_positive_hit_genes": join_genes(support_hits),
                    "context_positive_hit_genes": join_genes(context_hits),
                    "negative_expected_low_hit_genes": join_genes(negative_expected_low_hits),
                    "negative_contradiction_genes": join_genes(negative_contradictions),
                    "all_top_hit_genes": join_genes(all_hits),
                }
            )
    return pd.DataFrame(rows)


def build_gene_overlap_table(selected: pd.DataFrame, entries: list[TaxonomyEntry], *, run_name: str) -> pd.DataFrame:
    memberships = marker_memberships(entries)
    rows = []
    for row in selected.itertuples(index=False):
        gene = str(row.gene).upper()
        programs = memberships.get(gene, [])
        rows.append(
            {
                "run_name": run_name,
                "cluster": row.cluster,
                "gene": gene,
                "rank_for_cluster": row.rank_for_cluster,
                "mean_attr": row.mean_attr,
                "mean_abs_attr": row.mean_abs_attr,
                "attribution_direction": row.attribution_direction,
                "in_taxonomy_reference": bool(programs),
                "taxonomy_program_count": len({item["program_id"] for item in programs}),
                "taxonomy_programs": "; ".join(unique(item["program_id"] for item in programs)),
                "taxonomy_tiers": "; ".join(unique(f'{item["program_id"]}:{item["tier"]}' for item in programs)),
            }
        )
    return pd.DataFrame(rows)


def build_recurrent_gene_table(gene_overlap: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gene, sub in gene_overlap.groupby("gene", sort=False):
        clusters = unique(sub["cluster"].astype(str).tolist())
        taxonomy_programs = unique(
            program
            for value in sub["taxonomy_programs"].dropna().astype(str)
            for program in value.split("; ")
            if program
        )
        rows.append(
            {
                "gene": gene,
                "n_clusters_top_gene": len(clusters),
                "clusters": "; ".join(clusters),
                "mean_abs_attr_mean": float(sub["mean_abs_attr"].mean()),
                "mean_abs_attr_max": float(sub["mean_abs_attr"].max()),
                "in_taxonomy_reference": bool(taxonomy_programs),
                "taxonomy_program_count": len(taxonomy_programs),
                "taxonomy_programs": "; ".join(taxonomy_programs),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["n_clusters_top_gene", "mean_abs_attr_max", "gene"],
        ascending=[False, False, True],
    )


def marker_memberships(entries: list[TaxonomyEntry]) -> dict[str, list[dict[str, str]]]:
    memberships: dict[str, list[dict[str, str]]] = defaultdict(list)
    for entry in entries:
        pid = program_id(entry)
        layer = normalized_taxonomy_layer(entry)
        for tier, genes in entry.markers.items():
            for gene in genes:
                memberships[str(gene).upper()].append(
                    {
                        "program_id": pid,
                        "taxonomy_layer": layer,
                        "taxonomy_label": entry.canonical_label,
                        "tier": tier,
                    }
                )
    return memberships


def make_wide(long: pd.DataFrame, value_col: str) -> pd.DataFrame:
    wide = long.pivot_table(
        index="cluster",
        columns="program_id",
        values=value_col,
        aggfunc="max",
        fill_value=0,
    )
    return wide.loc[sorted(wide.index, key=cluster_sort_key), sorted(wide.columns)]


def top_matches_per_cluster(long: pd.DataFrame, n: int) -> pd.DataFrame:
    ranked = long.copy()
    ranked["support_level_rank"] = ranked["support_level"].map(support_level_rank)
    ranked = ranked.sort_values(
        [
            "cluster",
            "support_level_rank",
            "weighted_support_score",
            "n_core_positive_hits",
            "n_supporting_hits",
            "weighted_support_pct",
            "program_id",
        ],
        ascending=[True, False, False, False, False, False, True],
    )
    return ranked.groupby("cluster", group_keys=False, sort=False).head(n).reset_index(drop=True)


def plot_layer_heatmaps(long: pd.DataFrame, fig_dir: str) -> None:
    for layer in ["subtype", "state"]:
        sub = long.loc[long["taxonomy_layer"].eq(layer)].copy()
        if sub.empty:
            continue
        label = layer_display_name(layer)
        score_wide = make_wide(sub, "weighted_support_score")
        hit_wide = make_wide(sub, "n_supporting_hits")
        plot_heatmap(
            score_wide,
            fig_dir,
            f"concordance_weighted_score_heatmap_{layer}",
            f"{label} weighted concordance score",
            include_layer=False,
        )
        plot_heatmap(
            hit_wide,
            fig_dir,
            f"concordance_hit_count_heatmap_{layer}",
            f"{label} supporting taxonomy hits",
            include_layer=False,
        )


def plot_heatmap(
    wide: pd.DataFrame,
    fig_dir: str,
    name: str,
    cbar_label: str,
    *,
    include_layer: bool = True,
) -> None:
    if wide.empty:
        return
    fig_w = max(10, 0.34 * wide.shape[1])
    fig_h = max(5, 0.32 * wide.shape[0])
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    values = wide.to_numpy(dtype=float)
    vmax = float(np.nanmax(values)) if np.isfinite(values).any() else 1.0
    vmax = vmax if vmax > 0 else 1.0
    im = ax.imshow(values, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(wide.shape[1]))
    ax.set_xticklabels(
        [display_program_id(col, include_layer=include_layer) for col in wide.columns],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.set_yticks(np.arange(wide.shape[0]))
    ax.set_yticklabels(wide.index, fontsize=8)
    ax.set_xlabel("Curated NK taxonomy marker program")
    ax.set_ylabel("Data-driven Leiden cluster")
    ax.set_title(cbar_label, fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.01)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    savefig(fig, fig_dir, name)


def plot_top_match_heatmap(long: pd.DataFrame, fig_dir: str) -> None:
    if long.empty:
        return
    best = (
        long.loc[long["weighted_support_score"].gt(0)]
        .sort_values(
            ["cluster", "taxonomy_layer", "weighted_support_score", "n_supporting_hits"],
            ascending=[True, True, False, False],
        )
        .groupby(["cluster", "taxonomy_layer"], sort=False)
        .head(1)
        .copy()
    )
    clusters = sorted(long["cluster"].unique(), key=cluster_sort_key)
    fig_h = max(4, 0.36 * len(clusters))
    fig, ax = plt.subplots(figsize=(9, fig_h))
    y = np.arange(len(clusters))
    offsets = {"subtype": -0.18, "state": 0.18}
    colors = {"subtype": "#4c78a8", "state": "#f58518"}
    handles = []
    for layer in ["subtype", "state"]:
        layer_best = best.loc[best["taxonomy_layer"].eq(layer)].set_index("cluster")
        values = []
        labels = []
        for cluster in clusters:
            if cluster in layer_best.index:
                row = layer_best.loc[cluster]
                values.append(float(row["weighted_support_score"]))
                labels.append(display_program_id(row["program_id"], include_layer=False))
            else:
                values.append(0.0)
                labels.append("")
        ax.barh(y + offsets[layer], values, height=0.32, color=colors[layer], alpha=0.9)
        handles.append(Patch(facecolor=colors[layer], label=layer_display_name(layer)))
        for yi, value, label in zip(y + offsets[layer], values, labels):
            if value <= 0:
                continue
            ax.text(value + 0.15, yi, label, va="center", ha="left", fontsize=6)
    ax.set_yticks(y)
    ax.set_yticklabels(clusters, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Weighted concordance score")
    ax.set_title("Top subtype and state marker matches per data-driven cluster", fontsize=12, fontweight="bold")
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    savefig(fig, fig_dir, "top_taxonomy_match_by_cluster")


def plot_recurrent_genes(
    recurrent: pd.DataFrame,
    fig_dir: str,
    n: int,
    *,
    min_recurrent_clusters: int = 1,
) -> None:
    if recurrent.empty:
        return
    filtered = recurrent.loc[recurrent["n_clusters_top_gene"].ge(min_recurrent_clusters)].copy()
    if filtered.empty:
        return
    top = filtered.head(n).iloc[::-1]
    colors = np.where(top["in_taxonomy_reference"], "#c27c00", "#555555")
    fig_h = max(5, 0.23 * len(top))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.barh(top["gene"], top["n_clusters_top_gene"], color=colors, alpha=0.9)
    ax.set_xlabel("Number of clusters where gene appears in top attribution genes")
    ax.set_title(
        f"Recurrent top attribution genes (>= {min_recurrent_clusters} clusters)",
        fontsize=12,
        fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor="#c27c00", label="In curated NK taxonomy"),
            Patch(facecolor="#555555", label="Not in curated NK taxonomy"),
        ],
        frameon=False,
        loc="lower right",
        fontsize=8,
    )
    fig.tight_layout()
    savefig(fig, fig_dir, "recurrent_top_attribution_genes")


def plot_taxonomy_tree(long: pd.DataFrame, entries: list[TaxonomyEntry], fig_dir: str) -> None:
    if long.empty or not entries:
        return
    score_by_program = long.groupby("program_id")["weighted_support_score"].max().to_dict()
    cluster_hits_by_program = (
        long.loc[long["n_supporting_hits"].gt(0)]
        .groupby("program_id")["cluster"]
        .nunique()
        .to_dict()
    )
    programs_by_layer: dict[str, list[str]] = {"subtype": [], "state": []}
    for entry in entries:
        layer = normalized_taxonomy_layer(entry)
        programs_by_layer[layer].append(program_id(entry))

    total_leaves = sum(len(values) for values in programs_by_layer.values())
    if total_leaves == 0:
        return

    leaf_gap = 1.0
    layer_gap = 1.8
    y_cursor = 0.0
    coords: dict[str, tuple[float, float]] = {}
    layer_centers: dict[str, float] = {}
    for layer in ["subtype", "state"]:
        programs = programs_by_layer[layer]
        start_y = y_cursor
        for program in programs:
            coords[program] = (2.0, y_cursor)
            y_cursor += leaf_gap
        if programs:
            layer_centers[layer] = (start_y + y_cursor - leaf_gap) / 2.0
            y_cursor += layer_gap

    root_y = np.mean(list(layer_centers.values())) if layer_centers else 0.0
    vmax = max([float(v) for v in score_by_program.values()] + [1.0])
    norm = plt.Normalize(vmin=0, vmax=vmax)
    cmap = plt.get_cmap("viridis")

    fig_h = max(6, 0.28 * total_leaves)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.scatter([0.0], [root_y], s=280, color="#222222", zorder=3)
    ax.text(-0.08, root_y, "Curated NK taxonomy", ha="right", va="center", fontsize=10, fontweight="bold")

    for layer, center in layer_centers.items():
        ax.plot([0.05, 1.0], [root_y, center], color="#999999", lw=1.2, zorder=1)
        ax.scatter([1.0], [center], s=230, color="#dddddd", edgecolor="#444444", zorder=3)
        ax.text(0.92, center, layer_display_name(layer), ha="right", va="center", fontsize=9, fontweight="bold")
        for program in programs_by_layer[layer]:
            _, y = coords[program]
            score = float(score_by_program.get(program, 0.0))
            n_clusters = int(cluster_hits_by_program.get(program, 0))
            size = 45 + 18 * n_clusters
            ax.plot([1.08, 1.9], [center, y], color="#c0c0c0", lw=0.8, zorder=1)
            ax.scatter([2.0], [y], s=size, color=cmap(norm(score)), edgecolor="#333333", linewidth=0.4, zorder=3)
            ax.text(
                2.08,
                y,
                display_program_id(program, include_layer=False),
                ha="left",
                va="center",
                fontsize=7,
            )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Best weighted concordance score across clusters")
    ax.set_xlim(-0.65, 4.2)
    ax.set_ylim(y_cursor - 0.8, -1.0)
    ax.axis("off")
    ax.set_title("Curated NK taxonomy tree with post hoc attribution concordance", fontsize=12, fontweight="bold")
    fig.tight_layout()
    savefig(fig, fig_dir, "taxonomy_tree_concordance")


def savefig(fig, fig_dir: str, name: str) -> None:
    png = os.path.join(fig_dir, f"{name}.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVE] {png}")


def program_id(entry: TaxonomyEntry) -> str:
    return f"{normalized_taxonomy_layer(entry)}:{entry.canonical_label or entry.name}".replace("  ", " ")


def normalized_taxonomy_layer(entry: TaxonomyEntry) -> str:
    label = str(entry.canonical_label or entry.name).strip()
    if label in STATE_LABELS:
        return "state"
    if label in SUBTYPE_LABELS:
        return "subtype"
    layer = str(entry.layer).strip().lower()
    if layer in {"subtype", "state"}:
        return layer
    return "state"


def display_program_id(program: str, *, include_layer: bool = False) -> str:
    text = str(program)
    if text.startswith("subtype:"):
        label = text.removeprefix("subtype:")
        return f"{layer_display_name('subtype')}: {label}" if include_layer else label
    if text.startswith("state:"):
        label = text.removeprefix("state:")
        return f"{layer_display_name('state')}: {label}" if include_layer else label
    return text


def layer_display_name(layer: str) -> str:
    if str(layer) == "subtype":
        return "Subtype/lineage"
    if str(layer) == "state":
        return "State"
    return str(layer)


def intersect(markers: list[str], genes: set[str]) -> list[str]:
    return [str(gene).upper() for gene in markers if str(gene).upper() in genes]


def unique(values) -> list[str]:
    seen = set()
    out = []
    for value in values:
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def join_genes(genes: list[str]) -> str:
    return "; ".join(unique(genes))


def cluster_sort_key(value: str):
    text = str(value)
    if text.startswith("Leiden_"):
        suffix = text.removeprefix("Leiden_")
        if suffix.isdigit():
            return (0, int(suffix))
    if text.isdigit():
        return (0, int(text))
    return (1, text)


if __name__ == "__main__":
    main()
