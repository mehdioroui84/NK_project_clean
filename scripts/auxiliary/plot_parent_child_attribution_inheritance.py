#!/usr/bin/env python3
"""Compare adaptive SIGnature genes between Leiden 0.1 parents and 0.5 children.

The script supports both the original categorical inheritance heatmap and a
quantitative two-panel heatmap. The quantitative view shows normalized
attribution mass for the parent and children plus log2 child/parent attribution
change. Complete parent-child classifications for the pair-specific union of
the parent and child mass-50 gene sets, plus exact plotted matrices, are
written to CSV. The visualization limit affects figures only.

Categorical color meanings are fixed:
  green  = inherited (selected in both parent and child mass-50 sets)
  purple = child-emergent (selected in child, not selected in parent)
  white  = not selected in both clusters for that parent-child relationship

The detailed CSV continues to distinguish stable, increased, and decreased
inherited genes and parent-selected genes not retained by a child, even though
the simplified categorical heatmap does not display those distinctions.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from nk_project.annotation_agent.taxonomy_reference import load_taxonomy_entries


ATTRIBUTION_ROOT = PROJECT_ROOT / (
    "outputs/scanvi_leiden_0_5_agent_v2_cluster_labels/"
    "cellxgene_gene_attribution"
)
DEFAULT_PARENT_DIR = ATTRIBUTION_ROOT / "signature_mass50_leiden_0_1"
DEFAULT_CHILD_DIR = ATTRIBUTION_ROOT / "signature_mass50_leiden_0_5"
DEFAULT_H5AD = PROJECT_ROOT / (
    "outputs/annotation_agent/assay_only_leiden_0_5_gpt5mini_v2/"
    "full_scvi_leiden_refined_v1.h5ad"
)
DEFAULT_CHILD_MAPPING = PROJECT_ROOT / (
    "outputs/annotation_agent/assay_only_leiden_0_5_gpt5mini_v2/"
    "cluster_annotation_mapping.csv"
)
DEFAULT_TAXONOMY = PROJECT_ROOT / (
    "nk_project/annotation_agent/references/"
    "FINAL_UNIFIED_NK_TAXONOMY_REFERENCE_nolayer.md"
)

STATUS_COLORS = {
    "absent": "#FFFFFF",
    "inherited_stable": "#2CA25F",
    "inherited_increased": "#2CA25F",
    "inherited_decreased": "#2CA25F",
    "child_emergent": "#7B3294",
    "parent_not_retained": "#FFFFFF",
}
STATUS_CODES = {
    "absent": 0,
    "inherited_stable": 1,
    "inherited_increased": 2,
    "inherited_decreased": 3,
    "child_emergent": 4,
    "parent_not_retained": 5,
}
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent-selected-csv",
        type=Path,
        default=DEFAULT_PARENT_DIR / "tables/embedding_attribution_mass_selected_genes.csv",
    )
    parser.add_argument(
        "--parent-all-genes-csv",
        type=Path,
        default=DEFAULT_PARENT_DIR / "tables/embedding_attribution_per_label.csv",
    )
    parser.add_argument(
        "--child-selected-csv",
        type=Path,
        default=DEFAULT_CHILD_DIR / "tables/embedding_attribution_mass_selected_genes.csv",
    )
    parser.add_argument(
        "--child-all-genes-csv",
        type=Path,
        default=DEFAULT_CHILD_DIR / "tables/embedding_attribution_per_label.csv",
    )
    parser.add_argument("--hierarchy-h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--child-mapping-csv", type=Path, default=DEFAULT_CHILD_MAPPING)
    parser.add_argument("--taxonomy-reference", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--parent-key", default="leiden_0_1")
    parser.add_argument("--child-key", default="leiden_0_5")
    parser.add_argument(
        "--include-parents",
        default="",
        help=(
            "Optional comma-separated parent IDs. By default, include every "
            "Leiden 0.1 parent with a qualifying NK child."
        ),
    )
    parser.add_argument("--child-annotation-column", default="final_structured_label")
    parser.add_argument("--child-free-label-column", default="free_label")
    parser.add_argument("--non-nk-label", default="Non-NK")
    parser.add_argument(
        "--include-non-nk-children",
        action="store_true",
        help=(
            "Include every mapped Leiden 0.5 child, including B, T, myeloid, "
            "stromal, and other clusters labeled Non-NK."
        ),
    )
    parser.add_argument("--minimum-parent-fraction", type=float, default=0.01)
    parser.add_argument("--minimum-child-fraction", type=float, default=0.01)
    parser.add_argument(
        "--log2-change-threshold",
        type=float,
        default=1.0,
        help="Absolute log2 child/parent attribution-share threshold. Default 1 (twofold).",
    )
    parser.add_argument(
        "--top-genes-per-cluster",
        type=int,
        default=10,
        help=(
            "Show the union of the top N mass-selected genes from the parent "
            "and from each child. Default 10."
        ),
    )
    # Retained for compatibility with previously shared commands. Display-gene
    # selection is now controlled only by --top-genes-per-cluster.
    parser.add_argument("--parent-backbone-genes", type=int, default=8, help=argparse.SUPPRESS)
    parser.add_argument("--genes-per-status-per-child", type=int, default=3, help=argparse.SUPPRESS)
    parser.add_argument("--max-display-genes", type=int, default=50, help=argparse.SUPPRESS)
    parser.add_argument(
        "--plot-mode",
        choices=["categorical", "quantitative"],
        default="categorical",
        help=(
            "categorical preserves the original status heatmap; quantitative "
            "plots attribution mass and child/parent log2 change."
        ),
    )
    parser.add_argument(
        "--figure-scope",
        choices=["per-parent", "combined", "both"],
        default="per-parent",
        help=(
            "Write separate figures for each parent, one combined figure containing "
            "all parent-child edges, or both. Default: per-parent."
        ),
    )
    parser.add_argument(
        "--cluster-gene-order",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For the combined categorical figure, hierarchically cluster gene "
            "rows from their displayed inherited and emergent patterns across "
            "all parent-child edges."
        ),
    )
    parser.add_argument(
        "--annotate-values",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print numeric values inside quantitative heatmap tiles.",
    )
    parser.add_argument("--figure-dir", type=Path, default=DEFAULT_CHILD_DIR / "figures")
    parser.add_argument(
        "--matrix-dir",
        type=Path,
        default=DEFAULT_CHILD_DIR / "tables",
        help="Directory for exact per-parent heatmap matrices.",
    )
    parser.add_argument(
        "--gene-status-csv",
        type=Path,
        default=DEFAULT_CHILD_DIR / (
            "tables/attribution_parent_child_gene_relationship_mass50.csv"
        ),
    )
    parser.add_argument(
        "--edge-summary-csv",
        type=Path,
        default=DEFAULT_CHILD_DIR / "tables/attribution_parent_child_summary.csv",
    )
    parser.add_argument(
        "--pathway-gene-status-csv",
        type=Path,
        default=DEFAULT_CHILD_DIR
        / "tables/attribution_parent_child_pathway_gene_status_mass50.csv",
        help=(
            "Compact parent-child gene-status table intended for pathway "
            "enrichment analysis."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def natural_key(value: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(value))
    )


def require_columns(frame: pd.DataFrame, columns: set[str], path: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")


def read_attribution(path: Path) -> pd.DataFrame:
    print("[LOAD ATTRIBUTION]", path)
    frame = pd.read_csv(path, dtype={"label": str})
    require_columns(frame, {"label", "gene", "mean_abs_attr"}, path)
    frame = frame.copy()
    frame["label"] = frame["label"].astype(str)
    frame["gene"] = frame["gene"].astype(str).str.upper()
    frame["mean_abs_attr"] = pd.to_numeric(frame["mean_abs_attr"], errors="raise")
    frame = frame.sort_values(
        ["label", "mean_abs_attr", "gene"], ascending=[True, False, True]
    )
    if "rank" not in frame.columns:
        frame["rank"] = frame.groupby("label").cumcount() + 1
    else:
        frame["rank"] = pd.to_numeric(frame["rank"], errors="raise").astype(int)
    totals = frame.groupby("label")["mean_abs_attr"].transform("sum")
    frame["attribution_share"] = np.where(
        totals > 0, frame["mean_abs_attr"] / totals, 0.0
    )
    if frame.duplicated(["label", "gene"]).any():
        raise ValueError(f"Duplicate label/gene rows found in {path}")
    return frame


def read_selected(path: Path) -> dict[str, set[str]]:
    print("[LOAD SELECTED]", path)
    frame = pd.read_csv(path, dtype={"label": str})
    require_columns(frame, {"label", "gene"}, path)
    frame["label"] = frame["label"].astype(str)
    frame["gene"] = frame["gene"].astype(str).str.upper()
    return {
        str(label): set(group["gene"])
        for label, group in frame.groupby("label", observed=True)
    }


def build_taxonomy_maps(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    priority = {
        "core": 4,
        "support": 3,
        "context": 2,
        "negative_expected_low": 1,
    }
    gene_tiers: dict[str, tuple[int, str]] = {}
    gene_categories: dict[str, set[str]] = {}
    for entry in load_taxonomy_entries(path):
        category = entry.canonical_label or entry.name
        for tier, genes in entry.markers.items():
            for gene in genes:
                key = str(gene).upper()
                rank = priority.get(tier, 0)
                if rank > gene_tiers.get(key, (0, "not_in_taxonomy"))[0]:
                    gene_tiers[key] = (rank, tier)
                gene_categories.setdefault(key, set()).add(str(category))
    tier_map = {gene: value[1] for gene, value in gene_tiers.items()}
    category_map = {
        gene: "; ".join(sorted(categories))
        for gene, categories in gene_categories.items()
    }
    return tier_map, category_map


def significant_parent_child_flows(args: argparse.Namespace) -> pd.DataFrame:
    print("[LOAD HIERARCHY]", args.hierarchy_h5ad)
    adata = ad.read_h5ad(args.hierarchy_h5ad, backed="r")
    missing = [key for key in (args.parent_key, args.child_key) if key not in adata.obs]
    if missing:
        raise KeyError(f"Hierarchy h5ad is missing obs columns: {missing}")
    obs = adata.obs[[args.parent_key, args.child_key]].dropna().copy()
    adata.file.close()
    obs[args.parent_key] = obs[args.parent_key].astype(str)
    obs[args.child_key] = obs[args.child_key].astype(str)

    mapping = pd.read_csv(args.child_mapping_csv, dtype={args.child_key: str})
    require_columns(
        mapping,
        {args.child_key, args.child_annotation_column},
        args.child_mapping_csv,
    )
    annotation = mapping[args.child_annotation_column].fillna("").astype(str).str.strip()
    if mapping[args.child_key].duplicated().any():
        duplicates = sorted(
            mapping.loc[mapping[args.child_key].duplicated(False), args.child_key]
            .astype(str)
            .unique(),
            key=natural_key,
        )
        raise ValueError(f"Child mapping contains duplicate cluster IDs: {duplicates}")
    if args.include_non_nk_children:
        eligible_children = set(mapping[args.child_key].dropna().astype(str))
        analysis_scope = "all"
    else:
        eligible_children = set(
            mapping.loc[
                annotation.ne("") & annotation.ne(args.non_nk_label), args.child_key
            ].astype(str)
        )
        analysis_scope = "NK-only"
    included_parents = {
        value.strip()
        for value in str(args.include_parents).split(",")
        if value.strip()
    }

    flows = (
        obs.groupby([args.parent_key, args.child_key], observed=True)
        .size()
        .rename("flow_cells")
        .reset_index()
    )
    parent_totals = obs[args.parent_key].value_counts()
    child_totals = obs[args.child_key].value_counts()
    flows["parent_fraction"] = flows.apply(
        lambda row: row["flow_cells"] / parent_totals.loc[row[args.parent_key]], axis=1
    )
    flows["child_fraction"] = flows.apply(
        lambda row: row["flow_cells"] / child_totals.loc[row[args.child_key]], axis=1
    )
    parent_mask = (
        flows[args.parent_key].isin(included_parents)
        if included_parents
        else pd.Series(True, index=flows.index)
    )
    flows = flows.loc[
        parent_mask
        & flows[args.child_key].isin(eligible_children)
        & (flows["parent_fraction"] >= args.minimum_parent_fraction)
        & (flows["child_fraction"] >= args.minimum_child_fraction)
    ].copy()
    if flows.empty:
        raise ValueError("No parent-child flows passed the requested thresholds.")
    mapping_columns = [args.child_key, args.child_annotation_column]
    if args.child_free_label_column in mapping.columns:
        mapping_columns.append(args.child_free_label_column)
    flows = flows.merge(
        mapping[mapping_columns],
        on=args.child_key,
        how="left",
        validate="many_to_one",
    )
    flows = flows.rename(
        columns={
            args.child_annotation_column: "child_annotation",
            args.child_free_label_column: "child_free_label",
        }
    )
    if "child_free_label" not in flows.columns:
        flows["child_free_label"] = ""
    flows["child_annotation"] = flows["child_annotation"].fillna("").astype(str)
    flows["child_free_label"] = flows["child_free_label"].fillna("").astype(str)
    flows["parent_cluster"] = flows[args.parent_key].astype(str)
    flows["child_cluster"] = flows[args.child_key].astype(str)
    flows = flows.sort_values(
        ["parent_cluster", "parent_fraction"], ascending=[True, False]
    )
    print("[ANALYSIS SCOPE]", analysis_scope)
    print("[CHILDREN]", sorted(flows["child_cluster"].unique(), key=natural_key))
    print("[PARENTS]", sorted(flows["parent_cluster"].unique(), key=natural_key))
    return flows


def label_lookup(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    sub = frame.loc[frame["label"].eq(str(label))].copy()
    if sub.empty:
        raise KeyError(f"Attribution table has no label {label!r}")
    return sub.set_index("gene", drop=False)


def scalar(lookup: pd.DataFrame, gene: str, column: str, default=np.nan):
    if gene not in lookup.index:
        return default
    value = lookup.loc[gene, column]
    return value.iloc[0] if isinstance(value, pd.Series) else value


def classify_gene(
    parent_selected: bool,
    child_selected: bool,
    log2_change: float,
    threshold: float,
) -> str:
    if not parent_selected and not child_selected:
        return "absent"
    if parent_selected and child_selected:
        if log2_change >= threshold:
            return "inherited_increased"
        if log2_change <= -threshold:
            return "inherited_decreased"
        return "inherited_stable"
    if child_selected:
        return "child_emergent"
    return "parent_not_retained"


def build_gene_and_edge_tables(
    flows: pd.DataFrame,
    parent_all: pd.DataFrame,
    child_all: pd.DataFrame,
    parent_selected: dict[str, set[str]],
    child_selected: dict[str, set[str]],
    tier_map: dict[str, str],
    category_map: dict[str, str],
    threshold: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    gene_rows = []
    summary_rows = []
    epsilon = 1e-12

    # Internally retain the parent plus all sibling-child mass-50 genes for
    # every edge. This rectangular representation is needed by the heatmaps;
    # pair-absent rows are removed from the detailed CSV before export.
    parent_gene_unions: dict[str, set[str]] = {}
    for flow in flows.itertuples(index=False):
        parent = str(flow.parent_cluster)
        child = str(flow.child_cluster)
        parent_gene_unions.setdefault(parent, set()).update(
            parent_selected.get(parent, set())
        )
        parent_gene_unions[parent].update(child_selected.get(child, set()))

    for flow in flows.itertuples(index=False):
        parent = str(flow.parent_cluster)
        child = str(flow.child_cluster)
        parent_lookup = label_lookup(parent_all, parent)
        child_lookup = label_lookup(child_all, child)
        p_selected = parent_selected.get(parent, set())
        c_selected = child_selected.get(child, set())
        selected_union = parent_gene_unions[parent]
        edge_selected_union = p_selected | c_selected
        shared = p_selected & c_selected

        # Attribution across all genes is used only for the weighted-overlap
        # summary metric.
        all_genes = set(parent_lookup.index) | set(child_lookup.index)
        p_vector = np.array(
            [float(scalar(parent_lookup, gene, "attribution_share", 0.0)) for gene in all_genes]
        )
        c_vector = np.array(
            [float(scalar(child_lookup, gene, "attribution_share", 0.0)) for gene in all_genes]
        )
        weighted_denominator = float(np.maximum(p_vector, c_vector).sum())
        weighted_overlap = (
            float(np.minimum(p_vector, c_vector).sum()) / weighted_denominator
            if weighted_denominator > 0
            else np.nan
        )

        for gene in sorted(selected_union):
            p_share = float(scalar(parent_lookup, gene, "attribution_share", 0.0))
            c_share = float(scalar(child_lookup, gene, "attribution_share", 0.0))
            log2_change = float(np.log2((c_share + epsilon) / (p_share + epsilon)))
            p_is_selected = gene in p_selected
            c_is_selected = gene in c_selected
            status = classify_gene(
                p_is_selected, c_is_selected, log2_change, threshold
            )
            gene_rows.append(
                {
                    "parent_cluster": parent,
                    "child_cluster": child,
                    "child_annotation": str(flow.child_annotation),
                    "child_free_label": str(flow.child_free_label),
                    "flow_cells": int(flow.flow_cells),
                    "parent_fraction": float(flow.parent_fraction),
                    "child_fraction": float(flow.child_fraction),
                    "gene": gene,
                    "parent_selected_mass50": p_is_selected,
                    "child_selected_mass50": c_is_selected,
                    "parent_rank": scalar(parent_lookup, gene, "rank"),
                    "child_rank": scalar(child_lookup, gene, "rank"),
                    "parent_attribution_share": p_share,
                    "child_attribution_share": c_share,
                    "log2_child_parent_attribution_change": log2_change,
                    "gene_status": status,
                    "taxonomy_tier": tier_map.get(gene, "not_in_taxonomy"),
                    "taxonomy_categories": category_map.get(gene, ""),
                    "displayed_in_parent_heatmap": False,
                }
            )

        union_size = len(edge_selected_union)
        summary_rows.append(
            {
                "parent_cluster": parent,
                "child_cluster": child,
                "child_annotation": str(flow.child_annotation),
                "child_free_label": str(flow.child_free_label),
                "flow_cells": int(flow.flow_cells),
                "parent_fraction": float(flow.parent_fraction),
                "child_fraction": float(flow.child_fraction),
                "n_parent_selected": len(p_selected),
                "n_child_selected": len(c_selected),
                "n_shared": len(shared),
                "n_parent_not_retained": len(p_selected - c_selected),
                "n_child_emergent": len(c_selected - p_selected),
                "parent_retention_fraction": (
                    len(shared) / len(p_selected) if p_selected else np.nan
                ),
                "child_inheritance_fraction": (
                    len(shared) / len(c_selected) if c_selected else np.nan
                ),
                "child_novelty_fraction": (
                    len(c_selected - p_selected) / len(c_selected)
                    if c_selected
                    else np.nan
                ),
                "jaccard_selected_gene_overlap": (
                    len(shared) / union_size if union_size else np.nan
                ),
                "weighted_attribution_overlap": weighted_overlap,
            }
        )

    genes_df = pd.DataFrame(gene_rows)
    summary_df = pd.DataFrame(summary_rows)
    sibling_counts = (
        genes_df.loc[genes_df["child_selected_mass50"]]
        .groupby(["parent_cluster", "gene"])["child_cluster"]
        .nunique()
        .rename("n_sibling_children_selected")
        .reset_index()
    )
    genes_df = genes_df.merge(
        sibling_counts, on=["parent_cluster", "gene"], how="left"
    )
    genes_df["n_sibling_children_selected"] = (
        genes_df["n_sibling_children_selected"].fillna(0).astype(int)
    )
    return genes_df, summary_df


def choose_display_genes(
    parent_detail: pd.DataFrame,
    children: list[str],
    top_genes_per_cluster: int,
) -> list[str]:
    """Return the ordered union of each cluster's top mass-selected genes."""
    parent_rows = parent_detail.loc[parent_detail["parent_selected_mass50"]].copy()
    parent_top = (
        parent_rows.sort_values(["parent_rank", "gene"])
        .drop_duplicates("gene")
        .head(top_genes_per_cluster)["gene"]
        .astype(str)
        .tolist()
    )
    selected = list(parent_top)
    seen = set(selected)

    # Children are already ordered by decreasing fraction of the parent. Append
    # each child's top-ranked mass-selected genes and remove duplicates while
    # preserving this parent-then-children ordering.
    for child in children:
        child_top = (
            parent_detail.loc[
                parent_detail["child_cluster"].eq(child)
                & parent_detail["child_selected_mass50"]
            ]
            .sort_values(["child_rank", "gene"])
            .head(top_genes_per_cluster)["gene"]
            .astype(str)
            .tolist()
        )
        for gene in child_top:
            if gene not in seen:
                selected.append(gene)
                seen.add(gene)

    # Arrange rows for biological readability without changing which genes are
    # displayed: broadly inherited genes first, parent-only/not-retained genes
    # next, and child-emergent genes last.
    row_order = {}
    for gene in selected:
        rows = parent_detail.loc[parent_detail["gene"].eq(gene)].copy()
        inherited_count = int(rows["gene_status"].str.startswith("inherited_").sum())
        parent_is_selected = bool(rows["parent_selected_mass50"].any())
        emergent_count = int(rows["gene_status"].eq("child_emergent").sum())
        parent_rank = pd.to_numeric(rows["parent_rank"], errors="coerce").min()
        child_selected_rows = rows.loc[rows["child_selected_mass50"]]
        best_child_rank = pd.to_numeric(
            child_selected_rows["child_rank"], errors="coerce"
        ).min()
        parent_rank = float(parent_rank) if pd.notna(parent_rank) else np.inf
        best_child_rank = (
            float(best_child_rank) if pd.notna(best_child_rank) else np.inf
        )

        if inherited_count > 0:
            row_order[gene] = (0, -inherited_count, parent_rank, gene)
        elif parent_is_selected:
            row_order[gene] = (1, 0, parent_rank, gene)
        else:
            row_order[gene] = (2, -emergent_count, best_child_rank, gene)

    return sorted(selected, key=lambda gene: row_order[gene])


def broad_child_identity(annotation: str, free_label: str) -> str:
    """Return a compact display identity while preserving full labels in CSV."""
    annotation_text = str(annotation).strip()
    free_text = str(free_label).strip().lower()
    if annotation_text and annotation_text != "Non-NK":
        return "NK"
    if "b_cell" in free_text or "b cells" in free_text:
        return "B cell"
    if "t_cell" in free_text or "t cells" in free_text:
        return "T cell"
    if any(token in free_text for token in ("myeloid", "monocyte", "macrophage")):
        return "Myeloid"
    if any(token in free_text for token in ("stromal", "epithelial", "trophoblast")):
        return "Stromal/epithelial"
    if any(token in free_text for token in ("erythroid", "hepatocyte")):
        return "Other/contaminant"
    return "Non-NK"


def plot_parent_heatmap(
    parent: str,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    display_genes: list[str],
    top_genes_per_cluster: int,
    figure_dir: Path,
    dpi: int,
) -> None:
    edge_summary = summary.loc[summary["parent_cluster"].eq(parent)].sort_values(
        "parent_fraction", ascending=False
    )
    children = edge_summary["child_cluster"].astype(str).tolist()
    matrix = np.zeros((len(display_genes), len(children)), dtype=int)

    status_lookup = {
        (str(row.child_cluster), str(row.gene)): str(row.gene_status)
        for row in detail.itertuples(index=False)
    }
    for row_index, gene in enumerate(display_genes):
        for col_index, child in enumerate(children):
            status = status_lookup.get((child, gene), "absent")
            matrix[row_index, col_index] = STATUS_CODES[status]

    cmap = ListedColormap(
        [
            STATUS_COLORS["absent"],
            STATUS_COLORS["inherited_stable"],
            STATUS_COLORS["inherited_increased"],
            STATUS_COLORS["inherited_decreased"],
            STATUS_COLORS["child_emergent"],
            STATUS_COLORS["parent_not_retained"],
        ]
    )
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    fig_width = max(9.0, 1.55 * len(children) + 5.0)
    fig_height = max(7.0, 0.29 * len(display_genes) + 3.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="none")

    ylabels = display_genes
    xlabels = [f"Child {child}" for child in children]
    ax.set_xticks(np.arange(len(children)), labels=xlabels, fontsize=10, fontweight="bold")
    ax.set_yticks(np.arange(len(display_genes)), labels=ylabels, fontsize=9)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax.set_xticks(np.arange(-0.5, len(children), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(display_genes), 1), minor=True)
    ax.grid(which="minor", color="#BDBDBD", linewidth=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=STATUS_COLORS["inherited_stable"], label="Inherited"),
        Patch(facecolor=STATUS_COLORS["child_emergent"], label="Child-emergent"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.035),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    ax.set_title(
        f"Leiden 0.1 parent {parent}: attribution inheritance across Leiden 0.5 children\n"
        f"Union of top {top_genes_per_cluster} genes per cluster "
        f"({len(display_genes)} unique genes)",
        fontsize=14,
        fontweight="bold",
        pad=18,
    )
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"parent_{parent}_attribution_inheritance_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[SAVE]", path)


def choose_combined_display_genes(
    detail: pd.DataFrame,
    top_genes_per_cluster: int,
) -> list[str]:
    """Return the ordered union of top genes from every parent and child."""
    selected: set[str] = set()

    for parent in sorted(detail["parent_cluster"].unique(), key=natural_key):
        parent_top = (
            detail.loc[
                detail["parent_cluster"].eq(parent)
                & detail["parent_selected_mass50"]
            ]
            .sort_values(["parent_rank", "gene"])
            .drop_duplicates("gene")
            .head(top_genes_per_cluster)["gene"]
        )
        selected.update(parent_top.astype(str))

    for child in sorted(detail["child_cluster"].unique(), key=natural_key):
        child_top = (
            detail.loc[
                detail["child_cluster"].eq(child)
                & detail["child_selected_mass50"]
            ]
            .sort_values(["child_rank", "gene"])
            .drop_duplicates("gene")
            .head(top_genes_per_cluster)["gene"]
        )
        selected.update(child_top.astype(str))

    row_order: dict[str, tuple] = {}
    for gene in selected:
        rows = detail.loc[detail["gene"].eq(gene)]
        inherited_count = int(rows["gene_status"].str.startswith("inherited_").sum())
        lost_count = int(rows["gene_status"].eq("parent_not_retained").sum())
        emergent_count = int(rows["gene_status"].eq("child_emergent").sum())
        parent_selected = bool(rows["parent_selected_mass50"].any())
        parent_rank = pd.to_numeric(rows["parent_rank"], errors="coerce").min()
        child_rank = pd.to_numeric(rows["child_rank"], errors="coerce").min()
        parent_rank = float(parent_rank) if pd.notna(parent_rank) else np.inf
        child_rank = float(child_rank) if pd.notna(child_rank) else np.inf

        if inherited_count:
            row_order[gene] = (0, -inherited_count, parent_rank, child_rank, gene)
        elif parent_selected:
            row_order[gene] = (1, -lost_count, parent_rank, child_rank, gene)
        else:
            row_order[gene] = (2, -emergent_count, child_rank, gene)

    return sorted(selected, key=lambda gene: row_order[gene])


def plot_combined_categorical_heatmap(
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    top_genes_per_cluster: int,
    figure_dir: Path,
    dpi: int,
    cluster_gene_order: bool,
) -> list[str]:
    """Plot all qualifying parent-child edges in one categorical matrix."""
    genes = choose_combined_display_genes(detail, top_genes_per_cluster)
    edges = summary.copy()
    parent_order = {
        parent: index
        for index, parent in enumerate(
            sorted(edges["parent_cluster"].astype(str).unique(), key=natural_key)
        )
    }
    edges["_parent_order"] = edges["parent_cluster"].astype(str).map(parent_order)
    edges = edges.sort_values(
        ["_parent_order", "parent_fraction", "child_cluster"],
        ascending=[True, False, True],
    ).drop(columns="_parent_order")
    edge_pairs = list(
        edges[["parent_cluster", "child_cluster"]].itertuples(index=False, name=None)
    )

    matrix = np.zeros((len(genes), len(edge_pairs)), dtype=int)
    status_lookup = {
        (str(row.parent_cluster), str(row.child_cluster), str(row.gene)): str(row.gene_status)
        for row in detail.itertuples(index=False)
    }
    for row_index, gene in enumerate(genes):
        for col_index, (parent, child) in enumerate(edge_pairs):
            status = status_lookup.get((str(parent), str(child), gene), "absent")
            matrix[row_index, col_index] = STATUS_CODES[status]

    if cluster_gene_order and len(genes) > 1:
        # Collapse the three inherited subtypes because they share one visual
        # category. Not-retained and absent are both white in this simplified
        # display, so ordering is based only on visible green/purple patterns.
        categorical = np.zeros_like(matrix)
        categorical[np.isin(matrix, [1, 2, 3])] = 1  # inherited
        categorical[matrix == 4] = 2                 # child-emergent
        features = np.concatenate(
            [(categorical == code).astype(np.uint8) for code in (1, 2)],
            axis=1,
        )
        distances = pdist(features, metric="jaccard")
        if np.isfinite(distances).all() and np.any(distances > 0):
            order = leaves_list(
                linkage(distances, method="average", optimal_ordering=True)
            )
            matrix = matrix[order, :]
            genes = [genes[index] for index in order]

    cmap = ListedColormap(
        [
            STATUS_COLORS["absent"],
            STATUS_COLORS["inherited_stable"],
            STATUS_COLORS["inherited_increased"],
            STATUS_COLORS["inherited_decreased"],
            STATUS_COLORS["child_emergent"],
            STATUS_COLORS["parent_not_retained"],
        ]
    )
    norm = BoundaryNorm(np.arange(-0.5, 6.5, 1), cmap.N)
    fig_width = max(18.0, 0.48 * len(edge_pairs) + 8.0)
    fig_height = max(10.0, 0.22 * len(genes) + 4.0)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto", interpolation="none")

    labels = [f"P{parent}→C{child}" for parent, child in edge_pairs]
    ax.set_xticks(np.arange(len(edge_pairs)), labels=labels, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(genes)), labels=genes, fontsize=7)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    ax.set_xticks(np.arange(-0.5, len(edge_pairs), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(genes), 1), minor=True)
    ax.grid(which="minor", color="#D0D0D0", linewidth=0.30)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    parents = [str(parent) for parent, _ in edge_pairs]
    for index in range(1, len(parents)):
        if parents[index] != parents[index - 1]:
            ax.axvline(index - 0.5, color="#303030", linewidth=1.3)

    legend_handles = [
        Patch(facecolor=STATUS_COLORS["inherited_stable"], label="Inherited"),
        Patch(facecolor=STATUS_COLORS["child_emergent"], label="Child-emergent"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.025),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    ax.set_title(
        "Attribution inheritance across all Leiden 0.1→0.5 parent-child edges\n"
        f"Union of top {top_genes_per_cluster} genes per parent and child "
        f"({len(genes)} unique genes; "
        f"gene order: {'hierarchical clustering' if cluster_gene_order else 'status/rank'})",
        fontsize=15,
        fontweight="bold",
        pad=18,
    )
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / (
        f"all_parent_child_top{top_genes_per_cluster}_attribution_inheritance_heatmap.png"
    )
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[SAVE]", path)
    return genes


def build_quantitative_matrices(
    parent: str,
    detail: pd.DataFrame,
    children: list[str],
    display_genes: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return exact plotted attribution, change, and status matrices."""
    absolute = pd.DataFrame(
        index=display_genes,
        columns=[f"parent_{parent}"] + [f"child_{child}" for child in children],
        dtype=float,
    )
    change = pd.DataFrame(
        index=display_genes,
        columns=[f"child_{child}" for child in children],
        dtype=float,
    )
    status = pd.DataFrame(
        index=display_genes,
        columns=[f"child_{child}" for child in children],
        dtype=object,
    )

    for gene in display_genes:
        gene_rows = detail.loc[detail["gene"].eq(gene)]
        if gene_rows.empty:
            raise KeyError(f"No parent-child data available for displayed gene {gene!r}")
        absolute.loc[gene, f"parent_{parent}"] = (
            100 * float(gene_rows["parent_attribution_share"].iloc[0])
        )
        for child in children:
            match = gene_rows.loc[gene_rows["child_cluster"].eq(child)]
            if len(match) != 1:
                raise ValueError(
                    f"Expected one row for parent {parent}, child {child}, gene {gene}; "
                    f"found {len(match)}"
                )
            row = match.iloc[0]
            absolute.loc[gene, f"child_{child}"] = (
                100 * float(row["child_attribution_share"])
            )
            change.loc[gene, f"child_{child}"] = float(
                row["log2_child_parent_attribution_change"]
            )
            status.loc[gene, f"child_{child}"] = str(row["gene_status"])

    absolute.index.name = "gene"
    change.index.name = "gene"
    status.index.name = "gene"
    return absolute, change, status


def save_quantitative_matrices(
    parent: str,
    absolute: pd.DataFrame,
    change: pd.DataFrame,
    status: pd.DataFrame,
    matrix_dir: Path,
) -> None:
    matrix_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        matrix_dir / f"parent_{parent}_mass50_attribution_share_percent_matrix.csv": absolute,
        matrix_dir / f"parent_{parent}_mass50_log2_child_parent_matrix.csv": change,
        matrix_dir / f"parent_{parent}_mass50_gene_status_matrix.csv": status,
    }
    for path, frame in outputs.items():
        frame.to_csv(path)
        print("[SAVE]", path)


def _format_attribution_percent(value: float) -> str:
    if value >= 1:
        return f"{value:.1f}"
    if value >= 0.1:
        return f"{value:.2f}"
    if value >= 0.01:
        return f"{value:.3f}"
    return f"{value:.1e}"


def plot_parent_quantitative_heatmap(
    parent: str,
    detail: pd.DataFrame,
    summary: pd.DataFrame,
    display_genes: list[str],
    figure_dir: Path,
    matrix_dir: Path,
    dpi: int,
    annotate_values: bool,
) -> None:
    """Plot normalized attribution and child/parent changes for one parent."""
    edge_summary = summary.loc[summary["parent_cluster"].eq(parent)].sort_values(
        "parent_fraction", ascending=False
    )
    children = edge_summary["child_cluster"].astype(str).tolist()
    absolute, change, status = build_quantitative_matrices(
        parent, detail, children, display_genes
    )
    save_quantitative_matrices(parent, absolute, change, status, matrix_dir)

    absolute_values = absolute.to_numpy(dtype=float)
    change_values = change.to_numpy(dtype=float)
    positive = absolute_values[np.isfinite(absolute_values) & (absolute_values > 0)]
    if not positive.size:
        raise ValueError(f"Parent {parent} has no positive attribution values to plot.")
    absolute_vmin = max(float(np.nanpercentile(positive, 2)), 1e-12)
    absolute_vmax = float(np.nanmax(positive))
    if absolute_vmax <= absolute_vmin:
        absolute_vmax = absolute_vmin * 10

    finite_changes = change_values[np.isfinite(change_values)]
    change_limit = max(
        1.0,
        min(
            4.0,
            float(np.nanpercentile(np.abs(finite_changes), 95))
            if finite_changes.size
            else 1.0,
        ),
    )
    plotted_changes = np.clip(change_values, -change_limit, change_limit)

    fig_width = max(12.0, 1.35 * (absolute.shape[1] + change.shape[1]) + 4.5)
    fig_height = max(7.5, 0.30 * len(display_genes) + 4.5)
    fig, (ax_absolute, ax_change) = plt.subplots(
        ncols=2,
        figsize=(fig_width, fig_height),
        sharey=True,
        gridspec_kw={
            "width_ratios": [absolute.shape[1], max(1, change.shape[1])],
            "wspace": 0.08,
        },
    )

    absolute_cmap = plt.get_cmap("viridis").copy()
    absolute_cmap.set_bad("#F2F2F2")
    absolute_masked = np.ma.masked_invalid(absolute_values)
    image_absolute = ax_absolute.imshow(
        absolute_masked,
        cmap=absolute_cmap,
        norm=LogNorm(vmin=absolute_vmin, vmax=absolute_vmax),
        aspect="auto",
        interpolation="none",
    )
    image_change = ax_change.imshow(
        plotted_changes,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-change_limit, vcenter=0, vmax=change_limit),
        aspect="auto",
        interpolation="none",
    )

    child_summary = edge_summary.set_index("child_cluster")
    child_labels = [
        f"Child {child}\n"
        f"{broad_child_identity(child_summary.loc[child, 'child_annotation'], child_summary.loc[child, 'child_free_label'])}\n"
        f"{100 * float(child_summary.loc[child, 'parent_fraction']):.1f}% of parent"
        for child in children
    ]
    ax_absolute.set_xticks(
        np.arange(absolute.shape[1]),
        labels=[f"Parent {parent}"] + child_labels,
    )
    ax_change.set_xticks(np.arange(change.shape[1]), labels=child_labels)

    parent_selected_genes = set(
        detail.loc[detail["parent_selected_mass50"], "gene"].astype(str)
    )
    emergent_genes = set(
        detail.loc[detail["gene_status"].eq("child_emergent"), "gene"].astype(str)
    )
    ylabels = []
    for gene in display_genes:
        if gene in parent_selected_genes:
            ylabels.append(f"● {gene}")
        elif gene in emergent_genes:
            ylabels.append(f"◆ {gene}")
        else:
            ylabels.append(f"  {gene}")
    ax_absolute.set_yticks(np.arange(len(display_genes)), labels=ylabels)
    for tick, gene in zip(ax_absolute.get_yticklabels(), display_genes):
        if gene in emergent_genes and gene not in parent_selected_genes:
            tick.set_color(STATUS_COLORS["child_emergent"])
            tick.set_fontweight("bold")

    for axis, n_columns in (
        (ax_absolute, absolute.shape[1]),
        (ax_change, change.shape[1]),
    ):
        axis.tick_params(
            axis="x",
            top=True,
            labeltop=True,
            bottom=False,
            labelbottom=False,
            labelsize=9,
            length=0,
        )
        axis.tick_params(axis="y", labelsize=8.5, length=0)
        axis.set_xticks(np.arange(-0.5, n_columns, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(display_genes), 1), minor=True)
        axis.grid(which="minor", color="#D0D0D0", linewidth=0.45)
        axis.tick_params(which="minor", bottom=False, left=False)
        for spine in axis.spines.values():
            spine.set_visible(False)
    ax_change.tick_params(axis="y", labelleft=False)

    if annotate_values:
        for row_index in range(absolute.shape[0]):
            for col_index in range(absolute.shape[1]):
                value = absolute_values[row_index, col_index]
                ax_absolute.text(
                    col_index,
                    row_index,
                    _format_attribution_percent(value),
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="white" if value < np.sqrt(absolute_vmin * absolute_vmax) else "black",
                )
            for col_index in range(change.shape[1]):
                value = change_values[row_index, col_index]
                ax_change.text(
                    col_index,
                    row_index,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=5.8,
                    color="white" if abs(value) > 0.7 * change_limit else "black",
                )

    ax_absolute.set_title(
        "Attribution mass (%)\nlogarithmic color scale",
        fontsize=12,
        fontweight="bold",
        pad=18,
    )
    ax_change.set_title(
        "Change relative to parent\nlog2(child share / parent share)",
        fontsize=12,
        fontweight="bold",
        pad=18,
    )
    colorbar_absolute = fig.colorbar(
        image_absolute, ax=ax_absolute, fraction=0.035, pad=0.025
    )
    colorbar_absolute.set_label("Attribution mass (%)", fontsize=9)
    colorbar_change = fig.colorbar(
        image_change, ax=ax_change, fraction=0.05, pad=0.025
    )
    colorbar_change.set_label("log2 child/parent", fontsize=9)
    fig.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="black", markerfacecolor="black",
                linestyle="None", markersize=5, label="Selected in parent mass-50 set",
            ),
            Line2D(
                [0], [0], marker="D", color=STATUS_COLORS["child_emergent"],
                markerfacecolor=STATUS_COLORS["child_emergent"], linestyle="None",
                markersize=5, label="Child-emergent mass-50 gene",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle(
        f"Leiden 0.1 parent {parent}: quantitative SIGnature inheritance",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    fig.subplots_adjust(top=0.88, bottom=0.10, wspace=0.12)
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / f"parent_{parent}_mass50_quantitative_attribution_heatmap.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("[SAVE]", path)


def main() -> None:
    args = parse_args()
    if args.log2_change_threshold <= 0:
        raise ValueError("--log2-change-threshold must be positive.")
    if args.top_genes_per_cluster < 1:
        raise ValueError("--top-genes-per-cluster must be at least 1.")

    parent_all = read_attribution(args.parent_all_genes_csv)
    child_all = read_attribution(args.child_all_genes_csv)
    parent_selected = read_selected(args.parent_selected_csv)
    child_selected = read_selected(args.child_selected_csv)
    flows = significant_parent_child_flows(args)
    tier_map, category_map = build_taxonomy_maps(args.taxonomy_reference)
    detail, summary = build_gene_and_edge_tables(
        flows,
        parent_all,
        child_all,
        parent_selected,
        child_selected,
        tier_map,
        category_map,
        args.log2_change_threshold,
    )

    if args.figure_scope in {"per-parent", "both"}:
        for parent in sorted(detail["parent_cluster"].unique(), key=natural_key):
            parent_detail = detail.loc[detail["parent_cluster"].eq(parent)].copy()
            parent_summary = summary.loc[summary["parent_cluster"].eq(parent)].copy()
            children = (
                parent_summary.sort_values("parent_fraction", ascending=False)["child_cluster"]
                .astype(str)
                .tolist()
            )
            display_genes = choose_display_genes(
                parent_detail,
                children,
                args.top_genes_per_cluster,
            )
            detail.loc[
                detail["parent_cluster"].eq(parent)
                & detail["gene"].isin(display_genes),
                "displayed_in_parent_heatmap",
            ] = True
            if args.plot_mode == "quantitative":
                plot_parent_quantitative_heatmap(
                    parent,
                    parent_detail,
                    parent_summary,
                    display_genes,
                    args.figure_dir,
                    args.matrix_dir,
                    args.dpi,
                    args.annotate_values,
                )
            else:
                plot_parent_heatmap(
                    parent,
                    parent_detail,
                    parent_summary,
                    display_genes,
                    args.top_genes_per_cluster,
                    args.figure_dir,
                    args.dpi,
                )

    if args.figure_scope in {"combined", "both"}:
        if args.plot_mode != "categorical":
            raise ValueError(
                "The combined figure currently supports --plot-mode categorical only."
            )
        combined_genes = plot_combined_categorical_heatmap(
            detail,
            summary,
            args.top_genes_per_cluster,
            args.figure_dir,
            args.dpi,
            args.cluster_gene_order,
        )
        detail.loc[detail["gene"].isin(combined_genes), "displayed_in_parent_heatmap"] = True

    args.gene_status_csv.parent.mkdir(parents=True, exist_ok=True)
    args.edge_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    args.pathway_gene_status_csv.parent.mkdir(parents=True, exist_ok=True)
    # Export only the pair-specific union: every row must be selected in the
    # parent mass-50 set, the child mass-50 set, or both. Rows needed solely to
    # make sibling heatmaps rectangular are intentionally excluded.
    detail_export = detail.loc[
        detail["parent_selected_mass50"] | detail["child_selected_mass50"]
    ].copy()
    pathway_export = detail_export[
        ["parent_cluster", "child_cluster", "gene", "gene_status"]
    ].copy()
    detail_export.to_csv(args.gene_status_csv, index=False)
    pathway_export.to_csv(args.pathway_gene_status_csv, index=False)
    summary.to_csv(args.edge_summary_csv, index=False)
    print("[SAVE]", args.gene_status_csv, len(detail_export), "rows")
    print("[SAVE]", args.pathway_gene_status_csv, len(pathway_export), "rows")
    print("[SAVE]", args.edge_summary_csv, len(summary), "rows")


if __name__ == "__main__":
    main()
