#!/usr/bin/env python3
"""Plot adaptive SIGnature-selected genes against curated NK taxonomy genes.

By default, this script uses the Leiden 0.5 adaptive 50%-attribution-mass gene
list, excludes clusters annotated as Non-NK, and restricts both sets to the
genes present in the 2,007-gene model AnnData. It saves a proportional Venn
diagram and a gene-level membership table.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from nk_project.annotation_agent.taxonomy_reference import load_taxonomy_entries


DEFAULT_ATTRIBUTION = PROJECT_ROOT / (
    "outputs/scanvi_leiden_0_5_agent_v2_cluster_labels/"
    "cellxgene_gene_attribution/signature_mass50_leiden_0_5/"
    "tables/embedding_attribution_mass_selected_genes.csv"
)
DEFAULT_MAPPING = PROJECT_ROOT / (
    "outputs/annotation_agent/assay_only_leiden_0_5_gpt5mini_v2/"
    "cluster_annotation_mapping.csv"
)
DEFAULT_TAXONOMY = PROJECT_ROOT / (
    "nk_project/annotation_agent/references/"
    "FINAL_UNIFIED_NK_TAXONOMY_REFERENCE_nolayer.md"
)
DEFAULT_H5AD = PROJECT_ROOT / (
    "outputs/annotation_agent/assay_only_leiden_0_5_gpt5mini_v2/"
    "full_scvi_leiden_refined_v1.h5ad"
)
DEFAULT_FIGURE = PROJECT_ROOT / (
    "outputs/scanvi_leiden_0_5_agent_v2_cluster_labels/"
    "cellxgene_gene_attribution/signature_mass50_leiden_0_5/"
    "figures/signature_mass50_vs_curated_nk_taxonomy_venn.png"
)
DEFAULT_MEMBERSHIP = PROJECT_ROOT / (
    "outputs/scanvi_leiden_0_5_agent_v2_cluster_labels/"
    "cellxgene_gene_attribution/signature_mass50_leiden_0_5/"
    "tables/signature_mass50_vs_curated_nk_taxonomy_membership.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attribution-csv", type=Path, default=DEFAULT_ATTRIBUTION)
    parser.add_argument("--mapping-csv", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--taxonomy-reference", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--model-h5ad", type=Path, default=DEFAULT_H5AD)
    parser.add_argument("--label-key", default="leiden_0_5")
    parser.add_argument("--annotation-column", default="final_structured_label")
    parser.add_argument("--non-nk-label", default="Non-NK")
    parser.add_argument(
        "--include-clusters",
        default=None,
        help=(
            "Optional comma-separated cluster IDs to include. When supplied, "
            "this explicit string-valued list replaces filtering through the "
            "annotation mapping file."
        ),
    )
    parser.add_argument("--output-png", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument(
        "--output-membership-csv", type=Path, default=DEFAULT_MEMBERSHIP
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def natural_key(value: str) -> tuple:
    return tuple(
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(value))
    )


def require_columns(frame: pd.DataFrame, columns: set[str], source: Path) -> None:
    missing = columns.difference(frame.columns)
    if missing:
        raise KeyError(f"{source} is missing columns: {sorted(missing)}")


def main() -> None:
    args = parse_args()

    try:
        from matplotlib_venn import venn2
    except ImportError as exc:
        raise ImportError(
            "This script requires matplotlib-venn. Install it with: "
            "python -m pip install matplotlib-venn"
        ) from exc

    print("[LOAD H5AD]", args.model_h5ad)
    adata = ad.read_h5ad(args.model_h5ad, backed="r")
    model_genes = {str(gene).upper() for gene in adata.var_names}
    adata.file.close()

    if args.include_clusters:
        nk_clusters = {
            cluster.strip()
            for cluster in str(args.include_clusters).split(",")
            if cluster.strip()
        }
        print("[CLUSTER FILTER] Explicit --include-clusters list")
    else:
        print("[LOAD MAPPING]", args.mapping_csv)
        mapping = pd.read_csv(args.mapping_csv, dtype={args.label_key: str})
        require_columns(
            mapping,
            {args.label_key, args.annotation_column},
            args.mapping_csv,
        )
        annotations = mapping[args.annotation_column].fillna("").astype(str).str.strip()
        nk_mask = annotations.ne("") & annotations.ne(args.non_nk_label)
        nk_clusters = set(mapping.loc[nk_mask, args.label_key].astype(str))
    if not nk_clusters:
        raise ValueError("No NK clusters remained after applying the annotation filter.")
    print(
        "[NK CLUSTERS]",
        len(nk_clusters),
        sorted(nk_clusters, key=natural_key),
    )

    print("[LOAD ATTRIBUTION]", args.attribution_csv)
    attribution = pd.read_csv(args.attribution_csv, dtype={"label": str})
    require_columns(attribution, {"label", "gene"}, args.attribution_csv)
    attribution["label"] = attribution["label"].astype(str)
    attribution["gene"] = attribution["gene"].astype(str).str.upper()
    nk_attribution = attribution.loc[attribution["label"].isin(nk_clusters)].copy()
    attribution_genes = set(nk_attribution["gene"]) & model_genes
    if not attribution_genes:
        raise ValueError("No selected attribution genes remained for the NK clusters.")

    print("[LOAD TAXONOMY]", args.taxonomy_reference)
    entries = load_taxonomy_entries(args.taxonomy_reference)
    if not entries:
        raise ValueError("No taxonomy entries were loaded.")
    taxonomy_genes = {
        str(gene).upper()
        for entry in entries
        for genes in entry.markers.values()
        for gene in genes
    }
    taxonomy_genes_in_model = taxonomy_genes & model_genes

    shared = attribution_genes & taxonomy_genes_in_model
    attribution_only = attribution_genes - taxonomy_genes_in_model
    taxonomy_only = taxonomy_genes_in_model - attribution_genes

    print("[ATTRIBUTION GENES]", len(attribution_genes))
    print("[TAXONOMY GENES IN MODEL]", len(taxonomy_genes_in_model))
    print("[ATTRIBUTION ONLY]", len(attribution_only))
    print("[SHARED]", len(shared))
    print("[TAXONOMY ONLY]", len(taxonomy_only))

    membership = pd.DataFrame(
        {"gene": sorted(attribution_genes | taxonomy_genes_in_model)}
    )
    membership["in_signature_mass50"] = membership["gene"].isin(
        attribution_genes
    )
    membership["in_curated_nk_taxonomy"] = membership["gene"].isin(
        taxonomy_genes_in_model
    )
    membership["overlap_category"] = "curated_taxonomy_only"
    membership.loc[
        membership["in_signature_mass50"], "overlap_category"
    ] = "signature_selected_only"
    membership.loc[
        membership["in_signature_mass50"]
        & membership["in_curated_nk_taxonomy"],
        "overlap_category",
    ] = "shared"

    args.output_membership_csv.parent.mkdir(parents=True, exist_ok=True)
    membership.to_csv(args.output_membership_csv, index=False)
    print("[SAVE]", args.output_membership_csv)

    fig, ax = plt.subplots(figsize=(9, 7))
    venn = venn2(
        subsets=(
            len(attribution_only),
            len(taxonomy_only),
            len(shared),
        ),
        set_labels=(
            "SIGnature-selected genes\n"
            f"NK clusters (n={len(attribution_genes)})",
            "Curated NK taxonomy genes\n"
            f"in model genes (n={len(taxonomy_genes_in_model)})",
        ),
        set_colors=("#7EA6D8", "#F2A766"),
        alpha=0.72,
        ax=ax,
    )
    for region_id in ("10", "01", "11"):
        label = venn.get_label_by_id(region_id)
        if label is not None:
            label.set_fontsize(16)
            label.set_fontweight("bold")
    for label in venn.set_labels:
        if label is not None:
            label.set_fontsize(13)
            label.set_fontweight("bold")

    resolution = args.label_key.removeprefix("leiden_").replace("_", ".")
    ax.set_title(
        f"Leiden {resolution} NK-cluster attribution genes vs "
        "curated NK taxonomy genes\n"
        "Adaptive 50% attribution-mass selection",
        fontsize=16,
        fontweight="bold",
        pad=18,
    )
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output_png,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("[SAVE]", args.output_png)


if __name__ == "__main__":
    main()
