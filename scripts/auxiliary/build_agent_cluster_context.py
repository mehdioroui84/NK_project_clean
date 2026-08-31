#!/usr/bin/env python
"""Build compact Leiden cluster-context JSON files for the annotation agent.

The LLM-facing files contain only cluster evidence. Source paths, hashes, the
generation date, and processing rules are written to a separate manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESOLUTIONS = {
    "leiden_0_1": {
        "worksheet": "outputs/leiden_discovery_assay_only/full_leiden_0_1_annotation_worksheet.csv",
        "raw_pathway": "data/annotation_agent/pathway_enrichment/go_bp_enrichment_leiden01.csv",
        "raw_mitochondrial": "data/annotation_agent/cluster_qc/leiden_0_1_dataset_mt_percentage_summary.tsv",
        "attribution_side": "parent",
    },
    "leiden_0_5": {
        "worksheet": "outputs/leiden_discovery_assay_only/full_leiden_0_5_annotation_worksheet.csv",
        "raw_pathway": "data/annotation_agent/pathway_enrichment/go_bp_enrichment_all_leiden05.csv",
        "raw_mitochondrial": "data/annotation_agent/cluster_qc/leiden_0_5_dataset_mt_percentage_summary.tsv",
        "attribution_side": "child",
    },
}

ATTRIBUTION_RELATIONSHIP = (
    "data/annotation_agent/gene_attribution/"
    "attribution_parent_child_gene_relationship_mass50.csv"
)

WORKSHEET_REQUIRED_COLUMNS = {
    "n_cells",
    "top_tissue",
    "top_tissue_frac",
    "top_dataset_id",
    "top_dataset_id_frac",
    "top_assay_clean",
    "top_assay_clean_frac",
}
PATHWAY_REQUIRED_COLUMNS = {"cluster", "pathway_rank", "pathway"}
MITOCHONDRIAL_REQUIRED_COLUMNS = {
    "cluster",
    "percent_of_cluster_cells_covered_by_mt_summary",
    "cell_count_weighted_average_mt_percentage",
    "datasets_with_high_mt_percentage",
    "total_datasets_evaluated_for_mt",
    "percent_of_covered_cluster_cells_from_high_mt_datasets",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine cluster composition, ranked significant pathways, and "
            "mitochondrial context into compact JSON files for the LLM."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="NK_project repository root. Default: inferred from this script.",
    )
    parser.add_argument(
        "--generated-date",
        default=date.today().isoformat(),
        help="Date suffix identifying the processed inputs, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--high-mt-percentage-cutoff",
        type=float,
        default=10.0,
        help="Cutoff used by the mitochondrial post-processing script. Default: 10.0.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/annotation_agent/processed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/annotation_agent/agent_context.",
    )
    return parser.parse_args()


def validate_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"--generated-date must use YYYY-MM-DD: {value!r}") from exc
    return value


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")


def cluster_sort_key(value: object) -> tuple[int, object]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def percent_from_fraction(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    numeric = float(numeric)
    if numeric < 0 or numeric > 1:
        raise ValueError(f"Expected a fraction between 0 and 1; found {numeric}")
    return round(100.0 * numeric, 6)


def clean_text(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def clean_number(value: Any, *, integer: bool = False) -> int | float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    if integer:
        return int(numeric)
    return round(float(numeric), 6)


def load_worksheet(path: Path) -> pd.DataFrame:
    worksheet = pd.read_csv(path, index_col=0, low_memory=False)
    require_columns(worksheet, WORKSHEET_REQUIRED_COLUMNS, path)
    worksheet.index = worksheet.index.astype(str)
    if not worksheet.index.is_unique:
        raise ValueError(f"Duplicate cluster identifiers in {path}")
    worksheet["n_cells"] = pd.to_numeric(worksheet["n_cells"], errors="raise")
    if (worksheet["n_cells"] <= 0).any():
        raise ValueError(f"Nonpositive cluster sizes in {path}")
    return worksheet


def load_pathways(path: Path) -> dict[str, list[dict[str, Any]]]:
    pathways = pd.read_csv(path, low_memory=False, dtype={"cluster": str})
    require_columns(pathways, PATHWAY_REQUIRED_COLUMNS, path)
    pathways["cluster"] = pathways["cluster"].astype(str)
    pathways["pathway_rank"] = pd.to_numeric(pathways["pathway_rank"], errors="raise").astype(int)
    if pathways.duplicated(["cluster", "pathway_rank"]).any():
        raise ValueError(f"Duplicate cluster/pathway ranks in {path}")

    result: dict[str, list[dict[str, Any]]] = {}
    for cluster, group in pathways.groupby("cluster", sort=False):
        group = group.sort_values("pathway_rank", kind="mergesort")
        expected = list(range(1, len(group) + 1))
        if group["pathway_rank"].tolist() != expected:
            raise ValueError(f"Nonconsecutive pathway ranks for cluster {cluster!r} in {path}")
        result[str(cluster)] = [
            {"rank": int(row.pathway_rank), "pathway": str(row.pathway)}
            for row in group.itertuples(index=False)
        ]
    return result


def load_mitochondrial(path: Path) -> pd.DataFrame:
    mitochondrial = pd.read_csv(path, low_memory=False, dtype={"cluster": str})
    require_columns(mitochondrial, MITOCHONDRIAL_REQUIRED_COLUMNS, path)
    mitochondrial["cluster"] = mitochondrial["cluster"].astype(str)
    if not mitochondrial["cluster"].is_unique:
        raise ValueError(f"Expected one mitochondrial row per cluster in {path}")
    return mitochondrial.set_index("cluster")


def boolean_series(values: pd.Series, *, column: str, path: Path) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    unexpected = sorted(set(normalized).difference({"true", "false"}))
    if unexpected:
        raise ValueError(
            f"{path} column {column!r} contains values other than true/false: "
            f"{unexpected[:10]}"
        )
    return normalized.eq("true")


def load_mass50_attribution_genes(
    path: Path,
    *,
    side: str,
) -> dict[str, list[str]]:
    if side not in {"parent", "child"}:
        raise ValueError(f"Attribution side must be parent or child, found: {side!r}")

    cluster_column = f"{side}_cluster"
    selected_column = f"{side}_selected_mass50"
    rank_column = f"{side}_rank"
    required = {
        cluster_column,
        selected_column,
        rank_column,
        "gene",
    }
    frame = pd.read_csv(path, low_memory=False, dtype={cluster_column: str})
    require_columns(frame, required, path)
    selected = boolean_series(
        frame[selected_column], column=selected_column, path=path
    )
    frame = frame.loc[
        selected,
        [cluster_column, "gene", rank_column],
    ].copy()
    frame = frame.rename(
        columns={
            cluster_column: "cluster",
            rank_column: "attribution_rank_within_cluster",
        }
    )
    frame["cluster"] = frame["cluster"].astype(str)
    frame["gene"] = frame["gene"].astype(str).str.strip().str.upper()
    frame["attribution_rank_within_cluster"] = pd.to_numeric(
        frame["attribution_rank_within_cluster"], errors="raise"
    ).astype(int)
    if frame["gene"].eq("").any():
        raise ValueError(f"Empty gene name in {path}")

    disagreement = (
        frame.groupby(["cluster", "gene"])["attribution_rank_within_cluster"]
        .nunique()
        .gt(1)
    )
    if disagreement.any():
        raise ValueError(
            f"Conflicting duplicated cluster/gene attribution rows in {path}"
        )
    frame = frame.drop_duplicates(["cluster", "gene"])
    frame = frame.sort_values(
        ["cluster", "attribution_rank_within_cluster", "gene"],
        kind="mergesort",
    )

    result: dict[str, list[str]] = {}
    for cluster, group in frame.groupby("cluster", sort=False):
        ranks = group["attribution_rank_within_cluster"].tolist()
        if ranks != list(range(1, len(group) + 1)):
            raise ValueError(
                f"Mass-50 attribution ranks are not consecutive for cluster "
                f"{cluster!r} in {path}"
            )
        result[str(cluster)] = group["gene"].astype(str).tolist()
    return result


def build_resolution_context(
    *,
    resolution: str,
    worksheet_path: Path,
    pathway_path: Path,
    mitochondrial_path: Path,
    attribution_relationship_path: Path,
    attribution_side: str,
) -> dict[str, Any]:
    worksheet = load_worksheet(worksheet_path)
    pathways = load_pathways(pathway_path)
    mitochondrial = load_mitochondrial(mitochondrial_path)
    attribution = load_mass50_attribution_genes(
        attribution_relationship_path,
        side=attribution_side,
    )

    worksheet_clusters = set(worksheet.index)
    unexpected_pathway_clusters = set(pathways).difference(worksheet_clusters)
    unexpected_mitochondrial_clusters = set(mitochondrial.index).difference(worksheet_clusters)
    unexpected_attribution_clusters = set(attribution).difference(worksheet_clusters)
    if unexpected_pathway_clusters:
        raise ValueError(
            f"Pathway table has clusters absent from {worksheet_path}: "
            f"{sorted(unexpected_pathway_clusters, key=cluster_sort_key)}"
        )
    if unexpected_mitochondrial_clusters:
        raise ValueError(
            f"Mitochondrial table has clusters absent from {worksheet_path}: "
            f"{sorted(unexpected_mitochondrial_clusters, key=cluster_sort_key)}"
        )
    if unexpected_attribution_clusters:
        raise ValueError(
            f"Attribution table has clusters absent from {worksheet_path}: "
            f"{sorted(unexpected_attribution_clusters, key=cluster_sort_key)}"
        )

    missing_mitochondrial = worksheet_clusters.difference(mitochondrial.index)
    if missing_mitochondrial:
        raise ValueError(
            "Missing mitochondrial context for clusters: "
            f"{sorted(missing_mitochondrial, key=cluster_sort_key)}"
        )
    missing_attribution = worksheet_clusters.difference(attribution)
    if missing_attribution:
        raise ValueError(
            "Missing mass-50 attribution genes for clusters: "
            f"{sorted(missing_attribution, key=cluster_sort_key)}"
        )

    clusters: list[dict[str, Any]] = []
    for cluster in sorted(worksheet.index, key=cluster_sort_key):
        composition = worksheet.loc[cluster]
        mitochondrial_row = mitochondrial.loc[cluster]
        all_cluster_pathways = pathways.get(str(cluster), [])
        cluster_attribution_genes = attribution[str(cluster)]
        clusters.append(
            {
                "cluster": str(cluster),
                "n_cells": int(composition["n_cells"]),
                "metadata_context": {
                    "top_tissue": clean_text(composition["top_tissue"]),
                    "percent_of_cluster_from_top_tissue": percent_from_fraction(
                        composition["top_tissue_frac"]
                    ),
                    "top_dataset_id": clean_text(composition["top_dataset_id"]),
                    "percent_of_cluster_from_top_dataset": percent_from_fraction(
                        composition["top_dataset_id_frac"]
                    ),
                    "top_assay": clean_text(composition["top_assay_clean"]),
                    "percent_of_cluster_from_top_assay": percent_from_fraction(
                        composition["top_assay_clean_frac"]
                    ),
                },
                "attribution_gene_selection": {
                    "target_cumulative_attribution_mass_percent": 50,
                    "genes_ordered_from_highest_to_lowest_attribution": cluster_attribution_genes,
                },
                "total_significant_pathways": len(all_cluster_pathways),
                "mitochondrial_context": {
                    "percent_of_cluster_cells_covered_by_mt_summary": clean_number(
                        mitochondrial_row[
                            "percent_of_cluster_cells_covered_by_mt_summary"
                        ]
                    ),
                    "cell_count_weighted_average_mt_percentage": clean_number(
                        mitochondrial_row["cell_count_weighted_average_mt_percentage"]
                    ),
                    "datasets_with_high_mt_percentage": clean_number(
                        mitochondrial_row["datasets_with_high_mt_percentage"],
                        integer=True,
                    ),
                    "total_datasets_evaluated_for_mt": clean_number(
                        mitochondrial_row["total_datasets_evaluated_for_mt"],
                        integer=True,
                    ),
                    "percent_of_covered_cluster_cells_from_high_mt_datasets": clean_number(
                        mitochondrial_row[
                            "percent_of_covered_cluster_cells_from_high_mt_datasets"
                        ]
                    ),
                },
            }
        )

    if int(sum(item["n_cells"] for item in clusters)) != 311_471:
        raise ValueError(
            f"{resolution} context does not represent 311,471 cells: "
            f"{sum(item['n_cells'] for item in clusters):,}"
        )
    return {"leiden_resolution": resolution, "clusters": clusters}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_name(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def save_json(value: Any, path: Path) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    generated_date = validate_date(args.generated_date)
    project_root = args.project_root.resolve()
    processed_dir = (
        args.processed_dir.resolve()
        if args.processed_dir is not None
        else project_root / "data" / "annotation_agent" / "processed"
    )
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "data" / "annotation_agent" / "agent_context"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths: list[Path] = []
    context_outputs: list[Path] = []
    context_summaries: dict[str, dict[str, int]] = {}
    attribution_relationship_path = project_root / ATTRIBUTION_RELATIONSHIP
    if not attribution_relationship_path.exists():
        raise FileNotFoundError(attribution_relationship_path)

    for resolution, relative_paths in RESOLUTIONS.items():
        worksheet_path = project_root / relative_paths["worksheet"]
        raw_pathway_path = project_root / relative_paths["raw_pathway"]
        raw_mitochondrial_path = project_root / relative_paths["raw_mitochondrial"]
        pathway_path = processed_dir / (
            f"significant_go_bp_ranked_{resolution}_{generated_date}.csv"
        )
        mitochondrial_path = processed_dir / (
            f"mitochondrial_context_{resolution}_{generated_date}.csv"
        )
        required_paths = [
            worksheet_path,
            raw_pathway_path,
            raw_mitochondrial_path,
            pathway_path,
            mitochondrial_path,
            attribution_relationship_path,
        ]
        for path in required_paths:
            if not path.exists():
                raise FileNotFoundError(path)
        source_paths.extend(required_paths)

        context = build_resolution_context(
            resolution=resolution,
            worksheet_path=worksheet_path,
            pathway_path=pathway_path,
            mitochondrial_path=mitochondrial_path,
            attribution_relationship_path=attribution_relationship_path,
            attribution_side=relative_paths["attribution_side"],
        )
        context_output = output_dir / (
            f"cluster_context_{resolution}_{generated_date}.json"
        )
        save_json(context, context_output)
        context_outputs.append(context_output)
        context_summaries[resolution] = {
            "clusters": len(context["clusters"]),
            "cells": int(sum(item["n_cells"] for item in context["clusters"])),
            "clusters_with_significant_pathways": int(
                sum(
                    item["total_significant_pathways"] > 0
                    for item in context["clusters"]
                )
            ),
            "mass50_attribution_genes": int(
                sum(
                    len(
                        item["attribution_gene_selection"][
                            "genes_ordered_from_highest_to_lowest_attribution"
                        ]
                    )
                    for item in context["clusters"]
                )
            ),
        }
        print(
            f"[SAVE] {context_output} "
            f"({context_summaries[resolution]['clusters']} clusters; "
            f"{context_summaries[resolution]['cells']:,} cells)"
        )

    manifest_output = output_dir / f"agent_context_manifest_{generated_date}.json"
    manifest = {
        "generated_on": generated_date,
        "processing_rules": {
            "pathway_filter": "significant_fdr_0_05 == TRUE",
            "pathway_ranking": (
                "adjusted_p_value ascending, overlap_count descending, "
                "gene_ratio descending, pathway ascending"
            ),
            "pathway_delivery_to_agent": (
                "Pathway names are not embedded in cluster context files. The "
                "planner retrieves ranked pathway batches on demand from the "
                "processed significant-pathway tables."
            ),
            "maximum_pathways_per_retrieval": 30,
            "pathway_similarity_interpretation": (
                "No pathway collapsing was performed; the annotation agent is "
                "responsible for recognizing related biological pathways."
            ),
            "attribution_gene_selection": (
                "Cluster-specific smallest ranked SIGnature attribution gene set "
                "reaching 50 percent cumulative attribution mass."
            ),
            "high_mt_percentage_cutoff": float(args.high_mt_percentage_cutoff),
            "mt_percentage_summary": (
                "cell-count-weighted average of supplied cluster-dataset "
                "median_mt_percentage values"
            ),
        },
        "source_file_hashes": {
            relative_name(path, project_root): sha256(path)
            for path in sorted(set(source_paths))
        },
        "context_file_hashes": {
            relative_name(path, project_root): sha256(path) for path in context_outputs
        },
        "context_summaries": context_summaries,
    }
    save_json(manifest, manifest_output)
    print(f"[SAVE] {manifest_output} (provenance only; not passed to the LLM)")


if __name__ == "__main__":
    main()
