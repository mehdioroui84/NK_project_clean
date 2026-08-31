"""Outputs for the evidence-driven multi-agent annotation workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def mapping_columns(resolution: str) -> list[str]:
    return [
        resolution,
        "n_cells",
        "proposed_broad_identity",
        "proposed_subtype",
        "proposed_functional_state",
        "confidence_score",
        "requires_human_review",
        "human_review_reason",
        "evidence_summary",
        "biological_interpretation",
        "technical_caveats",
        "critical_review_decision",
        "mass50_gene_count",
        "significant_pathway_count",
        "pathways_retrieved",
        "top_tissue",
        "percent_of_cluster_from_top_tissue",
        "top_dataset_id",
        "percent_of_cluster_from_top_dataset",
        "top_assay",
        "percent_of_cluster_from_top_assay",
        "cell_count_weighted_average_mt_percentage",
        "mitochondrial_metadata_review_activated",
        "number_of_mitochondrial_genes_in_top_10",
        "number_of_mitochondrial_genes_in_top_20",
        "planner_calls",
        "biological_evidence_calls",
        "technical_context_calls",
        "critical_reviewer_calls",
    ]


def write_multiagent_outputs(
    *,
    results: list[dict[str, Any]],
    cluster_context: dict[str, dict[str, Any]],
    outdir: Path,
    resolution: str,
    run_config: dict[str, Any],
    failures: list[dict[str, str]],
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)

    annotations_path = outdir / "cluster_annotations.json"
    _write_json(
        {
            "leiden_resolution": resolution,
            "annotations": [result["final_decision"] for result in results],
        },
        annotations_path,
    )
    print(f"[SAVE] {annotations_path}")

    mapping = build_mapping_table(results, cluster_context, resolution)
    mapping_path = outdir / "cluster_annotations.csv"
    mapping.to_csv(mapping_path, index=False)
    print(f"[SAVE] {mapping_path}")

    report_path = outdir / "cluster_annotation_report.md"
    report_path.write_text(
        build_markdown_report(results, cluster_context, resolution, failures),
        encoding="utf-8",
    )
    print(f"[SAVE] {report_path}")

    trace_path = outdir / "agent_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
    print(f"[SAVE] {trace_path}")

    failures_path = outdir / "failed_clusters.json"
    _write_json(failures, failures_path)
    print(f"[SAVE] {failures_path}")

    config_path = outdir / "run_config.json"
    _write_json(run_config, config_path)
    print(f"[SAVE] {config_path}")


def build_mapping_table(
    results: list[dict[str, Any]],
    cluster_context: dict[str, dict[str, Any]],
    resolution: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for result in results:
        cluster = str(result["cluster"])
        final = result["final_decision"]
        context = cluster_context[cluster]
        metadata = context["metadata_context"]
        mitochondrial = context["mitochondrial_context"]
        mitochondrial_review = result["specialist_reports"][
            "mitochondrial_attribution_review"
        ]
        calls = result["role_call_counts"]
        rows.append(
            {
                resolution: cluster,
                "n_cells": context["n_cells"],
                "proposed_broad_identity": final["proposed_broad_identity"],
                "proposed_subtype": final["proposed_subtype"],
                "proposed_functional_state": final["proposed_functional_state"],
                "confidence_score": final["confidence_score"],
                "requires_human_review": final["requires_human_review"],
                "human_review_reason": final["human_review_reason"],
                "evidence_summary": final["evidence_summary"],
                "biological_interpretation": final["biological_interpretation"],
                "technical_caveats": "; ".join(final["technical_caveats"]),
                "critical_review_decision": final["critical_review_decision"],
                "mass50_gene_count": len(
                    context["attribution_gene_selection"][
                        "genes_ordered_from_highest_to_lowest_attribution"
                    ]
                ),
                "significant_pathway_count": context["total_significant_pathways"],
                "pathways_retrieved": sum(
                    len(batch["pathways_returned"])
                    for batch in result["pathway_batches_retrieved"]
                ),
                "top_tissue": metadata["top_tissue"],
                "percent_of_cluster_from_top_tissue": metadata[
                    "percent_of_cluster_from_top_tissue"
                ],
                "top_dataset_id": metadata["top_dataset_id"],
                "percent_of_cluster_from_top_dataset": metadata[
                    "percent_of_cluster_from_top_dataset"
                ],
                "top_assay": metadata["top_assay"],
                "percent_of_cluster_from_top_assay": metadata[
                    "percent_of_cluster_from_top_assay"
                ],
                "cell_count_weighted_average_mt_percentage": mitochondrial[
                    "cell_count_weighted_average_mt_percentage"
                ],
                "mitochondrial_metadata_review_activated": mitochondrial_review[
                    "mitochondrial_metadata_review_activated"
                ],
                "number_of_mitochondrial_genes_in_top_10": mitochondrial_review[
                    "number_of_mitochondrial_genes_in_top_10"
                ],
                "number_of_mitochondrial_genes_in_top_20": mitochondrial_review[
                    "number_of_mitochondrial_genes_in_top_20"
                ],
                "planner_calls": calls["planner"],
                "biological_evidence_calls": calls["biological_evidence"],
                "technical_context_calls": calls["technical_context"],
                "critical_reviewer_calls": calls["critical_reviewer"],
            }
        )
    return pd.DataFrame(rows, columns=mapping_columns(resolution))


def build_markdown_report(
    results: list[dict[str, Any]],
    cluster_context: dict[str, dict[str, Any]],
    resolution: str,
    failures: list[dict[str, str]],
) -> str:
    lines = [
        "# Evidence-driven Multi-agent Cluster Annotation Report",
        "",
        f"Leiden resolution: `{resolution}`",
        "",
        (
            "Each cluster was interpreted from its complete ordered SIGnature "
            "mass-50 gene list, the top 20 ranked significant pathways with one "
            "optional batch of 10, "
            "and an independent technical-context review. No DEG evidence, "
            "curated taxonomy, previous labels, or parent-child annotations were used."
        ),
        (
            "Mitochondrial metadata was exposed to the technical review only when "
            "at least 4 MT- genes were present among the top 10 attribution genes "
            "or at least 6 were present among the top 20."
        ),
        "",
        "## Summary",
        "",
    ]
    summary = build_mapping_table(results, cluster_context, resolution)
    summary_columns = [
        resolution,
        "n_cells",
        "proposed_broad_identity",
        "proposed_subtype",
        "proposed_functional_state",
        "confidence_score",
        "requires_human_review",
    ]
    lines.extend([_markdown_table(summary[summary_columns]), "", "## Cluster details", ""])

    for result in results:
        cluster = str(result["cluster"])
        final = result["final_decision"]
        context = cluster_context[cluster]
        technical = result["specialist_reports"]["technical_context"]
        reviews = result["specialist_reports"]["critical_reviews"]
        reviewer = reviews[-1]
        lines.extend(
            [
                f"### Cluster {cluster}",
                "",
                f"- Broad identity: {final['proposed_broad_identity']}",
                f"- Subtype: {final['proposed_subtype']}",
                f"- Functional state: {final['proposed_functional_state']}",
                f"- Confidence: {final['confidence_score']}",
                f"- Human review: {final['requires_human_review']}",
                f"- Review reason: {final['human_review_reason'] or 'None'}",
                f"- Mass-50 genes supplied: {len(context['attribution_gene_selection']['genes_ordered_from_highest_to_lowest_attribution'])}",
                f"- Significant pathways available: {context['total_significant_pathways']}",
                f"- Pathways retrieved: {sum(len(batch['pathways_returned']) for batch in result['pathway_batches_retrieved'])}",
                "",
                "**Evidence summary**",
                "",
                final["evidence_summary"],
                "",
                "**Biological interpretation**",
                "",
                final["biological_interpretation"],
                "",
                "**Technical-context assessment**",
                "",
                technical["technical_context_summary"],
                "",
                "**Technical caveats**",
                "",
                "; ".join(final["technical_caveats"]) or "None",
                "",
                "**Critical review**",
                "",
                f"{reviewer['review_decision']}: {reviewer['grounding_assessment']}",
                "",
            ]
        )

    if failures:
        lines.extend(["## Failed clusters", ""])
        lines.extend(
            f"- Cluster {item['cluster']}: {item['error_type']}: {item['error']}"
            for item in failures
        )
        lines.append("")
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def _write_json(value: Any, path: Path) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
