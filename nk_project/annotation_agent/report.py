from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

import pandas as pd


MAPPING_FILENAME = "cluster_annotation_mapping.csv"

MAPPING_COLUMNS = [
    "nk_subtype_call",
    "nk_state_call",
    "final_structured_label",
    "free_label",
    "free_label_reason",
    "needs_human_review",
    "human_review_reason",
    "n_cells",
    "confidence_score_0_5",
    "tissue_specificity_score_0_5",
    "dataset_assay_specificity_score_0_5",
    "top_tissue",
    "top_dataset",
    "top_assay",
    "taxonomy_top_matches",
    "n_iterations",
    "n_pairwise_de_compared",
]


def write_outputs(
    results: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    outdir: str,
    groupby: str,
    *,
    review_threshold: int,
    save_debug_trace: bool = False,
) -> None:
    mapping = build_mapping_table(results, evidence, groupby, review_threshold=review_threshold)
    mapping = mapping[[groupby] + [col for col in MAPPING_COLUMNS if col in mapping.columns]]
    mapping_path = os.path.join(outdir, MAPPING_FILENAME)
    mapping.to_csv(mapping_path, index=False)
    print(f"[SAVE] {mapping_path}")

    review_flags = mapping.loc[mapping["needs_human_review"].astype(bool)].copy()
    flags_path = os.path.join(outdir, "review_flags.csv")
    review_flags.to_csv(flags_path, index=False)
    print(f"[SAVE] {flags_path}")

    if save_debug_trace:
        debug_dir = os.path.join(outdir, "debug")
        os.makedirs(debug_dir, exist_ok=True)
        trace_path = os.path.join(debug_dir, "cluster_decision_trace.jsonl")
        with open(trace_path, "w", encoding="utf-8") as handle:
            for result in results:
                handle.write(json.dumps(result) + "\n")
        print(f"[SAVE] {trace_path}")

    report_path = os.path.join(outdir, "cluster_annotation_report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(build_markdown_report(results, evidence, mapping, groupby))
    print(f"[SAVE] {report_path}")

    summary_path = os.path.join(outdir, "annotation_summary.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(build_annotation_summary(mapping, groupby))
    print(f"[SAVE] {summary_path}")


def build_mapping_table(
    results: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    groupby: str,
    *,
    review_threshold: int,
) -> pd.DataFrame:
    rows = []
    for result in results:
        final = result["final_decision"]
        cluster_id = str(result["cluster_id"])
        comp = evidence[cluster_id]["composition"]
        confidence = int(final.get("confidence_score", 0))
        needs_review = bool(final.get("needs_human_review", False)) or confidence < review_threshold
        review_reason = str(final.get("human_review_reason") or "").strip()
        if needs_review and not review_reason:
            review_reason = f"confidence {confidence}<{review_threshold}"
        rows.append(
            {
                groupby: cluster_id,
                "nk_subtype_call": final.get("nk_subtype_call", "Non-NK"),
                "nk_state_call": final.get("nk_state_call", "NA"),
                "final_structured_label": final.get("final_structured_label", "Non-NK"),
                "free_label": final.get("free_label", final.get("final_structured_label", "Non-NK")),
                "free_label_reason": final.get("free_label_reason", ""),
                "needs_human_review": needs_review,
                "human_review_reason": review_reason,
                "n_cells": comp.get("n_cells"),
                "confidence_score_0_5": confidence,
                "tissue_specificity_score_0_5": final.get("tissue_specificity_score", 0),
                "dataset_assay_specificity_score_0_5": final.get("dataset_assay_specificity_score", 0),
                "top_tissue": comp.get("top_tissue"),
                "top_dataset": comp.get("top_dataset_id") or comp.get("top_dataset"),
                "top_assay": comp.get("top_assay_clean") or comp.get("top_assay"),
                "taxonomy_top_matches": format_taxonomy_matches(
                    evidence[cluster_id].get("taxonomy_marker_hits", {}).get("top_matches", []),
                    max_items=3,
                ),
                "n_iterations": len(result["iterations"]),
                "n_pairwise_de_compared": len(evidence[cluster_id].get("pairwise_de_evidence", [])),
            }
        )
    return pd.DataFrame(rows)


def build_markdown_report(
    results: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    mapping: pd.DataFrame,
    groupby: str,
) -> str:
    lines = [
        "# Cluster Annotation Report",
        "",
        f"Groupby: `{groupby}`",
        "",
        "The annotations below use cluster markers, Amina taxonomy marker support, metadata context, and pairwise DE when available.",
        "",
        "## Summary",
        "",
        markdown_table(
            mapping[
                [
                    groupby,
                    "n_cells",
                    "final_structured_label",
                    "free_label",
                    "needs_human_review",
                    "human_review_reason",
                    "confidence_score_0_5",
                    "top_tissue",
                ]
            ]
        ),
        "",
        "## Cluster Details",
        "",
    ]
    for result in results:
        final = result["final_decision"]
        cluster_id = str(result["cluster_id"])
        ev = evidence[cluster_id]
        comp = ev["composition"]
        lines.extend(
            [
                f"### Cluster {cluster_id}: {final.get('final_structured_label', 'Non-NK')}",
                "",
                f"- Subtype: {final.get('nk_subtype_call', 'Non-NK')}",
                f"- State: {final.get('nk_state_call', 'NA')}",
                f"- Free label: {final.get('free_label', final.get('final_structured_label', 'Non-NK'))}",
                f"- Confidence: {final.get('confidence_score', 0)}/5",
                f"- Tissue specificity: {final.get('tissue_specificity_score', 0)}/5",
                f"- Dataset/assay specificity: {final.get('dataset_assay_specificity_score', 0)}/5",
                f"- Needs human review: {final.get('needs_human_review', False)}",
                f"- Review reason: {final.get('human_review_reason', '') or 'None'}",
                f"- Top tissue: {comp.get('top_tissue')} ({comp.get('top_tissue_frac')})",
                f"- Top dataset: {comp.get('top_dataset_id') or comp.get('top_dataset')} ({comp.get('top_dataset_id_frac') or comp.get('top_dataset_frac')})",
                f"- Top assay: {comp.get('top_assay_clean') or comp.get('top_assay')} ({comp.get('top_assay_clean_frac') or comp.get('top_assay_frac')})",
                f"- Top taxonomy matches: {format_taxonomy_matches(ev.get('taxonomy_marker_hits', {}).get('top_matches', []), max_items=5) or 'None'}",
                f"- Pairwise DE comparisons loaded: {len(ev.get('pairwise_de_evidence', []))}",
                "",
                "Free-label reason:",
                final.get("free_label_reason", "") or "None",
                "",
                "Evidence summary:",
            ]
        )
        lines.extend(f"- {item}" for item in final.get("evidence_summary", []))
        if final.get("concerns"):
            lines.append("")
            lines.append("Concerns:")
            lines.extend(f"- {item}" for item in final["concerns"])
        lines.append("")
    return "\n".join(lines)


def build_annotation_summary(mapping: pd.DataFrame, groupby: str) -> str:
    structured_counts = Counter(mapping["final_structured_label"].astype(str))
    free_counts = Counter(mapping["free_label"].astype(str))
    n_review = int(mapping["needs_human_review"].astype(bool).sum())
    review_clusters = ", ".join(mapping.loc[mapping["needs_human_review"].astype(bool), groupby].astype(str))
    median_confidence = float(pd.to_numeric(mapping["confidence_score_0_5"], errors="coerce").median())
    return "\n".join(
        [
            "# Annotation Summary",
            "",
            f"Annotated {len(mapping)} clusters.",
            f"Structured labels: {format_counter(structured_counts, max_items=8)}.",
            f"Free labels: {format_counter(free_counts, max_items=8)}.",
            f"Median confidence: {median_confidence:.1f}/5.",
            f"Human-review clusters: {n_review} ({review_clusters or 'none'}).",
            "",
        ]
    )


def format_counter(counter: Counter, *, max_items: int) -> str:
    items = counter.most_common(max_items)
    text = ", ".join(f"{label}: {count}" for label, count in items)
    remaining = sum(counter.values()) - sum(count for _, count in items)
    if remaining:
        text += f", and {remaining} other cluster(s)"
    return text


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def format_taxonomy_matches(matches: list[dict[str, Any]], *, max_items: int) -> str:
    parts = []
    for item in (matches or [])[:max_items]:
        label = str(item.get("taxonomy_label", item.get("taxonomy_state", ""))).strip()
        if not label:
            continue
        support_level = str(item.get("support_level", "") or "").strip()
        core_hits = item.get("core_hits") or []
        support_hits = item.get("support_hits") or []
        evidence_bits = []
        if support_level:
            evidence_bits.append(support_level)
        if core_hits:
            evidence_bits.append("CORE " + ",".join(map(str, core_hits[:4])))
        if support_hits:
            evidence_bits.append("SUPPORT " + ",".join(map(str, support_hits[:4])))
        parts.append(f"{label} ({'; '.join(evidence_bits)})")
    return "; ".join(parts)
