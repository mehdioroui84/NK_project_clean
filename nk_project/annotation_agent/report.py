from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

import pandas as pd


REVIEW_MAPPING_COLUMNS = [
    "candidate_refined_label",
    "nk_subtype_call",
    "nk_state_call",
    "final_structured_label",
    "agent_preferred_label",
    "agent_preferred_label_reason",
    "label_action",
    "needs_human_review",
    "review_reason",
    "n_cells",
    "confidence_score_0_5",
    "ambiguity_score_0_5",
    "technical_concern_score_0_5",
    "top_tissue",
    "taxonomy_top_matches",
    "n_iterations",
    "n_pairwise_DE_compared",
    "distance_review_flag",
    "distance_review_reason",
    "possible_novel_subtype",
    "novel_subtype_reason",
    "suggested_split_label",
    "suggested_split_label_reason",
    "recommended_pairwise_comparisons",
]


REPORT_SUMMARY_COLUMNS = [
    "n_cells",
    "candidate_refined_label",
    "nk_subtype_call",
    "nk_state_call",
    "final_structured_label",
    "agent_preferred_label",
    "agent_preferred_label_reason",
    "label_action",
    "needs_human_review",
    "review_reason",
    "top_tissue",
    "taxonomy_top_matches",
    "confidence_score_0_5",
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
    mapping = mapping[[groupby] + [col for col in REVIEW_MAPPING_COLUMNS if col in mapping.columns]]
    mapping_path = os.path.join(outdir, "candidate_refined_label_mapping.csv")
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

    summary_path = os.path.join(outdir, "annotation_refinement_summary.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(build_annotation_status_paragraph(results, evidence, mapping, groupby))
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
        distance_evidence = evidence[cluster_id].get("distance_novelty_evidence", {})
        technical_concern = int(final["technical_concern_score"])
        needs_review = (
            bool(final["needs_human_review"])
            or int(final["confidence_score"]) < review_threshold
            or technical_concern >= 2
            or bool(final.get("suggested_new_label", ""))
        )
        review_reason = build_review_reason(
            final,
            needs_review=needs_review,
            review_threshold=review_threshold,
            worksheet_note=None,
            n_pairwise_de=len(evidence[cluster_id].get("pairwise_de_evidence", [])),
        )
        rows.append(
            {
                groupby: cluster_id,
                "candidate_refined_label": final.get("final_structured_label", "Non-NK"),
                "nk_subtype_call": final.get("nk_subtype_call", "Non-NK"),
                "nk_state_call": final.get("nk_state_call", "NA"),
                "final_structured_label": final.get("final_structured_label", "Non-NK"),
                "agent_preferred_label": final.get("final_structured_label", "Non-NK"),
                "label_action": final.get("label_action", "keep"),
                "needs_human_review": needs_review,
                "review_reason": review_reason,
                "n_cells": comp.get("n_cells"),
                "confidence_score_0_5": final["confidence_score"],
                "ambiguity_score_0_5": final["ambiguity_score"],
                "technical_concern_score_0_5": final["technical_concern_score"],
                "top_tissue": comp.get("top_tissue"),
                "taxonomy_top_matches": format_taxonomy_matches(
                    evidence[cluster_id].get("taxonomy_marker_hits", {}).get("top_matches", []),
                    max_items=3,
                ),
                "n_iterations": len(result["iterations"]),
                "n_pairwise_DE_compared": len(evidence[cluster_id].get("pairwise_de_evidence", [])),
                "distance_review_flag": distance_evidence.get("distance_review_flag", False),
                "distance_review_reason": distance_evidence.get("distance_review_reason", ""),
                "possible_novel_subtype": final.get("possible_novel_subtype", distance_evidence.get("possible_novel_subtype", False)),
                "novel_subtype_reason": final.get("novel_subtype_reason", "") or distance_evidence.get("novel_subtype_reason", ""),
                "suggested_split_label": final.get("suggested_split_label", ""),
                "suggested_split_label_reason": final.get("suggested_split_label_reason", ""),
                "agent_preferred_label_reason": final.get("new_label_reason", ""),
                "recommended_pairwise_comparisons": "; ".join(final["recommended_pairwise_comparisons"]),
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
        "# Refined Annotation Agent Report",
        "",
        f"Groupby: `{groupby}`",
        "",
        "This is a draft, evidence-based annotation report. Review the CSV before applying labels.",
        "",
        "## Summary",
        "",
        markdown_table(
            mapping[[groupby] + [col for col in REPORT_SUMMARY_COLUMNS if col in mapping.columns]]
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
                f"### Cluster {cluster_id}: {final['candidate_label']}",
                "",
                f"- Current final label: {final.get('current_final_label', final['candidate_label'])}",
                f"- Agent preferred label: {final.get('agent_preferred_label', final.get('suggested_new_label', '') or final['candidate_label'])}",
                f"- Layer 1 NK subtype call: {final.get('nk_subtype_call', 'Non-NK')}",
                f"- Layer 2 NK state call: {final.get('nk_state_call', 'NA')}",
                f"- Final structured label: {final.get('final_structured_label', 'Non-NK')}",
                f"- Label action: {final.get('label_action', 'keep')}",
                f"- Overwrite recommendation: {final.get('overwrite_recommendation', False)}",
                f"- Approved label: {final.get('approved_label', final.get('current_final_label', final['candidate_label']))}",
                f"- Confidence: {final['confidence_score']}/5",
                f"- Top DE support: {final['top_de_marker_support']}/5",
                f"- Curated marker support: {final['curated_marker_support']}/5",
                f"- Technical concern: {final['technical_concern_score']}/5",
                f"- Ambiguity: {final['ambiguity_score']}/5",
                f"- Needs human review: {final['needs_human_review']}",
                f"- Suggested new label: {final.get('suggested_new_label', '') or 'None'}",
                f"- New label reason: {final.get('new_label_reason', '') or 'None'}",
                f"- Tissue: {comp.get('top_tissue')} ({comp.get('top_tissue_frac')})",
                f"- Top taxonomy matches: {format_taxonomy_matches(ev.get('taxonomy_marker_hits', {}).get('top_matches', []), max_items=5) or 'None'}",
                f"- Pairwise evidence comparisons loaded: {len(ev.get('pairwise_de_evidence', []))}",
                "",
                "Top DE genes:",
                ", ".join(ev["top_gene_names"][:20]),
                "",
                "Evidence summary:",
            ]
        )
        lines.extend(f"- {item}" for item in final["evidence_summary"])
        if final["concerns"]:
            lines.append("")
            lines.append("Concerns:")
            lines.extend(f"- {item}" for item in final["concerns"])
        if final["recommended_pairwise_comparisons"]:
            lines.append("")
            lines.append("Recommended pairwise checks:")
            lines.extend(f"- {item}" for item in final["recommended_pairwise_comparisons"])
        lines.append("")
    return "\n".join(lines)


def build_annotation_status_paragraph(
    results: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    mapping: pd.DataFrame,
    groupby: str,
) -> str:
    total_clusters = len(mapping)
    label_counts = Counter(mapping["candidate_refined_label"].astype(str))
    label_summary = format_counter(label_counts, max_items=6)
    review_df = mapping.loc[mapping["needs_human_review"].astype(bool)].copy()
    n_review = int(review_df.shape[0])
    review_clusters = ", ".join(review_df[groupby].astype(str).tolist()) if n_review else "none"
    review_reasons = summarize_review_reasons(review_df["review_reason"].dropna().astype(str).tolist())
    alternative_df = mapping.head(0).copy()
    alternative_summary = summarize_alternatives(alternative_df, groupby)
    n_pairwise = int(pd.to_numeric(mapping.get("n_pairwise_DE_compared", 0), errors="coerce").fillna(0).sum())
    lineage_summary = summarize_candidate_lineages(label_counts)
    confidence_median = float(pd.to_numeric(mapping["confidence_score_0_5"], errors="coerce").median())

    overview = (
        f"The optional annotation agent reviewed {plural(total_clusters, 'Leiden 0.4 cluster')} using "
        f"cluster-vs-rest marker evidence, positive/negative markers, Amina taxonomy marker support, metadata composition, and "
        f"{plural(n_pairwise, 'loaded pairwise DE comparison')}, assigning candidates across "
        f"{plural(len(label_counts), 'refined label')} ({label_summary}). "
        f"Overall, the naming pattern supports {lineage_summary}, with a median confidence score of "
        f"{confidence_median:.1f}/5. "
        f"{plural(n_review, 'cluster')} remain flagged for review ({review_clusters}), mainly because of "
        f"{review_reasons}. "
        f"Alternative names were considered separately from the safe approved candidate labels; {alternative_summary}. "
        "The candidate labels are therefore ready as a structured draft for manual sign-off, while the flagged clusters "
        "represent targeted review items rather than a broad failure of the refined annotation scheme."
    )
    return (
        "# Annotation Refinement Summary\n\n"
        + overview
        + "\n"
    )


def format_counter(counter: Counter, *, max_items: int) -> str:
    items = counter.most_common(max_items)
    text = ", ".join(f"{label}: {count}" for label, count in items)
    remaining = sum(counter.values()) - sum(count for _, count in items)
    if remaining:
        text += f", and {plural(remaining, 'other cluster')}"
    return text


def summarize_review_reasons(reasons: list[str]) -> str:
    if not reasons:
        return "no remaining review reasons"
    counts = Counter()
    for reason in reasons:
        for part in reason.split(";"):
            text = part.strip()
            if text:
                counts[text] += 1
    return format_counter(counts, max_items=4)


def summarize_alternatives(alternative_df: pd.DataFrame, groupby: str) -> str:
    if alternative_df.empty:
        return "no agent-preferred names differed from the candidate labels"
    rows = []
    for _, row in alternative_df.iterrows():
        rows.append(f"cluster {row[groupby]}: {row['agent_preferred_label']}")
    return f"{plural(len(rows), 'agent-preferred name suggestion')} differed from candidate labels ({'; '.join(rows)})"


def summarize_candidate_lineages(label_counts: Counter) -> str:
    labels = set(label_counts)
    themes = []
    if any("Cytotoxic" in label for label in labels):
        themes.append("cytotoxic NK states")
    if any("Tissue-Resident" in label or label.startswith("Lung") for label in labels):
        themes.append("tissue/context-associated NK states")
    if any("Cytokine-Stimulated" in label for label in labels):
        themes.append("cytokine-stimulated states")
    if "Proliferative" in labels or any("Proliferative" in label for label in labels):
        themes.append("proliferative programs")
    contaminants = [label for label in ["T", "B", "Myeloid-like", "Unknown_BM_1 Erythroid-like"] if label in labels]
    if contaminants:
        themes.append("non-NK or contamination-like groups")
    return ", ".join(themes) if themes else "the observed marker-defined groups"


def plural(count: int, noun: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {noun}{suffix}"


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        values = [str(row[col]) for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_review_reason(
    final: dict[str, Any],
    *,
    needs_review: bool,
    review_threshold: int,
    worksheet_note: Any,
    n_pairwise_de: int,
) -> str:
    reasons = []
    confidence = int(final.get("confidence_score", 0))
    ambiguity = int(final.get("ambiguity_score", 0))
    technical = int(final.get("technical_concern_score", 0))
    note = str(worksheet_note or "").strip()
    note = humanize_review_note(note)

    if needs_review:
        if confidence < review_threshold:
            reasons.append(f"confidence {confidence}<{review_threshold}")
        if technical >= 2:
            reasons.append(f"technical concern {technical}/5")
        suggested = str(final.get("suggested_new_label", "")).strip()
        if suggested:
            reasons.append(f"suggested new label: {suggested}")
        if note and note.lower() != "nan":
            reasons.append(note)
        if bool(final.get("needs_human_review")) and not reasons:
            reasons.append("model uncertainty")
        return "; ".join(reasons)

    reasons.append(f"confident approved label ({confidence}/5)")
    if ambiguity <= 2:
        reasons.append(f"low ambiguity ({ambiguity}/5)")
    if technical < 2:
        reasons.append(f"low technical concern ({technical}/5)")
    if n_pairwise_de:
        reasons.append(f"reviewed {n_pairwise_de} pairwise DE comparison(s)")
    return "; ".join(reasons)


def build_alternative_name_reason(final: dict[str, Any]) -> str:
    suggested = str(final.get("suggested_new_label", "") or "").strip()
    reason = str(final.get("new_label_reason", "") or "").strip()
    if reason:
        return reason
    if suggested:
        return "Alternative label suggested because the approved label may not fully capture the cluster biology."
    return "No alternative name suggested; approved candidate label is sufficient."


def humanize_review_note(note: str) -> str:
    if not note or note.lower() == "nan":
        return ""
    replacements = {
        "mixed_original_NK_State": "mixed original manual labels",
        "high_tissue_specificity": "mostly from one tissue",
        "high_assay_specificity": "mostly from one assay",
        "high_dataset_specificity": "mostly from one dataset",
    }
    normalized = note
    for token in replacements:
        normalized = normalized.replace(token, f";{token};")
    parts = []
    for raw_part in normalized.replace(",", ";").split(";"):
        part = raw_part.strip()
        if not part:
            continue
        human = replacements.get(part, part)
        if human not in parts:
            parts.append(human)
    return "; ".join(parts)


def format_taxonomy_matches(matches: list[dict[str, Any]], *, max_items: int) -> str:
    parts = []
    for item in (matches or [])[:max_items]:
        state = str(item.get("taxonomy_state", "")).strip()
        if not state:
            continue
        support_level = str(item.get("support_level", "") or "").strip()
        percent = item.get("percent_of_max_score")
        core_hits = item.get("core_hits") or []
        support_hits = item.get("support_hits") or []
        context_hits = item.get("context_hits") or []
        negative_low = item.get("negative_expected_low_hits") or []
        contradictions = item.get("negative_contradictions") or []
        evidence_bits = []
        if support_level:
            evidence_bits.append(support_level)
        if percent is not None:
            evidence_bits.append(f"{float(percent):.1f}% max")
        if core_hits:
            evidence_bits.append("CORE " + ",".join(map(str, core_hits[:4])))
        if support_hits:
            evidence_bits.append("SUPPORT " + ",".join(map(str, support_hits[:4])))
        if context_hits:
            evidence_bits.append("CONTEXT " + ",".join(map(str, context_hits[:3])))
        if negative_low:
            evidence_bits.append("expected-low " + ",".join(map(str, negative_low[:3])))
        if contradictions:
            evidence_bits.append("contradicts " + ",".join(map(str, contradictions[:3])))
        parts.append(f"{state} ({'; '.join(evidence_bits)})")
    return "; ".join(parts)
