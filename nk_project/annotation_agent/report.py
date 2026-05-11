from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any

import pandas as pd


def write_outputs(
    results: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    outdir: str,
    groupby: str,
    *,
    review_threshold: int,
) -> None:
    mapping = build_mapping_table(results, evidence, groupby, review_threshold=review_threshold)
    mapping_path = os.path.join(outdir, "candidate_refined_label_mapping.csv")
    mapping.to_csv(mapping_path, index=False)
    print(f"[SAVE] {mapping_path}")

    review_flags = mapping.loc[mapping["needs_human_review"].astype(bool)].copy()
    flags_path = os.path.join(outdir, "review_flags.csv")
    review_flags.to_csv(flags_path, index=False)
    print(f"[SAVE] {flags_path}")

    trace_path = os.path.join(outdir, "cluster_decision_trace.jsonl")
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
            worksheet_note=comp.get("worksheet_review_note"),
            n_pairwise_de=len(evidence[cluster_id].get("pairwise_de_evidence", [])),
        )
        rows.append(
            {
                groupby: cluster_id,
                "candidate_refined_label": final["candidate_label"],
                "needs_human_review": needs_review,
                "review_reason": review_reason,
                "n_cells": comp.get("n_cells"),
                "top_original_label": comp.get("top_NK_State"),
                "top_original_label_frac": comp.get("top_NK_State_frac"),
                "confidence_score_0_5": final["confidence_score"],
                "ambiguity_score_0_5": final["ambiguity_score"],
                "technical_concern_score_0_5": final["technical_concern_score"],
                "top_tissue": comp.get("top_tissue"),
                "worksheet_initial_draft_label": comp.get("draft_refined_label"),
                "n_iterations": len(result["iterations"]),
                "n_pairwise_DE_compared": len(evidence[cluster_id].get("pairwise_de_evidence", [])),
                "possible_novel_subtype": distance_evidence.get("possible_novel_subtype", False),
                "novel_subtype_score_0_5": distance_evidence.get("novel_subtype_score_0_5", 0),
                "novel_subtype_reason": distance_evidence.get("novel_subtype_reason", ""),
                "alternative_name_suggestion": final.get("suggested_new_label", "") or "None",
                "alternative_name_reason": build_alternative_name_reason(final),
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
            mapping[
                [
                    groupby,
                    "n_cells",
                    "candidate_refined_label",
                    "needs_human_review",
                    "review_reason",
                    "top_original_label",
                    "top_original_label_frac",
                    "top_tissue",
                    "worksheet_initial_draft_label",
                    "confidence_score_0_5",
                    "alternative_name_suggestion",
                    "alternative_name_reason",
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
                f"### Cluster {cluster_id}: {final['candidate_label']}",
                "",
                f"- Confidence: {final['confidence_score']}/5",
                f"- Manual support: {final['manual_annotation_support']}/5",
                f"- Top DE support: {final['top_de_marker_support']}/5",
                f"- Curated marker support: {final['curated_marker_support']}/5",
                f"- Technical concern: {final['technical_concern_score']}/5",
                f"- Ambiguity: {final['ambiguity_score']}/5",
                f"- Needs human review: {final['needs_human_review']}",
                f"- Suggested new label: {final.get('suggested_new_label', '') or 'None'}",
                f"- New label reason: {final.get('new_label_reason', '') or 'None'}",
                f"- Original/manual composition: {comp.get('top_NK_State')} ({comp.get('top_NK_State_frac')})",
                f"- Original/manual composition top labels: {format_manual_composition(comp.get('manual_annotation_composition', [])) or 'None'}",
                f"- Tissue: {comp.get('top_tissue')} ({comp.get('top_tissue_frac')})",
                f"- Worksheet initial draft: {comp.get('draft_refined_label')}",
                f"- Worksheet/script review hint: {comp.get('worksheet_review_note') or 'None'}",
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
    alternative_df = mapping.loc[
        mapping["alternative_name_suggestion"].astype(str).str.lower().ne("none")
    ].copy()
    alternative_summary = summarize_alternatives(alternative_df, groupby)
    n_pairwise = int(pd.to_numeric(mapping.get("n_pairwise_DE_compared", 0), errors="coerce").fillna(0).sum())
    lineage_summary = summarize_candidate_lineages(label_counts)
    confidence_median = float(pd.to_numeric(mapping["confidence_score_0_5"], errors="coerce").median())

    overview = (
        f"The optional annotation agent reviewed {plural(total_clusters, 'Leiden 0.4 cluster')} using original/manual "
        f"label composition, cluster-vs-rest marker evidence, curated marker programs, and "
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
    kept = build_kept_as_is_paragraph(results, evidence, mapping, groupby)
    merged = build_merged_paragraph(results, evidence, mapping, groupby)
    refined = build_refined_further_paragraph(results, evidence, mapping, groupby)
    return (
        "# Annotation Refinement Summary\n\n"
        + overview
        + "\n\n"
        + kept
        + "\n\n"
        + merged
        + "\n\n"
        + refined
        + "\n"
    )


def build_kept_as_is_paragraph(
    results: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    mapping: pd.DataFrame,
    groupby: str,
) -> str:
    kept = mapping.loc[
        mapping["candidate_refined_label"].astype(str) == mapping["top_original_label"].astype(str)
    ].copy()
    if kept.empty:
        return (
            "Manual labels kept as-is: none of the top original/manual labels were retained exactly as the final "
            "candidate label; all clusters were either merged, renamed for specificity, or flagged for review."
        )
    clauses = []
    for label in sorted(kept["candidate_refined_label"].astype(str).unique()):
        sub = kept.loc[kept["candidate_refined_label"].astype(str) == label]
        clauses.append(label_rationale_clause(label, sub, results, evidence, groupby))
    return (
        "Manual labels kept as-is: "
        + "; ".join(clauses)
        + ". These labels were preserved because the original/manual composition, top DE genes, and curated marker "
        "programs were already concordant, so the refinement step mainly served as evidence confirmation rather than "
        "renaming."
    )


def build_merged_paragraph(
    results: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    mapping: pd.DataFrame,
    groupby: str,
) -> str:
    clauses = []
    for label in sorted(mapping["candidate_refined_label"].astype(str).unique()):
        sub = mapping.loc[mapping["candidate_refined_label"].astype(str) == label].copy()
        if sub.shape[0] < 2:
            continue
        original_labels = {
            str(item)
            for item in sub["top_original_label"].dropna().astype(str).tolist()
            if str(item) and str(item).lower() != "none"
        }
        if len(original_labels) < 2 and sub.shape[0] < 3:
            continue
        clauses.append(label_rationale_clause(label, sub, results, evidence, groupby))
    if not clauses:
        return (
            "Merged labels: no candidate refined class clearly merged multiple original/manual labels or several "
            "related clusters in this draft."
        )
    return (
        "Merged labels: "
        + "; ".join(clauses)
        + ". These merges collapse Leiden-level fragments into broader refined classes when related clusters shared "
        "the same dominant functional program, marker support, and pairwise/related-cluster evidence, even if the "
        "original top manual labels or tissue contexts differed."
    )


def build_refined_further_paragraph(
    results: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    mapping: pd.DataFrame,
    groupby: str,
) -> str:
    refined = mapping.loc[
        mapping["candidate_refined_label"].astype(str) != mapping["top_original_label"].astype(str)
    ].copy()
    if refined.empty:
        return (
            "Labels refined further: no cluster changed from its top original/manual label in this draft. Review should "
            "therefore focus on confidence, technical flags, and any alternative-name suggestions rather than major "
            "renaming decisions."
        )
    clauses = []
    for label in sorted(refined["candidate_refined_label"].astype(str).unique()):
        sub = refined.loc[refined["candidate_refined_label"].astype(str) == label]
        clauses.append(label_rationale_clause(label, sub, results, evidence, groupby))
    return (
        "Labels refined further: "
        + "; ".join(clauses)
        + ". These refinements are the main biological changes from the original/manual annotation: they add specificity "
        "when DE and curated markers support a more precise functional program, and they separate likely non-NK or "
        "contamination-like groups when lineage markers conflict with the original state."
    )


def label_rationale_clause(
    label: str,
    rows: pd.DataFrame,
    results: list[dict[str, Any]],
    evidence: dict[str, dict[str, Any]],
    groupby: str,
) -> str:
    cluster_ids = [str(item) for item in rows[groupby].astype(str).tolist()]
    originals = sorted(
        {
            str(item)
            for item in rows["top_original_label"].dropna().astype(str).tolist()
            if str(item) and str(item).lower() != "none"
        }
    )
    result_by_cluster = {str(result["cluster_id"]): result for result in results}
    support = collect_support_phrases(cluster_ids, result_by_cluster)
    genes = collect_top_genes(cluster_ids, evidence, max_genes=6)
    original_text = ", ".join(originals) if originals else "no dominant original label"
    support_text = "; ".join(support) if support else f"top genes included {', '.join(genes)}"
    if genes and support:
        support_text += f"; representative DE genes included {', '.join(genes)}"
    return (
        f"{label} (clusters {', '.join(cluster_ids)}; top original labels: {original_text}) "
        f"because {support_text}"
    )


def collect_support_phrases(
    cluster_ids: list[str],
    result_by_cluster: dict[str, dict[str, Any]],
    *,
    max_phrases: int = 2,
) -> list[str]:
    phrases = []
    seen = set()
    for cluster_id in cluster_ids:
        final = result_by_cluster.get(cluster_id, {}).get("final_decision", {})
        for phrase in final.get("evidence_summary", []):
            text = str(phrase).strip().rstrip(".")
            if not text or text in seen:
                continue
            seen.add(text)
            phrases.append(text)
            if len(phrases) >= max_phrases:
                return phrases
    return phrases


def collect_top_genes(
    cluster_ids: list[str],
    evidence: dict[str, dict[str, Any]],
    *,
    max_genes: int,
) -> list[str]:
    genes = []
    seen = set()
    for cluster_id in cluster_ids:
        for gene in evidence.get(cluster_id, {}).get("top_gene_names", [])[:10]:
            gene = str(gene)
            key = gene.upper()
            if key in seen:
                continue
            seen.add(key)
            genes.append(gene)
            if len(genes) >= max_genes:
                return genes
    return genes


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
        return "no alternative names were suggested because the approved candidate labels were judged sufficient"
    rows = []
    for _, row in alternative_df.iterrows():
        rows.append(f"cluster {row[groupby]}: {row['alternative_name_suggestion']}")
    return f"{plural(len(rows), 'alternative name suggestion')} were made ({'; '.join(rows)})"


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


def format_manual_composition(entries: list[dict[str, Any]]) -> str:
    parts = []
    for entry in entries or []:
        label = entry.get("label")
        fraction = entry.get("fraction")
        n_cells = entry.get("n_cells")
        if label is None:
            continue
        if fraction is None:
            parts.append(str(label))
        else:
            parts.append(f"{label} ({float(fraction):.3f}, n={n_cells})")
    return "; ".join(parts)
