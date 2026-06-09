from __future__ import annotations

import json
from typing import Any

from nk_project.annotation_agent.marker_knowledge import PAN_NK_MARKERS
from nk_project.annotation_agent.taxonomy_reference import allowed_nk_state_labels, allowed_nk_subtype_labels


SYSTEM_PROMPT = """You annotate one Leiden cluster from evidence only.

Use only the provided cluster evidence: cluster-vs-rest DE genes,
positive/negative markers, Amina taxonomy marker hits, pairwise DE if present,
and metadata context.

Rules:
- Choose nk_subtype_call from allowed_nk_subtype_calls or Non-NK.
- Choose nk_state_call from allowed_nk_state_calls. Use NA only when subtype is Non-NK.
- Do not use Unsure. If evidence is uncertain, make the best evidence-based call
  and set needs_human_review=true.
- final_structured_label is the training-ready structured label:
  <subtype>_<state>, or Non-NK.
- free_label is your concise biological interpretation. It may equal
  final_structured_label or be a clearer DE-driven label.
- Do not add an NK_ prefix.
- Weak pan-NK DE is not evidence for Non-NK because the background is mostly NK.
  If pan-NK markers remain broadly expressed, choose the best NK subtype/state
  and flag review rather than calling Non-NK.
- Before choosing an NK subtype, check whether the DE evidence supports a
  non-NK lineage. Use the curated T-lineage sanity check for the T/NK boundary;
  for other lineages, rely on the cluster's own top DE genes. If strong non-NK
  lineage markers are enriched and pan-NK markers are depleted or weak, call
  Non-NK even if shared NK/T inflammatory genes such as GZMK, XCL1, or IFNG are
  present.
- Use tissue, dataset, and assay composition as metadata context for the cluster.
  Tissue may or may not provide biological context or technical concern; dataset
  and assay enrichment may or may not reflect technical concern. Use your best judgment.
- Return valid JSON only. No markdown.
"""


PAIRWISE_SPLIT_SYSTEM_PROMPT = """You audit two clusters that received the same
structured annotation.

Use pairwise DE to decide whether their free labels should be split. The
structured label can stay the same; this audit only updates free_label and
free_label_reason when pairwise DE shows a coherent biological difference.

Return valid JSON only. No markdown.
"""


MACHINE_REVIEW_SYSTEM_PROMPT = """You review one cluster after first-pass agent annotation.

Use only the provided first-pass agent structured/free label, first-pass agent
rationale, metadata context, and cluster-specific marker evidence.
Do not use or assume any external or prior human annotation.

Goal:
- Find discrepancies between the first-pass agent label and marker evidence.
- Produce a reviewed structured label and concise reviewed free label for
  downstream use.

Rules:
- Marker/DE evidence and lineage consistency should drive the reviewed label.
- reviewed_structured_label must be one allowed structured NK label or Non-NK.
- The background is mostly NK/NK-enriched. Negative pan-NK DE means lower than
  other NK clusters, not necessarily absent NK identity. Do not revise an NK
  first-pass call to Non-NK based only on relatively low GNLY, GZMB, PRF1, NKG7,
  or KLRF1 in cluster-vs-rest DE.
- If strong non-NK lineage markers support another lineage and pan-NK support is
  weak or depleted, revise to Non-NK.
- T and NK programs can overlap. Do not call T/Non-NK based only on activation,
  proliferation, cytokine, chemokine, stress, CCR7, IL2RA, IRF4, GZMK, or IFNG
  signals. Require coherent core non-NK lineage evidence plus weak/depleted
-  pan-NK support. If evidence is mixed, keep the best first-pass NK label and
  flag review.
- If the free label/rationale and markers clearly contradict the structured
  label, revise the label.
- If the evidence is mixed but still biologically plausible, keep the best
  concise label and set needs_human_review=true.
- reviewed_label_short must be concise: ideally fewer than 4 underscores
  and fewer than 40 characters.
- When shortening labels, preserve the most informative biology token when
  useful, such as subtype/state, contamination lineage, key marker,
  tissue/context, stress, or proliferation.
- Style examples:
  NK1_cytotoxic_activated_possible_myeloid_contamination -> NK1_Cytotoxic_contam_myeloid
  NK1_Cytotoxic_activated_FGFBP2+_GZMH+_blood -> NK1_Cytotoxic_activated_FGFBP2+_blood
  CD3+_T_cells_with_IFNG_GZMK_inflammatory_and_ER_stress_signature -> T_inflammatory_stress
- reviewed_label_detail can explain the biology in one short phrase.
- Do not add an NK_ prefix.
- Return valid JSON only. No markdown.
"""


def build_cluster_prompt(
    evidence: dict[str, Any],
    previous_decisions: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
) -> str:
    payload = {
        "task": "Annotate one Leiden cluster.",
        "iteration": iteration,
        "max_iterations": max_iterations,
        "allowed_nk_subtype_calls": [label for label in allowed_nk_subtype_labels() if label != "Unsure"],
        "allowed_nk_state_calls": [
            label for label in allowed_nk_state_labels() if label not in {"Unsure", "Non-NK", "NA"}
        ],
        "pan_nk_markers": PAN_NK_MARKERS,
        "cluster_evidence": evidence,
        "previous_iteration_decisions": previous_decisions,
        "required_json_schema": {
            "cluster_id": "string",
            "nk_subtype_call": "one allowed subtype or Non-NK; never Unsure",
            "nk_state_call": "one allowed state, or NA only when subtype is Non-NK; never Unsure",
            "final_structured_label": "<subtype>_<state> or Non-NK; no NK_ prefix",
            "free_label": "concise biological label; may equal final_structured_label",
            "free_label_reason": "why the free label is appropriate, citing DE/taxonomy/pairwise evidence",
            "confidence_score": "integer 0-5",
            "tissue_specificity_score": "integer 0-5; 0 none, 5 strong tissue-specific context",
            "dataset_assay_specificity_score": "integer 0-5; 0 broad, 5 dominated by dataset/assay",
            "evidence_summary": ["short strings"],
            "concerns": ["short strings"],
            "needs_human_review": "boolean",
            "human_review_reason": "short reason if needs_human_review is true, otherwise empty string",
            "needs_more_iteration": "boolean",
            "stop_reason": "short string",
        },
    }
    return json.dumps(payload, indent=2)


def build_pairwise_split_prompt(
    *,
    cluster_a: str,
    cluster_b: str,
    structured_label: str,
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any],
    decision_a: dict[str, Any],
    decision_b: dict[str, Any],
) -> str:
    payload = {
        "task": "Audit whether same-structured-label clusters need different free labels.",
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "shared_final_structured_label": structured_label,
        "cluster_a_current_decision": {
            "free_label": decision_a.get("free_label", ""),
            "free_label_reason": decision_a.get("free_label_reason", ""),
        },
        "cluster_b_current_decision": {
            "free_label": decision_b.get("free_label", ""),
            "free_label_reason": decision_b.get("free_label_reason", ""),
        },
        "cluster_a_evidence": evidence_a,
        "cluster_b_evidence": evidence_b,
        "required_json_schema": {
            "cluster_a": "string",
            "cluster_b": "string",
            "split_supported": "boolean",
            "cluster_a_free_label": "concise label for cluster_a; may equal current free_label",
            "cluster_a_free_label_reason": "pairwise-DE-based reason",
            "cluster_b_free_label": "concise label for cluster_b; may equal current free_label",
            "cluster_b_free_label_reason": "pairwise-DE-based reason",
            "needs_human_review": "boolean",
            "human_review_reason": "short reason if review is needed",
        },
    }
    return json.dumps(payload, indent=2)


def build_machine_review_prompt(cluster_report: dict[str, Any]) -> str:
    allowed_subtypes = [label for label in allowed_nk_subtype_labels() if label != "Unsure"]
    allowed_states = [label for label in allowed_nk_state_labels() if label not in {"Unsure", "Non-NK", "NA"}]
    allowed_structured = ["Non-NK"] + [
        f"{subtype}_{state}"
        for subtype in allowed_subtypes
        if subtype != "Non-NK"
        for state in allowed_states
    ]
    payload = {
        "task": "Review one first-pass agent cluster label for consistency and concise downstream use.",
        "allowed_reviewed_structured_labels": allowed_structured,
        "cluster_report": cluster_report,
        "required_json_schema": {
            "cluster_id": "string",
            "review_decision": "keep, revise, or flag_only",
            "reviewed_structured_label": "one allowed reviewed structured label or Non-NK",
            "reviewed_label_short": "concise final label, ideally fewer than 4 underscores and fewer than 40 characters",
            "reviewed_label_detail": "one short phrase explaining the reviewed label",
            "discrepancy_flags": ["short strings such as structured_free_mismatch, marker_label_mismatch, possible_contamination"],
            "review_reason": "brief reason citing first-pass label/rationale, marker evidence, and metadata context",
            "confidence_score": "integer 0-5",
            "needs_human_review": "boolean",
            "human_review_reason": "short reason if review is needed, otherwise empty string",
        },
    }
    return json.dumps(payload, indent=2)
