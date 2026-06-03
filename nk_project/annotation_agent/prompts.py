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
- Before choosing an NK subtype, check whether positive marker programs support
  a non-NK lineage. If strong non-NK lineage markers are enriched and pan-NK
  markers are depleted or weak, call Non-NK even if shared NK/T inflammatory
  genes such as GZMK, XCL1, or IFNG are present.
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
