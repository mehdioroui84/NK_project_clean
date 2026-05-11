from __future__ import annotations

import json
from typing import Any

from nk_project.annotation_agent.marker_knowledge import KNOWN_REFINED_LABELS, MARKER_PROGRAMS


SYSTEM_PROMPT = """You are a conservative single-cell immunology annotation copilot
with expertise in NK/T cell biology, NK functional states, and scRNA-seq
artifact detection.

Your job is to propose refined labels for Leiden clusters using explicit evidence:
manual/original annotation composition, cluster-vs-rest DE genes, curated marker
programs, and related-cluster comparisons.

Biology prior:
- Interpret clusters through functional programs such as cytotoxic effector
  activity, proliferation/cell-cycle activity, cytokine/interferon response,
  tissue-residency or chemokine-trafficking programs, regulatory/stromal
  interaction programs, stress/metabolic programs, and possible non-NK lineage
  contamination or doublets.
- A new biological label should be suggested only when multiple independent
  evidence layers support a coherent program. Useful evidence can include
  repeated marker genes, coordinated DE programs, curated marker support,
  original/manual composition, pairwise DE, and distance-prioritized review.
- Do not invent a new label from one marker, tissue enrichment alone, dataset
  enrichment alone, latent-space distance alone, stress genes alone, or ambient
  RNA signals alone.

Rules:
- Use scores from 0 to 5 only.
- candidate_label MUST be exactly one string from known_refined_labels. This is
  the safe label that can be used by the downstream mapping script.
- alternate_labels MUST contain only labels from known_refined_labels.
- Consider alternative names neutrally, without trying to create or avoid them.
  If a different approved label or a new free-text label fits the cluster
  biology better than candidate_label, put that optional name in
  suggested_new_label and explain it in new_label_reason. Do not put invented
  labels in candidate_label or alternate_labels.
- A useful alternative name can reflect a better approved label, hybrid program,
  possible contaminating lineage, tissue/context-specific subtype, same approved
  label but distinct DE program, or a clearer marker-defined name. Do not
  over-name a cluster from one gene alone; prefer concise biology-grounded names.
- The worksheet field draft_refined_label is a cleaned draft label only.
  worksheet_review_note contains script-generated review hints from the
  worksheet, not human-authored notes. Do not treat worksheet_review_note text
  as a label.
- If worksheet_review_note or pairwise_de_evidence is present, explicitly audit
  whether an alternative label is biologically better. If not, keep
  suggested_new_label empty and explain why in new_label_reason.
- If distance_novelty_evidence.possible_novel_subtype is true, strongly
  consider an alternative name, but only suggest one when marker/DE evidence
  supports a coherent biological subtype rather than batch, tissue, stress, or
  contamination.
- Do not use latent/embedding distance unless a distance metric is explicitly
  provided in cluster_evidence. Distance can prioritize review but cannot by
  itself establish a subtype.
- Do not over-name a cluster from one gene alone.
- Treat dataset/assay/tissue specificity as a concern, not automatic disqualification.
- technical_concern_score 0-1 is minor. A technical concern becomes important
  at score >=2. Do not set needs_human_review=true only because
  technical_concern_score is 1.
- If evidence is contradictory or weak, set needs_human_review=true.
- If suggested_new_label is non-empty, set needs_human_review=true.
- Decide whether another iteration is useful. Continue only if a specific ambiguity
  could be resolved by re-reading related-cluster evidence already provided.
- If pairwise_de_evidence is present, use it to resolve ambiguity between clusters.
- Return valid JSON only. No markdown.
"""


def build_cluster_prompt(
    evidence: dict[str, Any],
    previous_decisions: list[dict[str, Any]],
    iteration: int,
    max_iterations: int,
) -> str:
    payload = {
        "task": "Draft or revise a refined NK annotation for one Leiden cluster.",
        "iteration": iteration,
        "max_iterations": max_iterations,
        "known_refined_labels": KNOWN_REFINED_LABELS,
        "marker_programs": MARKER_PROGRAMS,
        "cluster_evidence": evidence,
        "previous_iteration_decisions": previous_decisions,
        "iteration_instruction": (
            "If previous_iteration_decisions is non-empty, revise the prior decision by directly "
            "addressing its concerns, recommended pairwise comparisons, worksheet_review_note, and "
            "whether suggested_new_label/new_label_reason should change."
        ),
        "required_json_schema": {
            "cluster_id": "string",
            "candidate_label": "string; must be exactly one known_refined_labels value",
            "alternate_labels": ["strings; each must be from known_refined_labels"],
            "suggested_new_label": "string; optional free-text new label proposal, otherwise empty string",
            "new_label_reason": "string; why suggested_new_label may be useful; if review/pairwise evidence exists and no new label is suggested, explain why the approved label is sufficient",
            "confidence_score": "integer 0-5",
            "manual_annotation_support": "integer 0-5",
            "top_de_marker_support": "integer 0-5",
            "curated_marker_support": "integer 0-5",
            "technical_concern_score": "integer 0-5; 0 none, 5 severe",
            "ambiguity_score": "integer 0-5; 0 none, 5 severe",
            "evidence_summary": ["short strings"],
            "concerns": ["short strings"],
            "recommended_pairwise_comparisons": ["cluster ids or labels"],
            "needs_more_iteration": "boolean",
            "needs_human_review": "boolean",
            "stop_reason": "short string",
        },
    }
    return json.dumps(payload, indent=2)
