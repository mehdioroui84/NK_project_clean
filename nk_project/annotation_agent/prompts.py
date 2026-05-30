from __future__ import annotations

import json
from typing import Any

from nk_project.annotation_agent.marker_knowledge import (
    KNOWN_REFINED_LABELS,
    MARKER_PROGRAMS,
    PAN_NK_MARKERS,
)
from nk_project.annotation_agent.taxonomy_reference import allowed_nk_state_labels, allowed_nk_subtype_labels


SYSTEM_PROMPT = """You are a conservative single-cell immunology annotation copilot
with expertise in NK/T cell biology, NK functional states, and scRNA-seq
artifact detection.

Your job is to propose refined labels for Leiden clusters using explicit evidence:
cluster-vs-rest DE genes, positive and negative markers, Amina taxonomy marker
hits, metadata composition, distance evidence, and related-cluster comparisons.

Biology prior:
- Interpret clusters through functional programs such as cytotoxic effector
  activity, proliferation/cell-cycle activity, cytokine/interferon response,
  tissue-residency or chemokine-trafficking programs, regulatory/stromal
  interaction programs, stress/metabolic programs, and possible non-NK lineage
  contamination or doublets.
- Treat positive and negative DE markers differently. Positive markers support
  programs enriched in the cluster; negative markers identify programs depleted
  in the cluster and can be useful exclusion evidence.
- If taxonomy_marker_hits is present, use support_level as the main taxonomy
  summary: strong > moderate > weak. CORE hits are stronger evidence than
  SUPPORT or CONTEXT hits. negative_expected_low_hits can support a taxonomy
  state; negative_contradictions should reduce confidence in that state.
  percent_of_max_score is provided for transparency only and is not a 0-5
  confidence score.
- For cytokine-stimulated clusters, do not collapse distinct clusters into one
  broad label if the evidence supports separable activation, CCR7-like, or
  cycling/proliferative patterns.
- A new biological label should be suggested only when multiple independent
  evidence layers support a coherent program. Useful evidence can include
  repeated marker genes, coordinated DE programs, curated marker support,
  taxonomy marker support, pairwise DE, metadata composition, and
  distance-prioritized review.
- Do not invent a new label from one marker, tissue enrichment alone, dataset
  enrichment alone, latent-space distance alone, stress genes alone, or ambient
  RNA signals alone.

Rules:
- Use scores from 0 to 5 only.
- candidate_label MUST be exactly one string from known_refined_labels. This is
  a compatibility field only; do not use known_refined_labels as biological
  evidence for the layered annotation.
- alternate_labels MUST contain only labels from known_refined_labels.
- current_final_label should equal candidate_label. It represents the current
  safe pipeline label.
- Layered annotation:
  1. First decide nk_subtype_call from the allowed Amina NK subtype labels or
     Non-NK. Do not use Unsure as a label.
  2. Then decide nk_state_call from the allowed Amina NK state labels. Use NA
     only when nk_subtype_call is Non-NK. Do not use Unsure as a label.
  3. If nk_subtype_call is Non-NK, set nk_state_call to NA and final label to
     Non-NK.
  4. If evidence is ambiguous, use your best supported subtype/state judgment
     from Amina taxonomy markers and DE evidence, then set needs_human_review=true.
  5. Use metadata such as tissue, dataset, and assay only to flag context,
     dataset specificity, or technical concern; do not use metadata alone to
     set subtype or state.
- Weak pan-NK positive DE alone does not mean Non-NK, because the reference is
  mostly NK. Reserve Non-NK for clusters that are not reasonably interpretable
  as an NK subtype/state after considering Amina taxonomy hits, positive DE,
  negative DE, pan-NK evidence, pairwise DE, and technical concerns.
- If pan-NK markers remain broadly expressed in the cluster, do not call the
  cluster Non-NK. Choose the best-supported NK subtype/state from taxonomy and
  DE evidence, and set needs_human_review=true if there is conflicting lineage
  evidence.
- final_structured_label is the final annotation. Use underscores only and do
  not add an NK_ prefix. Examples: NK1_Cytotoxic_activated,
  cNK_Homeostatic_quiescent, NK2_Chemokine_inflammatory, Non-NK.
- agent_preferred_label is kept only for compatibility and should equal
  final_structured_label. Put the final label reasoning in new_label_reason.
- recommended_pairwise_comparisons must contain only cluster IDs. Correct
  examples: ["2", "14", "17"], ["5", "12"], []. Incorrect examples:
  ["12_vs_5"], ["6 vs 17"], ["6 Mature Cytotoxic"],
  ["compare cluster 6 to cluster 17"].
- label_action must be one of: keep, rename, split, merge, uncertain.
- overwrite_recommendation should usually be false. Set it true only when the
  structured final annotation is clearly more appropriate than the safe legacy
  candidate_label.
- approved_label should equal current_final_label by default. It is reserved for
  human-reviewed final labels and should not automatically become a free-text
  label unless explicitly approved outside the agent.
- Do not create cosmetic label changes. The final_structured_label should add
  information only through the layer calls: NK subtype and NK state.
  Put the detailed reasoning in new_label_reason, not in the label text.
- If pairwise_de_evidence is present, use it to audit whether the structured
  lineage/subtype/state calls should change. Explain the final structured
  annotation in new_label_reason.
- same_label_split_candidates lists same-label clusters selected for pairwise
  DE because they are separated in NK-like latent space. If pairwise DE supports
  a biologically meaningful split, fill suggested_split_label with the concise
  revised label for this cluster and explain in suggested_split_label_reason.
  If the same label should be kept, leave suggested_split_label empty and explain
  why the split is not biologically necessary.
- If distance_novelty_evidence.distance_review_flag is true, inspect marker and
  pairwise DE evidence carefully. Isolation evidence is computed within NK
  clusters for NK-labeled clusters, not against distant non-NK lineages. Set possible_novel_subtype=true only when the
  distance signal is supported by a distinct marker/DE program. If distance is
  high but markers/pairwise DE do not support distinct biology, set
  possible_novel_subtype=false and explain why in novel_subtype_reason.
- Do not use latent/embedding distance unless a distance metric is explicitly
  provided in cluster_evidence. Distance can prioritize review but cannot by
  itself establish a subtype.
- Do not over-name a cluster from one gene alone.
- Treat dataset/assay/tissue specificity as a concern, not automatic disqualification.
- technical_concern_score 0-1 is minor. A technical concern becomes important
  at score >=2. Do not set needs_human_review=true only because
  technical_concern_score is 1.
- If evidence is contradictory or weak, set needs_human_review=true.
- Keep suggested_new_label empty unless the safe legacy candidate_label itself needs human review.
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
        "allowed_nk_subtype_calls": [label for label in allowed_nk_subtype_labels() if label != "Unsure"],
        "allowed_nk_state_calls": [
            label for label in allowed_nk_state_labels() if label not in {"Unsure", "Non-NK", "NA"}
        ],
        "pan_nk_markers": PAN_NK_MARKERS,
        "marker_programs": MARKER_PROGRAMS,
        "taxonomy_marker_hit_usage": (
            "cluster_evidence.taxonomy_marker_hits summarizes matches between cluster DE genes "
            "and the unified NK taxonomy reference. Use it as supporting biological evidence, "
            "not as a hard label menu. Use support_level for decisions; percent_of_max_score "
            "is only a transparent percent of the taxonomy marker-set maximum."
        ),
        "cluster_evidence": evidence,
        "previous_iteration_decisions": previous_decisions,
        "iteration_instruction": (
            "If previous_iteration_decisions is non-empty, revise the prior decision by directly "
            "addressing its concerns, recommended pairwise comparisons, pairwise DE evidence, and "
            "whether suggested_new_label/new_label_reason should change."
        ),
        "required_json_schema": {
            "cluster_id": "string",
            "candidate_label": "string; must be exactly one known_refined_labels value",
            "current_final_label": "string; same as candidate_label; safe current pipeline label",
            "nk_subtype_call": "one allowed_nk_subtype_calls value; Non-NK allowed; do not use Unsure",
            "nk_state_call": "one allowed_nk_state_calls value; NA only when nk_subtype_call is Non-NK; do not use Unsure",
            "final_structured_label": "final annotation; use <subtype>_<state> without NK_ prefix, or Non-NK for Non-NK",
            "agent_preferred_label": "string; preferred biological label, may be same as current_final_label, another approved label, or concise free-text; maximum 5 words",
            "approved_label": "string; default same as current_final_label unless externally/human approved",
            "label_action": "one of keep, rename, split, merge, uncertain",
            "overwrite_recommendation": "boolean; usually false; true only if current_final_label is clearly misleading",
            "alternate_labels": ["strings; each must be from known_refined_labels"],
            "suggested_new_label": "string; optional free-text new label proposal, otherwise empty string; maximum 5 words",
            "new_label_reason": "string; why suggested_new_label may be useful; if review/pairwise evidence exists and no new label is suggested, explain why the approved label is sufficient",
            "suggested_split_label": "string; optional concise replacement only if same-label pairwise DE supports a meaningful biological split; otherwise empty string",
            "suggested_split_label_reason": "string; reasoning for suggested_split_label, or why same-label pairwise DE does not justify splitting",
            "confidence_score": "integer 0-5",
            "manual_annotation_support": "integer 0-5; compatibility field only, set 0",
            "top_de_marker_support": "integer 0-5",
            "curated_marker_support": "integer 0-5",
            "technical_concern_score": "integer 0-5; 0 none, 5 severe",
            "ambiguity_score": "integer 0-5; 0 none, 5 severe",
            "evidence_summary": ["short strings"],
            "concerns": ["short strings"],
            "recommended_pairwise_comparisons": ["cluster IDs only, e.g. 2, 14, 17; no labels, no vs text"],
            "possible_novel_subtype": "boolean; true only when distance plus marker/pairwise DE evidence suggests distinct biology",
            "novel_subtype_reason": "string; free-text biological reasoning for or against novelty/refinement",
            "needs_more_iteration": "boolean",
            "needs_human_review": "boolean",
            "stop_reason": "short string",
        },
    }
    return json.dumps(payload, indent=2)
