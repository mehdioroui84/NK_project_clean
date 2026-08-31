"""LangGraph workflow for independent evidence-driven cluster annotation."""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, TypedDict

from nk_project.annotation_agent.evidence_multiagent_prompts import (
    BIOLOGICAL_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REVIEWER_SYSTEM_PROMPT,
    TECHNICAL_SYSTEM_PROMPT,
    biological_prompt,
    planner_integration_prompt,
    planner_revision_prompt,
    reviewer_prompt,
    technical_prompt,
)
from nk_project.annotation_agent.evidence_tools import get_ranked_pathways


MAX_CALLS_PER_ROLE = 3
INITIAL_PATHWAY_COUNT = 20
ADDITIONAL_PATHWAY_COUNT = 10
CONFIDENCE_PATTERN = re.compile(r"^[1-5]/5$")


class EvidenceAgentState(TypedDict, total=False):
    resolution: str
    cluster_context: dict[str, Any]
    project_root: str
    generated_date: str
    active_llm: str
    temperature: float
    llm_retries: int
    retry_sleep: float
    planner_calls: int
    biological_calls: int
    technical_calls: int
    reviewer_calls: int
    pending_pathway_request: dict[str, Any]
    planner_actions: list[dict[str, Any]]
    pathway_batches: list[dict[str, Any]]
    biological_reports: list[dict[str, Any]]
    mitochondrial_attribution_review: dict[str, Any]
    technical_report: dict[str, Any]
    planner_draft: dict[str, Any]
    reviewer_report: dict[str, Any]
    reviewer_reports: list[dict[str, Any]]
    final_decision: dict[str, Any]
    trace: list[dict[str, Any]]


def run_evidence_multiagent(
    cluster_context: dict[str, Any],
    *,
    resolution: str,
    project_root: Path,
    generated_date: str,
    active_llm: str,
    temperature: float = 0.0,
    llm_retries: int = 5,
    retry_sleep: float = 5.0,
) -> dict[str, Any]:
    workflow = build_evidence_multiagent_graph()
    initial: EvidenceAgentState = {
        "resolution": resolution,
        "cluster_context": cluster_context,
        "project_root": str(project_root.resolve()),
        "generated_date": generated_date,
        "active_llm": active_llm,
        "temperature": float(temperature),
        "llm_retries": int(llm_retries),
        "retry_sleep": float(retry_sleep),
        "planner_calls": 0,
        "biological_calls": 0,
        "technical_calls": 0,
        "reviewer_calls": 0,
        "pending_pathway_request": {},
        "planner_actions": [],
        "pathway_batches": [],
        "biological_reports": [],
        "mitochondrial_attribution_review": {},
        "technical_report": {},
        "planner_draft": {},
        "reviewer_report": {},
        "reviewer_reports": [],
        "final_decision": {},
        "trace": [],
    }
    result = workflow.invoke(initial, config={"recursion_limit": 30})
    return {
        "cluster": str(cluster_context["cluster"]),
        "leiden_resolution": resolution,
        "final_decision": result["final_decision"],
        "specialist_reports": {
            "biological_evidence": result["biological_reports"],
            "mitochondrial_attribution_review": result[
                "mitochondrial_attribution_review"
            ],
            "technical_context": result["technical_report"],
            "critical_reviews": result["reviewer_reports"],
        },
        "planner_actions": result["planner_actions"],
        "pathway_batches_retrieved": result["pathway_batches"],
        "role_call_counts": {
            "planner": int(result["planner_calls"]),
            "biological_evidence": int(result["biological_calls"]),
            "technical_context": int(result["technical_calls"]),
            "critical_reviewer": int(result["reviewer_calls"]),
        },
        "trace": result["trace"],
    }


def build_evidence_multiagent_graph():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "LangGraph is required. Install dependencies with "
            "`pip install langgraph langchain-openai`."
        ) from exc

    graph = StateGraph(EvidenceAgentState)
    graph.add_node("planner_select_pathways", planner_select_pathways)
    graph.add_node("retrieve_pathways", retrieve_pathways)
    graph.add_node("biological_evidence", biological_evidence)
    graph.add_node("technical_context", technical_context)
    graph.add_node("planner_integrate", planner_integrate)
    graph.add_node("critical_review", critical_review)
    graph.add_node("planner_revise", planner_revise)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("planner_select_pathways")
    graph.add_conditional_edges(
        "planner_select_pathways",
        route_after_initial_planner,
        {"retrieve": "retrieve_pathways", "biological": "biological_evidence"},
    )
    graph.add_edge("retrieve_pathways", "biological_evidence")
    graph.add_conditional_edges(
        "biological_evidence",
        route_after_biological,
        {"technical": "technical_context", "planner": "planner_integrate"},
    )
    graph.add_edge("technical_context", "planner_integrate")
    graph.add_conditional_edges(
        "planner_integrate",
        route_after_planner_integration,
        {"retrieve": "retrieve_pathways", "review": "critical_review"},
    )
    graph.add_conditional_edges(
        "critical_review",
        route_after_review,
        {"revise": "planner_revise", "finalize": "finalize"},
    )
    graph.add_edge("planner_revise", "critical_review")
    graph.add_edge("finalize", END)
    return graph.compile()


def planner_select_pathways(state: EvidenceAgentState) -> dict[str, Any]:
    total = int(state["cluster_context"]["total_significant_pathways"])
    if total == 0:
        request = {
            "number_of_pathways": 0,
            "pathway_request_reason": "No significant pathways are available.",
        }
        return {
            "pending_pathway_request": request,
            "planner_actions": list(state["planner_actions"])
            + [{"action": "skip_pathway_retrieval", **request}],
            "trace": _append_trace(state, "workflow", "No pathways available; skipped initial retrieval."),
        }

    number = min(INITIAL_PATHWAY_COUNT, total)
    request = {
        "number_of_pathways": number,
        "pathway_request_reason": (
            "Deterministic initial retrieval of the top 20 significant pathways."
        ),
    }
    return {
        "pending_pathway_request": {
            "start_rank": 1,
            **request,
        },
        "planner_actions": list(state["planner_actions"])
        + [{"action": "request_initial_pathways", "start_rank": 1, **request}],
        "trace": _append_trace(
            state,
            "workflow",
            f"Selected pathway ranks 1-{number} for initial review without using a planner call.",
        ),
    }


def route_after_initial_planner(state: EvidenceAgentState) -> str:
    return (
        "retrieve"
        if int(state.get("pending_pathway_request", {}).get("number_of_pathways", 0)) > 0
        else "biological"
    )


def retrieve_pathways(state: EvidenceAgentState) -> dict[str, Any]:
    request = state["pending_pathway_request"]
    batch = get_ranked_pathways(
        project_root=Path(state["project_root"]),
        resolution=state["resolution"],
        cluster=str(state["cluster_context"]["cluster"]),
        start_rank=int(request["start_rank"]),
        number_of_pathways=int(request["number_of_pathways"]),
        generated_date=state["generated_date"],
    )
    batches = list(state["pathway_batches"])
    batches.append(batch)
    returned = batch["pathways_returned"]
    rank_text = (
        f"{returned[0]['rank']}-{returned[-1]['rank']}" if returned else "none"
    )
    return {
        "pathway_batches": batches,
        "pending_pathway_request": {},
        "trace": _append_trace(
            state,
            "pathway_tool",
            f"Retrieved pathway ranks {rank_text} ({len(returned)} pathways).",
        ),
    }


def biological_evidence(state: EvidenceAgentState) -> dict[str, Any]:
    _require_role_capacity(state, "biological")
    if state["pathway_batches"]:
        current_batch = state["pathway_batches"][-1]
    else:
        current_batch = {
            "leiden_resolution": state["resolution"],
            "cluster": str(state["cluster_context"]["cluster"]),
            "total_significant_pathways": 0,
            "pathways_returned": [],
            "next_start_rank_if_more_pathways_are_available": None,
        }
    previous = state["biological_reports"][-1] if state["biological_reports"] else None
    response = _call_role(
        state,
        role="biological",
        system_prompt=BIOLOGICAL_SYSTEM_PROMPT,
        user_prompt=biological_prompt(
            cluster_context=state["cluster_context"],
            pathway_batch=current_batch,
            previous_biological_report=previous,
        ),
    )
    report = _validate_biological_report(response)
    reports = list(state["biological_reports"])
    reports.append(report)
    return {
        "biological_calls": int(state["biological_calls"]) + 1,
        "biological_reports": reports,
        "trace": _append_trace(
            state,
            "biological_evidence",
            f"Completed biological review {len(reports)}.",
        ),
    }


def route_after_biological(state: EvidenceAgentState) -> str:
    return "technical" if not state.get("technical_report") else "planner"


def technical_context(state: EvidenceAgentState) -> dict[str, Any]:
    _require_role_capacity(state, "technical")
    mitochondrial_review = _mitochondrial_attribution_review(
        state["cluster_context"]
    )
    response = _call_role(
        state,
        role="technical",
        system_prompt=TECHNICAL_SYSTEM_PROMPT,
        user_prompt=technical_prompt(
            state["cluster_context"], mitochondrial_review
        ),
    )
    report = _validate_technical_report(response)
    return {
        "technical_calls": int(state["technical_calls"]) + 1,
        "mitochondrial_attribution_review": mitochondrial_review,
        "technical_report": report,
        "trace": _append_trace(state, "technical_context", "Completed technical-context review."),
    }


def planner_integrate(state: EvidenceAgentState) -> dict[str, Any]:
    _require_role_capacity(state, "planner")
    call_number = int(state["planner_calls"]) + 1
    response = _call_role(
        state,
        role="planner",
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=planner_integration_prompt(
            cluster_context=state["cluster_context"],
            pathway_batches=state["pathway_batches"],
            biological_reports=state["biological_reports"],
            technical_report=state["technical_report"],
            planner_call_number=call_number,
            maximum_planner_calls=MAX_CALLS_PER_ROLE,
        ),
    )
    response = _validate_planner_integration(response, state, call_number)
    update: dict[str, Any] = {
        "planner_calls": call_number,
        "planner_actions": list(state["planner_actions"]) + [response],
        "trace": _append_trace(
            state,
            "planner",
            f"Integration decision: {response['action']}.",
        ),
    }
    if response["action"] == "request_more_pathways":
        next_rank = _next_pathway_rank(state["pathway_batches"])
        update["pending_pathway_request"] = {
            "start_rank": next_rank,
            "number_of_pathways": response["number_of_additional_pathways"],
            "pathway_request_reason": response["reason_for_action"],
        }
    else:
        update["planner_draft"] = response["draft_annotation"]
    return update


def route_after_planner_integration(state: EvidenceAgentState) -> str:
    return "retrieve" if state.get("pending_pathway_request") else "review"


def critical_review(state: EvidenceAgentState) -> dict[str, Any]:
    _require_role_capacity(state, "reviewer")
    response = _call_role(
        state,
        role="reviewer",
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        user_prompt=reviewer_prompt(
            cluster_context=state["cluster_context"],
            pathway_batches=state["pathway_batches"],
            biological_reports=state["biological_reports"],
            technical_report=state["technical_report"],
            draft_annotation=state["planner_draft"],
        ),
    )
    report = _validate_reviewer_report(response)
    reports = list(state["reviewer_reports"])
    reports.append(report)
    return {
        "reviewer_calls": int(state["reviewer_calls"]) + 1,
        "reviewer_report": report,
        "reviewer_reports": reports,
        "trace": _append_trace(
            state,
            "critical_reviewer",
            f"Review decision: {report['review_decision']}.",
        ),
    }


def route_after_review(state: EvidenceAgentState) -> str:
    wants_revision = state["reviewer_report"]["review_decision"] == "revise"
    has_capacity = int(state["planner_calls"]) < MAX_CALLS_PER_ROLE
    return "revise" if wants_revision and has_capacity else "finalize"


def planner_revise(state: EvidenceAgentState) -> dict[str, Any]:
    _require_role_capacity(state, "planner")
    response = _call_role(
        state,
        role="planner",
        system_prompt=PLANNER_SYSTEM_PROMPT,
        user_prompt=planner_revision_prompt(
            cluster_context=state["cluster_context"],
            pathway_batches=state["pathway_batches"],
            biological_reports=state["biological_reports"],
            technical_report=state["technical_report"],
            draft_annotation=state["planner_draft"],
            reviewer_report=state["reviewer_report"],
        ),
    )
    draft = _validate_draft(response, str(state["cluster_context"]["cluster"]))
    return {
        "planner_calls": int(state["planner_calls"]) + 1,
        "planner_draft": draft,
        "planner_actions": list(state["planner_actions"])
        + [{"action": "revise_annotation", "draft_annotation": draft}],
        "trace": _append_trace(state, "planner", "Revised draft after critical review."),
    }


def finalize(state: EvidenceAgentState) -> dict[str, Any]:
    final = dict(state["planner_draft"])
    review = state.get("reviewer_report", {})
    revision_unavailable = (
        review.get("review_decision") == "revise"
        and int(state["planner_calls"]) >= MAX_CALLS_PER_ROLE
    )
    if revision_unavailable:
        final["requires_human_review"] = True
        prior = str(final.get("human_review_reason") or "").strip()
        limit_reason = (
            "Critical reviewer requested revision after the planner reached its "
            "three-call limit."
        )
        final["human_review_reason"] = "; ".join(
            part for part in [prior, limit_reason] if part
        )
    final["critical_review_decision"] = review.get("review_decision", "not_run")
    return {
        "final_decision": final,
        "trace": _append_trace(state, "workflow", "Finalized cluster annotation."),
    }


def _call_role(
    state: EvidenceAgentState,
    *,
    role: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    try:
        from nk_project.annotation_agent.llm_factory import get_active_llm
    except ImportError as exc:
        raise ImportError(
            "LLM dependencies are required. Install `langgraph` and "
            "`langchain-openai`."
        ) from exc

    llm = get_active_llm(
        temperature=float(state["temperature"]),
        active_llm=state["active_llm"],
    )
    cluster = str(state["cluster_context"]["cluster"])
    role_iteration = _role_count(state, role) + 1
    started = time.time()
    print(
        f"[LLM_START] cluster={cluster} role={role} call={role_iteration} "
        f"model={state['active_llm']}",
        flush=True,
    )
    response = _invoke_with_retry(
        llm,
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        retries=int(state["llm_retries"]),
        sleep_seconds=float(state["retry_sleep"]),
        cluster_id=f"{cluster}:{role}",
        iteration=role_iteration,
    )
    print(
        f"[LLM_DONE] cluster={cluster} role={role} "
        f"elapsed={time.time() - started:.1f}s",
        flush=True,
    )
    parsed = _parse_json_response(response.content)
    if not isinstance(parsed, dict):
        raise TypeError(f"{role} response must be a JSON object")
    return parsed


def _invoke_with_retry(
    llm,
    messages: list[dict[str, str]],
    *,
    retries: int,
    sleep_seconds: float,
    cluster_id: str,
    iteration: int,
):
    """Use the established annotation-agent retry behavior without old graph state."""
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"[LLM_ATTEMPT] cluster={cluster_id} iteration={iteration} "
                f"attempt={attempt}/{retries}",
                flush=True,
            )
            return llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - transient API errors are retried.
            last_error = exc
            if attempt >= retries:
                break
            wait = sleep_seconds * attempt
            print(
                f"[WARN] LLM call failed on attempt {attempt}/{retries}: "
                f"{type(exc).__name__}: {exc}. Retrying in {wait:.1f}s...",
                flush=True,
            )
            time.sleep(wait)
    if last_error is None:
        raise RuntimeError("LLM invocation failed without an exception")
    raise last_error


def _parse_json_response(content: str) -> dict[str, Any]:
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(content[start : end + 1])
    if not isinstance(value, dict):
        raise TypeError("LLM response must be a JSON object")
    return value


def _validate_biological_report(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "biological_summary": _required_text(response, "biological_summary"),
        "supported_biological_programs": _string_list(response, "supported_biological_programs"),
        "single_best_supported_biological_interpretation": _required_text(
            response, "single_best_supported_biological_interpretation"
        ),
        "broad_identity_evidence": _required_text(response, "broad_identity_evidence"),
        "subtype_evidence": _required_text(response, "subtype_evidence"),
        "functional_state_evidence": _required_text(response, "functional_state_evidence"),
        "conflicting_evidence": _string_list(response, "conflicting_evidence"),
        "evidence_limitations": _string_list(response, "evidence_limitations"),
        "additional_pathways_might_help": _required_bool(response, "additional_pathways_might_help"),
        "question_for_additional_pathways": _optional_text(response, "question_for_additional_pathways"),
    }


def _validate_technical_report(response: dict[str, Any]) -> dict[str, Any]:
    level = _required_text(response, "technical_concern_level").lower()
    if level not in {"low", "moderate", "high"}:
        raise ValueError("technical_concern_level must be low, moderate, or high")
    return {
        "technical_context_summary": _required_text(response, "technical_context_summary"),
        "tissue_representation_assessment": _required_text(response, "tissue_representation_assessment"),
        "dataset_and_assay_representation_assessment": _required_text(
            response, "dataset_and_assay_representation_assessment"
        ),
        "mitochondrial_signal_assessment": _required_text(response, "mitochondrial_signal_assessment"),
        "technical_concern_level": level,
        "limitations": _string_list(response, "limitations"),
    }


def _validate_planner_integration(
    response: dict[str, Any],
    state: EvidenceAgentState,
    call_number: int,
) -> dict[str, Any]:
    action = _required_text(response, "action")
    if action not in {"request_more_pathways", "draft_annotation"}:
        raise ValueError("Planner action must be request_more_pathways or draft_annotation")
    reason = _required_text(response, "reason_for_action")
    if action == "draft_annotation":
        draft = response.get("draft_annotation")
        if not isinstance(draft, dict):
            raise TypeError("draft_annotation must be a JSON object")
        return {
            "action": action,
            "number_of_additional_pathways": 0,
            "reason_for_action": reason,
            "draft_annotation": _validate_draft(
                draft, str(state["cluster_context"]["cluster"])
            ),
        }

    if call_number >= MAX_CALLS_PER_ROLE:
        raise ValueError("Planner must draft on its final allowed call")
    if int(state["biological_calls"]) >= MAX_CALLS_PER_ROLE:
        raise ValueError("Biological agent has reached its call limit")
    next_rank = _next_pathway_rank(state["pathway_batches"])
    total = int(state["cluster_context"]["total_significant_pathways"])
    if next_rank is None or next_rank > total:
        raise ValueError("No additional ranked pathways are available")
    requested_number = _required_int(response, "number_of_additional_pathways")
    if len(state["pathway_batches"]) >= 2:
        raise ValueError("Only one additional pathway batch is allowed")
    remaining = total - next_rank + 1
    actual_number = min(ADDITIONAL_PATHWAY_COUNT, remaining)
    allowed_requests = {ADDITIONAL_PATHWAY_COUNT, actual_number}
    if requested_number not in allowed_requests:
        raise ValueError(
            f"Additional pathway request must ask for 10 pathways (or the "
            f"{actual_number} remaining pathways); found {requested_number}"
        )
    if response.get("draft_annotation") not in ({}, None):
        raise ValueError("draft_annotation must be empty when requesting pathways")
    return {
        "action": action,
        "number_of_additional_pathways": actual_number,
        "reason_for_action": reason,
        "draft_annotation": {},
    }


def _validate_reviewer_report(response: dict[str, Any]) -> dict[str, Any]:
    decision = _required_text(response, "review_decision").lower()
    if decision not in {"pass", "revise"}:
        raise ValueError("review_decision must be pass or revise")
    result = {
        "review_decision": decision,
        "grounding_assessment": _required_text(response, "grounding_assessment"),
        "unsupported_or_overstated_claims": _string_list(response, "unsupported_or_overstated_claims"),
        "missing_major_technical_caveats": _string_list(
            response, "missing_major_technical_caveats"
        ),
        "confidence_assessment": _required_text(response, "confidence_assessment"),
        "required_changes": _string_list(response, "required_changes"),
    }
    if decision == "revise" and not result["required_changes"]:
        raise ValueError("Reviewer must provide required_changes when requesting revision")
    return result


def _validate_draft(response: dict[str, Any], cluster: str) -> dict[str, Any]:
    response_cluster = _required_text(response, "cluster")
    if response_cluster != cluster:
        raise ValueError(
            f"Draft cluster {response_cluster!r} does not match requested cluster {cluster!r}"
        )
    confidence = _required_text(response, "confidence_score")
    if not CONFIDENCE_PATTERN.fullmatch(confidence):
        raise ValueError("confidence_score must be formatted from 1/5 through 5/5")
    requires_review = _required_bool(response, "requires_human_review")
    review_reason = _optional_text(response, "human_review_reason")
    if requires_review and not review_reason:
        raise ValueError("human_review_reason is required when requires_human_review is true")
    return {
        "cluster": cluster,
        "evidence_summary": _required_text(response, "evidence_summary"),
        "biological_interpretation": _required_text(response, "biological_interpretation"),
        "proposed_broad_identity": _required_text(response, "proposed_broad_identity"),
        "proposed_subtype": _required_text(response, "proposed_subtype"),
        "proposed_functional_state": _required_text(response, "proposed_functional_state"),
        "technical_caveats": _string_list(response, "technical_caveats"),
        "confidence_score": confidence,
        "requires_human_review": requires_review,
        "human_review_reason": review_reason,
    }


def _next_pathway_rank(batches: list[dict[str, Any]]) -> int | None:
    if not batches:
        return 1
    value = batches[-1].get("next_start_rank_if_more_pathways_are_available")
    return int(value) if value is not None else None


def _mitochondrial_attribution_review(
    cluster_context: dict[str, Any],
) -> dict[str, Any]:
    """Apply the agreed rank-based gate before exposing mitochondrial metadata."""
    genes = cluster_context["attribution_gene_selection"][
        "genes_ordered_from_highest_to_lowest_attribution"
    ]
    top10 = [str(gene) for gene in genes[:10]]
    top20 = [str(gene) for gene in genes[:20]]
    mt_top10 = [gene for gene in top10 if gene.upper().startswith("MT-")]
    mt_top20 = [gene for gene in top20 if gene.upper().startswith("MT-")]
    activated = len(mt_top10) >= 4 or len(mt_top20) >= 6
    return {
        "mitochondrial_metadata_review_activated": activated,
        "activation_rule": (
            "at least 4 MT- genes among the top 10 attribution genes or "
            "at least 6 MT- genes among the top 20"
        ),
        "mitochondrial_genes_in_top_10": mt_top10,
        "number_of_mitochondrial_genes_in_top_10": len(mt_top10),
        "mitochondrial_genes_in_top_20": mt_top20,
        "number_of_mitochondrial_genes_in_top_20": len(mt_top20),
    }


def _role_count(state: EvidenceAgentState, role: str) -> int:
    key = {
        "planner": "planner_calls",
        "biological": "biological_calls",
        "technical": "technical_calls",
        "reviewer": "reviewer_calls",
    }[role]
    return int(state.get(key, 0))


def _require_role_capacity(state: EvidenceAgentState, role: str) -> None:
    if _role_count(state, role) >= MAX_CALLS_PER_ROLE:
        raise RuntimeError(f"{role} agent exceeded its {MAX_CALLS_PER_ROLE}-call limit")


def _append_trace(
    state: EvidenceAgentState, role: str, event: str
) -> list[dict[str, Any]]:
    trace = list(state.get("trace", []))
    trace.append({"step": len(trace) + 1, "role": role, "event": event})
    return trace


def _required_text(response: dict[str, Any], key: str) -> str:
    if key not in response:
        raise KeyError(f"Missing required response field: {key}")
    value = str(response[key]).strip()
    if not value:
        raise ValueError(f"Response field {key!r} cannot be empty")
    return value


def _optional_text(response: dict[str, Any], key: str) -> str:
    value = response.get(key, "")
    return "" if value is None else str(value).strip()


def _required_int(response: dict[str, Any], key: str) -> int:
    if key not in response or isinstance(response[key], bool):
        raise KeyError(f"Missing required integer response field: {key}")
    try:
        value = int(response[key])
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Response field {key!r} must be an integer") from exc
    if str(response[key]).strip() not in {str(value), f"{value}.0"}:
        raise TypeError(f"Response field {key!r} must be an integer")
    return value


def _required_bool(response: dict[str, Any], key: str) -> bool:
    if key not in response or not isinstance(response[key], bool):
        raise TypeError(f"Response field {key!r} must be a boolean")
    return bool(response[key])


def _string_list(response: dict[str, Any], key: str) -> list[str]:
    if key not in response or not isinstance(response[key], list):
        raise TypeError(f"Response field {key!r} must be a list")
    return [str(value).strip() for value in response[key] if str(value).strip()]
