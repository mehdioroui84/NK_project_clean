from __future__ import annotations

import json
import time
from typing import Any, TypedDict

from nk_project.annotation_agent.marker_knowledge import KNOWN_REFINED_LABELS
from nk_project.annotation_agent.prompts import SYSTEM_PROMPT, build_cluster_prompt


class ClusterAgentState(TypedDict):
    evidence: dict[str, Any]
    previous_decisions: list[dict[str, Any]]
    iteration: int
    max_iterations: int
    active_llm: str
    temperature: float
    llm_retries: int
    retry_sleep: float
    final_decision: dict[str, Any]


def run_cluster_agent(
    evidence: dict[str, Any],
    *,
    active_llm: str,
    max_iterations: int = 5,
    temperature: float = 0.0,
    llm_retries: int = 5,
    retry_sleep: float = 5.0,
) -> dict[str, Any]:
    workflow = build_graph()
    state: ClusterAgentState = {
        "evidence": evidence,
        "previous_decisions": [],
        "iteration": 1,
        "max_iterations": max_iterations,
        "active_llm": active_llm,
        "temperature": temperature,
        "llm_retries": llm_retries,
        "retry_sleep": retry_sleep,
        "final_decision": {},
    }
    result = workflow.invoke(state)
    return {
        "cluster_id": evidence["cluster_id"],
        "iterations": result["previous_decisions"],
        "final_decision": result["final_decision"],
    }


def build_graph():
    try:
        from langgraph.graph import END, StateGraph
    except ImportError as exc:
        raise ImportError(
            "LangGraph is required for the annotation agent. Install local dependencies with "
            "`pip install langgraph langchain-openai`."
        ) from exc

    graph = StateGraph(ClusterAgentState)
    graph.add_node("draft_or_revise", draft_or_revise)
    graph.add_node("finalize", finalize)
    graph.set_entry_point("draft_or_revise")
    graph.add_conditional_edges(
        "draft_or_revise",
        should_continue,
        {
            "continue": "draft_or_revise",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def draft_or_revise(state: ClusterAgentState) -> ClusterAgentState:
    decision = call_llm_for_decision(state)
    decision = normalize_decision(decision, state["evidence"])
    previous = list(state["previous_decisions"])
    previous.append(decision)
    state["previous_decisions"] = previous
    state["iteration"] = int(state["iteration"]) + 1
    return state


def should_continue(state: ClusterAgentState) -> str:
    last = state["previous_decisions"][-1]
    wants_more = bool(last.get("needs_more_iteration", False))
    needs_review_pass = bool(last.get("needs_human_review", False)) and len(state["previous_decisions"]) < 2
    below_limit = int(state["iteration"]) <= int(state["max_iterations"])
    if (wants_more or needs_review_pass) and below_limit:
        return "continue"
    return "finalize"


def finalize(state: ClusterAgentState) -> ClusterAgentState:
    final = dict(state["previous_decisions"][-1])
    if int(state["iteration"]) > int(state["max_iterations"]):
        final["needs_more_iteration"] = False
        final["stop_reason"] = "Reached max iteration limit."
        final["needs_human_review"] = True
    state["final_decision"] = final
    return state


def call_llm_for_decision(state: ClusterAgentState) -> dict[str, Any]:
    try:
        from nk_project.annotation_agent.llm_factory import get_active_llm
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for LLM calls. Install local dependencies with "
            "`pip install langchain-openai`."
        ) from exc

    llm = get_active_llm(
        temperature=float(state["temperature"]),
        active_llm=state["active_llm"],
    )
    prompt = build_cluster_prompt(
        state["evidence"],
        state["previous_decisions"],
        state["iteration"],
        state["max_iterations"],
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    cluster_id = state["evidence"].get("cluster_id", "unknown")
    iteration = state["iteration"]
    started = time.time()
    print(
        f"[LLM_START] cluster={cluster_id} iteration={iteration} "
        f"model={state['active_llm']}",
        flush=True,
    )
    response = invoke_with_retry(
        llm,
        messages,
        retries=int(state["llm_retries"]),
        sleep_seconds=float(state["retry_sleep"]),
        cluster_id=str(cluster_id),
        iteration=int(iteration),
    )
    elapsed = time.time() - started
    print(
        f"[LLM_DONE] cluster={cluster_id} iteration={iteration} "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    return parse_json_response(response.content)


def invoke_with_retry(
    llm,
    messages: list[dict[str, str]],
    *,
    retries: int,
    sleep_seconds: float,
    cluster_id: str,
    iteration: int,
):
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            print(
                f"[LLM_ATTEMPT] cluster={cluster_id} iteration={iteration} "
                f"attempt={attempt}/{retries}",
                flush=True,
            )
            return llm.invoke(messages)
        except Exception as exc:  # noqa: BLE001 - retry transient local APIM/network errors.
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
    raise last_error


def parse_json_response(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def normalize_decision(decision: dict[str, Any], evidence: dict[str, Any]) -> dict[str, Any]:
    decision = dict(decision)
    cluster_id = str(evidence["cluster_id"])
    decision["cluster_id"] = str(decision.get("cluster_id") or cluster_id)
    for key in [
        "confidence_score",
        "manual_annotation_support",
        "top_de_marker_support",
        "curated_marker_support",
        "technical_concern_score",
        "ambiguity_score",
    ]:
        decision[key] = clamp_score(decision.get(key))
    for key in [
        "alternate_labels",
        "evidence_summary",
        "concerns",
        "recommended_pairwise_comparisons",
    ]:
        value = decision.get(key, [])
        if isinstance(value, str):
            value = [value]
        decision[key] = [str(item) for item in value]
    suggested_new_label = str(decision.get("suggested_new_label") or "").strip()
    new_label_reason = str(decision.get("new_label_reason") or "").strip()
    candidate = str(decision.get("candidate_label") or "").strip()
    alternate_labels = [label for label in decision["alternate_labels"] if label in KNOWN_REFINED_LABELS]

    if candidate not in KNOWN_REFINED_LABELS:
        if candidate and not suggested_new_label:
            suggested_new_label = candidate
            if not new_label_reason:
                new_label_reason = "The model proposed this label, but it is not in the approved refined-label vocabulary."
        candidate = choose_fallback_label(decision, alternate_labels)
        decision["concerns"].append(
            "Candidate label was not in the approved refined-label vocabulary and was moved to suggested_new_label."
        )
    for key in ["needs_more_iteration", "needs_human_review"]:
        decision[key] = bool(decision.get(key, False))
    if suggested_new_label:
        decision["needs_human_review"] = True
    if requires_new_label_audit(evidence) and not suggested_new_label and not new_label_reason:
        new_label_reason = (
            "No new label suggested: the approved candidate label was judged sufficient after "
            "reviewing worksheet notes and/or pairwise evidence."
        )
    decision["candidate_label"] = candidate
    decision["alternate_labels"] = alternate_labels
    decision["suggested_new_label"] = suggested_new_label
    decision["new_label_reason"] = new_label_reason
    decision["stop_reason"] = str(decision.get("stop_reason") or "")
    return decision


def requires_new_label_audit(evidence: dict[str, Any]) -> bool:
    composition = evidence.get("composition", {})
    worksheet_note = str(composition.get("worksheet_review_note") or "").strip()
    pairwise = evidence.get("pairwise_de_evidence") or []
    return bool(worksheet_note or pairwise)


def choose_fallback_label(decision: dict[str, Any], alternate_labels: list[str]) -> str:
    if alternate_labels:
        return alternate_labels[0]
    label_text = " ".join(
        [
            str(decision.get("candidate_label", "")),
            " ".join(map(str, decision.get("evidence_summary", []))),
            " ".join(map(str, decision.get("concerns", []))),
        ]
    ).lower()
    for label in KNOWN_REFINED_LABELS:
        if label.lower() in label_text:
            return label
    return "Mature Cytotoxic"


def clamp_score(value: Any) -> int:
    try:
        return max(0, min(5, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0
