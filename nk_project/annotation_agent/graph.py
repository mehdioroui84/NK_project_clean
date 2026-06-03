from __future__ import annotations

import json
import re
import time
from typing import Any, TypedDict

from nk_project.annotation_agent.prompts import (
    PAIRWISE_SPLIT_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_cluster_prompt,
    build_pairwise_split_prompt,
)
from nk_project.annotation_agent.taxonomy_reference import allowed_nk_state_labels, allowed_nk_subtype_labels


ALLOWED_NK_SUBTYPE_CALLS = [label for label in allowed_nk_subtype_labels() if label != "Unsure"]
ALLOWED_NK_STATE_CALLS = [
    label for label in allowed_nk_state_labels() if label not in {"Unsure", "Non-NK", "NA"}
]


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
    max_iterations: int = 1,
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


def run_pairwise_split_agent(
    *,
    cluster_a: str,
    cluster_b: str,
    structured_label: str,
    evidence_a: dict[str, Any],
    evidence_b: dict[str, Any],
    decision_a: dict[str, Any],
    decision_b: dict[str, Any],
    active_llm: str,
    temperature: float = 0.0,
    llm_retries: int = 5,
    retry_sleep: float = 5.0,
) -> dict[str, Any]:
    try:
        from nk_project.annotation_agent.llm_factory import get_active_llm
    except ImportError as exc:
        raise ImportError(
            "langchain-openai is required for LLM calls. Install local dependencies with "
            "`pip install langchain-openai`."
        ) from exc

    llm = get_active_llm(temperature=float(temperature), active_llm=active_llm)
    prompt = build_pairwise_split_prompt(
        cluster_a=cluster_a,
        cluster_b=cluster_b,
        structured_label=structured_label,
        evidence_a=evidence_a,
        evidence_b=evidence_b,
        decision_a=decision_a,
        decision_b=decision_b,
    )
    messages = [
        {"role": "system", "content": PAIRWISE_SPLIT_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    started = time.time()
    print(f"[SPLIT_LLM_START] {cluster_a} vs {cluster_b} model={active_llm}", flush=True)
    response = invoke_with_retry(
        llm,
        messages,
        retries=int(llm_retries),
        sleep_seconds=float(retry_sleep),
        cluster_id=f"{cluster_a}_vs_{cluster_b}",
        iteration=1,
    )
    elapsed = time.time() - started
    print(f"[SPLIT_LLM_DONE] {cluster_a} vs {cluster_b} elapsed={elapsed:.1f}s", flush=True)
    return normalize_pairwise_split_decision(
        parse_json_response(response.content),
        cluster_a=cluster_a,
        cluster_b=cluster_b,
        structured_label=structured_label,
        decision_a=decision_a,
        decision_b=decision_b,
    )


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
        {"continue": "draft_or_revise", "finalize": "finalize"},
    )
    graph.add_edge("finalize", END)
    return graph.compile()


def draft_or_revise(state: ClusterAgentState) -> ClusterAgentState:
    decision = normalize_decision(call_llm_for_decision(state), state["evidence"])
    previous = list(state["previous_decisions"])
    previous.append(decision)
    state["previous_decisions"] = previous
    state["iteration"] = int(state["iteration"]) + 1
    return state


def should_continue(state: ClusterAgentState) -> str:
    if not state["previous_decisions"]:
        return "continue"
    last = state["previous_decisions"][-1]
    wants_more = bool(last.get("needs_more_iteration", False))
    below_limit = int(state["iteration"]) <= int(state["max_iterations"])
    return "continue" if wants_more and below_limit else "finalize"


def finalize(state: ClusterAgentState) -> ClusterAgentState:
    final = dict(state["previous_decisions"][-1])
    if int(state["iteration"]) > int(state["max_iterations"]):
        final["needs_more_iteration"] = False
        final["stop_reason"] = "Reached max iteration limit."
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

    llm = get_active_llm(temperature=float(state["temperature"]), active_llm=state["active_llm"])
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
    cluster_id = str(state["evidence"].get("cluster_id", "unknown"))
    iteration = int(state["iteration"])
    started = time.time()
    print(f"[LLM_START] cluster={cluster_id} iteration={iteration} model={state['active_llm']}", flush=True)
    response = invoke_with_retry(
        llm,
        messages,
        retries=int(state["llm_retries"]),
        sleep_seconds=float(state["retry_sleep"]),
        cluster_id=cluster_id,
        iteration=iteration,
    )
    elapsed = time.time() - started
    print(f"[LLM_DONE] cluster={cluster_id} iteration={iteration} elapsed={elapsed:.1f}s", flush=True)
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
        except Exception as exc:  # noqa: BLE001 - transient API/network errors are retried.
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

    subtype = normalize_allowed_label(decision.get("nk_subtype_call"), ALLOWED_NK_SUBTYPE_CALLS + ["Non-NK"])
    state = normalize_allowed_label(decision.get("nk_state_call"), ALLOWED_NK_STATE_CALLS + ["NA"])
    review_from_fallback = False

    if subtype == "Non-NK":
        if pan_nk_broadly_expressed(evidence):
            subtype = best_taxonomy_label(evidence, "subtype") or "cNK"
            state = best_taxonomy_label(evidence, "state") or "Homeostatic_quiescent"
            review_from_fallback = True
        else:
            state = "NA"
    else:
        if subtype in {"", "Unsure", "NA"}:
            subtype = best_taxonomy_label(evidence, "subtype") or "cNK"
            review_from_fallback = True
        if state in {"", "Unsure", "Non-NK", "NA"}:
            state = best_taxonomy_label(evidence, "state") or "Homeostatic_quiescent"
            review_from_fallback = True

    final_label = "Non-NK" if subtype == "Non-NK" else build_final_structured_label(subtype, state)
    free_label = standardize_free_label(decision.get("free_label") or final_label)
    if final_label != "Non-NK" and free_label == "Non-NK":
        free_label = final_label
    if not free_label:
        free_label = final_label

    for key in ["evidence_summary", "concerns"]:
        value = decision.get(key, [])
        if isinstance(value, str):
            value = [value]
        decision[key] = [str(item).strip() for item in value if str(item).strip()]

    if review_from_fallback:
        decision["concerns"].append(
            "Layer call was normalized from weak/invalid output; review the evidence."
        )
    if subtype != "Non-NK" and pan_nk_broadly_expressed(evidence) and str(decision.get("nk_subtype_call")) == "Non-NK":
        decision["concerns"].append(
            "Model proposed Non-NK, but pan-NK markers remain broadly expressed; kept as NK for review."
        )

    needs_review = bool(decision.get("needs_human_review", False)) or review_from_fallback
    human_review_reason = str(decision.get("human_review_reason") or "").strip()
    if needs_review and not human_review_reason:
        human_review_reason = "; ".join(decision["concerns"][:3]) or "Evidence is uncertain."

    return {
        "cluster_id": decision["cluster_id"],
        "nk_subtype_call": subtype,
        "nk_state_call": state,
        "final_structured_label": final_label,
        "free_label": free_label,
        "free_label_reason": str(decision.get("free_label_reason") or "").strip(),
        "confidence_score": clamp_score(decision.get("confidence_score")),
        "tissue_specificity_score": clamp_score(decision.get("tissue_specificity_score")),
        "dataset_assay_specificity_score": clamp_score(decision.get("dataset_assay_specificity_score")),
        "evidence_summary": decision["evidence_summary"],
        "concerns": decision["concerns"],
        "needs_human_review": needs_review,
        "human_review_reason": human_review_reason,
        "needs_more_iteration": bool(decision.get("needs_more_iteration", False)),
        "stop_reason": str(decision.get("stop_reason") or ""),
    }


def normalize_pairwise_split_decision(
    decision: dict[str, Any],
    *,
    cluster_a: str,
    cluster_b: str,
    structured_label: str,
    decision_a: dict[str, Any],
    decision_b: dict[str, Any],
) -> dict[str, Any]:
    decision = dict(decision)
    split_supported = bool(decision.get("split_supported", False))
    label_a = standardize_free_label(decision.get("cluster_a_free_label") or decision_a.get("free_label") or structured_label)
    label_b = standardize_free_label(decision.get("cluster_b_free_label") or decision_b.get("free_label") or structured_label)
    return {
        "cluster_a": cluster_a,
        "cluster_b": cluster_b,
        "final_structured_label": structured_label,
        "split_supported": split_supported,
        "cluster_a_free_label": label_a,
        "cluster_a_free_label_reason": str(decision.get("cluster_a_free_label_reason") or "").strip(),
        "cluster_b_free_label": label_b,
        "cluster_b_free_label_reason": str(decision.get("cluster_b_free_label_reason") or "").strip(),
        "needs_human_review": bool(decision.get("needs_human_review", split_supported)),
        "human_review_reason": str(decision.get("human_review_reason") or "").strip(),
    }


def best_taxonomy_label(evidence: dict[str, Any], layer: str) -> str:
    allowed_labels = ALLOWED_NK_SUBTYPE_CALLS if layer == "subtype" else ALLOWED_NK_STATE_CALLS
    allowed = set(allowed_labels)
    for item in evidence.get("taxonomy_marker_hits", {}).get("top_matches", []):
        if item.get("taxonomy_layer") != layer:
            continue
        label = normalize_allowed_label(item.get("taxonomy_label"), allowed_labels)
        if label in allowed:
            return label
    return ""


def pan_nk_broadly_expressed(evidence: dict[str, Any]) -> bool:
    summary = evidence.get("pan_nk_marker_summary", {})
    try:
        n_tested = int(summary.get("n_pan_nk_genes_tested") or 0)
        median_pct = float(summary.get("median_pan_nk_pct_nz_cluster"))
    except (TypeError, ValueError):
        return False
    return n_tested >= 4 and median_pct >= 0.5


def normalize_allowed_label(value: Any, allowed: list[str]) -> str:
    text = str(value or "").strip()
    if text in allowed:
        return text
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    for label in allowed:
        if re.sub(r"[^a-z0-9]+", "", label.lower()) == compact:
            return label
    return "Unsure"


def build_final_structured_label(subtype: str, state: str) -> str:
    return "_".join([standardize_label_part(subtype), standardize_label_part(state)])


def standardize_label_part(value: Any) -> str:
    text = str(value or "").strip()
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "Unknown"


def standardize_free_label(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_+.-]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def clamp_score(value: Any) -> int:
    try:
        return max(0, min(5, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0
