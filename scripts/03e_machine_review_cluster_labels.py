#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nk_project.annotation_agent.graph import invoke_with_retry, parse_json_response
from nk_project.annotation_agent.llm_factory import get_active_llm
from nk_project.annotation_agent.prompts import MACHINE_REVIEW_SYSTEM_PROMPT, build_machine_review_prompt
from nk_project.annotation_agent.taxonomy_reference import allowed_nk_state_labels, allowed_nk_subtype_labels


DEFAULT_GROUPBY = "leiden_0_4"
DEFAULT_REPORT_DIR = "reports/cellxgene_cluster_report_all_annotations_plus_agent_evidence_v1"


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    table_dir = outdir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_report_export:
        run_report_export(args)

    report = load_report_summary(args.report_dir, args.groupby, args.evidence_report_csv)
    first_pass = load_first_pass_mapping(args.mapping_csv, args.groupby)
    report = merge_first_pass_labels(report, first_pass, args.groupby)
    markers = load_marker_table(args.report_dir, args.groupby)
    cluster_ids = select_cluster_ids(report, args)
    trace_path = table_dir / "machine_review_trace.jsonl"

    if args.use_existing_trace:
        results = load_results_from_trace(trace_path, cluster_ids)
    else:
        results = run_llm_reviews(report, markers, cluster_ids, trace_path, args)

    reviewed = build_reviewed_table(report, pd.DataFrame(results), args.groupby)
    reviewed_path = table_dir / "machine_reviewed_cluster_labels.csv"
    changed_path = table_dir / "machine_review_changed_or_flagged_labels.csv"
    md_path = outdir / "machine_reviewed_cluster_labels.md"

    reviewed.to_csv(reviewed_path, index=False)
    changed = changed_or_flagged_rows(reviewed)
    changed.to_csv(changed_path, index=False)
    write_markdown(md_path, reviewed, changed, args.groupby)

    print(f"[SAVE] {reviewed_path}")
    print(f"[SAVE] {changed_path}")
    print(f"[SAVE] {trace_path}")
    print(f"[SAVE] {md_path}")
    print("[DONE] Machine review complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic CellXGene cluster report export, then use an LLM review agent "
            "to find label discrepancies and produce concise reviewed labels."
        )
    )
    parser.add_argument("--groupby", default=DEFAULT_GROUPBY)
    parser.add_argument("--composition-counts-csv", default="outputs/leiden_discovery/leiden_0_4_by_NK_State.csv")
    parser.add_argument(
        "--mapping-csv",
        default="outputs/annotation_agent/leiden_0_4_subtype_state_gpt5mini_sanity_v1/cluster_annotation_mapping.csv",
        help="First-pass agent mapping CSV. Only agent labels/rationale are used; CellXGene/Yuntao/manual annotations are not used.",
    )
    parser.add_argument(
        "--markers-csv",
        default="outputs/markers/full/leiden_0_4/leiden_0_4_markers_top50_per_cluster.csv",
    )
    parser.add_argument(
        "--annotation-figure",
        default="outputs/refined_annotation_v1_agent_preferred_gpt5mini_sanity_v1/figures/annotation_umap_review_panels.png",
    )
    parser.add_argument(
        "--marker-figure",
        default="outputs/markers/full/leiden_0_4/leiden_0_4_dotplot_top_markers.png",
    )
    parser.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--evidence-report-csv",
        default="",
        help=(
            "Evidence-only cluster report CSV. Defaults to "
            "<report-dir>/tables/cluster_evidence_only_report_for_agent.csv."
        ),
    )
    parser.add_argument("--outdir", default="reports/cellxgene_cluster_machine_review_gpt5mini_sanity_v1")
    parser.add_argument("--top-n-markers", type=int, default=50)
    parser.add_argument("--marker-detail-n", type=int, default=20)
    parser.add_argument("--cluster-id", default=None, help="Review only one cluster.")
    parser.add_argument("--skip-report-export", action="store_true")
    parser.add_argument(
        "--use-existing-trace",
        action="store_true",
        help="Do not call the LLM; rebuild output tables from tables/machine_review_trace.jsonl.",
    )
    parser.add_argument(
        "--active-llm",
        default=os.environ.get("NK_ANNOTATION_AGENT_LLM", "41_mini"),
        choices=["4o", "41", "41_mini", "5_mini"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    args = parser.parse_args()
    if args.marker_detail_n < 1:
        raise ValueError("--marker-detail-n must be at least 1.")
    if args.top_n_markers < 1:
        raise ValueError("--top-n-markers must be at least 1.")
    return args


def run_report_export(args: argparse.Namespace) -> None:
    script_path = Path(__file__).with_name("03d_export_cellxgene_cluster_report.py")
    cmd = [
        sys.executable,
        str(script_path),
        "--cluster-key",
        args.groupby,
        "--composition-counts-csv",
        args.composition_counts_csv,
        "--mapping-csv",
        args.mapping_csv,
        "--markers-csv",
        args.markers_csv,
        "--annotation-figure",
        args.annotation_figure,
        "--marker-figure",
        args.marker_figure,
        "--outdir",
        args.report_dir,
        "--top-n-markers",
        str(args.top_n_markers),
    ]
    print("[REPORT_EXPORT] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_llm_reviews(
    report: pd.DataFrame,
    markers: pd.DataFrame,
    cluster_ids: list[str],
    trace_path: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    llm = get_active_llm(temperature=float(args.temperature), active_llm=args.active_llm)
    results = []
    if trace_path.exists():
        trace_path.unlink()
    for idx, cluster_id in enumerate(cluster_ids, start=1):
        row = report.loc[report[args.groupby].astype(str).eq(cluster_id)].iloc[0].to_dict()
        marker_rows = markers.loc[markers[args.groupby].astype(str).eq(cluster_id)]
        payload = build_cluster_review_payload(row, marker_rows, args)
        print(f"[REVIEW] cluster {cluster_id} ({idx}/{len(cluster_ids)}) model={args.active_llm}", flush=True)
        decision = call_review_llm(llm, payload, args)
        decision = normalize_review_decision(decision, payload)
        results.append(decision)
        with open(trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"cluster_report": payload, "review_decision": decision}) + "\n")
    return results


def load_results_from_trace(trace_path: Path, cluster_ids: list[str]) -> list[dict[str, Any]]:
    require_file(trace_path)
    wanted = set(map(str, cluster_ids))
    results_by_cluster: dict[str, dict[str, Any]] = {}
    with open(trace_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            decision = item.get("review_decision", {})
            cluster_id = str(decision.get("cluster_id") or item.get("cluster_report", {}).get("cluster_id", ""))
            if cluster_id in wanted:
                results_by_cluster[cluster_id] = decision
    missing = sorted(wanted - set(results_by_cluster), key=cluster_sort_key)
    if missing:
        raise RuntimeError(f"Existing trace is missing review decisions for clusters: {missing}")
    print(f"[LOAD] existing review trace: {trace_path} ({len(results_by_cluster)} clusters)")
    return [results_by_cluster[cluster_id] for cluster_id in sorted(wanted, key=cluster_sort_key)]


def load_report_summary(report_dir: str, groupby: str, evidence_report_csv: str = "") -> pd.DataFrame:
    path = Path(evidence_report_csv) if evidence_report_csv else Path(report_dir) / "tables" / "cluster_evidence_only_report_for_agent.csv"
    require_file(path)
    df = pd.read_csv(path, low_memory=False).fillna("")
    if groupby not in df.columns:
        raise KeyError(f"{path} must contain {groupby!r}.")
    df[groupby] = df[groupby].astype(str)
    print(f"[LOAD] {path}")
    return df


def load_marker_table(report_dir: str, groupby: str) -> pd.DataFrame:
    path = Path(report_dir) / "tables" / "cluster_top_markers.csv"
    require_file(path)
    df = pd.read_csv(path, low_memory=False).fillna("")
    if groupby not in df.columns:
        raise KeyError(f"{path} must contain {groupby!r}.")
    df[groupby] = df[groupby].astype(str)
    print(f"[LOAD] {path}")
    return df


def load_first_pass_mapping(path: str, groupby: str) -> pd.DataFrame:
    path = Path(path)
    require_file(path)
    df = pd.read_csv(path, low_memory=False).fillna("")
    if groupby not in df.columns:
        raise KeyError(f"{path} must contain {groupby!r}.")
    keep = [
        groupby,
        "nk_subtype_call",
        "nk_state_call",
        "final_structured_label",
        "free_label",
        "free_label_reason",
        "needs_human_review",
        "human_review_reason",
        "confidence_score_0_5",
        "taxonomy_top_matches",
    ]
    keep = [col for col in keep if col in df.columns]
    out = df[keep].copy()
    out[groupby] = out[groupby].astype(str)
    print(f"[LOAD] first-pass agent labels: {path}")
    return out


def merge_first_pass_labels(report: pd.DataFrame, first_pass: pd.DataFrame, groupby: str) -> pd.DataFrame:
    report = report.copy()
    report[groupby] = report[groupby].astype(str)
    first_pass = first_pass.copy()
    first_pass[groupby] = first_pass[groupby].astype(str)
    return report.merge(first_pass, on=groupby, how="left")


def select_cluster_ids(report: pd.DataFrame, args: argparse.Namespace) -> list[str]:
    if args.cluster_id is not None:
        cluster_id = str(args.cluster_id)
        if not report[args.groupby].astype(str).eq(cluster_id).any():
            raise KeyError(f"Cluster {cluster_id!r} not found in report.")
        return [cluster_id]
    return sorted(report[args.groupby].astype(str).unique(), key=cluster_sort_key)


def build_cluster_review_payload(row: dict[str, Any], marker_rows: pd.DataFrame, args: argparse.Namespace) -> dict[str, Any]:
    cluster_id = str(row.get(args.groupby, ""))
    markers = []
    if not marker_rows.empty and "marker_direction" in marker_rows.columns:
        marker_rows = marker_rows.copy()
        marker_rows["_direction_order"] = marker_rows["marker_direction"].map({"up": 0, "down": 1}).fillna(2)
        marker_rows = marker_rows.sort_values(["_direction_order", "marker_rank"])
    else:
        marker_rows = marker_rows.sort_values("marker_rank") if "marker_rank" in marker_rows.columns else marker_rows
    for _, marker in marker_rows.head(args.marker_detail_n).iterrows():
        markers.append(
            {
                "direction": str(marker.get("marker_direction", "")),
                "rank": as_int(marker.get("marker_rank")),
                "gene": str(marker.get("gene", "")),
                "logfoldchange": as_float(marker.get("logfoldchanges")),
                "pct_cluster": as_float(first_nonempty(marker, ["pct_nz_group", "pct_expr_group"])),
                "pct_reference": as_float(first_nonempty(marker, ["pct_nz_reference", "pct_expr_reference"])),
                "adjusted_p_value": as_float(marker.get("pvals_adj")),
            }
        )
    return {
        "cluster_id": cluster_id,
        "n_cells": as_int(row.get("n_cells")),
        "first_pass_agent_labels": {
            "structured_label": str(row.get("final_structured_label", "")),
            "free_label": str(row.get("free_label", "")),
            "nk_subtype_call": str(row.get("nk_subtype_call", "")),
            "nk_state_call": str(row.get("nk_state_call", "")),
            "confidence_score_0_5": as_int(row.get("confidence_score_0_5")),
            "needs_human_review": parse_bool(row.get("needs_human_review")),
            "human_review_reason": str(row.get("human_review_reason", "")),
        },
        "first_pass_agent_reasoning": {
            "free_label_reason": str(row.get("free_label_reason", "")),
            "taxonomy_top_matches": str(row.get("taxonomy_top_matches", "")),
        },
        "metadata_context": {
            "top_tissue": str(row.get("top_tissue", "")),
            "top_tissue_percent": as_float(row.get("top_tissue_percent")),
            "top_dataset": str(row.get("top_dataset", "")),
            "top_dataset_percent": as_float(row.get("top_dataset_percent")),
            "top_assay": str(row.get("top_assay", "")),
            "top_assay_percent": as_float(row.get("top_assay_percent")),
        },
        "n_pairwise_de_compared": as_int(row.get("n_pairwise_de_compared")),
        "positive_marker_count": as_int(row.get("positive_marker_count")),
        "positive_marker_list": str(row.get("positive_marker_list", "")),
        "negative_marker_count": as_int(row.get("negative_marker_count")),
        "negative_marker_list": str(row.get("negative_marker_list", "")),
        "marker_details": markers,
    }


def call_review_llm(llm, payload: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": MACHINE_REVIEW_SYSTEM_PROMPT},
        {"role": "user", "content": build_machine_review_prompt(payload)},
    ]
    cluster_id = str(payload["cluster_id"])
    started = time.time()
    response = invoke_with_retry(
        llm,
        messages,
        retries=int(args.llm_retries),
        sleep_seconds=float(args.retry_sleep),
        cluster_id=f"review_{cluster_id}",
        iteration=1,
    )
    elapsed = time.time() - started
    print(f"[REVIEW_DONE] cluster={cluster_id} elapsed={elapsed:.1f}s", flush=True)
    return parse_json_response(response.content)


def normalize_review_decision(decision: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    decision = dict(decision)
    cluster_id = str(payload["cluster_id"])
    decision["cluster_id"] = str(decision.get("cluster_id") or cluster_id)
    decision["review_decision"] = normalize_choice(
        decision.get("review_decision"),
        allowed={"keep", "revise", "flag_only"},
        default="flag_only",
    )
    fallback_structured = str(payload["first_pass_agent_labels"].get("structured_label", "")).strip() or "Non-NK"
    fallback_label = short_label_from_first_pass(payload)
    decision["reviewed_structured_label"] = normalize_structured_label(
        decision.get("reviewed_structured_label"),
        default=fallback_structured,
    )
    decision["reviewed_label_short"] = clean_short_label(decision.get("reviewed_label_short") or fallback_label)
    decision["reviewed_label_detail"] = clean_text(decision.get("reviewed_label_detail"))
    flags = decision.get("discrepancy_flags", [])
    if isinstance(flags, str):
        flags = [flags] if flags.strip() else []
    decision["discrepancy_flags"] = [clean_text(flag) for flag in flags if clean_text(flag)]
    decision["review_reason"] = clean_text(decision.get("review_reason"))
    decision["confidence_score"] = clamp_int(decision.get("confidence_score"), low=0, high=5, default=3)
    decision["needs_human_review"] = parse_bool(decision.get("needs_human_review"))
    decision["human_review_reason"] = clean_text(decision.get("human_review_reason"))
    if decision["needs_human_review"] and not decision["human_review_reason"]:
        decision["human_review_reason"] = "Machine review found mixed or uncertain evidence."
    return decision


def build_reviewed_table(report: pd.DataFrame, reviews: pd.DataFrame, groupby: str) -> pd.DataFrame:
    report = report.copy()
    reviews = reviews.copy()
    report[groupby] = report[groupby].astype(str)
    reviews[groupby] = reviews["cluster_id"].astype(str)
    reviews["discrepancy_flags"] = reviews["discrepancy_flags"].map(
        lambda x: "; ".join(x) if isinstance(x, list) else str(x)
    )
    keep_report_cols = [
        groupby,
        "n_cells",
        "final_structured_label",
        "free_label",
        "needs_human_review",
        "human_review_reason",
        "free_label_reason",
        "top_cluster_markers",
    ]
    keep_review_cols = [
        groupby,
        "review_decision",
        "reviewed_structured_label",
        "reviewed_label_short",
        "reviewed_label_detail",
        "discrepancy_flags",
        "review_reason",
        "confidence_score",
        "needs_human_review_reviewed",
        "human_review_reason_reviewed",
    ]
    reviews = reviews.rename(
        columns={
            "needs_human_review": "needs_human_review_reviewed",
            "human_review_reason": "human_review_reason_reviewed",
        }
    )
    keep_report_cols = [col for col in keep_report_cols if col in report.columns]
    keep_review_cols = [col for col in keep_review_cols if col in reviews.columns]
    out = report[keep_report_cols].merge(reviews[keep_review_cols], on=groupby, how="left")
    return out.sort_values(groupby, key=lambda s: s.map(cluster_sort_key))


def changed_or_flagged_rows(reviewed: pd.DataFrame) -> pd.DataFrame:
    review_flag = reviewed.get("needs_human_review_reviewed", pd.Series(False, index=reviewed.index)).map(parse_bool)
    return reviewed.loc[
        reviewed["review_decision"].astype(str).isin(["revise", "flag_only"])
        | review_flag
    ].copy()


def write_markdown(path: Path, reviewed: pd.DataFrame, changed: pd.DataFrame, groupby: str) -> None:
    lines = [
        "# Machine-Reviewed Cluster Labels",
        "",
        "This file summarizes the second-pass machine review of first-pass cluster labels.",
        "",
        "## Changed Or Flagged",
        "",
    ]
    if changed.empty:
        lines.append("No revised or flagged labels.")
    else:
        cols = [
            groupby,
            "final_structured_label",
            "free_label",
            "review_decision",
            "reviewed_structured_label",
            "reviewed_label_short",
            "discrepancy_flags",
            "needs_human_review_reviewed",
            "review_reason",
        ]
        lines.append(dataframe_to_markdown(changed[[col for col in cols if col in changed.columns]]))
    lines.extend(["", "## All Reviewed Labels", ""])
    cols = [
        groupby,
        "final_structured_label",
        "free_label",
        "review_decision",
        "reviewed_structured_label",
        "reviewed_label_short",
        "confidence_score",
        "needs_human_review_reviewed",
    ]
    lines.append(dataframe_to_markdown(reviewed[[col for col in cols if col in reviewed.columns]]))
    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def short_label_from_first_pass(payload: dict[str, Any]) -> str:
    free_label = str(payload["first_pass_agent_labels"].get("free_label", "")).strip()
    structured = str(payload["first_pass_agent_labels"].get("structured_label", "")).strip()
    return free_label or structured or "Unlabeled"


def allowed_structured_labels() -> set[str]:
    subtypes = [label for label in allowed_nk_subtype_labels() if label not in {"Unsure", "Non-NK"}]
    states = [label for label in allowed_nk_state_labels() if label not in {"Unsure", "Non-NK", "NA"}]
    return {"Non-NK"} | {f"{subtype}_{state}" for subtype in subtypes for state in states}


def normalize_structured_label(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    allowed = allowed_structured_labels()
    if text in allowed:
        return text
    if default in allowed:
        return default
    return "Non-NK"


def clean_short_label(value: Any) -> str:
    text = clean_text(value).replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")[:80] or "Unlabeled"


def clean_text(value: Any) -> str:
    return str(value or "").strip()


def normalize_choice(value: Any, *, allowed: set[str], default: str) -> str:
    text = str(value or "").strip().lower()
    return text if text in allowed else default


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clamp_int(value: Any, *, low: int, high: int, default: int) -> int:
    try:
        parsed = int(float(value))
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def as_float(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def first_nonempty(row: pd.Series, columns: list[str]) -> Any:
    for col in columns:
        if col in row and str(row[col]).strip() != "":
            return row[col]
    return ""


def cluster_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path)


if __name__ == "__main__":
    main()
