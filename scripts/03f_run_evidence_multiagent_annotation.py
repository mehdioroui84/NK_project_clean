#!/usr/bin/env python
"""Run independent evidence-driven multi-agent annotation for one resolution."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nk_project.annotation_agent.evidence_multiagent_graph import (  # noqa: E402
    run_evidence_multiagent,
)
from nk_project.annotation_agent.evidence_multiagent_report import (  # noqa: E402
    write_multiagent_outputs,
)
from nk_project.annotation_agent.evidence_tools import (  # noqa: E402
    SUPPORTED_RESOLUTIONS,
    cluster_sort_key,
    processed_pathway_path,
)


EXPECTED_CLUSTER_COUNTS = {"leiden_0_1": 11, "leiden_0_5": 26}
FORBIDDEN_CONTEXT_FIELDS = {
    "top_ranked_pathways",
    "deg",
    "taxonomy",
    "previous_annotation",
    "draft_refined_label",
    "nk_state",
    "cell_type",
}
REQUIRED_CLUSTER_FIELDS = {
    "cluster",
    "n_cells",
    "metadata_context",
    "attribution_gene_selection",
    "total_significant_pathways",
    "mitochondrial_context",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evidence-driven LangGraph annotation using planner, biological, "
            "technical-context, and critical-reviewer roles."
        )
    )
    parser.add_argument(
        "--resolution", required=True, choices=sorted(SUPPORTED_RESOLUTIONS)
    )
    parser.add_argument(
        "--generated-date",
        default=None,
        help=(
            "Optional evidence-version date in YYYY-MM-DD format. By default, "
            "the newest matching context/pathway version is selected automatically."
        ),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--context-json",
        type=Path,
        default=None,
        help="Default: dated context JSON under data/annotation_agent/agent_context.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=None,
        help=(
            "Default: outputs/annotation_agent/"
            "evidence_multiagent_<resolution>_<date>."
        ),
    )
    parser.add_argument(
        "--cluster-id",
        default=None,
        help="Run one cluster only for testing; cluster identifiers remain strings.",
    )
    parser.add_argument(
        "--active-llm",
        default=os.environ.get("NK_ANNOTATION_AGENT_LLM", "5_mini"),
        choices=["4o", "41", "41_mini", "5_mini"],
        help="Reuse the existing MDA/Azure LLM factory selection.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all evidence inputs without creating outputs or calling an LLM.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of files in an existing new-agent output directory.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Keep completed clusters in an existing output directory and run "
            "only clusters that previously failed or are missing."
        ),
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed cluster instead of recording it and continuing.",
    )
    args = parser.parse_args()
    if args.generated_date is not None:
        try:
            date.fromisoformat(args.generated_date)
        except ValueError as exc:
            raise ValueError("--generated-date must use YYYY-MM-DD") from exc
    if args.llm_retries < 1:
        raise ValueError("--llm-retries must be at least 1")
    if args.retry_sleep < 0:
        raise ValueError("--retry-sleep cannot be negative")
    if args.overwrite and args.resume:
        raise ValueError("Use either --overwrite or --resume, not both")
    return args


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    generated_date = args.generated_date or newest_matching_evidence_date(
        project_root=project_root,
        resolution=args.resolution,
    )
    context_path = (
        args.context_json.resolve()
        if args.context_json is not None
        else project_root
        / "data"
        / "annotation_agent"
        / "agent_context"
        / f"cluster_context_{args.resolution}_{generated_date}.json"
    )
    pathway_path = processed_pathway_path(
        project_root=project_root,
        resolution=args.resolution,
        generated_date=generated_date,
    )
    context = load_and_validate_context(
        context_path=context_path,
        pathway_path=pathway_path,
        resolution=args.resolution,
    )
    by_cluster = {str(item["cluster"]): item for item in context["clusters"]}
    cluster_ids = sorted(by_cluster, key=cluster_sort_key)
    if args.cluster_id is not None:
        cluster_id = str(args.cluster_id)
        if cluster_id not in by_cluster:
            raise KeyError(
                f"Cluster {cluster_id!r} not found. Available: {cluster_ids}"
            )
        cluster_ids = [cluster_id]

    print("=" * 80)
    print("Evidence-driven multi-agent cluster annotation")
    print("=" * 80)
    print(f"[RESOLUTION] {args.resolution}")
    print(f"[EVIDENCE_VERSION] {generated_date}")
    print(f"[CONTEXT] {context_path}")
    print(f"[PATHWAYS] {pathway_path}")
    print(f"[CLUSTERS] {len(cluster_ids)}: {cluster_ids}")
    print(f"[CELLS_IN_FULL_CONTEXT] {sum(item['n_cells'] for item in context['clusters']):,}")
    print(f"[ACTIVE_LLM] {args.active_llm}")
    print("[EXCLUDED_EVIDENCE] DEG, curated taxonomy, previous labels, parent-child context")

    if args.dry_run:
        print("[DRY-RUN] Inputs validated; no output directory or LLM calls created.")
        return

    outdir = (
        args.outdir.resolve()
        if args.outdir is not None
        else project_root
        / "outputs"
        / "annotation_agent"
        / f"evidence_multiagent_{args.resolution}_{generated_date}"
    )
    if args.resume:
        results, failures = _load_existing_progress(outdir)
        completed_clusters = {str(result["cluster"]) for result in results}
        cluster_ids = [
            cluster_id
            for cluster_id in cluster_ids
            if cluster_id not in completed_clusters
        ]
        print(
            f"[RESUME] kept={len(results)} remaining={len(cluster_ids)} "
            f"previously_failed={len(failures)}"
        )
    else:
        _guard_output_directory(outdir, overwrite=args.overwrite)
        results = []
        failures = []
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"[OUTDIR] {outdir}")

    run_config = {
        "resolution": args.resolution,
        "generated_date_of_input_evidence": generated_date,
        "context_json": str(context_path),
        "processed_pathway_csv": str(pathway_path),
        "active_llm": args.active_llm,
        "temperature": args.temperature,
        "llm_retries": args.llm_retries,
        "retry_sleep": args.retry_sleep,
        "cluster_ids": cluster_ids,
        "maximum_calls_per_role_per_cluster": 3,
        "study_scope_prior": (
            "NK-focused study with possible non-NK contaminants; cytotoxic "
            "lymphocytes require coherent alternative-lineage evidence before "
            "being assigned away from NK"
        ),
        "t_cell_lineage_requirement": (
            "coherent TCR/CD3 attribution program; CD247 or generic T-cell GO "
            "terms alone are insufficient"
        ),
        "initial_pathways_retrieved": 20,
        "optional_additional_pathways_retrieved": 10,
        "maximum_total_pathways_retrieved": 30,
        "mitochondrial_review_trigger": (
            "at least 4 MT- genes in the top 10 attribution genes or at least "
            "6 MT- genes in the top 20"
        ),
        "evidence_excluded": [
            "differential_expression",
            "curated_taxonomy",
            "previous_annotations",
            "parent_child_context",
        ],
    }
    for position, cluster_id in enumerate(cluster_ids, start=1):
        print(f"\n[CLUSTER] {cluster_id} ({position}/{len(cluster_ids)})", flush=True)
        failures = [
            item for item in failures if str(item.get("cluster")) != cluster_id
        ]
        try:
            result = run_evidence_multiagent(
                by_cluster[cluster_id],
                resolution=args.resolution,
                project_root=project_root,
                generated_date=generated_date,
                active_llm=args.active_llm,
                temperature=args.temperature,
                llm_retries=args.llm_retries,
                retry_sleep=args.retry_sleep,
            )
            results.append(result)
            final = result["final_decision"]
            print(
                f"[ANNOTATION] {cluster_id}: {final['proposed_broad_identity']} | "
                f"{final['proposed_subtype']} | {final['proposed_functional_state']} | "
                f"confidence={final['confidence_score']}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001 - preserve other cluster results.
            failure = {
                "cluster": cluster_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            failures.append(failure)
            print(
                f"[ERROR] cluster={cluster_id} {failure['error_type']}: {failure['error']}",
                flush=True,
            )
            if args.fail_fast:
                _save_progress(
                    results, by_cluster, outdir, args.resolution, run_config, failures
                )
                raise
        _save_progress(
            results, by_cluster, outdir, args.resolution, run_config, failures
        )

    print(
        f"[DONE] completed={len(results)} failed={len(failures)} outdir={outdir}",
        flush=True,
    )


def newest_matching_evidence_date(*, project_root: Path, resolution: str) -> str:
    context_dir = project_root / "data" / "annotation_agent" / "agent_context"
    prefix = f"cluster_context_{resolution}_"
    suffix = ".json"
    candidates: list[str] = []
    for context_path in context_dir.glob(f"{prefix}*{suffix}"):
        filename = context_path.name
        date_text = filename[len(prefix) : -len(suffix)]
        try:
            date.fromisoformat(date_text)
        except ValueError:
            continue
        pathway_path = processed_pathway_path(
            project_root=project_root,
            resolution=resolution,
            generated_date=date_text,
        )
        if pathway_path.exists():
            candidates.append(date_text)
    if not candidates:
        raise FileNotFoundError(
            f"No matching dated context and processed pathway files found for "
            f"{resolution} under {context_dir}"
        )
    return max(candidates)


def load_and_validate_context(
    *, context_path: Path, pathway_path: Path, resolution: str
) -> dict[str, Any]:
    if not context_path.exists():
        raise FileNotFoundError(context_path)
    if not pathway_path.exists():
        raise FileNotFoundError(pathway_path)
    context = json.loads(context_path.read_text(encoding="utf-8"))
    if context.get("leiden_resolution") != resolution:
        raise ValueError(
            f"Context resolution {context.get('leiden_resolution')!r} does not "
            f"match requested {resolution!r}"
        )
    clusters = context.get("clusters")
    if not isinstance(clusters, list):
        raise TypeError("Context must contain a clusters list")
    expected_count = EXPECTED_CLUSTER_COUNTS[resolution]
    if len(clusters) != expected_count:
        raise ValueError(
            f"Expected {expected_count} clusters for {resolution}; found {len(clusters)}"
        )
    cluster_ids = [str(item.get("cluster")) for item in clusters]
    if len(cluster_ids) != len(set(cluster_ids)):
        raise ValueError("Context contains duplicate cluster identifiers")
    if sum(int(item.get("n_cells", 0)) for item in clusters) != 311_471:
        raise ValueError("Full context must represent exactly 311,471 cells")

    for item in clusters:
        missing = sorted(REQUIRED_CLUSTER_FIELDS.difference(item))
        if missing:
            raise KeyError(f"Cluster {item.get('cluster')} is missing fields: {missing}")
        forbidden = sorted(FORBIDDEN_CONTEXT_FIELDS.intersection(_nested_keys(item)))
        if forbidden:
            raise ValueError(
                f"Cluster {item['cluster']} contains prohibited evidence fields: {forbidden}"
            )
        selection = item["attribution_gene_selection"]
        if selection.get("target_cumulative_attribution_mass_percent") != 50:
            raise ValueError(f"Cluster {item['cluster']} does not use mass-50 genes")
        genes = selection.get("genes_ordered_from_highest_to_lowest_attribution")
        if not isinstance(genes, list) or not genes:
            raise ValueError(f"Cluster {item['cluster']} has no mass-50 gene list")
        if len(genes) != len(set(genes)) or not all(
            isinstance(gene, str) and gene.strip() for gene in genes
        ):
            raise ValueError(
                f"Cluster {item['cluster']} has invalid or duplicate mass-50 genes"
            )

    pathway_counts = _pathway_counts(pathway_path)
    for item in clusters:
        cluster = str(item["cluster"])
        observed = int(item["total_significant_pathways"])
        expected = int(pathway_counts.get(cluster, 0))
        if observed != expected:
            raise ValueError(
                f"Cluster {cluster} context reports {observed} significant pathways, "
                f"but {pathway_path} contains {expected}"
            )
    return context


def _nested_keys(value: Any) -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, nested in value.items():
            keys.add(str(key).lower())
            keys.update(_nested_keys(nested))
    elif isinstance(value, list):
        for nested in value:
            keys.update(_nested_keys(nested))
    return keys


def _pathway_counts(path: Path) -> dict[str, int]:
    frame = pd.read_csv(path, dtype={"cluster": str}, low_memory=False)
    required = {"cluster", "pathway_rank", "pathway"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing pathway fields: {missing}")
    frame["cluster"] = frame["cluster"].astype(str)
    return frame.groupby("cluster", sort=False).size().astype(int).to_dict()


def _guard_output_directory(outdir: Path, *, overwrite: bool) -> None:
    expected_outputs = {
        "cluster_annotations.json",
        "cluster_annotations.csv",
        "cluster_annotation_report.md",
        "agent_trace.jsonl",
        "failed_clusters.json",
        "run_config.json",
    }
    existing = [path.name for path in outdir.glob("*") if path.name in expected_outputs]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output files already exist in {outdir}: {sorted(existing)}. "
            "Use --overwrite only when replacement is intended."
        )


def _load_existing_progress(
    outdir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    trace_path = outdir / "agent_trace.jsonl"
    failures_path = outdir / "failed_clusters.json"
    if not trace_path.exists() or not failures_path.exists():
        raise FileNotFoundError(
            f"Cannot resume {outdir}: agent_trace.jsonl and "
            "failed_clusters.json are both required"
        )
    results: list[dict[str, Any]] = []
    with trace_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            result = json.loads(line)
            if not isinstance(result, dict) or "cluster" not in result:
                raise ValueError(
                    f"Invalid saved result on line {line_number} of {trace_path}"
                )
            results.append(result)
    completed = [str(result["cluster"]) for result in results]
    if len(completed) != len(set(completed)):
        raise ValueError(f"Duplicate completed clusters in {trace_path}")
    failures = json.loads(failures_path.read_text(encoding="utf-8"))
    if not isinstance(failures, list):
        raise TypeError(f"{failures_path} must contain a JSON list")
    return results, failures


def _save_progress(
    results: list[dict[str, Any]],
    cluster_context: dict[str, dict[str, Any]],
    outdir: Path,
    resolution: str,
    run_config: dict[str, Any],
    failures: list[dict[str, str]],
) -> None:
    results = sorted(
        results,
        key=lambda result: cluster_sort_key(result["cluster"]),
    )
    failures = sorted(
        failures,
        key=lambda item: cluster_sort_key(item["cluster"]),
    )
    write_multiagent_outputs(
        results=results,
        cluster_context=cluster_context,
        outdir=outdir,
        resolution=resolution,
        run_config=run_config,
        failures=failures,
    )


if __name__ == "__main__":
    main()
