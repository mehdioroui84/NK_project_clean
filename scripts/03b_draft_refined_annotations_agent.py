#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.annotation_agent.evidence import (
    EvidencePaths,
    compact_cluster_evidence,
    load_cluster_evidence,
    save_compact_evidence_json,
    save_evidence_json,
)
from nk_project.annotation_agent.graph import run_cluster_agent, run_pairwise_split_agent
from nk_project.annotation_agent.pairwise import (
    centroid_distance_table,
    cluster_sort_key,
    existing_pair_set,
    run_pairwise_de_for_pairs,
    same_label_distance_pairs_from_results,
    structured_label_from_final,
)
from nk_project.annotation_agent.report import write_outputs
from nk_project.io_utils import ensure_dirs


DEFAULT_GROUPBY = "leiden_0_4"
PAIRWISE_TOP_N = 100
DEFAULT_MAX_DISTANCE_PAIRS_PER_ROUND = 25


def main() -> None:
    args = parse_args()
    leiden_dir = args.leiden_dir or os.path.join(cfg.BASE_OUTDIR, "leiden_discovery")
    marker_dir = args.marker_dir or os.path.join(cfg.BASE_OUTDIR, "markers", "full", args.groupby)
    outdir = args.outdir or os.path.join(cfg.BASE_OUTDIR, "annotation_agent", args.groupby)
    input_h5ad = args.input_h5ad or os.path.join(leiden_dir, "full_scvi_leiden.h5ad")
    pairwise_dir = args.pairwise_dir or os.path.join(outdir, "pairwise_de")
    ensure_dirs(outdir)

    print("=" * 80)
    print("NK subtype/state annotation agent")
    print("=" * 80)
    print(f"[LEIDEN_DIR] {leiden_dir}")
    print(f"[MARKER_DIR] {marker_dir}")
    print(f"[OUTDIR] {outdir}")
    print(f"[GROUPBY] {args.groupby}")
    print(f"[MAX_ITERATIONS] {args.max_iterations}")
    print(f"[PAIRWISE_SPLIT_AUDIT] {args.run_pairwise_split_audit}")
    print(f"[PAIRWISE_DIR] {pairwise_dir}")
    if args.run_pairwise_split_audit:
        print(f"[PAIRWISE_DISTANCE_QUANTILE] {args.distance_quantile}")
        print(f"[PAIRWISE_MAX_PAIRS] {args.max_distance_pairs_per_round}")
        print(f"[PAIRWISE_DE_METHOD] {args.pairwise_de_method}")
        print(f"[PAIRWISE_MODEL_DIR] {args.pairwise_model_dir}")
        print(f"[PAIRWISE_TRAIN_NAMES] {args.pairwise_train_names}")
    print(f"[ACTIVE_LLM] {args.active_llm}")

    evidence = load_evidence(leiden_dir, marker_dir, pairwise_dir, args)
    save_evidence_outputs(evidence, outdir, args)
    print(f"[CLUSTERS] {len(evidence)}")

    if args.dry_run:
        print("[DRY-RUN] Evidence loaded successfully; skipping LLM calls.")
        return

    if args.test_llm:
        cluster_id = args.cluster_id or sorted(evidence, key=cluster_sort_key)[0]
        print(f"[TEST_LLM] Running one cluster only: {cluster_id}")
        result = run_one_cluster(evidence, cluster_id, args)
        print(json.dumps(result["final_decision"], indent=2))
        return

    results = run_agent_round(evidence, outdir, args)

    if args.run_pairwise_split_audit and not args.cluster_id:
        results, evidence = run_pairwise_split_audit(
            results=results,
            evidence=evidence,
            leiden_dir=leiden_dir,
            marker_dir=marker_dir,
            input_h5ad=input_h5ad,
            pairwise_dir=pairwise_dir,
            outdir=outdir,
            args=args,
        )

    write_outputs(
        results,
        evidence,
        outdir,
        args.groupby,
        review_threshold=args.review_threshold,
        save_debug_trace=args.save_debug_evidence,
    )
    save_evidence_outputs(evidence, outdir, args)
    print("[DONE] Annotation agent complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evidence-only NK subtype/state annotation agent."
    )
    parser.add_argument("--leiden-dir", default=None)
    parser.add_argument("--marker-dir", default=None)
    parser.add_argument("--input-h5ad", default=None)
    parser.add_argument("--pairwise-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--groupby", default=DEFAULT_GROUPBY)
    parser.add_argument("--top-de-genes", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=1)
    parser.add_argument("--review-threshold", type=int, default=4)
    parser.add_argument("--cluster-id", default=None, help="Run only one cluster, useful for testing.")
    parser.add_argument("--test-llm", action="store_true", help="Make one LLM call on one cluster and print the result.")
    parser.add_argument(
        "--run-pairwise-split-audit",
        action="store_true",
        help=(
            "After the one-pass annotation, compare distant clusters that received "
            "the same structured label and let pairwise DE refine the free_label."
        ),
    )
    parser.add_argument(
        "--distance-quantile",
        type=float,
        default=0.50,
        help=(
            "Only used with --run-pairwise-split-audit. Same-label cluster pairs "
            "must be at or above this NK-cluster centroid-distance quantile to be "
            "audited by pairwise DE. Default: 0.50."
        ),
    )
    parser.add_argument(
        "--max-distance-pairs-per-round",
        type=int,
        default=DEFAULT_MAX_DISTANCE_PAIRS_PER_ROUND,
        help=(
            "Only used with --run-pairwise-split-audit. Maximum same-label distant "
            f"cluster pairs to audit. Default: {DEFAULT_MAX_DISTANCE_PAIRS_PER_ROUND}."
        ),
    )
    parser.add_argument(
        "--latent-key",
        default=None,
        help="AnnData.obsm key for centroid distances. Default: auto-detect X_scVI or similar.",
    )
    parser.add_argument(
        "--pairwise-de-method",
        choices=["scvi", "scanpy"],
        default="scvi",
        help="DE method for pairwise split audit. Default uses model.differential_expression().",
    )
    parser.add_argument(
        "--pairwise-model-dir",
        default=None,
        help="Trained scVI/scANVI model directory for pairwise model-based DE.",
    )
    parser.add_argument(
        "--pairwise-model-class",
        choices=["auto", "SCVI", "SCANVI"],
        default="auto",
        help="Model class for pairwise DE. Default auto tries SCVI then SCANVI.",
    )
    parser.add_argument(
        "--pairwise-train-names",
        default=None,
        help="Optional train_obs_names.txt for loading the pairwise DE model with train cells first.",
    )
    parser.add_argument(
        "--pairwise-marker-fdr",
        type=float,
        default=0.02,
        help="FDR target passed to model.differential_expression() for pairwise DE.",
    )
    parser.add_argument(
        "--pairwise-scvi-de-mode",
        choices=["change", "vanilla"],
        default="change",
        help="mode argument for pairwise model.differential_expression().",
    )
    parser.add_argument(
        "--pairwise-scvi-delta",
        type=float,
        default=0.25,
        help="delta argument for pairwise model.differential_expression().",
    )
    parser.add_argument(
        "--pairwise-scvi-de-batch-size",
        type=int,
        default=32768,
        help="Batch size for pairwise model.differential_expression(). Default: 32768.",
    )
    parser.add_argument(
        "--no-pairwise-scvi-batch-correction",
        dest="pairwise_scvi_batch_correction",
        action="store_false",
        help="Turn off batch_correction for pairwise model-based DE. Default is on.",
    )
    parser.set_defaults(pairwise_scvi_batch_correction=True)
    parser.add_argument("--llm-retries", type=int, default=5)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--active-llm",
        default=os.environ.get("NK_ANNOTATION_AGENT_LLM", "41_mini"),
        choices=["4o", "41", "41_mini", "5_mini"],
        help="Local MDA/Azure LLM factory selection.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--save-debug-evidence",
        action="store_true",
        help="Save full raw evidence and JSONL decision traces under debug/.",
    )
    args = parser.parse_args()
    if not 1 <= args.max_iterations <= 3:
        raise ValueError("--max-iterations must be between 1 and 3.")
    if not 0 <= args.review_threshold <= 5:
        raise ValueError("--review-threshold must be between 0 and 5.")
    if not 0 <= args.distance_quantile <= 1:
        raise ValueError("--distance-quantile must be between 0 and 1.")
    if args.max_distance_pairs_per_round < 1:
        raise ValueError("--max-distance-pairs-per-round must be at least 1.")
    if args.run_pairwise_split_audit and args.pairwise_de_method == "scvi" and not args.pairwise_model_dir:
        raise ValueError("--run-pairwise-split-audit with --pairwise-de-method scvi requires --pairwise-model-dir.")
    return args


def load_evidence(leiden_dir, marker_dir, pairwise_dir, args):
    paths = EvidencePaths(
        leiden_dir=leiden_dir,
        marker_dir=marker_dir,
        groupby=args.groupby,
        pairwise_dir=pairwise_dir,
    )
    return load_cluster_evidence(paths, top_n=args.top_de_genes)


def run_agent_round(evidence, outdir, args):
    ensure_dirs(outdir)
    results = []
    cluster_ids = [args.cluster_id] if args.cluster_id else sorted(evidence, key=cluster_sort_key)
    partial_path = debug_trace_path(outdir) if args.save_debug_evidence else None
    if partial_path and os.path.exists(partial_path):
        os.remove(partial_path)
    for idx, cluster_id in enumerate(cluster_ids, start=1):
        if cluster_id not in evidence:
            raise KeyError(f"Cluster {cluster_id!r} not found in evidence.")
        print(f"\n[AGENT] cluster {cluster_id} ({idx}/{len(cluster_ids)})")
        result = run_one_cluster(evidence, cluster_id, args)
        final = result["final_decision"]
        print(
            "[DRAFT] "
            f"{cluster_id} -> {final['final_structured_label']} "
            f"(free={final['free_label']}, confidence={final['confidence_score']}/5, "
            f"review={final['needs_human_review']})"
        )
        results.append(result)
        if partial_path:
            with open(partial_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result) + "\n")
    return results


def run_pairwise_split_audit(
    *,
    results,
    evidence,
    leiden_dir,
    marker_dir,
    input_h5ad,
    pairwise_dir,
    outdir,
    args,
):
    print("\n" + "=" * 80)
    print("[SPLIT_AUDIT] Same-structured-label pairwise DE")
    print("=" * 80)
    distance_df, latent_key = centroid_distance_table(
        input_h5ad=input_h5ad,
        groupby=args.groupby,
        latent_key=args.latent_key,
    )
    distance_path = os.path.join(outdir, "cluster_centroid_distances.csv")
    distance_df.to_csv(distance_path, index=False)
    print(f"[DISTANCE_LATENT_KEY] {latent_key}")
    print(f"[SAVE] {distance_path}")

    pairs, summary = same_label_distance_pairs_from_results(
        results,
        distance_df,
        min_quantile=args.distance_quantile,
        max_pairs=args.max_distance_pairs_per_round,
    )
    summary_path = os.path.join(outdir, "same_label_distance_pair_candidates.csv")
    summary.to_csv(summary_path, index=False)
    print(f"[SAVE] {summary_path}")
    print(f"[SPLIT_AUDIT_PAIRS] {len(pairs)}")
    if not pairs:
        return results, evidence

    existing = existing_pair_set(pairwise_dir)
    new_pairs = [pair for pair in pairs if pair not in existing]
    print(f"[PAIRWISE_DE_NEW] {len(new_pairs)}")
    run_pairwise_de_for_pairs(
        input_h5ad=input_h5ad,
        groupby=args.groupby,
        pairs=new_pairs,
        outdir=pairwise_dir,
        top_n=PAIRWISE_TOP_N,
        de_method=args.pairwise_de_method,
        model_dir=args.pairwise_model_dir,
        model_class=args.pairwise_model_class,
        train_names=args.pairwise_train_names,
        marker_fdr=args.pairwise_marker_fdr,
        scvi_de_mode=args.pairwise_scvi_de_mode,
        scvi_delta=args.pairwise_scvi_delta,
        scvi_de_batch_size=args.pairwise_scvi_de_batch_size,
        scvi_batch_correction=args.pairwise_scvi_batch_correction,
    )

    evidence = load_evidence(leiden_dir, marker_dir, pairwise_dir, args)
    add_same_label_split_candidates(evidence, pairs)
    result_by_cluster = {str(result["cluster_id"]): result for result in results}
    audit_rows = []
    for cluster_a, cluster_b in pairs[: args.max_distance_pairs_per_round]:
        if cluster_a not in result_by_cluster or cluster_b not in result_by_cluster:
            continue
        decision_a = result_by_cluster[cluster_a]["final_decision"]
        decision_b = result_by_cluster[cluster_b]["final_decision"]
        structured_label = structured_label_from_final(decision_a)
        if structured_label != structured_label_from_final(decision_b):
            continue
        audit = run_pairwise_split_agent(
            cluster_a=cluster_a,
            cluster_b=cluster_b,
            structured_label=structured_label,
            evidence_a=evidence_for_agent(evidence[cluster_a], args),
            evidence_b=evidence_for_agent(evidence[cluster_b], args),
            decision_a=decision_a,
            decision_b=decision_b,
            active_llm=args.active_llm,
            temperature=args.temperature,
            llm_retries=args.llm_retries,
            retry_sleep=args.retry_sleep,
        )
        audit_rows.append(audit)
        apply_pairwise_audit(result_by_cluster, audit)

    audit_path = os.path.join(outdir, "pairwise_split_audit.csv")
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print(f"[SAVE] {audit_path}")
    return [result_by_cluster[c] for c in sorted(result_by_cluster, key=cluster_sort_key)], evidence


def apply_pairwise_audit(result_by_cluster: dict[str, dict[str, Any]], audit: dict[str, Any]) -> None:
    if not audit.get("split_supported", False):
        return
    for side in ["a", "b"]:
        cluster_id = str(audit[f"cluster_{side}"])
        final = result_by_cluster[cluster_id]["final_decision"]
        label = str(audit.get(f"cluster_{side}_free_label") or "").strip()
        reason = str(audit.get(f"cluster_{side}_free_label_reason") or "").strip()
        if label:
            final["free_label"] = label
        if reason:
            prior = str(final.get("free_label_reason") or "").strip()
            final["free_label_reason"] = f"{prior} Pairwise split audit: {reason}".strip()
        final["needs_human_review"] = True
        audit_reason = str(audit.get("human_review_reason") or "Pairwise DE supports a distinct free label.").strip()
        prior_review = str(final.get("human_review_reason") or "").strip()
        final["human_review_reason"] = "; ".join(part for part in [prior_review, audit_reason] if part)


def run_one_cluster(evidence, cluster_id, args):
    cluster_evidence = evidence_for_agent(evidence[cluster_id], args)
    return run_cluster_agent(
        cluster_evidence,
        active_llm=args.active_llm,
        max_iterations=args.max_iterations,
        temperature=args.temperature,
        llm_retries=args.llm_retries,
        retry_sleep=args.retry_sleep,
    )


def evidence_for_agent(cluster_evidence, args):
    return copy.deepcopy(cluster_evidence) if args.save_debug_evidence else compact_cluster_evidence(cluster_evidence)


def save_evidence_outputs(evidence, outdir, args):
    compact_path = os.path.join(outdir, "cluster_evidence_summary.json")
    save_compact_evidence_json(evidence, compact_path)
    print(f"[SAVE] {compact_path}")
    if args.save_debug_evidence:
        debug_dir = os.path.join(outdir, "debug")
        ensure_dirs(debug_dir)
        full_path = os.path.join(debug_dir, "cluster_evidence_full.json")
        save_evidence_json(evidence, full_path)
        print(f"[SAVE] {full_path}")


def add_same_label_split_candidates(evidence, pairs):
    by_cluster = {cluster_id: [] for cluster_id in evidence}
    for a, b in pairs:
        if a in by_cluster:
            by_cluster[a].append(b)
        if b in by_cluster:
            by_cluster[b].append(a)
    for cluster_id, others in by_cluster.items():
        if others:
            evidence[cluster_id]["same_label_split_candidates"] = sorted(set(others), key=cluster_sort_key)


def debug_trace_path(outdir):
    debug_dir = os.path.join(outdir, "debug")
    ensure_dirs(debug_dir)
    return os.path.join(debug_dir, "cluster_decision_trace.partial.jsonl")


if __name__ == "__main__":
    main()
