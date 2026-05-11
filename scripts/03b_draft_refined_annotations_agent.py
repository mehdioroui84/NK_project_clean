#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.annotation_agent.evidence import EvidencePaths, load_cluster_evidence, save_evidence_json
from nk_project.annotation_agent.graph import run_cluster_agent
from nk_project.annotation_agent.marker_knowledge import KNOWN_REFINED_LABELS
from nk_project.annotation_agent.pairwise import (
    centroid_distance_table,
    cluster_distance_evidence_from_results,
    existing_pair_set,
    recommended_pairs_from_results,
    run_pairwise_de_for_pairs,
    same_label_distance_pairs_from_results,
)
from nk_project.annotation_agent.report import write_outputs
from nk_project.io_utils import ensure_dirs


DEFAULT_GROUPBY = "leiden_0_4"


def main() -> None:
    args = parse_args()
    leiden_dir = args.leiden_dir or os.path.join(cfg.BASE_OUTDIR, "leiden_discovery")
    marker_dir = args.marker_dir or os.path.join(cfg.BASE_OUTDIR, "markers", "full", args.groupby)
    outdir = args.outdir or os.path.join(cfg.BASE_OUTDIR, "annotation_agent", args.groupby)
    input_h5ad = args.input_h5ad or os.path.join(leiden_dir, "full_scvi_leiden.h5ad")
    pairwise_dir = args.pairwise_dir or os.path.join(outdir, "pairwise_de")
    ensure_dirs(outdir)

    print("=" * 80)
    print("Optional refined annotation agent")
    print("=" * 80)
    print(f"[LEIDEN_DIR] {leiden_dir}")
    print(f"[MARKER_DIR] {marker_dir}")
    print(f"[OUTDIR] {outdir}")
    print(f"[GROUPBY] {args.groupby}")
    print(f"[MAX_ITERATIONS] {args.max_iterations}")
    print(f"[PAIRWISE_REFINEMENT_ITERATIONS] {args.pairwise_refinement_iterations}")
    print("[LOCK_RULE] confidence>=4, ambiguity<=2, technical_concern<2, approved label, no review flag")
    print("[LOW_PRIORITY_RULE] confidence>=3, ambiguity<=2, technical_concern<2, approved label")
    print(f"[PAIRWISE_DIR] {pairwise_dir}")
    print(f"[DISTANCE_PAIRWISE] {not args.disable_distance_pairwise}")
    print(f"[DISTANCE_QUANTILE] {args.distance_quantile}")
    print(f"[DISCOVERY_FIRST] {args.discovery_first}")
    print(f"[ACTIVE_LLM] {args.active_llm}")
    print(f"[LLM_RETRIES] {args.llm_retries}")
    print("[SCORE_RANGE] 0-5")

    paths = EvidencePaths(
        leiden_dir=leiden_dir,
        marker_dir=marker_dir,
        groupby=args.groupby,
        pairwise_dir=pairwise_dir,
    )
    evidence = load_cluster_evidence(paths, top_n=args.top_de_genes)
    evidence_path = os.path.join(outdir, "cluster_evidence_summaries.json")
    save_evidence_json(evidence, evidence_path)
    print(f"[SAVE] {evidence_path}")
    print(f"[CLUSTERS] {len(evidence)}")

    if args.dry_run:
        print("[DRY-RUN] Evidence loaded successfully; skipping LLM calls.")
        return

    if args.test_llm:
        cluster_id = args.cluster_id or sorted(evidence, key=cluster_sort_key)[0]
        print(f"[TEST_LLM] Running one cluster only: {cluster_id}")
        args.discovery_first_active = bool(args.discovery_first)
        result = run_one_cluster(evidence, cluster_id, args)
        print(json.dumps(result["final_decision"], indent=2))
        return

    if args.pairwise_refinement_iterations:
        run_pairwise_refinement(
            args=args,
            leiden_dir=leiden_dir,
            marker_dir=marker_dir,
            input_h5ad=input_h5ad,
            pairwise_dir=pairwise_dir,
            outdir=outdir,
        )
    else:
        args.discovery_first_active = bool(args.discovery_first)
        results = run_agent_round(evidence, outdir, args)
        args.discovery_first_active = False
        write_outputs(
            results,
            evidence,
            outdir,
            args.groupby,
            review_threshold=args.review_threshold,
        )
    print("[DONE] Annotation agent draft complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optional LangGraph annotation copilot that drafts refined labels from "
            "Leiden composition, top DE markers, curated marker means, and related-cluster evidence."
        )
    )
    parser.add_argument("--leiden-dir", default=None)
    parser.add_argument("--marker-dir", default=None)
    parser.add_argument("--input-h5ad", default=None)
    parser.add_argument("--pairwise-dir", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--groupby", default=DEFAULT_GROUPBY)
    parser.add_argument("--top-de-genes", type=int, default=50)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--review-threshold", type=int, default=4)
    parser.add_argument("--cluster-id", default=None, help="Run only one cluster, useful for testing.")
    parser.add_argument("--test-llm", action="store_true", help="Make one LLM call on one cluster and print the result.")
    parser.add_argument(
        "--pairwise-refinement-iterations",
        type=int,
        default=0,
        help="Optional agent -> recommended pairwise DE -> agent refinement rounds. Max 3.",
    )
    parser.add_argument("--max-pairwise-checks-per-round", type=int, default=20)
    parser.add_argument("--pairwise-top-n", type=int, default=100)
    parser.add_argument(
        "--disable-distance-pairwise",
        action="store_true",
        help=(
            "Disable same-label centroid-distance triggers. By default, pairwise "
            "refinement also checks same-label clusters that are far apart in SCVI latent space."
        ),
    )
    parser.add_argument(
        "--distance-quantile",
        type=float,
        default=0.90,
        help="Centroid-distance quantile used to trigger same-label pairwise DE checks.",
    )
    parser.add_argument(
        "--isolation-quantile",
        type=float,
        default=0.90,
        help="Nearest-centroid distance quantile used to flag isolated possible subtype clusters.",
    )
    parser.add_argument(
        "--max-distance-pairs-per-round",
        type=int,
        default=10,
        help="Maximum same-label distance-triggered pairwise DE checks per refinement round.",
    )
    parser.add_argument(
        "--latent-key",
        default=None,
        help="AnnData.obsm key for centroid distances. Default: auto-detect X_scVI or similar.",
    )
    parser.add_argument(
        "--discovery-first",
        action="store_true",
        help=(
            "In the first annotation pass, hide worksheet draft labels/review hints "
            "from the LLM so it interprets clusters from evidence before mapping to "
            "the approved refined-label vocabulary."
        ),
    )
    parser.add_argument("--disable-locking", action="store_true", help="Rerun all clusters in every refinement round.")
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
    args = parser.parse_args()
    if not 1 <= args.max_iterations <= 5:
        raise ValueError("--max-iterations must be between 1 and 5.")
    if not 0 <= args.review_threshold <= 5:
        raise ValueError("--review-threshold must be between 0 and 5.")
    if not 0 <= args.pairwise_refinement_iterations <= 3:
        raise ValueError("--pairwise-refinement-iterations must be between 0 and 3.")
    if not 0 <= args.distance_quantile <= 1:
        raise ValueError("--distance-quantile must be between 0 and 1.")
    if not 0 <= args.isolation_quantile <= 1:
        raise ValueError("--isolation-quantile must be between 0 and 1.")
    return args


def run_agent_round(evidence, outdir, args):
    ensure_dirs(outdir)
    results = []
    cluster_ids = [args.cluster_id] if args.cluster_id else sorted(evidence, key=cluster_sort_key)
    partial_path = os.path.join(outdir, "cluster_decision_trace.partial.jsonl")
    if os.path.exists(partial_path):
        os.remove(partial_path)
    for idx, cluster_id in enumerate(cluster_ids, start=1):
        if cluster_id not in evidence:
            raise KeyError(f"Cluster {cluster_id!r} not found in evidence.")
        print(f"\n[AGENT] cluster {cluster_id} ({idx}/{len(cluster_ids)})")
        result = run_one_cluster(evidence, cluster_id, args)
        final = result["final_decision"]
        print(
            "[DRAFT] "
            f"{cluster_id} -> {final['candidate_label']} "
            f"(confidence={final['confidence_score']}/5, "
            f"iterations={len(result['iterations'])}, "
            f"review={final['needs_human_review']})"
        )
        results.append(result)
        with open(partial_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return results


def run_pairwise_refinement(args, leiden_dir, marker_dir, input_h5ad, pairwise_dir, outdir):
    final_results_by_cluster = {}
    final_evidence = None
    active_cluster_ids = None
    distance_df = None
    distance_evidence_csv = None
    for round_idx in range(args.pairwise_refinement_iterations + 1):
        print("\n" + "=" * 80)
        print(f"[REFINEMENT_ROUND] {round_idx}")
        print("=" * 80)
        round_outdir = os.path.join(outdir, f"round_{round_idx}")
        ensure_dirs(round_outdir)
        paths = EvidencePaths(
            leiden_dir=leiden_dir,
            marker_dir=marker_dir,
            groupby=args.groupby,
            pairwise_dir=pairwise_dir,
            distance_evidence_csv=distance_evidence_csv,
        )
        evidence = load_cluster_evidence(paths, top_n=args.top_de_genes)
        save_evidence_json(evidence, os.path.join(round_outdir, "cluster_evidence_summaries.json"))
        if active_cluster_ids is None or args.disable_locking:
            round_cluster_ids = [args.cluster_id] if args.cluster_id else sorted(evidence, key=cluster_sort_key)
        else:
            round_cluster_ids = [cluster_id for cluster_id in active_cluster_ids if cluster_id in evidence]
        if not round_cluster_ids:
            print("[REFINEMENT_STOP] No active clusters left to re-annotate.")
            break

        original_cluster_id = args.cluster_id
        args.cluster_id = None
        args.discovery_first_active = bool(args.discovery_first and round_idx == 0)
        results = run_agent_round_for_clusters(evidence, round_outdir, args, round_cluster_ids)
        args.discovery_first_active = False
        args.cluster_id = original_cluster_id

        for result in results:
            final_results_by_cluster[str(result["cluster_id"])] = result

        combined_results = [
            final_results_by_cluster[cluster_id]
            for cluster_id in sorted(final_results_by_cluster, key=cluster_sort_key)
        ]
        if not args.disable_distance_pairwise:
            if distance_df is None:
                print("[DISTANCE_LOAD] Computing cluster centroid distances")
                distance_df, latent_key = centroid_distance_table(
                    input_h5ad=input_h5ad,
                    groupby=args.groupby,
                    latent_key=args.latent_key,
                )
                distance_path = os.path.join(outdir, "cluster_centroid_distances.csv")
                distance_df.to_csv(distance_path, index=False)
                print(f"[DISTANCE_LATENT_KEY] {latent_key}")
                print(f"[SAVE] {distance_path}")
            distance_evidence = cluster_distance_evidence_from_results(
                combined_results,
                distance_df,
                distance_quantile=args.distance_quantile,
                isolation_quantile=args.isolation_quantile,
            )
            distance_evidence_csv = os.path.join(round_outdir, "cluster_distance_evidence.csv")
            distance_evidence.to_csv(distance_evidence_csv, index=False)
            print(f"[SAVE] {distance_evidence_csv}")
            for cluster_id, item in distance_evidence.set_index("cluster_id").to_dict(orient="index").items():
                if str(cluster_id) in evidence:
                    evidence[str(cluster_id)]["distance_novelty_evidence"] = item
        write_outputs(
            combined_results,
            evidence,
            round_outdir,
            args.groupby,
            review_threshold=args.review_threshold,
        )
        final_evidence = evidence

        statuses = {
            str(result["cluster_id"]): classify_decision(result["final_decision"])
            for result in combined_results
        }
        status_path = os.path.join(round_outdir, "cluster_review_status.csv")
        save_status_table(combined_results, statuses, status_path)
        n_locked = sum(status == "locked" for status in statuses.values())
        n_low = sum(status == "low_priority_review" for status in statuses.values())
        n_active = sum(status == "active_review" for status in statuses.values())
        print(f"[STATUS] locked={n_locked}, low_priority_review={n_low}, active_review={n_active}")

        if round_idx >= args.pairwise_refinement_iterations:
            break

        valid_ids = set(evidence)
        active_cluster_ids = [
            cluster_id
            for cluster_id, status in statuses.items()
            if status != "locked"
        ]
        if not active_cluster_ids:
            print("[PAIRWISE_STOP] No review clusters remain.")
            break
        print(f"[REFINEMENT_CONTINUE] review/low-priority clusters={len(active_cluster_ids)}")
        active_results = [
            result
            for result in combined_results
            if str(result["cluster_id"]) in set(active_cluster_ids)
        ]
        llm_pairs = recommended_pairs_from_results(
            active_results,
            valid_ids,
            max_pairs=args.max_pairwise_checks_per_round,
        )
        distance_pairs = []
        if not args.disable_distance_pairwise:
            distance_pairs, distance_summary = same_label_distance_pairs_from_results(
                combined_results,
                distance_df,
                active_cluster_ids=set(active_cluster_ids),
                min_quantile=args.distance_quantile,
                max_pairs=args.max_distance_pairs_per_round,
            )
            distance_summary_path = os.path.join(round_outdir, "same_label_distance_pair_candidates.csv")
            distance_summary.to_csv(distance_summary_path, index=False)
            print(f"[SAVE] {distance_summary_path}")

        pairs = merge_pair_queues(
            llm_pairs,
            distance_pairs,
            max_pairs=args.max_pairwise_checks_per_round,
        )
        existing = existing_pair_set(pairwise_dir)
        pairs = [pair for pair in pairs if pair not in existing]
        print(f"[PAIRWISE_LLM_CANDIDATES] {len(llm_pairs)}")
        print(f"[PAIRWISE_DISTANCE_CANDIDATES] {len(distance_pairs)}")
        print(f"[PAIRWISE_RECOMMENDED_NEW] {len(pairs)}")
        if not pairs:
            print("[PAIRWISE_SKIP] No new cluster-pair recommendations; continuing annotation-only refinement.")
            continue

        for cluster_a, cluster_b in pairs:
            print(f"[PAIRWISE_QUEUE] {cluster_a} vs {cluster_b}")
        run_pairwise_de_for_pairs(
            input_h5ad=input_h5ad,
            groupby=args.groupby,
            pairs=pairs,
            outdir=pairwise_dir,
            top_n=args.pairwise_top_n,
        )

    if final_results_by_cluster and final_evidence is not None:
        final_results = [
            final_results_by_cluster[cluster_id]
            for cluster_id in sorted(final_results_by_cluster, key=cluster_sort_key)
        ]
        write_outputs(
            final_results,
            final_evidence,
            outdir,
            args.groupby,
            review_threshold=args.review_threshold,
        )


def run_agent_round_for_clusters(evidence, outdir, args, cluster_ids):
    ensure_dirs(outdir)
    results = []
    partial_path = os.path.join(outdir, "cluster_decision_trace.partial.jsonl")
    if os.path.exists(partial_path):
        os.remove(partial_path)
    for idx, cluster_id in enumerate(cluster_ids, start=1):
        if cluster_id not in evidence:
            raise KeyError(f"Cluster {cluster_id!r} not found in evidence.")
        print(f"\n[AGENT] cluster {cluster_id} ({idx}/{len(cluster_ids)})")
        result = run_one_cluster(evidence, cluster_id, args)
        final = result["final_decision"]
        status = classify_decision(final)
        print(
            "[DRAFT] "
            f"{cluster_id} -> {final['candidate_label']} "
            f"(confidence={final['confidence_score']}/5, "
            f"iterations={len(result['iterations'])}, "
            f"review={final['needs_human_review']}, "
            f"status={status})"
        )
        results.append(result)
        with open(partial_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return results


def classify_decision(final):
    label = str(final.get("candidate_label", ""))
    approved = label in set(KNOWN_REFINED_LABELS)
    confidence = int(final.get("confidence_score", 0))
    ambiguity = int(final.get("ambiguity_score", 5))
    technical = int(final.get("technical_concern_score", 5))
    needs_review = bool(final.get("needs_human_review", True))
    has_new_label = bool(str(final.get("suggested_new_label", "")).strip())
    if approved and confidence >= 4 and ambiguity <= 2 and technical < 2 and not needs_review and not has_new_label:
        return "locked"
    if approved and confidence >= 3 and ambiguity <= 2 and technical < 2 and not has_new_label:
        return "low_priority_review"
    return "active_review"


def save_status_table(results, statuses, path):
    rows = []
    for result in results:
        final = result["final_decision"]
        cluster_id = str(result["cluster_id"])
        rows.append(
            {
                "cluster_id": cluster_id,
                "candidate_refined_label": final.get("candidate_label"),
                "suggested_new_label": final.get("suggested_new_label", ""),
                "review_status": statuses[cluster_id],
                "confidence_score": final.get("confidence_score"),
                "ambiguity_score": final.get("ambiguity_score"),
                "technical_concern_score": final.get("technical_concern_score"),
                "needs_human_review": final.get("needs_human_review"),
            }
        )
    import pandas as pd

    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"[SAVE] {path}")


def merge_pair_queues(*queues, max_pairs=None):
    pairs = []
    seen = set()
    for queue in queues:
        for pair in queue:
            normalized = tuple(sorted((str(pair[0]), str(pair[1])), key=cluster_sort_key))
            if normalized in seen:
                continue
            seen.add(normalized)
            pairs.append(normalized)
            if max_pairs is not None and len(pairs) >= max_pairs:
                return pairs
    return pairs


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
    if not bool(getattr(args, "discovery_first_active", False)):
        return cluster_evidence
    stripped = copy.deepcopy(cluster_evidence)
    stripped["annotation_mode"] = "discovery_first_no_worksheet_draft"
    comp = stripped.get("composition", {})
    for key in [
        "draft_refined_label",
        "draft_refined_label_raw",
        "worksheet_review_note",
        "review_notes",
    ]:
        comp.pop(key, None)
    for related in stripped.get("related_clusters", []):
        related.pop("draft_refined_label", None)
    return stripped


def cluster_sort_key(value):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


if __name__ == "__main__":
    main()
