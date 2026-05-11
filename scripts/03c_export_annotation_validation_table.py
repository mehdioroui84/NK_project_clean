#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from nk_project.annotation_agent.marker_knowledge import MARKER_PROGRAMS, marker_program_hits
from nk_project.io_utils import ensure_dirs


DEFAULT_GROUPBY = "leiden_0_4"


def main() -> None:
    args = parse_args()
    groupby = args.groupby
    mapping = load_mapping(args.mapping_csv, groupby)
    worksheet = pd.read_csv(args.worksheet_csv, low_memory=False)
    worksheet[groupby] = worksheet[groupby].astype(str)
    manual_counts = pd.read_csv(args.manual_counts_csv, low_memory=False)
    manual_counts[groupby] = manual_counts[groupby].astype(str)
    markers = pd.read_csv(args.top_markers_csv, low_memory=False)
    markers["group"] = markers["group"].astype(str)
    curated = pd.read_csv(args.curated_means_csv, low_memory=False)
    curated[groupby] = curated[groupby].astype(str)
    agent_decisions = load_agent_decisions(args.agent_dir, args.agent_trace_jsonl)

    rows = []
    for _, row in mapping.sort_values(groupby, key=lambda s: s.map(cluster_sort_key)).iterrows():
        cluster_id = str(row[groupby])
        refined_label = str(row["refined_annotation"])
        wk = worksheet.loc[worksheet[groupby].astype(str) == cluster_id]
        wk_row = wk.iloc[0].to_dict() if not wk.empty else {}
        marker_rows = markers.loc[markers["group"].astype(str) == cluster_id].head(args.top_n_de)
        curated_rows = curated.loc[curated[groupby].astype(str) == cluster_id]
        curated_row = curated_rows.iloc[0].to_dict() if not curated_rows.empty else {}

        top_gene_names = marker_rows["names"].astype(str).tolist()
        program_hits = marker_program_hits(top_gene_names)
        agent = agent_decisions.get(cluster_id, {})

        rows.append(
            {
                groupby: cluster_id,
                "refined_annotation": refined_label,
                "agent_annotation": agent.get("candidate_label", ""),
                "agent_reasoning": agent_reasoning(agent),
                "agent_concerns": "; ".join(map(str, agent.get("concerns", []))),
                "agent_alternative_name": agent.get("suggested_new_label", "") or "None",
                "agent_alternative_name_reason": agent.get("new_label_reason", ""),
                "agent_confidence_score_0_5": agent.get("confidence_score", ""),
                "n_cells": wk_row.get("n_cells"),
                "top_manual_NK_State": wk_row.get("top_NK_State"),
                "top_manual_NK_State_percent": pct(wk_row.get("top_NK_State_frac")),
                "manual_NK_State_composition_top5": manual_composition(manual_counts, cluster_id, groupby, top_n=5),
                "top_tissue": wk_row.get("top_tissue"),
                "top_tissue_percent": pct(wk_row.get("top_tissue_frac")),
                "top_dataset_id": wk_row.get("top_dataset_id"),
                "top_dataset_percent": pct(wk_row.get("top_dataset_id_frac")),
                "top_assay_clean": wk_row.get("top_assay_clean"),
                "top_assay_percent": pct(wk_row.get("top_assay_clean_frac")),
                "script02_draft_label": wk_row.get("draft_refined_label"),
                "script02_review_notes": wk_row.get("review_notes"),
                "top_DE_genes": "; ".join(top_gene_names[: args.top_n_de]),
                "top_DE_gene_details": de_details(marker_rows, top_n=args.detail_n_de),
                "DE_marker_program_hits": format_program_hits(program_hits),
                "curated_NK_cytotoxic_markers": curated_program_summary(curated_row, "NK cytotoxic", top_n=args.top_n_curated),
                "curated_tissue_chemokine_markers": curated_program_summary(
                    curated_row, "NK tissue/regulatory/chemokine", top_n=args.top_n_curated
                ),
                "curated_proliferation_markers": curated_program_summary(curated_row, "proliferation", top_n=args.top_n_curated),
                "curated_interferon_cytokine_markers": curated_program_summary(
                    curated_row, "interferon/cytokine", top_n=args.top_n_curated
                ),
                "curated_non_NK_marker_flags": non_nk_marker_summary(curated_row, top_n=args.top_n_curated),
                "curated_top_marker_means": top_curated_means(curated_row, top_n=args.top_n_curated),
                "validation_question": validation_question(refined_label, wk_row.get("top_NK_State"), program_hits, curated_row),
            }
        )

    out = pd.DataFrame(rows)
    ensure_dirs(os.path.dirname(args.out_csv))
    out.to_csv(args.out_csv, index=False)
    print(f"[SAVE] {args.out_csv}")
    if args.out_xlsx:
        if write_xlsx(out, args.out_xlsx):
            print(f"[SAVE] {args.out_xlsx}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a colleague-friendly table for validating refined Leiden cluster annotations."
    )
    parser.add_argument("--groupby", default=DEFAULT_GROUPBY)
    parser.add_argument(
        "--mapping-csv",
        default="outputs/refined_annotation_v1/full_leiden_0_4_to_refined_v1_mapping.csv",
        help="Refined label mapping. Supports NK_State_refined or candidate_refined_label.",
    )
    parser.add_argument(
        "--worksheet-csv",
        default="outputs/leiden_discovery/full_leiden_0_4_annotation_worksheet.csv",
    )
    parser.add_argument(
        "--manual-counts-csv",
        default="outputs/leiden_discovery/leiden_0_4_by_NK_State.csv",
    )
    parser.add_argument(
        "--top-markers-csv",
        default="outputs/markers/full/leiden_0_4/leiden_0_4_markers_top50_per_cluster.csv",
    )
    parser.add_argument(
        "--curated-means-csv",
        default="outputs/markers/full/leiden_0_4/leiden_0_4_curated_marker_cluster_means.csv",
    )
    parser.add_argument(
        "--out-csv",
        default="reports/refined_annotation_validation_table.csv",
    )
    parser.add_argument(
        "--out-xlsx",
        default="reports/refined_annotation_validation_table.xlsx",
    )
    parser.add_argument(
        "--agent-dir",
        default=None,
        help=(
            "Optional annotation-agent output directory. If provided, the export includes "
            "agent annotation and reasoning from cluster_decision_trace.jsonl."
        ),
    )
    parser.add_argument(
        "--agent-trace-jsonl",
        default=None,
        help="Optional direct path to cluster_decision_trace.jsonl. Overrides --agent-dir.",
    )
    parser.add_argument("--top-n-de", type=int, default=20)
    parser.add_argument("--detail-n-de", type=int, default=10)
    parser.add_argument("--top-n-curated", type=int, default=8)
    return parser.parse_args()


def load_mapping(path: str, groupby: str) -> pd.DataFrame:
    mapping = pd.read_csv(path, low_memory=False)
    if groupby not in mapping.columns:
        raise KeyError(f"{path} must contain {groupby!r}.")
    label_col = first_present(mapping, ["NK_State_refined", "candidate_refined_label", "refined_annotation"])
    if label_col is None:
        raise KeyError(f"{path} must contain one of NK_State_refined, candidate_refined_label, refined_annotation.")
    out = mapping[[groupby, label_col]].copy()
    out[groupby] = out[groupby].astype(str)
    out = out.rename(columns={label_col: "refined_annotation"})
    return out


def load_agent_decisions(agent_dir: str | None, trace_jsonl: str | None) -> dict[str, dict[str, Any]]:
    path = trace_jsonl
    if path is None and agent_dir:
        direct = Path(agent_dir) / "cluster_decision_trace.jsonl"
        if direct.exists():
            path = str(direct)
        else:
            rounds = sorted(Path(agent_dir).glob("round_*/cluster_decision_trace.jsonl"))
            if rounds:
                path = str(rounds[-1])
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    decisions: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            cluster_id = str(item.get("cluster_id", ""))
            final = item.get("final_decision", {})
            if cluster_id:
                decisions[cluster_id] = final
    print(f"[LOAD] agent decisions: {path} ({len(decisions)} clusters)")
    return decisions


def agent_reasoning(agent: dict[str, Any]) -> str:
    if not agent:
        return ""
    summary = [str(item) for item in agent.get("evidence_summary", []) if str(item).strip()]
    stop_reason = str(agent.get("stop_reason", "") or "").strip()
    pieces = []
    if summary:
        pieces.append("; ".join(summary))
    if stop_reason:
        pieces.append(f"Decision: {stop_reason}")
    return " | ".join(pieces)


def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def pct(value: Any) -> float | None:
    try:
        return round(float(value) * 100, 1)
    except (TypeError, ValueError):
        return None


def manual_composition(df: pd.DataFrame, cluster_id: str, groupby: str, *, top_n: int) -> str:
    sub = df.loc[df[groupby].astype(str) == str(cluster_id)]
    if sub.empty:
        return ""
    row = sub.iloc[0].drop(labels=[groupby], errors="ignore")
    counts = pd.to_numeric(row, errors="coerce").dropna()
    counts = counts[counts > 0].sort_values(ascending=False)
    total = counts.sum()
    parts = []
    for label, count in counts.head(top_n).items():
        parts.append(f"{label}: {int(count):,} ({100 * float(count) / float(total):.1f}%)")
    return "; ".join(parts)


def de_details(marker_rows: pd.DataFrame, *, top_n: int) -> str:
    parts = []
    for _, row in marker_rows.head(top_n).iterrows():
        gene = row.get("names")
        logfc = numeric_text(row.get("logfoldchanges"), ndigits=2)
        pct_group = numeric_text(row.get("pct_nz_group"), ndigits=2)
        pct_ref = numeric_text(row.get("pct_nz_reference"), ndigits=2)
        parts.append(f"{gene} logFC={logfc}, pct={pct_group} vs {pct_ref}")
    return "; ".join(parts)


def format_program_hits(program_hits: dict[str, list[str]]) -> str:
    if not program_hits:
        return ""
    return "; ".join(f"{program}: {', '.join(genes)}" for program, genes in program_hits.items())


def curated_program_summary(curated_row: dict[str, Any], program: str, *, top_n: int) -> str:
    markers = MARKER_PROGRAMS.get(program, [])
    return marker_value_summary(curated_row, markers, top_n=top_n)


def non_nk_marker_summary(curated_row: dict[str, Any], *, top_n: int) -> str:
    markers = []
    for program in ["T cell", "B cell", "myeloid", "erythroid", "lung/stromal-like"]:
        markers.extend(MARKER_PROGRAMS.get(program, []))
    return marker_value_summary(curated_row, markers, top_n=top_n)


def marker_value_summary(curated_row: dict[str, Any], markers: list[str], *, top_n: int) -> str:
    values = []
    for marker in markers:
        if marker not in curated_row:
            continue
        try:
            value = float(curated_row[marker])
        except (TypeError, ValueError):
            continue
        values.append((marker, value))
    values = sorted(values, key=lambda item: item[1], reverse=True)
    return "; ".join(f"{gene}={value:.2f}" for gene, value in values[:top_n] if value > 0)


def top_curated_means(curated_row: dict[str, Any], *, top_n: int) -> str:
    excluded = {DEFAULT_GROUPBY, "leiden_0_4"}
    values = []
    for gene, value in curated_row.items():
        if gene in excluded:
            continue
        try:
            values.append((str(gene), float(value)))
        except (TypeError, ValueError):
            continue
    values = sorted(values, key=lambda item: item[1], reverse=True)
    return "; ".join(f"{gene}={value:.2f}" for gene, value in values[:top_n])


def validation_question(
    refined_label: str,
    top_manual_label: Any,
    program_hits: dict[str, list[str]],
    curated_row: dict[str, Any],
) -> str:
    top_manual = str(top_manual_label)
    if refined_label == top_manual:
        return "Does the marker evidence support keeping this manual label?"
    if refined_label in {"T", "B", "Myeloid-like", "Unknown_BM_1 Erythroid-like", "Unknown_Kidney"}:
        return "Does the non-NK/contamination-like marker evidence justify separating this from NK states?"
    if "proliferation" in program_hits:
        return "Does the proliferation program justify this refined functional-state label?"
    if "NK tissue/regulatory/chemokine" in program_hits:
        return "Does the tissue/chemokine marker program justify this refined NK state?"
    if has_high_non_nk(curated_row):
        return "Are the lineage-marker flags biological, ambient RNA, or possible doublets?"
    return "Does the DE and curated marker evidence support this refinement from the manual label?"


def has_high_non_nk(curated_row: dict[str, Any], *, threshold: float = 1.0) -> bool:
    for program in ["T cell", "B cell", "myeloid", "erythroid"]:
        for marker in MARKER_PROGRAMS.get(program, []):
            try:
                if float(curated_row.get(marker, 0)) >= threshold:
                    return True
            except (TypeError, ValueError):
                continue
    return False


def numeric_text(value: Any, *, ndigits: int) -> str:
    try:
        return str(round(float(value), ndigits))
    except (TypeError, ValueError):
        return "NA"


def write_xlsx(df: pd.DataFrame, path: str) -> bool:
    ensure_dirs(os.path.dirname(path))
    try:
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="validation_table")
            ws = writer.sheets["validation_table"]
            ws.freeze_panes = "A2"
            for cell in ws[1]:
                cell.style = "Headline 3"
            widths = {
                "A": 12,
                "B": 34,
                "C": 12,
                "D": 28,
                "E": 18,
                "F": 80,
                "O": 60,
                "P": 90,
                "Q": 60,
                "W": 70,
            }
            for col, width in widths.items():
                ws.column_dimensions[col].width = width
    except ImportError:
        print("[WARN] openpyxl not installed; skipped XLSX export.")
        return False
    return True


def cluster_sort_key(value: Any) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


if __name__ == "__main__":
    main()
