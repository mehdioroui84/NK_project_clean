#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_CLUSTER_KEY = "leiden_0_4"


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    table_dir = outdir / "tables"
    figure_dir = outdir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    composition_counts = load_composition_counts(args.composition_counts_csv, args.cluster_key)
    composition_long = make_composition_long(composition_counts, args.cluster_key)
    composition_summary = make_composition_summary(composition_counts, args.cluster_key)

    mapping = load_agent_mapping(args.mapping_csv, args.cluster_key)
    markers = load_top_markers(args.markers_csv, args.cluster_key, args.top_n_markers)

    report = (
        composition_summary.merge(mapping, on=args.cluster_key, how="left")
        .merge(marker_summary(markers, args.cluster_key), on=args.cluster_key, how="left")
        .sort_values(args.cluster_key, key=lambda s: s.map(cluster_sort_key))
    )

    summary_csv = table_dir / "cluster_report_summary.csv"
    comp_csv = table_dir / "cluster_original_annotation_composition.csv"
    markers_csv = table_dir / "cluster_top_markers.csv"
    report.to_csv(summary_csv, index=False)
    composition_long.to_csv(comp_csv, index=False)
    markers.to_csv(markers_csv, index=False)
    print(f"[SAVE] {summary_csv}")
    print(f"[SAVE] {comp_csv}")
    print(f"[SAVE] {markers_csv}")

    xlsx_path = table_dir / "cellxgene_cluster_report.xlsx"
    write_xlsx(
        xlsx_path,
        {
            "cluster_summary": report,
            "full_original_composition": composition_long,
            "top_markers": markers,
        },
    )

    copied_figures = copy_figures(args, figure_dir)
    md_path = outdir / "cellxgene_cluster_report.md"
    write_markdown_report(md_path, report, markers, copied_figures, args)
    print(f"[SAVE] {md_path}")
    print("[DONE] CellXGene cluster annotation report exported.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a colleague-facing CellXGene cluster report with original NK/T/B composition, "
            "agent annotations/rationale, and cluster-specific markers."
        )
    )
    parser.add_argument("--cluster-key", default=DEFAULT_CLUSTER_KEY)
    parser.add_argument(
        "--composition-counts-csv",
        default="outputs/leiden_discovery/leiden_0_4_by_NK_State.csv",
        help="Wide count table: one row per cluster, one column per original CellXGene/manual label.",
    )
    parser.add_argument(
        "--mapping-csv",
        default="outputs/annotation_agent/leiden_0_4_subtype_state_gpt5mini_sanity_v1/cluster_annotation_mapping.csv",
        help="Agent cluster annotation mapping CSV.",
    )
    parser.add_argument(
        "--markers-csv",
        default="outputs/markers/full/leiden_0_4/leiden_0_4_markers_top50_per_cluster.csv",
        help="Cluster marker table, usually the top50 marker table from marker analysis.",
    )
    parser.add_argument(
        "--annotation-figure",
        default="outputs/refined_annotation_v1_agent_preferred_gpt5mini_sanity_v1/figures/annotation_umap_review_panels.png",
        help="Optional annotation QC figure to copy into the report folder.",
    )
    parser.add_argument(
        "--marker-figure",
        default="outputs/markers/full/leiden_0_4/leiden_0_4_dotplot_top_markers.png",
        help="Optional marker figure to copy into the report folder.",
    )
    parser.add_argument(
        "--outdir",
        default="reports/cellxgene_cluster_annotation_report",
    )
    parser.add_argument("--top-n-markers", type=int, default=50)
    return parser.parse_args()


def load_composition_counts(path: str, cluster_key: str) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False)
    if cluster_key not in df.columns:
        raise KeyError(f"{path} must contain {cluster_key!r}.")
    df[cluster_key] = df[cluster_key].astype(str)
    value_cols = [c for c in df.columns if c != cluster_key]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    print(f"[LOAD] original CellXGene/manual composition: {path}")
    return df


def make_composition_long(df: pd.DataFrame, cluster_key: str) -> pd.DataFrame:
    long = df.melt(id_vars=[cluster_key], var_name="cellxgene_annotation", value_name="n_cells")
    totals = long.groupby(cluster_key)["n_cells"].transform("sum")
    long["percent_of_cluster"] = (100.0 * long["n_cells"] / totals.where(totals > 0)).fillna(0).round(2)
    long = long.loc[long["n_cells"] > 0].copy()
    long["annotation_class"] = long["cellxgene_annotation"].map(classify_original_label)
    return long.sort_values(
        [cluster_key, "n_cells"],
        ascending=[True, False],
        key=lambda s: s.map(cluster_sort_key) if s.name == cluster_key else s,
    )


def make_composition_summary(df: pd.DataFrame, cluster_key: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    label_cols = [c for c in df.columns if c != cluster_key]
    for _, row in df.iterrows():
        cluster_id = str(row[cluster_key])
        counts = pd.Series({label: int(row[label]) for label in label_cols})
        total = int(counts.sum())
        class_counts = {"NK-like": 0, "T": 0, "B": 0, "Other": 0}
        for label, count in counts.items():
            class_counts[classify_original_label(label)] += int(count)
        top_counts = counts[counts > 0].sort_values(ascending=False)
        top_label = str(top_counts.index[0]) if not top_counts.empty else ""
        top_label_pct = pct(int(top_counts.iloc[0]), total) if not top_counts.empty else 0.0
        rows.append(
            {
                cluster_key: cluster_id,
                "n_cells": total,
                "cellxgene_pct_NK_like": pct(class_counts["NK-like"], total),
                "cellxgene_pct_T": pct(class_counts["T"], total),
                "cellxgene_pct_B": pct(class_counts["B"], total),
                "cellxgene_pct_Other": pct(class_counts["Other"], total),
                "top_manual_annotation": top_label,
                "top_manual_annotation_pct": top_label_pct,
                "top_manual_annotation_composition": format_top_composition(top_counts, total, top_n=6),
            }
        )
    return pd.DataFrame(rows)


def classify_original_label(label: Any) -> str:
    text = str(label).strip()
    lower = text.lower().replace("_", " ")
    if text == "T" or "t cell" in lower or lower.startswith("cd3"):
        return "T"
    if text == "B" or "b cell" in lower or lower.startswith("ms4a1") or lower.startswith("cd79"):
        return "B"
    other_terms = ["myeloid", "monocyte", "macrophage", "stromal", "epithelial", "erythroid", "fibroblast"]
    if any(term in lower for term in other_terms):
        return "Other"
    return "NK-like"


def load_agent_mapping(path: str, cluster_key: str) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False).fillna("")
    if cluster_key not in df.columns:
        raise KeyError(f"{path} must contain {cluster_key!r}.")
    df[cluster_key] = df[cluster_key].astype(str)
    keep = [
        cluster_key,
        "nk_subtype_call",
        "nk_state_call",
        "final_structured_label",
        "free_label",
        "free_label_reason",
        "needs_human_review",
        "human_review_reason",
        "confidence_score_0_5",
        "tissue_specificity_score_0_5",
        "dataset_assay_specificity_score_0_5",
        "top_tissue",
        "top_dataset",
        "top_assay",
        "taxonomy_top_matches",
        "n_pairwise_de_compared",
    ]
    keep = [c for c in keep if c in df.columns]
    print(f"[LOAD] agent mapping: {path}")
    return df[keep].copy()


def load_top_markers(path: str, cluster_key: str, top_n: int) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False)
    cluster_col = first_present(df, [cluster_key, "group", "cluster"])
    gene_col = first_present(df, ["names", "gene", "Gene"])
    if cluster_col is None or gene_col is None:
        raise KeyError(f"{path} must contain a cluster column and a gene column.")
    df = df.rename(columns={cluster_col: cluster_key, gene_col: "gene"}).copy()
    df[cluster_key] = df[cluster_key].astype(str)
    if "marker_direction" in df.columns:
        df = df.loc[df["marker_direction"].astype(str).str.lower().eq("up")].copy()
    elif "logfoldchanges" in df.columns:
        df = df.loc[pd.to_numeric(df["logfoldchanges"], errors="coerce").fillna(0) > 0].copy()
    sort_cols = [c for c in ["scores", "logfoldchanges", "pct_expr_diff", "pct_nz_group"] if c in df.columns]
    if sort_cols:
        df["_sort_score"] = pd.to_numeric(df[sort_cols[0]], errors="coerce").fillna(0)
        df = df.sort_values([cluster_key, "_sort_score"], ascending=[True, False])
    else:
        df["_sort_score"] = 0
    df["marker_rank"] = df.groupby(cluster_key).cumcount() + 1
    df = df.loc[df["marker_rank"] <= top_n].copy()
    keep = [
        cluster_key,
        "marker_rank",
        "gene",
        "scores",
        "logfoldchanges",
        "pvals_adj",
        "pct_nz_group",
        "pct_nz_reference",
        "pct_expr_group",
        "pct_expr_reference",
        "pct_expr_diff",
    ]
    keep = [c for c in keep if c in df.columns]
    print(f"[LOAD] top cluster markers: {path}")
    return df[keep].sort_values([cluster_key, "marker_rank"], key=cluster_marker_sort_key)


def marker_summary(markers: pd.DataFrame, cluster_key: str) -> pd.DataFrame:
    rows = []
    for cluster_id, sub in markers.groupby(cluster_key, sort=False):
        genes = sub.sort_values("marker_rank")["gene"].astype(str).tolist()
        rows.append({cluster_key: str(cluster_id), "top_cluster_markers": "; ".join(genes)})
    return pd.DataFrame(rows)


def copy_figures(args: argparse.Namespace, figure_dir: Path) -> list[Path]:
    copied = []
    for label, path in [("annotation_qc", args.annotation_figure), ("marker_dotplot", args.marker_figure)]:
        if not path:
            continue
        src = Path(path)
        if not src.exists():
            print(f"[WARN] figure not found, skipping: {src}")
            continue
        dst = figure_dir / f"{label}{src.suffix}"
        shutil.copy2(src, dst)
        copied.append(dst)
        print(f"[SAVE] {dst}")
    return copied


def write_markdown_report(
    path: Path,
    report: pd.DataFrame,
    markers: pd.DataFrame,
    copied_figures: list[Path],
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# CellXGene Cluster Annotation Report")
    lines.append("")
    lines.append(
        "This report summarizes original CellXGene/manual label composition, current agent labels, "
        "agent rationale, and cluster-specific positive DE markers."
    )
    lines.append("")
    if copied_figures:
        lines.append("## Figures")
        lines.append("")
        for fig in copied_figures:
            rel = fig.relative_to(path.parent)
            lines.append(f"- `{rel}`")
        lines.append("")
    lines.append("## Cluster Summary")
    lines.append("")
    summary_cols = [
        args.cluster_key,
        "n_cells",
        "cellxgene_pct_NK_like",
        "cellxgene_pct_T",
        "cellxgene_pct_B",
        "top_manual_annotation",
        "final_structured_label",
        "free_label",
        "needs_human_review",
    ]
    summary_cols = [c for c in summary_cols if c in report.columns]
    lines.append(dataframe_to_markdown(report[summary_cols]))
    lines.append("")
    lines.append("## Cluster Details")
    lines.append("")
    for _, row in report.iterrows():
        cluster_id = str(row[args.cluster_key])
        lines.append(f"### Cluster {cluster_id}")
        lines.append("")
        lines.append(f"- n cells: {row.get('n_cells', '')}")
        lines.append(
            "- CellXGene/manual composition: "
            f"NK-like {row.get('cellxgene_pct_NK_like', '')}%, "
            f"T {row.get('cellxgene_pct_T', '')}%, "
            f"B {row.get('cellxgene_pct_B', '')}%, "
            f"Other {row.get('cellxgene_pct_Other', '')}%"
        )
        lines.append(f"- Top manual annotations: {row.get('top_manual_annotation_composition', '')}")
        lines.append(f"- Structured agent label: {row.get('final_structured_label', '')}")
        lines.append(f"- Free agent label: {row.get('free_label', '')}")
        if str(row.get("free_label_reason", "")).strip():
            lines.append(f"- Agent rationale: {row.get('free_label_reason', '')}")
        if str(row.get("human_review_reason", "")).strip():
            lines.append(f"- Human review reason: {row.get('human_review_reason', '')}")
        lines.append(f"- Top markers: {row.get('top_cluster_markers', '')}")
        marker_details = format_marker_details(markers.loc[markers[args.cluster_key].astype(str) == cluster_id])
        if marker_details:
            lines.append(f"- Marker details: {marker_details}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_xlsx(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    try:
        with pd.ExcelWriter(path) as writer:
            for sheet_name, df in sheets.items():
                df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        print(f"[SAVE] {path}")
    except Exception as exc:  # pragma: no cover - optional dependency can vary on HPC
        print(f"[WARN] could not write XLSX ({type(exc).__name__}: {exc})")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```\n" + df.to_string(index=False) + "\n```"


def format_top_composition(counts: pd.Series, total: int, top_n: int) -> str:
    if total <= 0:
        return ""
    parts = []
    for label, count in counts.head(top_n).items():
        parts.append(f"{label}: {int(count):,} ({pct(int(count), total)}%)")
    return "; ".join(parts)


def format_marker_details(sub: pd.DataFrame) -> str:
    if sub.empty:
        return ""
    parts = []
    for _, row in sub.head(10).iterrows():
        gene = row.get("gene", "")
        logfc = fmt(row.get("logfoldchanges", ""))
        pct_group = fmt(row.get("pct_nz_group", row.get("pct_expr_group", "")))
        pct_ref = fmt(row.get("pct_nz_reference", row.get("pct_expr_reference", "")))
        pieces = [str(gene)]
        if logfc != "":
            pieces.append(f"logFC={logfc}")
        if pct_group != "":
            pieces.append(f"pct={pct_group}")
        if pct_ref != "":
            pieces.append(f"ref_pct={pct_ref}")
        parts.append(" ".join(pieces))
    return "; ".join(parts)


def first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * float(numerator) / float(denominator), 2)


def fmt(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return ""


def cluster_sort_key(value: Any) -> tuple[int, Any]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def cluster_marker_sort_key(series: pd.Series) -> pd.Series:
    if series.name in {DEFAULT_CLUSTER_KEY, "cluster", "group"}:
        return series.map(cluster_sort_key)
    return series


def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


if __name__ == "__main__":
    main()
