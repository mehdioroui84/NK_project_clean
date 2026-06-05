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

    if args.cellxgene_h5ad:
        composition_counts = load_composition_counts_from_h5ad(
            args.cellxgene_h5ad,
            args.cluster_key,
            args.cellxgene_annotation_key,
            args.exclude_dataset_id,
        )
    else:
        composition_counts = load_composition_counts(args.composition_counts_csv, args.cluster_key)
    composition_long = make_composition_long(composition_counts, args.cluster_key)
    composition_summary = make_composition_summary(composition_counts, args.cluster_key)
    yuntao_summary = load_manual_annotation_summary(args)

    if args.cellxgene_h5ad and args.context_from_cellxgene_h5ad:
        cluster_context = load_cluster_context_from_h5ad(
            args.cellxgene_h5ad,
            args.cluster_key,
            args.exclude_dataset_id,
        )
    else:
        cluster_context = load_cluster_summary(args.cluster_summary_csv, args.cluster_key)
    mapping = load_agent_mapping(args.mapping_csv, args.cluster_key)
    top_n_positive = args.top_n_positive_markers
    if top_n_positive is None:
        top_n_positive = args.top_n_markers
    markers = load_top_markers(
        args.markers_csv,
        args.cluster_key,
        top_n_positive=top_n_positive,
        top_n_negative=args.top_n_negative_markers,
    )

    report = (
        composition_summary.merge(cluster_context, on=args.cluster_key, how="left")
        .merge(mapping, on=args.cluster_key, how="left")
        .merge(marker_summary(markers, args.cluster_key), on=args.cluster_key, how="left")
        .sort_values(args.cluster_key, key=lambda s: s.map(cluster_sort_key))
    )
    all_annotations_report = make_all_annotations_report(report, yuntao_summary, args.cluster_key)
    evidence_only_report = make_evidence_only_report(report, args.cluster_key)

    comp_csv = table_dir / "cluster_original_annotation_composition.csv"
    markers_csv = table_dir / "cluster_top_markers.csv"
    all_annotations_csv = table_dir / "cluster_tissue_marker_report.csv"
    evidence_only_csv = table_dir / "cluster_evidence_only_report_for_agent.csv"
    legacy_summary_csv = table_dir / "cluster_report_summary.csv"
    if legacy_summary_csv.exists():
        legacy_summary_csv.unlink()
        print(f"[REMOVE_STALE] {legacy_summary_csv}")
    composition_long.to_csv(comp_csv, index=False)
    markers.to_csv(markers_csv, index=False)
    all_annotations_report.to_csv(all_annotations_csv, index=False)
    evidence_only_report.to_csv(evidence_only_csv, index=False)
    print(f"[SAVE] {comp_csv}")
    print(f"[SAVE] {markers_csv}")
    print(f"[SAVE] {all_annotations_csv}")
    print(f"[SAVE] {evidence_only_csv}")

    xlsx_sheets = {
        "all_annotations_report": all_annotations_report,
        "evidence_only_for_agent": evidence_only_report,
        "cellxgene_composition": composition_long,
        "top_markers": markers,
    }
    xlsx_path = table_dir / "cellxgene_cluster_report.xlsx"
    write_xlsx(xlsx_path, xlsx_sheets)

    copied_figures = copy_figures(args, figure_dir)
    md_path = outdir / "cellxgene_cluster_report.md"
    write_markdown_report(md_path, all_annotations_report, markers, copied_figures, args)
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
        help="Wide count table: one row per cluster, one column per annotation label.",
    )
    parser.add_argument(
        "--cellxgene-h5ad",
        default="",
        help="Optional AnnData file. If set, build annotation composition directly from obs instead of --composition-counts-csv.",
    )
    parser.add_argument(
        "--cellxgene-annotation-key",
        default="cell_type",
        help="obs column to summarize as the CellXGene website annotation when --cellxgene-h5ad is set.",
    )
    parser.add_argument(
        "--manual-h5ad",
        default="",
        help="Optional AnnData file containing the previous/manual annotation. Defaults to --cellxgene-h5ad when unset.",
    )
    parser.add_argument(
        "--manual-annotation-key",
        default="",
        help="Optional obs column to summarize as previous/manual annotation, e.g. NK_State.",
    )
    parser.add_argument(
        "--manual-composition-counts-csv",
        default="",
        help="Optional wide cluster-by-manual-annotation count table. Used if --manual-annotation-key is unset.",
    )
    parser.add_argument(
        "--exclude-dataset-id",
        action="append",
        default=[],
        help="dataset_id value to exclude when building CellXGene annotation composition from H5AD. Can be repeated.",
    )
    parser.add_argument(
        "--context-from-cellxgene-h5ad",
        action="store_true",
        help="Compute top tissue/dataset/assay percentages from --cellxgene-h5ad using the same dataset exclusions.",
    )
    parser.add_argument(
        "--cluster-summary-csv",
        default="outputs/markers/full/leiden_0_4/leiden_0_4_cluster_summary.csv",
        help="Cluster summary table with top tissue/dataset/assay fractions.",
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
    parser.add_argument(
        "--top-n-markers",
        type=int,
        default=50,
        help="Backward-compatible alias for --top-n-positive-markers when that flag is not set.",
    )
    parser.add_argument(
        "--top-n-positive-markers",
        type=int,
        default=None,
        help="Number of positive/up markers to keep per cluster.",
    )
    parser.add_argument(
        "--top-n-negative-markers",
        type=int,
        default=0,
        help="Number of negative/depleted markers to keep per cluster.",
    )
    return parser.parse_args()


def load_composition_counts(path: str, cluster_key: str) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False)
    if cluster_key not in df.columns:
        raise KeyError(f"{path} must contain {cluster_key!r}.")
    df[cluster_key] = df[cluster_key].astype(str)
    value_cols = [c for c in df.columns if c != cluster_key]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    print(f"[LOAD] annotation composition: {path}")
    return df


def load_composition_counts_from_h5ad(
    path: str,
    cluster_key: str,
    annotation_key: str,
    exclude_dataset_ids: list[str],
    source_label: str = "CellXGene annotation",
) -> pd.DataFrame:
    require_file(path)
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    required = [cluster_key, annotation_key]
    missing = [c for c in required if c not in obs.columns]
    if missing:
        raise KeyError(f"{path} is missing obs columns: {missing}")

    use = pd.DataFrame(
        {
            cluster_key: obs[cluster_key].astype(str),
            annotation_key: obs[annotation_key].astype(str),
        },
        index=obs.index,
    )
    if exclude_dataset_ids:
        if "dataset_id" not in obs.columns:
            raise KeyError("--exclude-dataset-id was set, but obs['dataset_id'] is missing.")
        excluded = {str(x) for x in exclude_dataset_ids}
        use["dataset_id"] = obs["dataset_id"].astype(str)
        use = use.loc[~use["dataset_id"].isin(excluded)].copy()

    use = use.loc[
        use[cluster_key].notna()
        & use[annotation_key].notna()
        & ~use[cluster_key].isin(["", "nan", "None"])
        & ~use[annotation_key].isin(["", "nan", "None"])
    ].copy()
    counts = pd.crosstab(use[cluster_key], use[annotation_key]).reset_index()
    counts[cluster_key] = counts[cluster_key].astype(str)
    print(f"[LOAD] {source_label} composition from {path}: obs['{annotation_key}']")
    if exclude_dataset_ids:
        print(f"[FILTER] excluded dataset_id: {', '.join(map(str, exclude_dataset_ids))}")
    return counts


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
        class_counts = {"NK": 0, "T": 0, "B": 0, "Other": 0}
        for label, count in counts.items():
            class_counts[classify_original_label(label)] += int(count)
        top_counts = counts[counts > 0].sort_values(ascending=False)
        top_label = str(top_counts.index[0]) if not top_counts.empty else ""
        top_label_pct = pct(int(top_counts.iloc[0]), total) if not top_counts.empty else 0.0
        rows.append(
            {
                cluster_key: cluster_id,
                "n_cells": total,
                "cellxgene_annotation_n_cells": total,
                "cellxgene_pct_NK": pct(class_counts["NK"], total),
                "cellxgene_pct_T": pct(class_counts["T"], total),
                "cellxgene_pct_B": pct(class_counts["B"], total),
                "cellxgene_pct_Other": pct(class_counts["Other"], total),
                "top_cellxgene_annotation": top_label,
                "top_cellxgene_annotation_percent": top_label_pct,
                "cellxgene_annotation_composition": format_top_composition(top_counts, total, top_n=6),
            }
        )
    return pd.DataFrame(rows)


def load_manual_annotation_summary(args: argparse.Namespace) -> pd.DataFrame:
    cluster_key = args.cluster_key
    if args.manual_annotation_key:
        h5ad_path = args.manual_h5ad or args.cellxgene_h5ad
        if not h5ad_path:
            raise ValueError("--manual-annotation-key requires --manual-h5ad or --cellxgene-h5ad.")
        counts = load_composition_counts_from_h5ad(
            h5ad_path,
            cluster_key,
            args.manual_annotation_key,
            exclude_dataset_ids=[],
            source_label="manual annotation",
        )
    elif args.manual_composition_counts_csv:
        counts = load_composition_counts(args.manual_composition_counts_csv, cluster_key)
    else:
        return pd.DataFrame({cluster_key: []})

    base = make_composition_summary(counts, cluster_key)
    summary = pd.DataFrame(
        {
            cluster_key: base[cluster_key],
            "manual_annotation_n_cells": base["n_cells"],
            "manual_pct_NK": base["cellxgene_pct_NK"],
            "manual_pct_T": base["cellxgene_pct_T"],
            "manual_pct_B": base["cellxgene_pct_B"],
            "manual_pct_Other": base["cellxgene_pct_Other"],
            "top_manual_annotation": base["top_cellxgene_annotation"],
            "top_manual_annotation_percent": base["top_cellxgene_annotation_percent"],
            "manual_annotation_composition": base["cellxgene_annotation_composition"],
        }
    )
    print("[LOAD] previous/manual annotation summary")
    return summary.copy()


def load_cluster_summary(path: str, cluster_key: str) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False)
    if cluster_key not in df.columns:
        raise KeyError(f"{path} must contain {cluster_key!r}.")
    df[cluster_key] = df[cluster_key].astype(str)
    out = pd.DataFrame({cluster_key: df[cluster_key]})
    if "n_cells" in df.columns:
        out["total_cluster_n_cells"] = pd.to_numeric(df["n_cells"], errors="coerce").fillna(0).astype(int)

    rename_map = {
        "top_tissue": "top_tissue",
        "top_dataset_id": "top_dataset",
        "top_assay_clean": "top_assay",
    }
    frac_map = {
        "top_tissue_frac": "top_tissue_percent",
        "top_dataset_id_frac": "top_dataset_percent",
        "top_assay_clean_frac": "top_assay_percent",
    }
    for src, dst in rename_map.items():
        if src in df.columns:
            out[dst] = df[src].astype(str)
    for src, dst in frac_map.items():
        if src in df.columns:
            out[dst] = (100.0 * pd.to_numeric(df[src], errors="coerce")).round(2)
    print(f"[LOAD] cluster tissue/dataset/assay summary: {path}")
    return out


def load_cluster_context_from_h5ad(
    path: str,
    cluster_key: str,
    exclude_dataset_ids: list[str],
) -> pd.DataFrame:
    require_file(path)
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    obs = adata.obs
    if cluster_key not in obs.columns:
        raise KeyError(f"{path} is missing obs['{cluster_key}']")
    use = pd.DataFrame({cluster_key: obs[cluster_key].astype(str)}, index=obs.index)
    for col in ["tissue", "dataset_id", "assay_clean"]:
        if col in obs.columns:
            use[col] = obs[col].astype(str)
    if exclude_dataset_ids:
        if "dataset_id" not in use.columns:
            raise KeyError("--exclude-dataset-id was set, but obs['dataset_id'] is missing.")
        excluded = {str(x) for x in exclude_dataset_ids}
        use = use.loc[~use["dataset_id"].isin(excluded)].copy()
    use = use.loc[~use[cluster_key].isin(["", "nan", "None"])].copy()

    cluster_counts = use[cluster_key].value_counts().rename("total_cluster_n_cells").reset_index()
    cluster_counts.columns = [cluster_key, "total_cluster_n_cells"]
    out = pd.DataFrame({cluster_key: sorted(use[cluster_key].unique(), key=cluster_sort_key)})
    out = out.merge(cluster_counts, on=cluster_key, how="left")
    context_specs = [
        ("tissue", "top_tissue", "top_tissue_percent"),
        ("dataset_id", "top_dataset", "top_dataset_percent"),
        ("assay_clean", "top_assay", "top_assay_percent"),
    ]
    for source_col, top_col, pct_col in context_specs:
        if source_col not in use.columns:
            continue
        rows = []
        for cluster_id, sub in use.groupby(cluster_key):
            values = sub[source_col].loc[~sub[source_col].isin(["", "nan", "None"])]
            counts = values.value_counts()
            if counts.empty:
                rows.append({cluster_key: cluster_id, top_col: "", pct_col: 0.0})
            else:
                top_value = str(counts.index[0])
                rows.append({cluster_key: cluster_id, top_col: top_value, pct_col: pct(int(counts.iloc[0]), int(counts.sum()))})
        out = out.merge(pd.DataFrame(rows), on=cluster_key, how="left")
    print(f"[LOAD] cluster tissue/dataset/assay context from {path}")
    if exclude_dataset_ids:
        print(f"[FILTER] context excluded dataset_id: {', '.join(map(str, exclude_dataset_ids))}")
    return out


def classify_original_label(label: Any) -> str:
    text = str(label).strip()
    lower = text.lower().replace("_", " ")
    if text == "T" or "t cell" in lower or lower.startswith("cd3"):
        return "T"
    if text == "B" or "b cell" in lower or lower.startswith("ms4a1") or lower.startswith("cd79"):
        return "B"
    nk_terms = [
        "natural killer",
        " nk",
        "nk ",
        "cd56",
        "cd16",
        "cytokine-stimulated",
        "mature cytotoxic",
        "transitional cytotoxic",
        "proliferative",
        "developmental",
        "regulatory",
        "unconventional",
    ]
    if any(term in lower for term in nk_terms):
        return "NK"
    other_terms = ["myeloid", "monocyte", "macrophage", "stromal", "epithelial", "erythroid", "fibroblast"]
    if any(term in lower for term in other_terms):
        return "Other"
    return "Other"


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
        "taxonomy_top_matches",
        "n_pairwise_de_compared",
    ]
    keep = [c for c in keep if c in df.columns]
    print(f"[LOAD] agent mapping: {path}")
    return df[keep].copy()


def load_top_markers(
    path: str,
    cluster_key: str,
    top_n_positive: int,
    top_n_negative: int,
) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False)
    cluster_col = first_present(df, [cluster_key, "group", "cluster"])
    gene_col = first_present(df, ["names", "gene", "Gene"])
    if cluster_col is None or gene_col is None:
        raise KeyError(f"{path} must contain a cluster column and a gene column.")
    df = df.rename(columns={cluster_col: cluster_key, gene_col: "gene"}).copy()
    df[cluster_key] = df[cluster_key].astype(str)
    if "marker_direction" in df.columns:
        direction = df["marker_direction"].astype(str).str.lower()
        df["marker_direction"] = direction.map({"positive": "up", "pos": "up", "negative": "down", "neg": "down"}).fillna(direction)
    elif "logfoldchanges" in df.columns:
        logfc = pd.to_numeric(df["logfoldchanges"], errors="coerce").fillna(0)
        df["marker_direction"] = "neutral"
        df.loc[logfc > 0, "marker_direction"] = "up"
        df.loc[logfc < 0, "marker_direction"] = "down"
    else:
        df["marker_direction"] = "up"

    selected: list[pd.DataFrame] = []
    if top_n_positive != 0:
        selected.append(select_markers_by_direction(df, cluster_key, "up", top_n_positive))
    if top_n_negative != 0:
        selected.append(select_markers_by_direction(df, cluster_key, "down", top_n_negative))
    if selected:
        df = pd.concat(selected, ignore_index=True)
    else:
        df = df.iloc[0:0].copy()
    keep = [
        cluster_key,
        "marker_direction",
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
    df["_direction_order"] = df["marker_direction"].map({"up": 0, "down": 1}).fillna(2)
    df = df.sort_values(
        [cluster_key, "_direction_order", "marker_rank"],
        key=cluster_marker_sort_key,
    )
    return df[keep]


def select_markers_by_direction(
    df: pd.DataFrame,
    cluster_key: str,
    direction: str,
    top_n: int,
) -> pd.DataFrame:
    sub = df.loc[df["marker_direction"].astype(str).eq(direction)].copy()
    if sub.empty:
        return sub
    score_col = first_present(sub, ["scores", "logfoldchanges", "pct_expr_diff", "pct_nz_group"])
    if score_col is None:
        sub["_sort_score"] = 0.0
    else:
        sub["_sort_score"] = pd.to_numeric(sub[score_col], errors="coerce").fillna(0).abs()
    sub = sub.sort_values([cluster_key, "_sort_score"], ascending=[True, False])
    sub["marker_rank"] = sub.groupby(cluster_key).cumcount() + 1
    if top_n > 0:
        sub = sub.loc[sub["marker_rank"] <= top_n].copy()
    return sub


def marker_summary(markers: pd.DataFrame, cluster_key: str) -> pd.DataFrame:
    rows = []
    for cluster_id, sub in markers.groupby(cluster_key, sort=False):
        sub = sub.sort_values(["marker_direction", "marker_rank"])
        positive_genes = (
            sub.loc[sub["marker_direction"].astype(str).eq("up"), "gene"].astype(str).tolist()
            if "marker_direction" in sub.columns
            else sub["gene"].astype(str).tolist()
        )
        negative_genes = (
            sub.loc[sub["marker_direction"].astype(str).eq("down"), "gene"].astype(str).tolist()
            if "marker_direction" in sub.columns
            else []
        )
        marker_list = "; ".join(positive_genes)
        rows.append(
            {
                cluster_key: str(cluster_id),
                "positive_marker_count": len(positive_genes),
                "negative_marker_count": len(negative_genes),
                "positive_marker_list": marker_list,
                "negative_marker_list": "; ".join(negative_genes),
                "marker_count": len(positive_genes) + len(negative_genes),
                "cluster_marker_list": " | ".join(
                    part
                    for part in [
                        f"positive: {'; '.join(positive_genes)}" if positive_genes else "",
                        f"negative: {'; '.join(negative_genes)}" if negative_genes else "",
                    ]
                    if part
                ),
                "top_cluster_markers": marker_list,
            }
        )
    return pd.DataFrame(rows)


def make_all_annotations_report(
    report: pd.DataFrame,
    yuntao_summary: pd.DataFrame,
    cluster_key: str,
) -> pd.DataFrame:
    columns = [
        cluster_key,
        "n_cells",
        "top_tissue",
        "top_tissue_percent",
        "top_dataset",
        "top_dataset_percent",
        "top_assay",
        "top_assay_percent",
        "cellxgene_annotation_n_cells",
        "cellxgene_pct_NK",
        "cellxgene_pct_T",
        "cellxgene_pct_B",
        "final_structured_label",
        "free_label",
        "free_label_reason",
        "needs_human_review",
        "human_review_reason",
        "n_pairwise_de_compared",
        "positive_marker_count",
        "positive_marker_list",
        "negative_marker_count",
        "negative_marker_list",
    ]
    columns = [c for c in columns if c in report.columns]
    out = report[columns].copy()
    if yuntao_summary.empty:
        return out
    rename = {
        "manual_annotation_n_cells": "yuntao_annotation_n_cells",
        "top_manual_annotation": "top_yuntao_annotation",
        "top_manual_annotation_percent": "top_yuntao_annotation_percent",
        "manual_annotation_composition": "yuntao_annotation_composition",
    }
    yuntao = yuntao_summary.rename(columns=rename)
    keep = [
        cluster_key,
        "yuntao_annotation_n_cells",
        "top_yuntao_annotation",
        "top_yuntao_annotation_percent",
        "yuntao_annotation_composition",
    ]
    keep = [c for c in keep if c in yuntao.columns]
    out = out.merge(yuntao[keep], on=cluster_key, how="left")
    insert_after = "cellxgene_pct_B"
    base_cols = [c for c in columns if c in out.columns]
    yuntao_cols = [c for c in keep if c != cluster_key and c in out.columns]
    if insert_after in base_cols:
        idx = base_cols.index(insert_after) + 1
        ordered = base_cols[:idx] + yuntao_cols + base_cols[idx:]
        return out[ordered].copy()
    return out[base_cols + yuntao_cols].copy()


def make_evidence_only_report(report: pd.DataFrame, cluster_key: str) -> pd.DataFrame:
    columns = [
        cluster_key,
        "n_cells",
        "top_tissue",
        "top_tissue_percent",
        "top_dataset",
        "top_dataset_percent",
        "top_assay",
        "top_assay_percent",
        "n_pairwise_de_compared",
        "positive_marker_count",
        "positive_marker_list",
        "negative_marker_count",
        "negative_marker_list",
    ]
    columns = [c for c in columns if c in report.columns]
    return report[columns].copy()


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
        "This report summarizes CellXGene annotation composition, Yuntao annotation composition, "
        "current agent labels, agent rationale, and cluster-specific positive/negative DE markers."
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
        "top_tissue",
        "top_tissue_percent",
        "cellxgene_pct_NK",
        "cellxgene_pct_T",
        "cellxgene_pct_B",
        "top_cellxgene_annotation",
        "top_yuntao_annotation",
        "top_yuntao_annotation_percent",
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
        if str(row.get("top_tissue", "")).strip():
            lines.append(f"- Top tissue: {row.get('top_tissue', '')} ({row.get('top_tissue_percent', '')}%)")
        lines.append(
            "- CellXGene composition: "
            f"NK {row.get('cellxgene_pct_NK', '')}%, "
            f"T {row.get('cellxgene_pct_T', '')}%, "
            f"B {row.get('cellxgene_pct_B', '')}%, "
            f"Other {row.get('cellxgene_pct_Other', '')}%"
        )
        lines.append(f"- Top CellXGene annotations: {row.get('cellxgene_annotation_composition', '')}")
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
