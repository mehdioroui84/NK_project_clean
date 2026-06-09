from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from nk_project.annotation_agent.marker_knowledge import (
    NON_NK_MARKER_GROUPS,
    PAN_NK_MARKERS,
    marker_program_hits,
)
from nk_project.annotation_agent.taxonomy_reference import load_taxonomy_entries, taxonomy_marker_hits


TAXONOMY_DE_FDR = 0.05
TAXONOMY_DE_LOGFC = 0.25
TAXONOMY_DE_MIN_PCT = 0.10
TAXONOMY_DE_TOP_N_PER_DIRECTION = 50
TAXONOMY_DE_MAX_GENES = 2 * TAXONOMY_DE_TOP_N_PER_DIRECTION


@dataclass
class EvidencePaths:
    leiden_dir: str
    marker_dir: str
    groupby: str
    pairwise_dir: str | None = None
    distance_evidence_csv: str | None = None

    @property
    def worksheet(self) -> str:
        return os.path.join(self.leiden_dir, f"full_{self.groupby}_annotation_worksheet.csv")

    @property
    def cluster_summary(self) -> str:
        return os.path.join(self.marker_dir, f"{self.groupby}_cluster_summary.csv")

    @property
    def top_markers(self) -> str:
        explicit = os.path.join(self.marker_dir, f"{self.groupby}_markers_top50_pos_top50_neg_per_cluster.csv")
        if os.path.exists(explicit):
            return explicit
        return os.path.join(self.marker_dir, f"{self.groupby}_markers_top50_per_cluster.csv")

    @property
    def all_markers(self) -> str:
        return os.path.join(self.marker_dir, f"{self.groupby}_markers_all_wilcoxon.csv")

    @property
    def curated_means(self) -> str:
        return os.path.join(self.marker_dir, f"{self.groupby}_curated_marker_cluster_means.csv")

    @property
    def curated_marker_genes(self) -> str:
        return os.path.join(self.marker_dir, f"{self.groupby}_curated_marker_genes_present.csv")


def load_cluster_evidence(paths: EvidencePaths, *, top_n: int = 50) -> dict[str, dict[str, Any]]:
    require(paths.worksheet)
    require(paths.top_markers)

    worksheet = pd.read_csv(paths.worksheet, index_col=0, low_memory=False)
    worksheet.index = worksheet.index.astype(str)
    taxonomy_entries = load_taxonomy_entries()

    if os.path.exists(paths.cluster_summary):
        cluster_summary = pd.read_csv(paths.cluster_summary, index_col=0, low_memory=False)
        cluster_summary.index = cluster_summary.index.astype(str)
    else:
        cluster_summary = pd.DataFrame(index=worksheet.index)

    markers = pd.read_csv(paths.top_markers, low_memory=False)
    if "group" not in markers or "names" not in markers:
        raise KeyError(f"{paths.top_markers} must contain at least 'group' and 'names'.")
    markers["group"] = markers["group"].astype(str)
    all_markers = None
    if os.path.exists(paths.all_markers):
        all_markers = pd.read_csv(paths.all_markers, low_memory=False)
        if "group" in all_markers and "names" in all_markers:
            all_markers["group"] = all_markers["group"].astype(str)
        else:
            all_markers = None

    curated = None
    if os.path.exists(paths.curated_means):
        curated = pd.read_csv(paths.curated_means, index_col=0, low_memory=False)
        curated.index = curated.index.astype(str)

    curated_marker_defs = None
    if os.path.exists(paths.curated_marker_genes):
        curated_marker_defs = pd.read_csv(paths.curated_marker_genes, low_memory=False)
        if not {"marker_set", "gene"}.issubset(set(curated_marker_defs.columns)):
            curated_marker_defs = None

    evidence: dict[str, dict[str, Any]] = {}
    for cluster_id in sorted(worksheet.index.astype(str), key=cluster_sort_key):
        marker_rows = select_top_marker_rows(markers, cluster_id, top_n=top_n)
        top_gene_records = marker_records(marker_rows)
        top_genes = [row["gene"] for row in top_gene_records]
        top_positive_genes = [
            row["gene"]
            for row in top_gene_records
            if float(row.get("logfoldchanges") or 0.0) > 0
        ]
        top_negative_genes = [
            row["gene"]
            for row in top_gene_records
            if float(row.get("logfoldchanges") or 0.0) < 0
        ]
        taxonomy_de_records = select_taxonomy_de_records(
            all_markers,
            cluster_id,
            fallback_records=top_gene_records,
        )
        taxonomy_positive_genes = [
            row["gene"]
            for row in taxonomy_de_records
            if float(row.get("logfoldchanges") or 0.0) > 0
        ]
        taxonomy_negative_genes = [
            row["gene"]
            for row in taxonomy_de_records
            if float(row.get("logfoldchanges") or 0.0) < 0
        ]

        row = worksheet.loc[cluster_id].to_dict()
        summary = cluster_summary.loc[cluster_id].to_dict() if cluster_id in cluster_summary.index else {}
        curated_values = curated_summary(curated, cluster_id) if curated is not None else {}
        curated_set_values = curated_marker_set_summary(curated, curated_marker_defs, cluster_id)
        pan_nk_summary = summarize_pan_nk_markers(all_markers, cluster_id)
        non_nk_summary = summarize_non_nk_markers(all_markers, cluster_id)
        lineage_sanity = lineage_sanity_check(pan_nk_summary, non_nk_summary)

        composition = strip_prior_annotation_fields(clean_mapping({**summary, **row}))

        evidence[cluster_id] = {
            "cluster_id": cluster_id,
            "groupby": paths.groupby,
            "composition": composition,
            "top_de_genes": top_gene_records,
            "top_gene_names": top_genes,
            "top_positive_gene_names": top_positive_genes,
            "top_negative_gene_names": top_negative_genes,
            "taxonomy_de_filter": {
                "source": paths.all_markers if all_markers is not None else paths.top_markers,
                "positive": f"pvals_adj<{TAXONOMY_DE_FDR}, logfoldchanges>{TAXONOMY_DE_LOGFC}, pct_in_cluster>={TAXONOMY_DE_MIN_PCT}",
                "negative": f"pvals_adj<{TAXONOMY_DE_FDR}, logfoldchanges<-{TAXONOMY_DE_LOGFC}, pct_in_reference>={TAXONOMY_DE_MIN_PCT}",
                "ranking": "top positive logfoldchanges descending and top negative logfoldchanges ascending, each then pvals_adj ascending",
                "max_genes_per_direction": TAXONOMY_DE_TOP_N_PER_DIRECTION,
                "max_total_genes": TAXONOMY_DE_MAX_GENES,
                "n_selected_positive": len(taxonomy_positive_genes),
                "n_selected_negative": len(taxonomy_negative_genes),
            },
            "taxonomy_de_genes": taxonomy_de_records,
            "taxonomy_positive_gene_names": taxonomy_positive_genes,
            "taxonomy_negative_gene_names": taxonomy_negative_genes,
            "marker_program_hits": marker_program_hits(top_positive_genes),
            "negative_marker_program_hits": marker_program_hits(top_negative_genes),
            "taxonomy_marker_hits": taxonomy_marker_hits(
                taxonomy_positive_genes,
                taxonomy_negative_genes,
                entries=taxonomy_entries,
            ),
            "pan_nk_marker_summary": pan_nk_summary,
            "non_nk_marker_summary": non_nk_summary,
            "lineage_sanity_check": lineage_sanity,
            "curated_marker_means": curated_values,
            "curated_marker_set_means": curated_set_values,
        }
        if paths.pairwise_dir:
            from nk_project.annotation_agent.pairwise import load_pairwise_evidence

            evidence[cluster_id]["pairwise_de_evidence"] = load_pairwise_evidence(
                paths.pairwise_dir,
                cluster_id,
            )
        if paths.distance_evidence_csv:
            distance_evidence = load_distance_evidence(paths.distance_evidence_csv, cluster_id)
            if distance_evidence:
                evidence[cluster_id]["distance_novelty_evidence"] = distance_evidence

    add_related_cluster_summaries(evidence)
    return evidence


def load_distance_evidence(path: str, cluster_id: str) -> dict[str, Any]:
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, low_memory=False)
    if "cluster_id" not in df.columns:
        return {}
    sub = df.loc[df["cluster_id"].astype(str) == str(cluster_id)]
    if sub.empty:
        return {}
    return clean_mapping(sub.iloc[0].to_dict())


def marker_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    cols = [
        "names",
        "scores",
        "logfoldchanges",
        "pvals_adj",
        "pct_nz_group",
        "pct_nz_reference",
        "pct_expr_group",
        "pct_expr_reference",
        "pct_expr_diff",
        "marker_direction",
    ]
    available = [col for col in cols if col in df.columns]
    for _, row in df[available].iterrows():
        item = {
            "gene": str(row.get("names")),
        }
        for col in available:
            if col == "names":
                continue
            item[col] = clean_scalar(row[col])
        records.append(item)
    return records


def select_top_marker_rows(markers: pd.DataFrame, cluster_id: str, *, top_n: int) -> pd.DataFrame:
    rows = markers.loc[markers["group"].astype(str) == str(cluster_id)].copy()
    if rows.empty:
        return rows
    if "marker_direction" not in rows.columns:
        return rows.head(top_n).copy()
    return (
        rows.groupby("marker_direction", group_keys=False, sort=False)
        .head(top_n)
        .reset_index(drop=True)
    )


def select_taxonomy_de_records(
    all_markers: pd.DataFrame | None,
    cluster_id: str,
    *,
    fallback_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if all_markers is None:
        return fallback_records[:TAXONOMY_DE_MAX_GENES]
    required = {"group", "names", "logfoldchanges", "pvals_adj"}
    if not required.issubset(set(all_markers.columns)):
        return fallback_records[:TAXONOMY_DE_MAX_GENES]
    rows = all_markers.loc[all_markers["group"].astype(str) == str(cluster_id)].copy()
    if rows.empty:
        return fallback_records[:TAXONOMY_DE_MAX_GENES]

    rows["logfoldchanges"] = pd.to_numeric(rows["logfoldchanges"], errors="coerce")
    rows["pvals_adj"] = pd.to_numeric(rows["pvals_adj"], errors="coerce")
    pct_group_col = first_existing_column(rows, ["pct_nz_group", "pct_expr_group", "pct.1"])
    pct_ref_col = first_existing_column(rows, ["pct_nz_reference", "pct_expr_reference", "pct.2"])
    if pct_group_col:
        rows[pct_group_col] = pd.to_numeric(rows[pct_group_col], errors="coerce")
    if pct_ref_col:
        rows[pct_ref_col] = pd.to_numeric(rows[pct_ref_col], errors="coerce")

    positive = (rows["pvals_adj"] < TAXONOMY_DE_FDR) & (rows["logfoldchanges"] > TAXONOMY_DE_LOGFC)
    negative = (rows["pvals_adj"] < TAXONOMY_DE_FDR) & (rows["logfoldchanges"] < -TAXONOMY_DE_LOGFC)
    if pct_group_col:
        positive &= rows[pct_group_col] >= TAXONOMY_DE_MIN_PCT
    if pct_ref_col:
        negative &= rows[pct_ref_col] >= TAXONOMY_DE_MIN_PCT

    positive_rows = rows.loc[positive].copy()
    negative_rows = rows.loc[negative].copy()
    if positive_rows.empty and negative_rows.empty:
        return fallback_records[:TAXONOMY_DE_MAX_GENES]
    positive_rows = positive_rows.sort_values(
        ["logfoldchanges", "pvals_adj"],
        ascending=[False, True],
    ).head(TAXONOMY_DE_TOP_N_PER_DIRECTION)
    negative_rows = negative_rows.sort_values(
        ["logfoldchanges", "pvals_adj"],
        ascending=[True, True],
    ).head(TAXONOMY_DE_TOP_N_PER_DIRECTION)
    selected = pd.concat([positive_rows, negative_rows], ignore_index=True)
    return marker_records(selected)


def summarize_pan_nk_markers(all_markers: pd.DataFrame | None, cluster_id: str) -> dict[str, Any]:
    rows = select_marker_gene_rows(all_markers, cluster_id, PAN_NK_MARKERS)
    if rows.empty:
        return {}
    rows = add_numeric_de_columns(rows)
    pct_group_col = first_existing_column(rows, ["pct_nz_group", "pct_expr_group", "pct.1"])
    pct_ref_col = first_existing_column(rows, ["pct_nz_reference", "pct_expr_reference", "pct.2"])
    logfc = rows["logfoldchanges"].dropna()
    genes_tested = sorted(rows["names"].dropna().astype(str).unique().tolist())
    depleted = rows.loc[rows["logfoldchanges"] <= -1.0].copy()
    depleted = depleted.sort_values("logfoldchanges").head(12)
    summary: dict[str, Any] = {
        "genes_tested": genes_tested,
        "n_pan_nk_genes_tested": len(genes_tested),
        "n_pan_nk_logfc_le_minus_0_5": int((rows["logfoldchanges"] <= -0.5).sum()),
        "n_pan_nk_logfc_le_minus_1": int((rows["logfoldchanges"] <= -1.0).sum()),
        "median_pan_nk_logfc": clean_scalar(logfc.median()) if not logfc.empty else None,
        "strongly_depleted_genes": depleted["names"].astype(str).tolist(),
    }
    if pct_group_col:
        summary["median_pan_nk_pct_nz_cluster"] = clean_scalar(rows[pct_group_col].dropna().median())
    if pct_ref_col:
        summary["median_pan_nk_pct_nz_reference"] = clean_scalar(rows[pct_ref_col].dropna().median())
    return summary


def summarize_non_nk_markers(all_markers: pd.DataFrame | None, cluster_id: str) -> dict[str, Any]:
    marker_genes = [gene for genes in NON_NK_MARKER_GROUPS.values() for gene in genes]
    rows = select_marker_gene_rows(all_markers, cluster_id, marker_genes)
    if rows.empty:
        return {}
    rows = add_numeric_de_columns(rows)
    summaries = []
    for group_name, genes in NON_NK_MARKER_GROUPS.items():
        group_rows = select_marker_gene_rows(rows, cluster_id, genes)
        if group_rows.empty:
            continue
        positive = group_rows.loc[
            (group_rows["pvals_adj"] < 0.05)
            & (group_rows["logfoldchanges"] >= 0.5)
        ].copy()
        strong = group_rows.loc[
            (group_rows["pvals_adj"] < 0.05)
            & (group_rows["logfoldchanges"] >= 1.0)
        ].copy()
        positive = positive.sort_values(["logfoldchanges", "pvals_adj"], ascending=[False, True])
        summaries.append(
            {
                "group": group_name,
                "genes_tested": sorted(group_rows["names"].dropna().astype(str).unique().tolist()),
                "n_positive_markers_logfc_ge_0_5": int(len(positive)),
                "n_positive_markers_logfc_ge_1": int(len(strong)),
                "matched_positive_markers": positive["names"].astype(str).head(12).tolist(),
                "median_positive_logfc": clean_scalar(positive["logfoldchanges"].median()) if not positive.empty else None,
            }
        )
    if not summaries:
        return {}
    summaries.sort(
        key=lambda item: (
            item["n_positive_markers_logfc_ge_1"],
            item["n_positive_markers_logfc_ge_0_5"],
            item["median_positive_logfc"] or -999,
        ),
        reverse=True,
    )
    strongest = summaries[0]
    if strongest["n_positive_markers_logfc_ge_0_5"] == 0:
        return {
            "strongest_non_nk_group": None,
            "matched_positive_non_nk_markers": [],
            "n_positive_non_nk_markers_logfc_ge_0_5": 0,
            "n_positive_non_nk_markers_logfc_ge_1": 0,
            "median_positive_non_nk_logfc": None,
            "group_summaries": summaries,
        }
    return {
        "strongest_non_nk_group": strongest["group"],
        "matched_positive_non_nk_markers": strongest["matched_positive_markers"],
        "n_positive_non_nk_markers_logfc_ge_0_5": strongest["n_positive_markers_logfc_ge_0_5"],
        "n_positive_non_nk_markers_logfc_ge_1": strongest["n_positive_markers_logfc_ge_1"],
        "median_positive_non_nk_logfc": strongest["median_positive_logfc"],
        "group_summaries": summaries,
    }


def lineage_sanity_check(
    pan_nk_summary: dict[str, Any],
    non_nk_summary: dict[str, Any],
) -> dict[str, Any]:
    strongest_group = non_nk_summary.get("strongest_non_nk_group")
    matched_non_nk = non_nk_summary.get("matched_positive_non_nk_markers", []) or []
    n_non_nk_strong = int(non_nk_summary.get("n_positive_non_nk_markers_logfc_ge_1") or 0)
    n_non_nk_positive = int(non_nk_summary.get("n_positive_non_nk_markers_logfc_ge_0_5") or 0)
    depleted_pan_nk = pan_nk_summary.get("strongly_depleted_genes", []) or []
    n_pan_nk_depleted = int(pan_nk_summary.get("n_pan_nk_logfc_le_minus_1") or 0)
    pan_nk_pct_cluster = pan_nk_summary.get("median_pan_nk_pct_nz_cluster")
    pan_nk_logfc = pan_nk_summary.get("median_pan_nk_logfc")

    if not strongest_group or n_non_nk_positive == 0:
        return {
            "summary": "No strong T-lineage marker signal detected.",
            "interpretation": "No T-lineage sanity concern from the curated T marker panel.",
        }

    pan_nk_broad = (
        (isinstance(pan_nk_pct_cluster, (int, float)) and pan_nk_pct_cluster >= 0.65)
        or (isinstance(pan_nk_logfc, (int, float)) and pan_nk_logfc >= -0.5)
        or n_pan_nk_depleted <= 2
    )
    pan_nk_weak = (
        n_pan_nk_depleted >= 4
        or (isinstance(pan_nk_pct_cluster, (int, float)) and pan_nk_pct_cluster < 0.55)
        or (isinstance(pan_nk_logfc, (int, float)) and pan_nk_logfc <= -1.0)
    )

    if strongest_group == "T" and n_non_nk_strong >= 4 and pan_nk_weak:
        return {
            "summary": (
                "Strong T-lineage evidence: "
                f"{', '.join(matched_non_nk[:8])} positive. "
                f"Pan-NK markers are depleted: {', '.join(depleted_pan_nk[:8])}."
            ),
            "interpretation": "Evidence favors Non-NK/T-like over NK subtype.",
        }

    if strongest_group == "T" and n_non_nk_positive > 0 and pan_nk_broad:
        return {
            "summary": (
                "Some T-lineage markers are positive: "
                f"{', '.join(matched_non_nk[:8])}. "
                "Pan-NK markers remain broadly detected."
            ),
            "interpretation": "Evidence does not support a clear Non-NK call from T markers alone.",
        }

    if n_non_nk_strong >= 3 and pan_nk_weak:
        return {
            "summary": (
                f"Strong {strongest_group} marker signal: "
                f"{', '.join(matched_non_nk[:8])} positive. "
                f"Pan-NK markers are depleted: {', '.join(depleted_pan_nk[:8])}."
            ),
            "interpretation": f"Evidence favors Non-NK/{strongest_group}-like over NK subtype.",
        }

    return {
        "summary": (
            f"{strongest_group} marker signal detected: "
            f"{', '.join(matched_non_nk[:8])}."
        ),
        "interpretation": "Lineage evidence is mixed; interpret with pan-NK and taxonomy evidence.",
    }


def select_marker_gene_rows(all_markers: pd.DataFrame | None, cluster_id: str, genes: list[str]) -> pd.DataFrame:
    if all_markers is None or "group" not in all_markers or "names" not in all_markers:
        return pd.DataFrame()
    wanted = {gene.upper() for gene in genes}
    rows = all_markers.loc[
        (all_markers["group"].astype(str) == str(cluster_id))
        & (all_markers["names"].astype(str).str.upper().isin(wanted))
    ].copy()
    return rows


def add_numeric_de_columns(rows: pd.DataFrame) -> pd.DataFrame:
    rows = rows.copy()
    for col in ["logfoldchanges", "pvals_adj", "pct_nz_group", "pct_nz_reference", "pct_expr_group", "pct_expr_reference", "pct.1", "pct.2"]:
        if col in rows.columns:
            rows[col] = pd.to_numeric(rows[col], errors="coerce")
    return rows


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def curated_summary(curated: pd.DataFrame, cluster_id: str) -> dict[str, float]:
    if cluster_id not in curated.index:
        return {}
    row = curated.loc[cluster_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    numeric = pd.to_numeric(row, errors="coerce").dropna()
    if numeric.empty:
        return {}
    top = numeric.sort_values(ascending=False).head(30)
    return {str(gene): float(value) for gene, value in top.items()}


def curated_marker_set_summary(
    curated: pd.DataFrame | None,
    marker_defs: pd.DataFrame | None,
    cluster_id: str,
) -> dict[str, float]:
    if curated is None or marker_defs is None or cluster_id not in curated.index:
        return {}
    row = curated.loc[cluster_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    values = pd.to_numeric(row, errors="coerce")
    scores = {}
    for marker_set, group in marker_defs.groupby("marker_set", sort=False):
        genes = [str(gene) for gene in group["gene"].dropna().astype(str) if str(gene) in values.index]
        if not genes:
            continue
        score = values.loc[genes].dropna().mean()
        if pd.notna(score):
            scores[str(marker_set)] = float(score)
    return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:30])


def strip_prior_annotation_fields(composition: dict[str, Any]) -> dict[str, Any]:
    prior_columns = [
        "top_" + "NK_State",
        "top_" + "NK_State" + "_frac",
        "man" + "ual_" + "annotation" + "_composition",
        "draft_" + "refined_label",
        "draft_" + "refined_label_raw",
        "worksheet_" + "review_note",
        "review_" + "notes",
    ]
    for key in prior_columns:
        composition.pop(key, None)
    return composition


def add_related_cluster_summaries(evidence: dict[str, dict[str, Any]]) -> None:
    compact = {
        cluster_id: {
            "cluster_id": cluster_id,
            "n_cells": data["composition"].get("n_cells"),
            "top_tissue": data["composition"].get("top_tissue"),
            "top_tissue_frac": data["composition"].get("top_tissue_frac"),
            "top_genes": data["top_gene_names"][:12],
            "marker_program_hits": data["marker_program_hits"],
        }
        for cluster_id, data in evidence.items()
    }

    for cluster_id, data in evidence.items():
        related = []
        comp = data["composition"]
        genes = set(map(str.upper, data["top_gene_names"][:25]))
        for other_id, other in evidence.items():
            if other_id == cluster_id:
                continue
            other_comp = other["composition"]
            other_genes = set(map(str.upper, other["top_gene_names"][:25]))
            overlap = len(genes.intersection(other_genes))
            same_tissue = other_comp.get("top_tissue") == comp.get("top_tissue")
            if overlap >= 3:
                score = overlap / 10 + int(same_tissue) / 4
                related.append((score, other_id))
        related_ids = [other_id for _, other_id in sorted(related, reverse=True)[:6]]
        data["related_clusters"] = [compact[other_id] for other_id in related_ids]


def compact_evidence_for_json(evidence: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        cluster_id: compact_cluster_evidence(data)
        for cluster_id, data in sorted(evidence.items(), key=lambda item: cluster_sort_key(item[0]))
    }


def compact_cluster_evidence(data: dict[str, Any]) -> dict[str, Any]:
    comp = data.get("composition", {})
    taxonomy_filter = data.get("taxonomy_de_filter", {})
    pairwise = data.get("pairwise_de_evidence", [])
    compact_related = []
    for related in data.get("related_clusters", [])[:6]:
        compact_related.append(
            {
                "cluster_id": related.get("cluster_id"),
                "n_cells": related.get("n_cells"),
                "top_tissue": related.get("top_tissue"),
                "top_tissue_frac": related.get("top_tissue_frac"),
                "top_genes": related.get("top_genes", [])[:8],
            }
        )

    return {
        "cluster_id": data.get("cluster_id"),
        "groupby": data.get("groupby"),
        "composition": {
            "n_cells": comp.get("n_cells"),
            "top_tissue": comp.get("top_tissue"),
            "top_tissue_frac": comp.get("top_tissue_frac"),
            "top_dataset_id": comp.get("top_dataset_id"),
            "top_dataset_id_frac": comp.get("top_dataset_id_frac"),
            "top_assay_clean": comp.get("top_assay_clean"),
            "top_assay_clean_frac": comp.get("top_assay_clean_frac"),
        },
        "top_de_genes": compact_gene_records(data.get("top_de_genes", []), max_items=25),
        "top_positive_gene_names": data.get("top_positive_gene_names", [])[:25],
        "top_negative_gene_names": data.get("top_negative_gene_names", [])[:25],
        "taxonomy_de_filter": {
            "positive": taxonomy_filter.get("positive"),
            "negative": taxonomy_filter.get("negative"),
            "ranking": taxonomy_filter.get("ranking"),
            "max_genes_per_direction": taxonomy_filter.get("max_genes_per_direction"),
            "max_total_genes": taxonomy_filter.get("max_total_genes"),
            "n_selected_positive": taxonomy_filter.get("n_selected_positive"),
            "n_selected_negative": taxonomy_filter.get("n_selected_negative"),
        },
        "taxonomy_positive_gene_names": data.get("taxonomy_positive_gene_names", [])[:50],
        "taxonomy_negative_gene_names": data.get("taxonomy_negative_gene_names", [])[:50],
        "taxonomy_marker_hits": {
            "top_matches": data.get("taxonomy_marker_hits", {}).get("top_matches", [])[:8],
        },
        "pan_nk_marker_summary": data.get("pan_nk_marker_summary", {}),
        "non_nk_marker_summary": data.get("non_nk_marker_summary", {}),
        "lineage_sanity_check": data.get("lineage_sanity_check", {}),
        "marker_program_hits": data.get("marker_program_hits", {}),
        "negative_marker_program_hits": data.get("negative_marker_program_hits", {}),
        "curated_marker_means_top": top_numeric_items(data.get("curated_marker_means", {}), max_items=20),
        "curated_marker_set_means_top": top_numeric_items(data.get("curated_marker_set_means", {}), max_items=20),
        "related_clusters": compact_related,
        "same_label_split_candidates": data.get("same_label_split_candidates", []),
        "pairwise_de_summary": {
            "n_comparisons": len(pairwise),
            "comparisons": [
                {
                    "comparison": item.get("comparison"),
                    "other_cluster_id": item.get("other_cluster_id"),
                    "top_genes_for_this_cluster": compact_gene_records(
                        item.get("top_genes_for_this_cluster", []),
                        max_items=10,
                    ),
                }
                for item in pairwise[:10]
            ],
        },
        "distance_novelty_evidence": data.get("distance_novelty_evidence", {}),
    }


def compact_gene_records(records: list[dict[str, Any]], *, max_items: int) -> list[dict[str, Any]]:
    keep = ["gene", "logfoldchanges", "pvals_adj", "pct_nz_group", "pct_nz_reference", "pct_expr_diff"]
    compact = []
    for record in records[:max_items]:
        compact.append({key: record.get(key) for key in keep if key in record})
    return compact


def top_numeric_items(values: dict[str, Any], *, max_items: int) -> dict[str, float]:
    numeric = []
    for key, value in (values or {}).items():
        try:
            numeric.append((str(key), float(value)))
        except (TypeError, ValueError):
            continue
    numeric.sort(key=lambda item: item[1], reverse=True)
    return {key: value for key, value in numeric[:max_items]}


def save_compact_evidence_json(evidence: dict[str, dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(compact_evidence_for_json(evidence), handle, indent=2)


def save_evidence_json(evidence: dict[str, dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2)


def clean_mapping(mapping: dict[str, Any]) -> dict[str, Any]:
    return {str(key): clean_scalar(value) for key, value in mapping.items()}


def clean_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, (int, str, bool)) or value is None:
        return value
    return str(value)


def require(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def cluster_sort_key(value: str) -> tuple[int, str]:
    text = str(value)
    return (0, f"{int(text):08d}") if text.isdigit() else (1, text)
