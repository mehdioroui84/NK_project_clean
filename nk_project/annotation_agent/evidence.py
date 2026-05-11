from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from configs import default_config as cfg
from nk_project.annotation_agent.marker_knowledge import KNOWN_REFINED_LABELS, marker_program_hits


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
        return os.path.join(self.marker_dir, f"{self.groupby}_markers_top50_per_cluster.csv")

    @property
    def all_markers(self) -> str:
        return os.path.join(self.marker_dir, f"{self.groupby}_markers_all_wilcoxon.csv")

    @property
    def curated_means(self) -> str:
        return os.path.join(self.marker_dir, f"{self.groupby}_curated_marker_cluster_means.csv")


def load_cluster_evidence(paths: EvidencePaths, *, top_n: int = 50) -> dict[str, dict[str, Any]]:
    require(paths.worksheet)
    require(paths.top_markers)

    worksheet = pd.read_csv(paths.worksheet, index_col=0, low_memory=False)
    worksheet.index = worksheet.index.astype(str)
    manual_compositions = load_manual_annotation_compositions(paths)

    if os.path.exists(paths.cluster_summary):
        cluster_summary = pd.read_csv(paths.cluster_summary, index_col=0, low_memory=False)
        cluster_summary.index = cluster_summary.index.astype(str)
    else:
        cluster_summary = pd.DataFrame(index=worksheet.index)

    markers = pd.read_csv(paths.top_markers, low_memory=False)
    if "group" not in markers or "names" not in markers:
        raise KeyError(f"{paths.top_markers} must contain at least 'group' and 'names'.")
    markers["group"] = markers["group"].astype(str)

    curated = None
    if os.path.exists(paths.curated_means):
        curated = pd.read_csv(paths.curated_means, index_col=0, low_memory=False)
        curated.index = curated.index.astype(str)

    evidence: dict[str, dict[str, Any]] = {}
    for cluster_id in sorted(worksheet.index.astype(str), key=cluster_sort_key):
        marker_rows = (
            markers.loc[markers["group"].astype(str) == cluster_id]
            .head(top_n)
            .copy()
        )
        top_gene_records = marker_records(marker_rows)
        top_genes = [row["gene"] for row in top_gene_records]

        row = worksheet.loc[cluster_id].to_dict()
        summary = cluster_summary.loc[cluster_id].to_dict() if cluster_id in cluster_summary.index else {}
        curated_values = curated_summary(curated, cluster_id) if curated is not None else {}

        composition = clean_mapping({**summary, **row})
        composition = normalize_worksheet_draft_label(composition)
        if cluster_id in manual_compositions:
            composition["manual_annotation_composition"] = manual_compositions[cluster_id]

        evidence[cluster_id] = {
            "cluster_id": cluster_id,
            "groupby": paths.groupby,
            "composition": composition,
            "top_de_genes": top_gene_records,
            "top_gene_names": top_genes,
            "marker_program_hits": marker_program_hits(top_genes),
            "curated_marker_means": curated_values,
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


def load_manual_annotation_compositions(paths: EvidencePaths, *, top_n: int = 8) -> dict[str, list[dict[str, Any]]]:
    h5ad_path = os.path.join(paths.leiden_dir, "full_scvi_leiden.h5ad")
    if not os.path.exists(h5ad_path):
        return {}

    try:
        import anndata as ad
    except ImportError:
        return {}

    adata = ad.read_h5ad(h5ad_path, backed="r")
    try:
        if paths.groupby not in adata.obs or cfg.LABEL_KEY not in adata.obs:
            return {}
        obs = adata.obs[[paths.groupby, cfg.LABEL_KEY]].copy()
    finally:
        adata.file.close()

    obs[paths.groupby] = obs[paths.groupby].astype(str)
    obs[cfg.LABEL_KEY] = obs[cfg.LABEL_KEY].astype(str)
    tab = pd.crosstab(obs[paths.groupby], obs[cfg.LABEL_KEY])
    totals = tab.sum(axis=1)
    compositions: dict[str, list[dict[str, Any]]] = {}
    for cluster_id, row in tab.iterrows():
        counts = row[row > 0].sort_values(ascending=False).head(top_n)
        entries = []
        total = float(totals.loc[cluster_id])
        for label, count in counts.items():
            entries.append(
                {
                    "label": str(label),
                    "n_cells": int(count),
                    "fraction": float(count / total) if total else 0.0,
                }
            )
        compositions[str(cluster_id)] = entries
    return compositions


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


def normalize_worksheet_draft_label(composition: dict[str, Any]) -> dict[str, Any]:
    """Separate worksheet draft labels from embedded script review hints.

    Early annotation worksheets sometimes stored values such as
    "Transitional Cytotoxic Tissue-Resident review" in the draft-label column.
    The biological draft label and the script-generated review hint should be separate
    evidence fields so neither the report nor the LLM treats the full phrase as
    a candidate cell-state name.
    """
    raw_value = composition.get("draft_refined_label")
    if raw_value is None:
        composition["draft_refined_label_raw"] = None
        composition["worksheet_review_note"] = clean_review_note(composition.get("review_notes"))
        return composition

    raw = str(raw_value).strip()
    label, embedded_note = split_draft_label_and_note(raw)
    explicit_note = clean_review_note(composition.get("review_notes"))
    composition["draft_refined_label_raw"] = raw
    composition["draft_refined_label"] = label
    composition["worksheet_review_note"] = combine_notes(embedded_note, explicit_note)
    return composition


def split_draft_label_and_note(raw: str) -> tuple[str, str]:
    text = raw.strip()
    if not text:
        return "", ""

    lower = text.lower()
    if " review:" in lower:
        idx = lower.index(" review:")
        label = text[:idx].strip()
        note = text[idx + len(" review:") :].strip()
        return label, note

    if lower.endswith(" review"):
        without_review = text[: -len(" review")].strip()
        label, note = split_known_label_prefix(without_review)
        note = combine_notes(note, "review")
        return label, note

    label, note = split_known_label_prefix(text)
    return label, note


def split_known_label_prefix(text: str) -> tuple[str, str]:
    text_lower = text.lower()
    for label in sorted(KNOWN_REFINED_LABELS, key=len, reverse=True):
        label_lower = label.lower()
        if text_lower == label_lower:
            return label, ""
        if text_lower.startswith(label_lower + " "):
            return label, text[len(label) :].strip()
    return text, ""


def clean_review_note(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return ""
    return text


def combine_notes(*notes: str) -> str:
    seen = set()
    combined = []
    for note in notes:
        text = clean_review_note(note)
        if not text:
            continue
        for part in [item.strip() for item in text.split(";")]:
            if not part or part in seen:
                continue
            seen.add(part)
            combined.append(part)
    return "; ".join(combined)


def add_related_cluster_summaries(evidence: dict[str, dict[str, Any]]) -> None:
    compact = {
        cluster_id: {
            "cluster_id": cluster_id,
            "n_cells": data["composition"].get("n_cells"),
            "top_NK_State": data["composition"].get(f"top_{cfg.LABEL_KEY}"),
            "top_NK_State_frac": data["composition"].get(f"top_{cfg.LABEL_KEY}_frac"),
            "draft_refined_label": data["composition"].get("draft_refined_label"),
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
        top_label = comp.get(f"top_{cfg.LABEL_KEY}")
        draft = comp.get("draft_refined_label")
        genes = set(map(str.upper, data["top_gene_names"][:25]))
        for other_id, other in evidence.items():
            if other_id == cluster_id:
                continue
            other_comp = other["composition"]
            other_genes = set(map(str.upper, other["top_gene_names"][:25]))
            overlap = len(genes.intersection(other_genes))
            same_label = other_comp.get(f"top_{cfg.LABEL_KEY}") == top_label
            same_draft = other_comp.get("draft_refined_label") == draft
            if same_label or same_draft or overlap >= 3:
                score = int(same_label) + int(same_draft) + overlap / 10
                related.append((score, other_id))
        related_ids = [other_id for _, other_id in sorted(related, reverse=True)[:6]]
        data["related_clusters"] = [compact[other_id] for other_id in related_ids]


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
