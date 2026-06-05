#!/usr/bin/env python
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


GROUPBY = "leiden_0_4"
N_TOP_MARKERS = 12
WORKSHEET_RULE_VERSION = "v6_locked_validation_labels"

# Locked validation-set labels for the current refined annotation pass.
CURATED_REFINED_LABELS = {
    "0": "Mature Cytotoxic",
    "1": "Transitional Cytotoxic",
    "2": "Mature Cytotoxic TCF7+",
    "3": "T",
    "4": "Unknown_Kidney",
    "5": "Mature Cytotoxic Engineered",
    "6": "Mature Cytotoxic",
    "7": "Proliferative",
    "8": "Lung Cytotoxic NK",
    "9": "Cytokine-Stimulated Effector",
    "10": "Transitional Cytotoxic Tissue-Resident",
    "11": "Transitional Cytotoxic Tissue-Resident",
    "12": "Cytokine-Stimulated CCR7+",
    "13": "Lung GZMK+ XCL1+ NK",
    "14": "Unconventional",
    "15": "B",
    "16": "Unknown_BM_1 Erythroid-like",
    "17": "Unknown_Lung_4",
    "18": "Regulatory",
    "19": "Unknown_Lung_1",
    "20": "Myeloid-like",
}

CURATED_LABEL_NOTES = {
    "2": "Mature cytotoxic-majority cluster with TCF7/IL7R/SELL/GZMK-type signal; keep separate from main mature cytotoxic clusters for review.",
    "5": "CB07/Flex-enriched mature cytotoxic-majority cluster; engineered NK context may be real biology.",
    "8": "Direct DE versus cluster 13 supports a lung-enriched cytotoxic NK program with FGFBP2/FCGR3A/GZMB/PRF1/NKG7.",
    "9": "Direct DE versus cluster 12 supports cytotoxic/chemokine-high cytokine-stimulated state.",
    "10": "Decidua/tissue-enriched transitional cytotoxic cluster with tissue-regulatory marker signal.",
    "11": "Small decidua/tissue-enriched transitional cytotoxic cluster with tissue-regulatory marker signal.",
    "12": "Direct DE versus cluster 9 supports CCR7/SELL/IRF4/HSPB1 activation/trafficking-like cytokine-stimulated state; proliferation should be treated as a possible modifier.",
    "13": "Direct DE versus cluster 8 supports a lung-enriched GZMK/XCL1/DUSP/RGS1 tissue-activated program.",
    "14": "Direct DE versus cluster 1 supports keeping this distinct from Transitional Cytotoxic.",
    "16": "Unknown_BM_1 label preserved with erythroid-like modifier because HBB/HBA/HBD/AHSP/CA1 are strong.",
    "18": "Direct DE versus cluster 10 supports keeping Regulatory distinct from Transitional Cytotoxic Tissue-Resident.",
    "17": "Unknown_Lung_4 label preserved; epithelial/lung marker signal should be reviewed before final annotation.",
    "20": "Manual label is Proliferative, but top marker program is myeloid-like; review carefully before final annotation.",
}

CURATED_BROAD_COMPARTMENTS = {
    "3": "T_cell",
    "15": "B_cell",
    "20": "myeloid_like",
}

MARKER_SET_PRIORITIES = [
    "B_cell",
    "T_cell",
    "myeloid",
    "epithelial_lung",
    "erythroid",
    "proliferation",
    "NK_regulatory_tissue",
    "interferon_cytokine",
    "NK_cytotoxic",
    "stress_mito",
]

NON_NK_REFERENCE_CALLS = {
    "B_cell": ("B_cell", "B"),
    "T_cell": ("T_cell", "T"),
}

NON_NK_REVIEW_CALLS = {
    "myeloid": ("myeloid_like", "myeloid_like_review"),
    "epithelial_lung": ("epithelial_like", "epithelial_lung_like_review"),
    "erythroid": ("erythroid_like", "erythroid_like_review"),
}


def main():
    print(f"[RULES] cluster interpretation worksheet rules: {WORKSHEET_RULE_VERSION}")

    marker_dir = os.path.join(cfg.BASE_OUTDIR, "markers", "validation", GROUPBY)
    leiden_dir = os.path.join(cfg.BASE_OUTDIR, "leiden_validation")
    ensure_dirs(marker_dir)

    cluster_summary = read_cluster_summary(marker_dir, leiden_dir)
    top_markers = read_top_markers(marker_dir)
    marker_scores = read_marker_scores(marker_dir)

    worksheet = cluster_summary.copy()
    worksheet.index = worksheet.index.astype(str)
    worksheet.index.name = "cluster"

    worksheet = worksheet.join(top_marker_strings(top_markers), how="left")
    worksheet = worksheet.join(marker_scores, how="left")

    calls = []
    for cluster, row in worksheet.iterrows():
        calls.append(call_cluster(row, marker_scores.columns.tolist()))

    calls_df = pd.DataFrame(calls, index=worksheet.index)
    worksheet = worksheet.join(calls_df)
    apply_curated_draft_labels(worksheet)

    ordered_cols = [
        "n_cells",
        f"top_{cfg.LABEL_KEY}",
        f"top_{cfg.LABEL_KEY}_frac",
        "top_tissue",
        "top_tissue_frac",
        f"top_{cfg.DATASET_KEY}",
        f"top_{cfg.DATASET_KEY}_frac",
        f"top_{cfg.ASSAY_CLEAN_KEY}",
        f"top_{cfg.ASSAY_CLEAN_KEY}_frac",
        "top_positive_markers",
        "top_marker_set",
        "top_marker_set_score",
        "broad_compartment_draft",
        "refined_NK_state_draft",
        "auto_refined_NK_state_draft",
        "curated_label_note",
        "review_priority",
        "review_warnings",
        "manual_final_label",
        "manual_notes",
    ]
    score_cols = [c for c in marker_scores.columns if c in worksheet.columns]
    remaining = [c for c in worksheet.columns if c not in ordered_cols + score_cols]
    worksheet = worksheet[
        [c for c in ordered_cols if c in worksheet.columns] + score_cols + remaining
    ]

    out_csv = os.path.join(marker_dir, f"{GROUPBY}_interpretation_worksheet.csv")
    worksheet.to_csv(out_csv)
    print(f"[SAVE] {out_csv}")

    out_xlsx = os.path.join(marker_dir, f"{GROUPBY}_interpretation_worksheet.xlsx")
    try:
        with pd.ExcelWriter(out_xlsx) as writer:
            worksheet.to_excel(writer, sheet_name="worksheet")
            write_readme(writer)
        print(f"[SAVE] {out_xlsx}")
    except Exception as exc:
        print(f"[WARN] Could not write Excel file: {exc}")

    print("\n[PREVIEW]")
    preview_cols = [
        "n_cells",
        f"top_{cfg.LABEL_KEY}",
        f"top_{cfg.LABEL_KEY}_frac",
        "top_marker_set",
        "broad_compartment_draft",
        "refined_NK_state_draft",
        "review_priority",
        "review_warnings",
    ]
    print(worksheet[[c for c in preview_cols if c in worksheet.columns]].to_string())


def apply_curated_draft_labels(worksheet):
    worksheet["auto_refined_NK_state_draft"] = worksheet["refined_NK_state_draft"]
    worksheet["curated_label_note"] = ""

    for cluster, label in CURATED_REFINED_LABELS.items():
        if cluster not in worksheet.index:
            continue
        worksheet.loc[cluster, "refined_NK_state_draft"] = label
        worksheet.loc[cluster, "curated_label_note"] = CURATED_LABEL_NOTES.get(cluster, "")
        if cluster in CURATED_BROAD_COMPARTMENTS:
            worksheet.loc[cluster, "broad_compartment_draft"] = CURATED_BROAD_COMPARTMENTS[cluster]


def read_cluster_summary(marker_dir, leiden_dir):
    paths = [
        os.path.join(marker_dir, f"{GROUPBY}_cluster_summary.csv"),
        os.path.join(leiden_dir, f"{GROUPBY}_final_cluster_summary.csv"),
    ]
    for path in paths:
        if os.path.exists(path):
            print(f"[LOAD] {path}")
            return pd.read_csv(path, index_col=0)
    raise FileNotFoundError(
        "Could not find a cluster summary. Run marker analysis first, or create "
        f"{paths[0]}"
    )


def read_top_markers(marker_dir):
    path = os.path.join(marker_dir, f"{GROUPBY}_markers_top50_per_cluster.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing marker table: {path}")
    print(f"[LOAD] {path}")
    df = pd.read_csv(path)
    df["group"] = df["group"].astype(str)
    if "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"] > 0].copy()
    return df


def read_marker_scores(marker_dir):
    means_path = os.path.join(marker_dir, f"{GROUPBY}_curated_marker_cluster_means.csv")
    genes_path = os.path.join(marker_dir, f"{GROUPBY}_curated_marker_genes_present.csv")
    if not os.path.exists(means_path) or not os.path.exists(genes_path):
        raise FileNotFoundError(
            "Missing curated marker score inputs. Run "
            "scripts/plot_curated_markers_validation.py first."
        )

    print(f"[LOAD] {means_path}")
    means = pd.read_csv(means_path, index_col=0)
    means.index = means.index.astype(str)

    print(f"[LOAD] {genes_path}")
    genes = pd.read_csv(genes_path)

    scores = pd.DataFrame(index=means.index)
    for marker_set in MARKER_SET_PRIORITIES:
        set_genes = genes.loc[genes["marker_set"] == marker_set, "gene"].astype(str)
        present = [g for g in set_genes if g in means.columns]
        if present:
            scores[f"score_{marker_set}"] = means[present].mean(axis=1)

    return minmax_by_column(scores)


def minmax_by_column(df):
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        vals = pd.to_numeric(df[col], errors="coerce")
        vmin = vals.min(skipna=True)
        vmax = vals.max(skipna=True)
        if pd.isna(vmin) or pd.isna(vmax) or vmax == vmin:
            out[col] = 0.0
        else:
            out[col] = (vals - vmin) / (vmax - vmin)
    return out


def top_marker_strings(top_markers):
    rows = {}
    for group, df in top_markers.groupby("group"):
        df = df.copy()
        sort_cols = [c for c in ["pvals_adj", "scores"] if c in df.columns]
        ascending = [True, False][: len(sort_cols)]
        if sort_cols:
            df = df.sort_values(sort_cols, ascending=ascending)
        genes = df["names"].astype(str).head(N_TOP_MARKERS).tolist()
        rows[str(group)] = {"top_positive_markers": ", ".join(genes)}
    return pd.DataFrame.from_dict(rows, orient="index")


def call_cluster(row, score_cols):
    score_values = {
        col.replace("score_", ""): safe_float(row.get(col, np.nan))
        for col in score_cols
        if col.startswith("score_")
    }
    top_marker_set = max(score_values, key=score_values.get) if score_values else "NA"
    top_score = score_values.get(top_marker_set, np.nan)

    warnings = []
    dataset_frac = safe_float(row.get(f"top_{cfg.DATASET_KEY}_frac", np.nan))
    assay_frac = safe_float(row.get(f"top_{cfg.ASSAY_CLEAN_KEY}_frac", np.nan))
    tissue_frac = safe_float(row.get("top_tissue_frac", np.nan))

    if dataset_frac >= 0.85:
        warnings.append("high_dataset_specificity")
    if assay_frac >= 0.85:
        warnings.append("high_assay_specificity")
    if tissue_frac >= 0.95:
        warnings.append("high_tissue_specificity")
    if score_values.get("stress_mito", 0.0) >= 0.75:
        warnings.append("stress_mito_high")

    top_label = str(row.get(f"top_{cfg.LABEL_KEY}", ""))
    top_label_frac = safe_float(row.get(f"top_{cfg.LABEL_KEY}_frac", np.nan))

    broad = "NK"
    refined = infer_nk_state(row, score_values)

    # B/T cells are intentional immune reference compartments. Prefer the
    # existing high-level annotation when it clearly supports that compartment.
    if top_label == "B" and top_label_frac >= 0.50:
        broad, refined = NON_NK_REFERENCE_CALLS["B_cell"]
    elif top_label == "T" and top_label_frac >= 0.50:
        broad, refined = NON_NK_REFERENCE_CALLS["T_cell"]
    elif should_call_reference(top_marker_set, top_score, score_values):
        broad, refined = NON_NK_REFERENCE_CALLS[top_marker_set]
        warnings.append(f"{top_marker_set}_marker_only_reference_call")
    elif should_call_non_nk_review(top_marker_set, top_score, score_values, top_label, top_label_frac):
        broad, refined = NON_NK_REVIEW_CALLS[top_marker_set]
        warnings.append("non_NK_like_review")

    if broad == "NK" and score_values.get("T_cell", 0.0) >= 0.70:
        warnings.append("T_cell_marker_signal")
    if broad == "NK" and score_values.get("B_cell", 0.0) >= 0.70:
        warnings.append("B_cell_marker_signal")
    if broad == "NK" and score_values.get("myeloid", 0.0) >= 0.70:
        warnings.append("myeloid_marker_signal")
    if broad == "NK" and score_values.get("epithelial_lung", 0.0) >= 0.70:
        warnings.append("epithelial_lung_marker_signal")
    if broad == "NK" and score_values.get("erythroid", 0.0) >= 0.70:
        warnings.append("erythroid_marker_signal")

    review_priority = "medium"
    if broad.endswith("_like") or "non_NK_like_review" in warnings:
        review_priority = "high"
    elif not refined or refined == "unresolved_NK":
        review_priority = "high"
    elif warnings:
        review_priority = "medium"
    else:
        review_priority = "low"

    return {
        "top_marker_set": top_marker_set,
        "top_marker_set_score": round(float(top_score), 3)
        if not pd.isna(top_score)
        else np.nan,
        "broad_compartment_draft": broad,
        "refined_NK_state_draft": refined,
        "review_priority": review_priority,
        "review_warnings": ";".join(warnings) if warnings else "",
        "manual_final_label": "",
        "manual_notes": "",
    }


def should_call_reference(top_marker_set, top_score, scores):
    # For this project, B and T cells are intentional reference compartments,
    # but marker-only B/T calls are too easy to over-trigger in cytotoxic NK
    # neighborhoods. Prefer the existing annotation for B/T, and report
    # marker-only B/T signal as a warning instead of changing compartment.
    return False


def should_call_non_nk_review(top_marker_set, top_score, scores, top_label, top_label_frac):
    if top_marker_set not in NON_NK_REVIEW_CALLS or top_score < 0.70:
        return False

    # Do not override a very pure existing immune/NK annotation with a single
    # non-NK marker set. Keep it as NK/reference and expose the marker warning.
    protected_labels = {
        "B",
        "T",
        "Mature Cytotoxic",
        "Transitional Cytotoxic",
        "Cytokine-Stimulated",
        "Proliferative",
        "Regulatory",
        "Unconventional",
    }
    if top_label in protected_labels and top_label_frac >= 0.80:
        return False

    nk_score = max(
        scores.get("NK_cytotoxic", 0.0),
        scores.get("NK_regulatory_tissue", 0.0),
        scores.get("interferon_cytokine", 0.0),
        scores.get("proliferation", 0.0),
    )
    return top_score >= nk_score + 0.15


def infer_nk_state(row, scores):
    top_label = str(row.get(f"top_{cfg.LABEL_KEY}", ""))
    label_frac = safe_float(row.get(f"top_{cfg.LABEL_KEY}_frac", np.nan))

    if scores.get("proliferation", 0.0) >= 0.60:
        return "Proliferative_NK_candidate"
    if scores.get("NK_regulatory_tissue", 0.0) >= 0.60:
        return "Tissue_regulatory_NK_candidate"
    if scores.get("interferon_cytokine", 0.0) >= 0.60:
        return "Cytokine_interferon_NK_candidate"
    if scores.get("NK_cytotoxic", 0.0) >= 0.50:
        if top_label and label_frac >= 0.60:
            return top_label
        return "Cytotoxic_NK_candidate"
    if top_label and label_frac >= 0.70:
        return top_label
    return "unresolved_NK"


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def write_readme(writer):
    readme = pd.DataFrame(
        {
            "field": [
                "broad_compartment_draft",
                "refined_NK_state_draft",
                "review_warnings",
                "manual_final_label",
                "manual_notes",
            ],
            "meaning": [
                "Computer-assisted broad compartment call. B/T are reference compartments, not contamination.",
                "Draft state call. Use only after manual biological review.",
                "Reasons to be cautious, such as dataset specificity or non-NK-like marker signal.",
                "Fill this manually after review.",
                "Free-text rationale for the final manual decision.",
            ],
        }
    )
    readme.to_excel(writer, sheet_name="README", index=False)


if __name__ == "__main__":
    main()
