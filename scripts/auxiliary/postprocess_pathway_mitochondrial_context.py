#!/usr/bin/env python
"""Create compact pathway and mitochondrial context tables for annotation.

This is a one-time auxiliary preparation step for the current Leiden 0.1 and
0.5 evidence files. It does not modify the preserved source tables.
"""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd


RESOLUTIONS = {
    "leiden_0_1": {
        "pathway": "data/annotation_agent/pathway_enrichment/go_bp_enrichment_leiden01.csv",
        "mitochondrial": "data/annotation_agent/cluster_qc/leiden_0_1_dataset_mt_percentage_summary.tsv",
        "worksheet": "outputs/leiden_discovery_assay_only/full_leiden_0_1_annotation_worksheet.csv",
    },
    "leiden_0_5": {
        "pathway": "data/annotation_agent/pathway_enrichment/go_bp_enrichment_all_leiden05.csv",
        "mitochondrial": "data/annotation_agent/cluster_qc/leiden_0_5_dataset_mt_percentage_summary.tsv",
        "worksheet": "outputs/leiden_discovery_assay_only/full_leiden_0_5_annotation_worksheet.csv",
    },
}

PATHWAY_REQUIRED_COLUMNS = {
    "cluster",
    "pathway",
    "overlap_count",
    "gene_ratio",
    "adjusted_p_value",
    "significant_fdr_0_05",
}
MITOCHONDRIAL_REQUIRED_COLUMNS = {
    "cluster",
    "dataset_id",
    "n_cells",
    "cluster_percentage",
    "median_mt_percentage",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter and rank significant GO-BP pathways and summarize "
            "cluster-level mitochondrial context."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="NK_project repository root. Default: inferred from this script.",
    )
    parser.add_argument(
        "--generated-date",
        default=date.today().isoformat(),
        help="Date suffix for outputs in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--high-mt-percentage-cutoff",
        type=float,
        default=10.0,
        help=(
            "A cluster-dataset stratum is high-MT when its median mitochondrial "
            "percentage is at least this value. Default: 10.0."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <project-root>/data/annotation_agent/processed.",
    )
    return parser.parse_args()


def validate_date(value: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"--generated-date must use YYYY-MM-DD: {value!r}") from exc
    return value


def require_columns(frame: pd.DataFrame, required: set[str], path: Path) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"{path} is missing required columns: {missing}")


def cluster_sort_key(value: object) -> tuple[int, object]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def readable_pathway_name(value: object) -> str:
    text = re.sub(r"^GOBP_", "", str(value).strip(), flags=re.IGNORECASE)
    text = re.sub(r"_+", " ", text).strip().lower()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    replacements = {
        r"\bAtp\b": "ATP",
        r"\bDna\b": "DNA",
        r"\bRna\b": "RNA",
        r"\bNadh\b": "NADH",
        r"\bNadph\b": "NADPH",
    }
    for pattern, replacement in replacements.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def significant_pathway_table(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    require_columns(frame, PATHWAY_REQUIRED_COLUMNS, path)

    frame = frame.copy()
    frame["cluster"] = frame["cluster"].astype(str)
    significant = frame["significant_fdr_0_05"].astype(str).str.strip().str.upper().eq("TRUE")
    frame = frame.loc[significant].copy()

    for column in ["adjusted_p_value", "overlap_count", "gene_ratio"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[["adjusted_p_value", "overlap_count", "gene_ratio"]].isna().any().any():
        raise ValueError(f"Significant pathway rows contain nonnumeric ranking values: {path}")

    frame = frame.sort_values(
        ["cluster", "adjusted_p_value", "overlap_count", "gene_ratio", "pathway"],
        ascending=[True, True, False, False, True],
        kind="mergesort",
    )
    frame = frame.drop_duplicates(["cluster", "pathway"], keep="first")
    frame["pathway_rank"] = frame.groupby("cluster", sort=False).cumcount() + 1
    frame["pathway"] = frame["pathway"].map(readable_pathway_name)
    frame = frame[["cluster", "pathway_rank", "pathway"]]

    ordered_clusters = sorted(frame["cluster"].unique(), key=cluster_sort_key)
    frame["cluster"] = pd.Categorical(frame["cluster"], categories=ordered_clusters, ordered=True)
    frame = frame.sort_values(["cluster", "pathway_rank"], kind="mergesort")
    frame["cluster"] = frame["cluster"].astype(object).astype(str)
    return frame.reset_index(drop=True)


def mitochondrial_context_table(
    mitochondrial_path: Path,
    worksheet_path: Path,
    *,
    high_mt_percentage_cutoff: float,
) -> pd.DataFrame:
    mitochondrial = pd.read_csv(mitochondrial_path, sep="\t", low_memory=False)
    require_columns(mitochondrial, MITOCHONDRIAL_REQUIRED_COLUMNS, mitochondrial_path)
    mitochondrial = mitochondrial.copy()
    mitochondrial["cluster"] = mitochondrial["cluster"].astype(str)
    mitochondrial["dataset_id"] = mitochondrial["dataset_id"].astype(str)

    if mitochondrial.duplicated(["cluster", "dataset_id"]).any():
        duplicates = mitochondrial.loc[
            mitochondrial.duplicated(["cluster", "dataset_id"], keep=False),
            ["cluster", "dataset_id"],
        ]
        raise ValueError(
            "Expected one row per cluster-dataset combination in "
            f"{mitochondrial_path}; duplicates include:\n{duplicates.head()}"
        )

    for column in ["n_cells", "cluster_percentage", "median_mt_percentage"]:
        mitochondrial[column] = pd.to_numeric(mitochondrial[column], errors="coerce")
    if mitochondrial[["n_cells", "median_mt_percentage"]].isna().any().any():
        raise ValueError(f"Missing numeric mitochondrial values in {mitochondrial_path}")
    if (mitochondrial["n_cells"] < 0).any():
        raise ValueError(f"Negative n_cells values in {mitochondrial_path}")

    worksheet = pd.read_csv(worksheet_path, index_col=0, low_memory=False)
    if "n_cells" not in worksheet.columns:
        raise KeyError(f"{worksheet_path} is missing required column 'n_cells'")
    worksheet.index = worksheet.index.astype(str)
    worksheet["n_cells"] = pd.to_numeric(worksheet["n_cells"], errors="raise")
    if not worksheet.index.is_unique:
        raise ValueError(f"Duplicate cluster identifiers in {worksheet_path}")

    rows: list[dict[str, object]] = []
    for cluster in sorted(worksheet.index, key=cluster_sort_key):
        subset = mitochondrial.loc[mitochondrial["cluster"] == cluster].copy()
        if subset.empty:
            raise ValueError(
                f"No mitochondrial rows for cluster {cluster!r} in {mitochondrial_path}"
            )

        total_cluster_cells = int(worksheet.loc[cluster, "n_cells"])
        covered_cells = float(subset["n_cells"].sum())
        if total_cluster_cells <= 0 or covered_cells <= 0:
            raise ValueError(f"Invalid cell totals for cluster {cluster!r}")
        if covered_cells > total_cluster_cells + 0.5:
            raise ValueError(
                f"Mitochondrial covered cells exceed total cells for cluster {cluster!r}: "
                f"{covered_cells} > {total_cluster_cells}"
            )

        high_mask = subset["median_mt_percentage"] >= high_mt_percentage_cutoff
        weighted_mt = float(
            np.average(subset["median_mt_percentage"], weights=subset["n_cells"])
        )
        high_cells = float(subset.loc[high_mask, "n_cells"].sum())

        rows.append(
            {
                "cluster": cluster,
                "percent_of_cluster_cells_covered_by_mt_summary": 100.0
                * covered_cells
                / total_cluster_cells,
                "cell_count_weighted_average_mt_percentage": weighted_mt,
                "datasets_with_high_mt_percentage": int(high_mask.sum()),
                "total_datasets_evaluated_for_mt": int(subset["dataset_id"].nunique()),
                "percent_of_covered_cluster_cells_from_high_mt_datasets": 100.0
                * high_cells
                / covered_cells,
            }
        )

    result = pd.DataFrame(rows)
    numeric_columns = [
        "percent_of_cluster_cells_covered_by_mt_summary",
        "cell_count_weighted_average_mt_percentage",
        "percent_of_covered_cluster_cells_from_high_mt_datasets",
    ]
    result[numeric_columns] = result[numeric_columns].round(6)
    return result


def main() -> None:
    args = parse_args()
    generated_date = validate_date(args.generated_date)
    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else project_root / "data" / "annotation_agent" / "processed"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PROJECT] {project_root}")
    print(f"[OUTPUT]  {output_dir}")
    print(f"[HIGH_MT] median_mt_percentage >= {args.high_mt_percentage_cutoff:g}")

    for resolution, relative_paths in RESOLUTIONS.items():
        pathway_path = project_root / relative_paths["pathway"]
        mitochondrial_path = project_root / relative_paths["mitochondrial"]
        worksheet_path = project_root / relative_paths["worksheet"]
        for path in [pathway_path, mitochondrial_path, worksheet_path]:
            if not path.exists():
                raise FileNotFoundError(path)

        pathways = significant_pathway_table(pathway_path)
        mitochondrial = mitochondrial_context_table(
            mitochondrial_path,
            worksheet_path,
            high_mt_percentage_cutoff=args.high_mt_percentage_cutoff,
        )

        pathway_output = output_dir / (
            f"significant_go_bp_ranked_{resolution}_{generated_date}.csv"
        )
        mitochondrial_output = output_dir / (
            f"mitochondrial_context_{resolution}_{generated_date}.csv"
        )
        pathways.to_csv(pathway_output, index=False)
        mitochondrial.to_csv(mitochondrial_output, index=False)

        print(
            f"[SAVE] {pathway_output} "
            f"({len(pathways):,} significant pathways; "
            f"{pathways['cluster'].nunique():,} clusters with pathways)"
        )
        print(
            f"[SAVE] {mitochondrial_output} "
            f"({len(mitochondrial):,} cluster summaries)"
        )


if __name__ == "__main__":
    main()
