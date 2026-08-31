"""Read-only evidence tools used by the evidence-driven annotation agent."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_RESOLUTIONS = {"leiden_0_1", "leiden_0_5"}
MAX_PATHWAYS_PER_REQUEST = 30
PATHWAY_REQUIRED_COLUMNS = {"cluster", "pathway_rank", "pathway"}


def cluster_sort_key(value: object) -> tuple[int, object]:
    text = str(value)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def processed_pathway_path(
    *,
    project_root: Path,
    resolution: str,
    generated_date: str | None = None,
) -> Path:
    if resolution not in SUPPORTED_RESOLUTIONS:
        raise ValueError(
            f"Unsupported resolution {resolution!r}; expected one of "
            f"{sorted(SUPPORTED_RESOLUTIONS)}"
        )
    date_suffix = generated_date or date.today().isoformat()
    return (
        project_root.resolve()
        / "data"
        / "annotation_agent"
        / "processed"
        / f"significant_go_bp_ranked_{resolution}_{date_suffix}.csv"
    )


def get_ranked_pathways(
    *,
    project_root: Path,
    resolution: str,
    cluster: str,
    start_rank: int = 1,
    number_of_pathways: int = 10,
    generated_date: str | None = None,
) -> dict[str, Any]:
    """Return one ranked pathway batch without modifying a source file."""
    if start_rank < 1:
        raise ValueError("start_rank must be at least 1")
    if not 1 <= number_of_pathways <= MAX_PATHWAYS_PER_REQUEST:
        raise ValueError(
            "number_of_pathways must be between 1 and "
            f"{MAX_PATHWAYS_PER_REQUEST}"
        )

    pathway_path = processed_pathway_path(
        project_root=project_root,
        resolution=resolution,
        generated_date=generated_date,
    )
    if not pathway_path.exists():
        raise FileNotFoundError(pathway_path)

    frame = pd.read_csv(pathway_path, dtype={"cluster": str}, low_memory=False)
    missing = sorted(PATHWAY_REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise KeyError(f"{pathway_path} is missing required columns: {missing}")

    frame["cluster"] = frame["cluster"].astype(str)
    frame["pathway_rank"] = pd.to_numeric(
        frame["pathway_rank"], errors="raise"
    ).astype(int)
    cluster_text = str(cluster)
    cluster_frame = frame.loc[frame["cluster"].eq(cluster_text)].sort_values(
        "pathway_rank", kind="mergesort"
    )

    if cluster_frame.empty:
        return {
            "leiden_resolution": resolution,
            "cluster": cluster_text,
            "total_significant_pathways": 0,
            "requested_start_rank": int(start_rank),
            "requested_number_of_pathways": int(number_of_pathways),
            "pathways_returned": [],
            "next_start_rank_if_more_pathways_are_available": None,
        }

    if cluster_frame.duplicated("pathway_rank").any():
        raise ValueError(
            f"Duplicate pathway ranks for cluster {cluster_text!r} in {pathway_path}"
        )
    expected = list(range(1, len(cluster_frame) + 1))
    if cluster_frame["pathway_rank"].tolist() != expected:
        raise ValueError(
            f"Nonconsecutive pathway ranks for cluster {cluster_text!r} in {pathway_path}"
        )

    total = int(len(cluster_frame))
    end_rank = min(total, start_rank + number_of_pathways - 1)
    selected = cluster_frame.loc[
        cluster_frame["pathway_rank"].between(start_rank, end_rank)
    ]
    pathways = [
        {"rank": int(row.pathway_rank), "pathway": str(row.pathway)}
        for row in selected.itertuples(index=False)
    ]
    next_start_rank = end_rank + 1 if end_rank < total else None
    return {
        "leiden_resolution": resolution,
        "cluster": cluster_text,
        "total_significant_pathways": total,
        "requested_start_rank": int(start_rank),
        "requested_number_of_pathways": int(number_of_pathways),
        "pathways_returned": pathways,
        "next_start_rank_if_more_pathways_are_available": next_start_rank,
    }
