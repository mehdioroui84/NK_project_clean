#!/usr/bin/env python
"""Command-line wrapper for the annotation agent's read-only evidence tools."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from nk_project.annotation_agent.evidence_tools import (  # noqa: E402
    SUPPORTED_RESOLUTIONS,
    get_ranked_pathways,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retrieve one read-only batch of ranked significant pathways."
    )
    parser.add_argument("--resolution", required=True, choices=sorted(SUPPORTED_RESOLUTIONS))
    parser.add_argument("--cluster", required=True, help="Cluster identifier as a string.")
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--number-of-pathways", type=int, default=10)
    parser.add_argument(
        "--generated-date",
        default=date.today().isoformat(),
        help="Date suffix of the processed pathway table. Default: today.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = get_ranked_pathways(
        project_root=args.project_root,
        resolution=args.resolution,
        cluster=str(args.cluster),
        start_rank=args.start_rank,
        number_of_pathways=args.number_of_pathways,
        generated_date=args.generated_date,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
