from __future__ import annotations

import numpy as np
import pandas as pd


def build_assay_clean(
    obs: pd.DataFrame,
    *,
    assay_key: str = "assay",
    assay_clean_key: str = "assay_clean",
    flex_fill: str = "Flex Gene Expression",
    verbose: bool = True,
) -> pd.DataFrame:
    """Return obs copy with missing assay labels filled into assay_clean."""
    obs = obs.copy()
    assay = obs[assay_key].astype(str)
    n_nan = int((assay == "nan").sum())
    assay = assay.replace("nan", flex_fill)
    obs[assay_clean_key] = assay.astype(str)

    if verbose and n_nan:
        print(f"[ASSAY] Filled {n_nan:,} nan assay values with '{flex_fill}'")

    return obs


def build_composite_batch_key(
    obs: pd.DataFrame,
    *,
    dataset_key: str = "dataset_id",
    assay_key: str = "assay",
    assay_clean_key: str = "assay_clean",
    flex_fill: str = "Flex Gene Expression",
    merge_threshold: int = 100,
    composite_col: str = "batch_composite",
    verbose: bool = True,
) -> pd.DataFrame:
    """Create assay_clean and dataset_id || assay_clean composite key.

    Rare assay groups within each dataset are relabelled to that dataset's
    dominant assay before the composite key is built.
    """
    obs = build_assay_clean(
        obs,
        assay_key=assay_key,
        assay_clean_key=assay_clean_key,
        flex_fill=flex_fill,
        verbose=verbose,
    )

    merged_count = 0
    for ds, grp in obs.groupby(dataset_key):
        counts = grp[assay_clean_key].value_counts()
        rare = counts[counts < merge_threshold].index.tolist()
        if not rare:
            continue

        dominant = counts.idxmax()
        mask = (obs[dataset_key].astype(str) == str(ds)) & obs[assay_clean_key].isin(rare)
        obs.loc[mask, assay_clean_key] = dominant
        merged_count += int(mask.sum())

        if verbose:
            for assay in rare:
                print(
                    f"[COMPOSITE] {str(ds)[:8]}... assay '{assay}' "
                    f"({int(counts[assay])} cells) -> '{dominant}'"
                )

    obs[composite_col] = obs[dataset_key].astype(str) + " || " + obs[assay_clean_key].astype(str)

    if verbose:
        print(
            f"[COMPOSITE] Relabelled {merged_count:,} cells to dominant assay "
            f"(threshold < {merge_threshold})"
        )
        print(f"[COMPOSITE] Final composite batches: {obs[composite_col].nunique():,}")

    return obs


def profile_batch_combinations(
    obs: pd.DataFrame,
    *,
    dataset_key: str = "dataset_id",
    assay_key: str = "assay",
    low_cell_warn: int = 500,
    min_cell_hard: int = 100,
) -> dict:
    """Return count tables and risk summaries for dataset x assay batches."""
    tmp = obs[[dataset_key, assay_key]].astype(str).copy()
    tmp["composite"] = tmp[dataset_key] + " || " + tmp[assay_key]

    combo_counts = tmp["composite"].value_counts().sort_values(ascending=False)
    dataset_counts = tmp[dataset_key].value_counts()
    assay_counts = tmp[assay_key].value_counts()

    return {
        "combo_counts": combo_counts,
        "dataset_counts": dataset_counts,
        "assay_counts": assay_counts,
        "n_datasets": int(dataset_counts.size),
        "n_assays": int(assay_counts.size),
        "n_combos": int(combo_counts.size),
        "worst_case": int(dataset_counts.size * assay_counts.size),
        "n_below_warn": int((combo_counts < low_cell_warn).sum()),
        "n_below_hard": int((combo_counts < min_cell_hard).sum()),
        "n_ok": int((combo_counts >= low_cell_warn).sum()),
    }


def log10_pivot_counts(
    obs: pd.DataFrame,
    *,
    dataset_key: str = "dataset_id",
    assay_key: str = "assay",
) -> pd.DataFrame:
    pivot = obs.groupby([dataset_key, assay_key]).size().unstack(fill_value=0)
    return np.log10(pivot.replace(0, np.nan))
