from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def compute_class_summary(
    labels: pd.Series,
    *,
    weight_mode: str = "inv_percent",
    weight_clip: tuple[float, float] = (0.1, 10.0),
    weight_normalize_mean: bool = True,
) -> pd.DataFrame:
    counts = labels.astype(str).value_counts().sort_values(ascending=False)
    total = int(counts.sum())
    class_summary = pd.DataFrame(
        {"n_cells": counts.astype(int), "percent": (counts / total) * 100.0}
    )

    eps = 1e-8
    if weight_mode == "inv_percent":
        weights = 1.0 / (class_summary["percent"].values + eps)
    elif weight_mode == "inv_freq":
        weights = 1.0 / (class_summary["n_cells"].values + eps)
    elif weight_mode == "sqrt_inv_freq":
        weights = 1.0 / np.sqrt(class_summary["n_cells"].values + eps)
    else:
        raise ValueError("weight_mode must be inv_percent|inv_freq|sqrt_inv_freq")

    weights = np.clip(weights, weight_clip[0], weight_clip[1])
    if weight_normalize_mean:
        weights = weights / (np.mean(weights) + eps)
    class_summary["weight"] = weights
    return class_summary


def qc_and_balance_anndata(
    adata,
    *,
    label_key: str = "NK_State",
    batch_key: str = "assay_clean",
    dataset_key: str = "dataset_id",
    protected_batch_value: str | None = None,
    low_cut: int = 200,
    max_counts: int | None = None,
    high_quantile: float | None = None,
    min_class_size: int = 780,
    cap_classes: dict[str, int] | None = None,
    major_class: str = "Mature Cytotoxic",
    major_ratio: float = 0.50,
    min_dataset_size: int = 100,
    weight_mode: str = "inv_percent",
    weight_clip: tuple[float, float] = (0.1, 10.0),
    weight_normalize_mean: bool = True,
    seed: int = 0,
    verbose: bool = True,
):
    """Apply the project QC/balancing logic and return AnnData plus weights."""
    rng = np.random.default_rng(seed)
    cap_classes = {"T": 20000, "B": 20000} if cap_classes is None else cap_classes

    ad_all = adata.copy()
    ad_all.obs_names_make_unique()
    ad_all.var_names_make_unique()

    if label_key not in ad_all.obs:
        raise KeyError(f"{label_key!r} not found in adata.obs")
    if batch_key not in ad_all.obs:
        raise KeyError(f"{batch_key!r} not found in adata.obs")

    if protected_batch_value is None:
        protected_mask0 = np.zeros(ad_all.n_obs, dtype=bool)
    else:
        protected_mask0 = ad_all.obs[dataset_key].astype(str).values == str(protected_batch_value)
        if verbose:
            print(f"[PROTECT] protected dataset '{protected_batch_value}': {protected_mask0.sum():,} cells")

    X = ad_all.X
    libsize = np.asarray(X.sum(axis=1)).ravel() if sparse.issparse(X) else np.asarray(X.sum(axis=1)).ravel()

    if max_counts is not None:
        qc_mask = (libsize >= low_cut) & (libsize <= float(max_counts))
        if verbose:
            print(f"[QC] libsize: low_cut={low_cut}, max_counts={max_counts}")
    else:
        if high_quantile is None:
            raise ValueError("If max_counts is None, set high_quantile.")
        hi = float(np.quantile(libsize, high_quantile))
        qc_mask = (libsize >= low_cut) & (libsize <= hi)
        if verbose:
            print(f"[QC] libsize: low_cut={low_cut}, high_quantile={high_quantile} (thr={hi:.1f})")

    ad_all = ad_all[qc_mask].copy()
    protected_mask = protected_mask0[qc_mask]
    ad_prot = ad_all[protected_mask].copy()
    ad = ad_all[~protected_mask].copy()

    if verbose:
        print(f"[QC] kept {ad_all.n_obs:,} cells after libsize filtering")
        print(f"[SPLIT] non-protected: {ad.n_obs:,} | protected: {ad_prot.n_obs:,}")

    if ad.n_obs > 0:
        labels = ad.obs[label_key].astype(str)
        keep_classes = labels.value_counts()[lambda s: s >= min_class_size].index
        ad = ad[labels.isin(keep_classes)].copy()

        if verbose:
            print(f"[QC] kept {ad.n_obs:,} non-protected cells after dropping classes < {min_class_size}")
            print(f"[QC] remaining classes: {ad.obs[label_key].astype(str).nunique():,}")

        for cls, cap in (cap_classes or {}).items():
            lab = ad.obs[label_key].astype(str).values
            idx_cls = np.where(lab == cls)[0]
            if idx_cls.size > cap:
                keep_cls = rng.choice(idx_cls, size=cap, replace=False)
                keep_other = np.where(lab != cls)[0]
                ad = ad[np.sort(np.concatenate([keep_other, keep_cls]))].copy()
                if verbose:
                    print(f"[QC] capped '{cls}': {idx_cls.size:,} -> {cap:,}")

        lab = ad.obs[label_key].astype(str).values
        vc = pd.Series(lab).value_counts()
        if major_class in vc.index:
            n_other = int(vc.drop(major_class).sum())
            target_major = int(major_ratio * n_other)
            current_major = int(vc.loc[major_class])
            if target_major > 0 and current_major > target_major:
                major_idx = np.where(lab == major_class)[0]
                other_idx = np.where(lab != major_class)[0]
                major_keep = rng.choice(major_idx, size=target_major, replace=False)
                ad = ad[np.sort(np.concatenate([other_idx, major_keep]))].copy()
                if verbose:
                    print(f"[QC] subsampled '{major_class}': {current_major:,} -> {target_major:,}")

        batch_counts = ad.obs[batch_key].astype(str).value_counts()
        keep_batches = batch_counts[batch_counts >= min_dataset_size].index
        ad = ad[ad.obs[batch_key].astype(str).isin(keep_batches)].copy()

        if verbose:
            print(f"[QC] kept {ad.n_obs:,} non-protected cells after batch filter")
            print(f"[QC] remaining batches: {ad.obs[batch_key].astype(str).nunique():,}")

    ad_out = sc.concat([ad, ad_prot], join="outer")
    ad_out.obs_names_make_unique()
    class_summary = compute_class_summary(
        ad_out.obs[label_key],
        weight_mode=weight_mode,
        weight_clip=weight_clip,
        weight_normalize_mean=weight_normalize_mean,
    )

    if verbose:
        display_df = class_summary.copy()
        display_df["percent"] = display_df["percent"].round(2)
        display_df["weight"] = display_df["weight"].round(3)
        print("\n[FINAL] class summary:")
        print(display_df)

    return ad_out, class_summary
