from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def prepare_label_categories(*adatas, label_key: str = "NK_State", unlabeled: str = "Unknown") -> None:
    for adata in adatas:
        adata.obs[label_key] = adata.obs[label_key].astype("category")
        if unlabeled not in adata.obs[label_key].cat.categories:
            adata.obs[label_key] = adata.obs[label_key].cat.add_categories([unlabeled])


def make_train_val_heldout_split(
    adata,
    *,
    dataset_key: str,
    label_key: str,
    held_out_datasets: list[str],
    test_size: float = 0.20,
    seed: int = 0,
):
    heldout_mask = adata.obs[dataset_key].astype(str).isin(held_out_datasets)
    adata_heldout = adata[heldout_mask].copy()
    adata_remaining = adata[~heldout_mask].copy()

    idx = np.arange(adata_remaining.n_obs)
    y = adata_remaining.obs[label_key].astype(str).values
    labels, counts = np.unique(y, return_counts=True)
    rare_labels = set(labels[counts < 2])
    if rare_labels:
        rare_mask = np.isin(y, list(rare_labels))
        common_idx = idx[~rare_mask]
        rare_idx = idx[rare_mask]
        common_y = y[~rare_mask]
        if len(common_idx) == 0 or len(np.unique(common_y)) < 2:
            train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=None)
        else:
            train_idx, val_idx = train_test_split(
                common_idx,
                test_size=test_size,
                random_state=seed,
                stratify=common_y,
            )
            train_idx = np.concatenate([train_idx, rare_idx])
    else:
        train_idx, val_idx = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y)

    adata_train = adata_remaining[train_idx].copy()
    adata_val = adata_remaining[val_idx].copy()
    train_names = adata_train.obs_names.tolist()
    val_names = adata_val.obs_names.tolist()
    heldout_names = adata_heldout.obs_names.tolist()

    return {
        "adata_train": adata_train,
        "adata_val": adata_val,
        "adata_heldout": adata_heldout,
        "train_names": train_names,
        "val_names": val_names,
        "heldout_names": heldout_names,
    }


def subset_by_names(adata, names):
    return adata[names].copy()
