from __future__ import annotations

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import (
    adjusted_rand_score,
    classification_report,
    f1_score,
    normalized_mutual_info_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors


def filtered_classification_metrics(
    true_labels,
    pred_labels,
    *,
    split_name: str,
    training_classes: set[str],
    unlabeled: str = "Unknown",
    min_class_eval: int = 30,
):
    true_labels = np.asarray(true_labels).astype(str)
    pred_labels = np.asarray(pred_labels).astype(str)

    gate_fail = ~np.isin(true_labels, list(training_classes))
    unseen_classes = sorted(set(true_labels[gate_fail]) - {unlabeled})
    after_gate = ~gate_fail
    counts_after_gate = pd.Series(true_labels[after_gate]).value_counts()
    rare_classes = sorted(counts_after_gate[counts_after_gate < min_class_eval].index.tolist())
    drop_classes = set(unseen_classes) | set(rare_classes)
    kept_mask = ~np.isin(true_labels, list(drop_classes))

    true_kept = true_labels[kept_mask]
    pred_kept = pred_labels[kept_mask]
    if len(true_kept) == 0:
        return {
            "macro_f1": np.nan,
            "weighted_f1": np.nan,
            "per_class": pd.DataFrame(),
            "kept_mask": kept_mask,
            "dropped_unseen": unseen_classes,
            "dropped_rare": rare_classes,
        }

    kept_classes = sorted(set(true_kept))
    macro_f1 = f1_score(true_kept, pred_kept, average="macro", labels=kept_classes, zero_division=0)
    weighted_f1 = f1_score(true_kept, pred_kept, average="weighted", labels=kept_classes, zero_division=0)
    report = classification_report(
        true_kept, pred_kept, labels=kept_classes, output_dict=True, zero_division=0
    )
    per_class = pd.DataFrame(
        {
            cls: {
                "n_true": int(counts_after_gate.get(cls, 0)),
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
            }
            for cls in kept_classes
        }
    ).T.sort_values("f1", ascending=False)

    print(f"\n{'=' * 60}")
    print(f"[{split_name}] Macro F1   : {macro_f1:.4f} ({len(kept_classes)} classes, {kept_mask.sum():,} cells)")
    print(f"[{split_name}] Weighted F1: {weighted_f1:.4f}")
    print(per_class.round(3).to_string())

    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
        "kept_mask": kept_mask,
        "dropped_unseen": unseen_classes,
        "dropped_rare": rare_classes,
    }


def subsample_for_metrics(X, labels, batch_labels, max_cells=50000, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    batch_labels = np.asarray(batch_labels)
    if X.shape[0] <= max_cells:
        return X, labels, batch_labels

    idx_all = np.arange(X.shape[0])
    keep = []
    props = pd.Series(labels).value_counts() / len(labels)
    for lab, prop in props.items():
        idx_lab = idx_all[labels == lab]
        n_lab = min(len(idx_lab), max(1, int(round(prop * max_cells))))
        keep.append(rng.choice(idx_lab, size=n_lab, replace=False))
    keep = np.unique(np.concatenate(keep))
    if len(keep) > max_cells:
        keep = rng.choice(keep, size=max_cells, replace=False)
    keep = np.sort(keep)
    return X[keep], labels[keep], batch_labels[keep]


def compute_knn_label_accuracy(X, y, k=30):
    y = np.asarray(y).astype(str)
    if len(np.unique(y)) < 2 or len(y) <= k:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    inds = nn.kneighbors(X, return_distance=False)[:, 1:]
    pred = []
    for row in y[inds]:
        vals, counts = np.unique(row, return_counts=True)
        pred.append(vals[np.argmax(counts)])
    return float(np.mean(np.asarray(pred) == y))


def compute_knn_batch_accuracy(X, batch_labels, *, within_labels=None, k=30):
    """Predict batch from nearest neighbors; lower values imply less batch signal.

    If within_labels is provided, neighbors are searched separately within each
    biological label to reduce the chance that biology drives apparent batch
    predictability.
    """
    batch_labels = np.asarray(batch_labels).astype(str)
    if within_labels is None:
        return _knn_majority_accuracy(X, batch_labels, k=k)

    within_labels = np.asarray(within_labels).astype(str)
    scores = []
    weights = []
    for lab in sorted(pd.unique(within_labels)):
        mask = within_labels == lab
        if mask.sum() <= k + 1 or len(np.unique(batch_labels[mask])) < 2:
            continue
        score = _knn_majority_accuracy(X[mask], batch_labels[mask], k=min(k, mask.sum() - 1))
        if not np.isnan(score):
            scores.append(score)
            weights.append(int(mask.sum()))
    return float(np.average(scores, weights=weights)) if scores else np.nan


def _knn_majority_accuracy(X, labels, k=30):
    labels = np.asarray(labels).astype(str)
    if len(np.unique(labels)) < 2 or len(labels) <= k:
        return np.nan
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean").fit(X)
    inds = nn.kneighbors(X, return_distance=False)[:, 1:]
    pred = []
    for row in labels[inds]:
        vals, counts = np.unique(row, return_counts=True)
        pred.append(vals[np.argmax(counts)])
    return float(np.mean(np.asarray(pred) == labels))


def compute_label_asw(X, labels):
    labels = np.asarray(labels).astype(str)
    if len(np.unique(labels)) < 2:
        return np.nan
    try:
        return (float(silhouette_score(X, labels, metric="euclidean")) + 1.0) / 2.0
    except Exception:
        return np.nan


def compute_batch_asw_label_aware(X, batch_labels, celltype_labels, min_cells=10):
    batch_labels = np.asarray(batch_labels).astype(str)
    celltype_labels = np.asarray(celltype_labels).astype(str)
    scores = []
    weights = []
    for ct in sorted(pd.unique(celltype_labels)):
        m = celltype_labels == ct
        if m.sum() < max(min_cells, 3):
            continue
        b = batch_labels[m]
        keep_batches = pd.Series(b).value_counts()[lambda s: s >= 2].index
        m2 = m.copy()
        m2[m] = np.isin(b, keep_batches)
        if m2.sum() < max(min_cells, 3) or len(np.unique(batch_labels[m2])) < 2:
            continue
        try:
            sil = silhouette_samples(X[m2], batch_labels[m2], metric="euclidean")
            scores.append(1.0 - abs(float(np.mean(sil))))
            weights.append(int(m2.sum()))
        except Exception:
            pass
    return float(np.average(scores, weights=weights)) if scores else np.nan


def compute_graph_connectivity(X, labels, n_neighbors=30):
    labels = np.asarray(labels).astype(str)
    if len(labels) < 5:
        return np.nan
    ad = sc.AnnData(X=np.asarray(X, dtype=np.float32))
    sc.pp.neighbors(ad, use_rep="X", n_neighbors=min(n_neighbors, len(labels) - 1))
    conn = ad.obsp["connectivities"].tocsr()
    scores = []
    for lab in sorted(pd.unique(labels)):
        idx = np.where(labels == lab)[0]
        if len(idx) < 3:
            continue
        n_comp, comp = connected_components(conn[idx][:, idx], directed=False, connection="weak")
        if n_comp:
            scores.append(np.bincount(comp).max() / len(idx))
    return float(np.mean(scores)) if scores else np.nan


def compute_cluster_metrics(X, labels, unlabeled="Unknown", leiden_resolution=1.0, n_neighbors=30):
    labels = np.asarray(labels).astype(str)
    valid = labels != unlabeled
    if valid.sum() < 10 or len(np.unique(labels[valid])) < 2:
        return np.nan, np.nan, pd.Series(dtype=int)
    ad = sc.AnnData(X=np.asarray(X[valid], dtype=np.float32))
    sc.pp.neighbors(ad, use_rep="X", n_neighbors=min(n_neighbors, ad.n_obs - 1))
    sc.tl.leiden(ad, resolution=leiden_resolution, key_added="leiden_tmp")
    clusters = ad.obs["leiden_tmp"].astype(str).values
    return (
        float(normalized_mutual_info_score(labels[valid], clusters)),
        float(adjusted_rand_score(labels[valid], clusters)),
        pd.Series(clusters).value_counts().sort_index(),
    )


def compute_integration_metrics_from_latent(
    z,
    labels,
    batch_labels,
    *,
    strategy_name: str,
    unlabeled: str = "Unknown",
    n_neighbors: int = 30,
    leiden_resolution: float = 1.0,
    max_metric_cells: int = 50000,
    seed: int = 0,
    knn_k: int = 30,
):
    labels = np.asarray(labels).astype(str)
    batch_labels = np.asarray(batch_labels).astype(str)
    labeled = labels != unlabeled
    z_lab, y_lab, b_lab = subsample_for_metrics(
        z[labeled], labels[labeled], batch_labels[labeled], max_cells=max_metric_cells, seed=seed
    )
    print(f"[{strategy_name}] Metric subsample: {z_lab.shape[0]:,} labeled cells")
    nmi, ari, cluster_sizes = compute_cluster_metrics(
        z_lab, y_lab, unlabeled=unlabeled, leiden_resolution=leiden_resolution, n_neighbors=n_neighbors
    )
    return {
        "strategy": strategy_name,
        "n_cells_metric": int(len(y_lab)),
        "n_batches_metric": int(pd.Series(b_lab).nunique()),
        "asw_nk_state": compute_label_asw(z_lab, y_lab),
        "asw_batch": compute_batch_asw_label_aware(z_lab, b_lab, y_lab),
        "graph_connectivity": compute_graph_connectivity(z_lab, y_lab, n_neighbors=n_neighbors),
        "knn_label_acc": compute_knn_label_accuracy(z_lab, y_lab, k=min(knn_k, max(2, len(y_lab) - 1))),
        "nmi": nmi,
        "ari": ari,
        "ilisi_batch": np.nan,
        "cluster_sizes": cluster_sizes,
    }


def minmax_normalize_series(s, higher_is_better=True):
    s = pd.Series(s, dtype=float)
    if s.notna().sum() <= 1:
        return pd.Series(np.ones(len(s)), index=s.index, dtype=float)
    smin = s.min(skipna=True)
    smax = s.max(skipna=True)
    out = pd.Series(np.ones(len(s)), index=s.index, dtype=float) if smax == smin else (s - smin) / (smax - smin)
    return out if higher_is_better else 1.0 - out
