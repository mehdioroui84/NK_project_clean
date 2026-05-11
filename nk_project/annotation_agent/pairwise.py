from __future__ import annotations

import os
import re
import warnings
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

PAIRWISE_TOP_N = 100


def recommended_pairs_from_results(
    results: list[dict[str, Any]],
    valid_cluster_ids: set[str],
    *,
    max_pairs: int | None = None,
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    seen = set()
    for result in results:
        cluster_id = str(result["cluster_id"])
        recs = result["final_decision"].get("recommended_pairwise_comparisons", [])
        for rec in recs:
            for other_id in extract_cluster_ids(str(rec), valid_cluster_ids):
                if other_id == cluster_id:
                    continue
                pair = tuple(sorted((cluster_id, other_id), key=cluster_sort_key))
                if pair in seen:
                    continue
                seen.add(pair)
                pairs.append(pair)
                if max_pairs is not None and len(pairs) >= max_pairs:
                    return pairs
    return pairs


def centroid_distance_table(
    *,
    input_h5ad: str,
    groupby: str,
    latent_key: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Compute pairwise Euclidean distances between cluster centroids.

    This is used only to trigger extra pairwise DE checks. It should not be
    interpreted as enough evidence, by itself, to split or rename a cluster.
    """
    if not os.path.exists(input_h5ad):
        raise FileNotFoundError(input_h5ad)

    import scanpy as sc

    adata = sc.read_h5ad(input_h5ad, backed="r")
    try:
        if groupby not in adata.obs:
            raise KeyError(f"{groupby!r} not found in AnnData.obs.")
        key = resolve_latent_key(adata, latent_key)
        z = np.asarray(adata.obsm[key], dtype=np.float32)
        labels = adata.obs[groupby].astype(str).to_numpy()
    finally:
        adata.file.close()

    centroids = {}
    for cluster_id in sorted(pd.unique(labels), key=cluster_sort_key):
        mask = labels == cluster_id
        if not np.any(mask):
            continue
        centroids[str(cluster_id)] = z[mask].mean(axis=0)

    rows = []
    for cluster_a, cluster_b in combinations(sorted(centroids, key=cluster_sort_key), 2):
        distance = float(np.linalg.norm(centroids[cluster_a] - centroids[cluster_b]))
        rows.append(
            {
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "latent_key": key,
                "centroid_distance": distance,
            }
        )
    return pd.DataFrame(rows), key


def resolve_latent_key(adata, latent_key: str | None) -> str:
    if latent_key:
        if latent_key not in adata.obsm:
            available = ", ".join(map(str, adata.obsm.keys()))
            raise KeyError(f"{latent_key!r} not found in AnnData.obsm. Available keys: {available}")
        return latent_key
    preferred = ["X_scVI", "X_SCANVI", "X_scANVI", "X_scanvi", "X_scvi", "X_pca", "X_umap"]
    for key in preferred:
        if key in adata.obsm:
            return key
    available = ", ".join(map(str, adata.obsm.keys()))
    raise KeyError(f"No latent key found in AnnData.obsm. Available keys: {available}")


def same_label_distance_pairs_from_results(
    results: list[dict[str, Any]],
    distance_table: pd.DataFrame,
    *,
    active_cluster_ids: set[str] | None = None,
    min_quantile: float = 0.90,
    max_pairs: int | None = None,
) -> tuple[list[tuple[str, str]], pd.DataFrame]:
    """Select same-label pairs whose centroids are unusually far apart.

    At least one cluster must be active if active_cluster_ids is provided.
    """
    if distance_table.empty:
        return [], pd.DataFrame()
    if not 0 <= min_quantile <= 1:
        raise ValueError("min_quantile must be between 0 and 1.")

    label_by_cluster = {}
    suggested_by_cluster = {}
    for result in results:
        cluster_id = str(result["cluster_id"])
        final = result.get("final_decision", {})
        label = str(final.get("candidate_label", "")).strip()
        suggested = str(final.get("suggested_new_label", "")).strip()
        if label:
            label_by_cluster[cluster_id] = label
            suggested_by_cluster[cluster_id] = suggested

    candidate_rows = []
    for _, row in distance_table.iterrows():
        cluster_a = str(row["cluster_a"])
        cluster_b = str(row["cluster_b"])
        label_a = label_by_cluster.get(cluster_a)
        label_b = label_by_cluster.get(cluster_b)
        if not label_a or label_a != label_b:
            continue
        if active_cluster_ids is not None and cluster_a not in active_cluster_ids and cluster_b not in active_cluster_ids:
            continue
        distance = float(row["centroid_distance"])
        candidate_rows.append(
            {
                "cluster_a": cluster_a,
                "cluster_b": cluster_b,
                "candidate_label": label_a,
                "centroid_distance": distance,
                "latent_key": row.get("latent_key"),
                "suggested_new_label_a": suggested_by_cluster.get(cluster_a, ""),
                "suggested_new_label_b": suggested_by_cluster.get(cluster_b, ""),
            }
        )

    if not candidate_rows:
        return [], pd.DataFrame()

    candidate_df = pd.DataFrame(candidate_rows)
    threshold = float(candidate_df["centroid_distance"].quantile(min_quantile))
    rows = []
    status_rows = []
    for item in candidate_rows:
        distance = float(item["centroid_distance"])
        status = "selected" if distance >= threshold else "below_threshold"
        item = {
            **item,
            "distance_quantile_threshold": threshold,
            "distance_quantile": min_quantile,
            "selection_status": status,
        }
        status_rows.append(item)
        if status == "selected":
            rows.append(item)

    selected = pd.DataFrame(rows).sort_values("centroid_distance", ascending=False) if rows else pd.DataFrame()
    if max_pairs is not None and not selected.empty:
        selected = selected.head(max_pairs)
    pairs = [
        tuple(sorted((str(row["cluster_a"]), str(row["cluster_b"])), key=cluster_sort_key))
        for _, row in selected.iterrows()
    ]
    summary = pd.DataFrame(status_rows)
    if not summary.empty:
        summary = summary.sort_values(["selection_status", "centroid_distance"], ascending=[False, False])
    return pairs, summary


def cluster_distance_evidence_from_results(
    results: list[dict[str, Any]],
    distance_table: pd.DataFrame,
    *,
    distance_quantile: float = 0.90,
    isolation_quantile: float = 0.90,
) -> pd.DataFrame:
    """Summarize centroid-distance evidence per cluster for novelty triage."""
    if distance_table.empty:
        return pd.DataFrame()
    label_by_cluster = {}
    alternative_by_cluster = {}
    for result in results:
        cluster_id = str(result["cluster_id"])
        final = result.get("final_decision", {})
        label_by_cluster[cluster_id] = str(final.get("candidate_label", "")).strip()
        alternative_by_cluster[cluster_id] = str(final.get("suggested_new_label", "")).strip()

    all_distances = pd.to_numeric(distance_table["centroid_distance"], errors="coerce").dropna()
    if all_distances.empty:
        return pd.DataFrame()
    isolation_threshold = float(all_distances.quantile(isolation_quantile))

    rows = []
    for cluster_id in sorted(label_by_cluster, key=cluster_sort_key):
        sub = distance_table[
            (distance_table["cluster_a"].astype(str) == cluster_id)
            | (distance_table["cluster_b"].astype(str) == cluster_id)
        ].copy()
        if sub.empty:
            continue
        sub["other_cluster"] = sub.apply(
            lambda row: str(row["cluster_b"]) if str(row["cluster_a"]) == cluster_id else str(row["cluster_a"]),
            axis=1,
        )
        sub["other_label"] = sub["other_cluster"].map(label_by_cluster)
        sub = sub.sort_values("centroid_distance")
        nearest = sub.iloc[0]
        nearest_same = sub.loc[sub["other_label"] == label_by_cluster[cluster_id]]
        nearest_same_row = nearest_same.iloc[0] if not nearest_same.empty else None
        farthest_same_row = nearest_same.sort_values("centroid_distance", ascending=False).iloc[0] if not nearest_same.empty else None
        same_label_distances = nearest_same["centroid_distance"].astype(float)
        same_label_threshold = (
            float(same_label_distances.quantile(distance_quantile))
            if not same_label_distances.empty
            else np.nan
        )
        nearest_distance = float(nearest["centroid_distance"])
        isolation_percentile = float((all_distances <= nearest_distance).mean())
        farthest_same_distance = (
            float(farthest_same_row["centroid_distance"]) if farthest_same_row is not None else np.nan
        )
        same_label_distance_flag = bool(
            farthest_same_row is not None
            and not np.isnan(same_label_threshold)
            and farthest_same_distance >= same_label_threshold
        )
        isolation_flag = bool(nearest_distance >= isolation_threshold)
        possible_subtype = bool(same_label_distance_flag or isolation_flag)
        novelty_score = novelty_score_from_flags(
            same_label_distance_flag=same_label_distance_flag,
            isolation_flag=isolation_flag,
            has_alternative=bool(alternative_by_cluster.get(cluster_id)),
        )
        rows.append(
            {
                "cluster_id": cluster_id,
                "candidate_refined_label": label_by_cluster[cluster_id],
                "nearest_cluster": str(nearest["other_cluster"]),
                "nearest_label": str(nearest["other_label"]),
                "nearest_distance": nearest_distance,
                "isolation_percentile": isolation_percentile,
                "isolation_flag": isolation_flag,
                "nearest_same_label_cluster": str(nearest_same_row["other_cluster"]) if nearest_same_row is not None else "",
                "nearest_same_label_distance": float(nearest_same_row["centroid_distance"]) if nearest_same_row is not None else np.nan,
                "farthest_same_label_cluster": str(farthest_same_row["other_cluster"]) if farthest_same_row is not None else "",
                "farthest_same_label_distance": farthest_same_distance,
                "same_label_distance_threshold": same_label_threshold,
                "same_label_distance_flag": same_label_distance_flag,
                "possible_novel_subtype": possible_subtype,
                "novel_subtype_score_0_5": novelty_score,
                "novel_subtype_reason": novelty_reason(
                    same_label_distance_flag=same_label_distance_flag,
                    isolation_flag=isolation_flag,
                    farthest_same_label_cluster=str(farthest_same_row["other_cluster"]) if farthest_same_row is not None else "",
                    nearest_cluster=str(nearest["other_cluster"]),
                    nearest_distance=nearest_distance,
                    isolation_percentile=isolation_percentile,
                ),
            }
        )
    return pd.DataFrame(rows)


def novelty_score_from_flags(*, same_label_distance_flag: bool, isolation_flag: bool, has_alternative: bool) -> int:
    score = 0
    if same_label_distance_flag:
        score += 3
    if isolation_flag:
        score += 2
    if has_alternative:
        score += 1
    return min(score, 5)


def novelty_reason(
    *,
    same_label_distance_flag: bool,
    isolation_flag: bool,
    farthest_same_label_cluster: str,
    nearest_cluster: str,
    nearest_distance: float,
    isolation_percentile: float,
) -> str:
    reasons = []
    if same_label_distance_flag:
        reasons.append(f"far from same-label cluster {farthest_same_label_cluster}")
    if isolation_flag:
        reasons.append(
            f"isolated from nearest cluster {nearest_cluster} "
            f"(distance={nearest_distance:.3f}, percentile={isolation_percentile:.2f})"
        )
    if not reasons:
        reasons.append("no strong centroid-distance novelty signal")
    return "; ".join(reasons)


def extract_cluster_ids(text: str, valid_cluster_ids: set[str]) -> list[str]:
    found = []
    for token in re.findall(r"\b\d+\b", text):
        if token in valid_cluster_ids and token not in found:
            found.append(token)
    if text.strip() in valid_cluster_ids and text.strip() not in found:
        found.append(text.strip())
    return found


def run_pairwise_de_for_pairs(
    *,
    input_h5ad: str,
    groupby: str,
    pairs: list[tuple[str, str]],
    outdir: str,
    top_n: int = PAIRWISE_TOP_N,
) -> list[str]:
    if not pairs:
        return []
    if not os.path.exists(input_h5ad):
        raise FileNotFoundError(input_h5ad)

    import scanpy as sc

    os.makedirs(outdir, exist_ok=True)
    print(f"[PAIRWISE_LOAD] {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    if groupby not in adata.obs:
        raise KeyError(f"{groupby!r} not found in AnnData.obs.")
    adata.obs[groupby] = adata.obs[groupby].astype(str)

    written = []
    for cluster_a, cluster_b in pairs:
        comp_name = pair_name(groupby, cluster_a, cluster_b)
        comp_dir = os.path.join(outdir, comp_name)
        top_path = os.path.join(comp_dir, f"{comp_name}_top{top_n}_per_group.csv")
        if os.path.exists(top_path):
            print(f"[PAIRWISE_SKIP] existing {top_path}")
            written.append(top_path)
            continue

        os.makedirs(comp_dir, exist_ok=True)
        mask = adata.obs[groupby].isin([cluster_a, cluster_b]).values
        ad = adata[mask].copy()
        if ad.n_obs == 0:
            print(f"[PAIRWISE_WARN] no cells for {cluster_a} vs {cluster_b}")
            continue

        label_map = {
            cluster_a: f"cluster_{cluster_a}",
            cluster_b: f"cluster_{cluster_b}",
        }
        ad.obs["pairwise_label"] = ad.obs[groupby].map(label_map).astype("category")

        meta = summarize_pair_metadata(ad, groupby)
        meta_path = os.path.join(comp_dir, f"{comp_name}_metadata_summary.csv")
        meta.to_csv(meta_path)
        print(f"[SAVE] {meta_path}")

        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        print(f"[PAIRWISE_DE] {cluster_a} vs {cluster_b}: {ad.n_obs:,} cells")
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in divide",
                category=RuntimeWarning,
                module=r"scanpy\.tools\._rank_genes_groups",
            )
            sc.tl.rank_genes_groups(
                ad,
                groupby="pairwise_label",
                method="wilcoxon",
                pts=True,
                tie_correct=True,
            )
        all_markers = sc.get.rank_genes_groups_df(ad, group=None)
        all_path = os.path.join(comp_dir, f"{comp_name}_all_markers_wilcoxon.csv")
        all_markers.to_csv(all_path, index=False)
        print(f"[SAVE] {all_path}")

        top_markers = select_top_markers(all_markers, top_n=top_n)
        top_markers.to_csv(top_path, index=False)
        print(f"[SAVE] {top_path}")
        written.append(top_path)
    return written


def load_pairwise_evidence(pairwise_dir: str | None, cluster_id: str, *, top_n: int = 20) -> list[dict[str, Any]]:
    if not pairwise_dir or not os.path.isdir(pairwise_dir):
        return []
    evidence = []
    for root, _, files in os.walk(pairwise_dir):
        for filename in files:
            if not filename.endswith("_per_group.csv"):
                continue
            path = os.path.join(root, filename)
            pair = pair_from_filename(filename)
            if pair is None or cluster_id not in pair:
                continue
            df = pd.read_csv(path, low_memory=False)
            if "group" not in df.columns or "names" not in df.columns:
                continue
            group_name = f"cluster_{cluster_id}"
            sub = df.loc[df["group"].astype(str) == group_name].head(top_n)
            other_id = pair[1] if pair[0] == cluster_id else pair[0]
            evidence.append(
                {
                    "comparison": f"{cluster_id}_vs_{other_id}",
                    "other_cluster_id": other_id,
                    "top_genes_for_this_cluster": pairwise_marker_records(sub),
                    "path": path,
                }
            )
    evidence.sort(key=lambda item: cluster_sort_key(item["other_cluster_id"]))
    return evidence


def pairwise_marker_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records = []
    cols = ["names", "scores", "logfoldchanges", "pvals_adj", "pct_nz_group", "pct_nz_reference"]
    available = [col for col in cols if col in df.columns]
    for _, row in df[available].iterrows():
        item = {"gene": str(row.get("names"))}
        for col in available:
            if col == "names":
                continue
            value = row[col]
            item[col] = None if pd.isna(value) else float(value)
        records.append(item)
    return records


def select_top_markers(markers: pd.DataFrame, *, top_n: int) -> pd.DataFrame:
    df = markers.copy()
    if "logfoldchanges" in df.columns:
        df = df[df["logfoldchanges"] > 0].copy()
    sort_cols = [col for col in ["group", "pvals_adj", "scores"] if col in df.columns]
    ascending = [True, True, False][: len(sort_cols)]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascending)
    return df.groupby("group", group_keys=False).head(top_n)


def summarize_pair_metadata(adata, groupby: str) -> pd.DataFrame:
    rows = []
    for label, obs in adata.obs.groupby("pairwise_label", observed=True):
        row = {"pairwise_label": label, "n_cells": int(obs.shape[0])}
        for col in [groupby, "NK_State", "tissue", "dataset_id", "assay_clean"]:
            if col not in obs:
                continue
            vc = obs[col].astype(str).value_counts()
            row[f"top_{col}"] = vc.index[0]
            row[f"top_{col}_frac"] = float(vc.iloc[0] / vc.sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("pairwise_label")


def pair_name(groupby: str, cluster_a: str, cluster_b: str) -> str:
    return f"{groupby}_{cluster_a}_vs_{cluster_b}"


def pair_from_filename(filename: str) -> tuple[str, str] | None:
    match = re.search(r"_(\d+)_vs_(\d+)_top\d+_per_group\.csv$", filename)
    if not match:
        return None
    return match.group(1), match.group(2)


def existing_pair_set(pairwise_dir: str | None) -> set[tuple[str, str]]:
    if not pairwise_dir or not os.path.isdir(pairwise_dir):
        return set()
    pairs = set()
    for _, _, files in os.walk(pairwise_dir):
        for filename in files:
            pair = pair_from_filename(filename)
            if pair is not None:
                pairs.add(tuple(sorted(pair, key=cluster_sort_key)))
    return pairs


def all_cluster_pairs(cluster_ids: list[str]) -> list[tuple[str, str]]:
    return [tuple(pair) for pair in combinations(cluster_ids, 2)]


def cluster_sort_key(value: str) -> tuple[int, int | str]:
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)
