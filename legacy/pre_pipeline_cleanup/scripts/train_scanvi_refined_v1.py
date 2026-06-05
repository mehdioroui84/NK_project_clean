#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs
from nk_project.workflows import train_scanvi


GROUPBY = "leiden_0_4"
DEFAULT_OUTDIR_NAME = "refined_scanvi_v1"


def main():
    args = parse_args()
    outdir = args.outdir or os.path.join(cfg.BASE_OUTDIR, DEFAULT_OUTDIR_NAME)
    ensure_dirs(outdir)

    refined_h5ad = os.path.join(outdir, "refined_training_input.h5ad")
    refined_obs_csv = os.path.join(outdir, "refined_training_obs_metadata.csv")
    provenance_csv = os.path.join(outdir, "refined_label_transfer_provenance.csv")

    if args.reuse_refined_input and os.path.exists(refined_h5ad):
        print(f"[CACHE] Reusing refined input: {refined_h5ad}")
    else:
        full = build_refined_training_input(args)
        full.write(refined_h5ad)
        full.obs.to_csv(refined_obs_csv)
        write_provenance(full, provenance_csv)
        print(f"[SAVE] {refined_h5ad}")
        print(f"[SAVE] {refined_obs_csv}")
        print(f"[SAVE] {provenance_csv}")

    if args.dry_run:
        print("[DRY-RUN] Refined input was built; skipping SCANVI training.")
        return

    run_cfg = make_run_config(outdir, refined_h5ad, args.max_epochs)
    train_scanvi(run_cfg, label_key=run_cfg.REFINED_LABEL_KEY, batch_key=run_cfg.PRODUCTION_BATCH_KEY)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build full filtered NK_State_refined labels from the locked validation "
            "leiden_0_4 map, then retrain SCANVI with batch_key=assay_clean."
        )
    )
    parser.add_argument("--outdir", default=None, help="Default: outputs/refined_scanvi_v1")
    parser.add_argument("--knn-k", type=int, default=30, help="kNN transfer neighbors from validation cells.")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override cfg.MAX_EPOCHS for smoke tests.")
    parser.add_argument("--dry-run", action="store_true", help="Build refined labels/input only; do not train.")
    parser.add_argument(
        "--reuse-refined-input",
        action="store_true",
        help="Reuse outdir/refined_training_input.h5ad if it already exists.",
    )
    return parser.parse_args()


def build_refined_training_input(args):
    scvi_path = os.path.join(cfg.LATENT_OUTDIR, "scvi_full_with_latent.h5ad")
    val_path = os.path.join(cfg.BASE_OUTDIR, "leiden_validation", "validation_scvi_leiden.h5ad")
    mapping = load_cluster_mapping()

    print(f"[LOAD] {scvi_path}")
    full = sc.read_h5ad(scvi_path)
    if "X_scVI" not in full.obsm:
        raise KeyError("X_scVI not found in full SCVI h5ad.")

    print(f"[LOAD] {val_path}")
    val = sc.read_h5ad(val_path)
    if GROUPBY not in val.obs:
        raise KeyError(f"{GROUPBY!r} not found in validation Leiden h5ad.")

    assign_splits(full)
    val_labels = val.obs[GROUPBY].astype(str).map(mapping)
    if val_labels.isna().any():
        missing = sorted(val.obs.loc[val_labels.isna(), GROUPBY].astype(str).unique())
        raise ValueError(f"Missing refined labels for {GROUPBY} clusters: {missing}")

    ref_names = [name for name in val.obs_names.astype(str) if name in full.obs_names]
    if not ref_names:
        raise ValueError("No validation cells overlap between validation Leiden h5ad and full SCVI h5ad.")

    ref_label_by_name = pd.Series(val_labels.values, index=val.obs_names.astype(str)).loc[ref_names]
    ref_pos = full.obs_names.get_indexer(ref_names)
    X_ref = np.asarray(full.obsm["X_scVI"][ref_pos], dtype=np.float32)
    X_all = np.asarray(full.obsm["X_scVI"], dtype=np.float32)

    k = min(args.knn_k, max(1, len(ref_names)))
    print(f"[TRANSFER] {len(ref_names):,} validation reference cells; k={k}")
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(X_ref)
    distances, indices = nn.kneighbors(X_all)

    ref_labels = ref_label_by_name.to_numpy(dtype=str)
    pred_labels = []
    pred_frac = []
    for neigh in indices:
        labels = ref_labels[neigh]
        counts = Counter(labels)
        # Deterministic tie-break: nearest neighbor among tied labels wins.
        top_count = max(counts.values())
        tied = {label for label, count in counts.items() if count == top_count}
        winner = next(label for label in labels if label in tied)
        pred_labels.append(winner)
        pred_frac.append(top_count / len(labels))

    full.obs[cfg.REFINED_LABEL_KEY] = pd.Series(pred_labels, index=full.obs_names, dtype="object")
    full.obs["NK_State_refined_transfer_confidence"] = pred_frac
    full.obs["NK_State_refined_source"] = "knn_transfer_from_validation"

    # For validation cells, keep the exact locked cluster mapping instead of the kNN vote.
    exact = pd.Series(ref_label_by_name.values, index=ref_label_by_name.index)
    full.obs.loc[exact.index, cfg.REFINED_LABEL_KEY] = exact
    full.obs.loc[exact.index, "NK_State_refined_source"] = "validation_leiden_locked"
    full.obs.loc[exact.index, "NK_State_refined_transfer_confidence"] = 1.0

    # Keep provenance columns that make later review easier.
    full.obs["NK_State_original"] = full.obs[cfg.LABEL_KEY].astype(str)
    full.obs[f"{GROUPBY}_validation"] = pd.NA
    full.obs.loc[exact.index, f"{GROUPBY}_validation"] = val.obs.loc[exact.index, GROUPBY].astype(str)

    print("\n[REFINED LABEL COUNTS]")
    print(full.obs[cfg.REFINED_LABEL_KEY].astype(str).value_counts().to_string())
    print("\n[REFINED LABEL SOURCE]")
    print(full.obs["NK_State_refined_source"].astype(str).value_counts().to_string())
    return full


def load_cluster_mapping():
    candidate_paths = [
        os.path.join(
            cfg.BASE_OUTDIR,
            "markers",
            "validation",
            GROUPBY,
            "figures",
            f"{GROUPBY}_interpretation_cluster_mapping.csv",
        ),
        os.path.join(
            cfg.BASE_OUTDIR,
            "markers",
            "validation",
            GROUPBY,
            f"{GROUPBY}_interpretation_worksheet.csv",
        ),
    ]
    for path in candidate_paths:
        if not os.path.exists(path):
            continue
        print(f"[LOAD] {path}")
        df = pd.read_csv(path)
        if GROUPBY in df.columns and "refined_NK_state_draft" in df.columns:
            return dict(zip(df[GROUPBY].astype(str), df["refined_NK_state_draft"].astype(str)))
        if "cluster" in df.columns and "refined_NK_state_draft" in df.columns:
            return dict(zip(df["cluster"].astype(str), df["refined_NK_state_draft"].astype(str)))

    raise FileNotFoundError(
        "Could not find refined cluster mapping. Expected one of:\n  "
        + "\n  ".join(candidate_paths)
    )


def assign_splits(adata):
    adata.obs["_split"] = "Unknown"
    for split_name, file_name in [
        ("Train", "train_obs_names.txt"),
        ("Val", "val_obs_names.txt"),
        ("Held-out", "heldout_obs_names.txt"),
    ]:
        path = os.path.join(cfg.TABLE_OUTDIR, file_name)
        if not os.path.exists(path):
            print(f"[WARN] Split file not found: {path}")
            continue
        names = pd.read_csv(path, header=None)[0].astype(str)
        names = names[names.isin(adata.obs_names)]
        adata.obs.loc[names.values, "_split"] = split_name
    print("\n[SPLITS]")
    print(adata.obs["_split"].astype(str).value_counts().to_string())


def write_provenance(adata, path):
    cols = [
        cfg.LABEL_KEY,
        "NK_State_original",
        cfg.REFINED_LABEL_KEY,
        "NK_State_refined_source",
        "NK_State_refined_transfer_confidence",
        "_split",
        cfg.DATASET_KEY,
        cfg.ASSAY_CLEAN_KEY,
        f"{GROUPBY}_validation",
    ]
    cols = [col for col in cols if col in adata.obs]
    adata.obs[cols].to_csv(path)


def make_run_config(outdir, refined_h5ad, max_epochs):
    values = {name: getattr(cfg, name) for name in dir(cfg) if name.isupper()}
    values["BASE_OUTDIR"] = outdir
    values["FIG_OUTDIR"] = os.path.join(outdir, "figures")
    values["MODEL_OUTDIR"] = os.path.join(outdir, "models")
    values["TABLE_OUTDIR"] = os.path.join(outdir, "tables")
    values["LATENT_OUTDIR"] = os.path.join(outdir, "latents")
    values["MERGED_PATH"] = refined_h5ad
    values["SPLIT_ID_SOURCE_DIR"] = cfg.TABLE_OUTDIR
    if max_epochs is not None:
        values["MAX_EPOCHS"] = int(max_epochs)
    return SimpleNamespace(**values)


if __name__ == "__main__":
    main()
