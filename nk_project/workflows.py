from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scanpy as sc
import scarches as sca
import scvi
import torch

from .evaluate import evaluate_scanvi_split, probability_summary
from .io_utils import ensure_dirs, save_latent_npz, save_run_config, save_split_ids
from .preprocessing import build_assay_clean, build_composite_batch_key
from .qc import qc_and_balance_anndata
from .splits import make_train_val_heldout_split, prepare_label_categories
from .training_plan import WeightedSemiSupervisedTrainingPlan


def prepare_filtered_data(
    cfg,
    *,
    batch_key: str | None = None,
    label_key: str | None = None,
    verbose: bool = True,
):
    batch_key = batch_key or cfg.PRODUCTION_BATCH_KEY
    label_key = label_key or cfg.LABEL_KEY
    merged = sc.read_h5ad(cfg.MERGED_PATH)
    merged.obs_names_make_unique()
    merged.var_names_make_unique()
    if verbose:
        print(f"[LOAD] {merged.n_obs:,} cells x {merged.n_vars:,} genes")

    merged.obs = build_assay_clean(
        merged.obs,
        assay_key=cfg.ASSAY_KEY,
        assay_clean_key=cfg.ASSAY_CLEAN_KEY,
        flex_fill=cfg.FLEX_ASSAY_FILL,
        verbose=verbose,
    )
    if cfg.COMPOSITE_BATCH_KEY not in merged.obs:
        merged.obs = build_composite_batch_key(
            merged.obs,
            dataset_key=cfg.DATASET_KEY,
            assay_key=cfg.ASSAY_KEY,
            assay_clean_key=cfg.ASSAY_CLEAN_KEY,
            flex_fill=cfg.FLEX_ASSAY_FILL,
            merge_threshold=cfg.COMPOSITE_MERGE_THRESHOLD,
            composite_col=cfg.COMPOSITE_BATCH_KEY,
            verbose=False,
        )

    adata_filtered, class_summary = qc_and_balance_anndata(
        merged,
        label_key=label_key,
        batch_key=batch_key,
        dataset_key=cfg.DATASET_KEY,
        protected_batch_value=cfg.PROTECTED_DATASET,
        low_cut=cfg.QC_LOW_CUT,
        max_counts=cfg.QC_MAX_COUNTS,
        min_class_size=cfg.MIN_CLASS_SIZE,
        cap_classes=cfg.CAP_CLASSES,
        major_class=cfg.MAJOR_CLASS,
        major_ratio=cfg.MAJOR_RATIO,
        min_dataset_size=cfg.MIN_BATCH_SIZE,
        weight_mode=cfg.WEIGHT_MODE,
        weight_clip=cfg.WEIGHT_CLIP,
        seed=cfg.SEED,
        verbose=verbose,
    )
    return adata_filtered, class_summary


def train_scvi(cfg, *, batch_key: str | None = None):
    batch_key = batch_key or cfg.PRODUCTION_BATCH_KEY
    ensure_dirs(cfg.MODEL_OUTDIR, cfg.TABLE_OUTDIR, cfg.LATENT_OUTDIR)
    save_run_config(os.path.join(cfg.BASE_OUTDIR, "run_config_scvi.json"), cfg)

    adata_filtered, class_summary = prepare_filtered_data(
        cfg,
        batch_key=batch_key,
        label_key=cfg.LABEL_KEY,
    )
    split = make_train_val_heldout_split(
        adata_filtered,
        dataset_key=cfg.DATASET_KEY,
        label_key=cfg.LABEL_KEY,
        held_out_datasets=cfg.HELD_OUT_DATASETS,
        test_size=cfg.TRAIN_VAL_TEST_SIZE,
        seed=cfg.SEED,
    )
    adata_train = split["adata_train"]
    adata_val = split["adata_val"]
    adata_heldout = split["adata_heldout"]
    save_split_ids(
        cfg.TABLE_OUTDIR,
        train_names=split["train_names"],
        val_names=split["val_names"],
        heldout_names=split["heldout_names"],
    )
    class_summary.to_csv(os.path.join(cfg.TABLE_OUTDIR, "class_summary.csv"))

    scvi.model.SCVI.setup_anndata(adata_train, batch_key=batch_key)
    model = scvi.model.SCVI(
        adata_train,
        n_layers=cfg.N_LAYERS,
        n_hidden=cfg.N_HIDDEN,
        n_latent=cfg.N_LATENT,
        gene_likelihood=cfg.GENE_LIKELIHOOD,
    )
    model.train(max_epochs=cfg.MAX_EPOCHS, batch_size=cfg.BATCH_SIZE, plan_kwargs={"lr": cfg.LR})

    model_dir = os.path.join(cfg.MODEL_OUTDIR, f"scvi_{_safe_name(batch_key)}_model")
    model.save(model_dir, overwrite=True)

    full = sc.concat([adata_train, adata_val, adata_heldout], join="outer")
    full.obs_names_make_unique()
    scvi.model.SCVI.prepare_query_anndata(full, model)
    new_manager = model.adata_manager.transfer_fields(full, extend_categories=True)
    model._register_manager_for_instance(new_manager)
    z_full = model.get_latent_representation(full)
    full.obsm["X_scVI"] = z_full
    save_latent_npz(os.path.join(cfg.LATENT_OUTDIR, "scvi_latents.npz"), X_scVI=z_full, obs_names=full.obs_names.values)
    full.obs.to_csv(os.path.join(cfg.TABLE_OUTDIR, "scvi_obs_metadata.csv"))
    full.write(os.path.join(cfg.LATENT_OUTDIR, "scvi_full_with_latent.h5ad"))

    print(f"[SAVE] SCVI model: {model_dir}")
    return model, full


def train_scanvi(cfg, *, label_key: str | None = None, batch_key: str | None = None):
    label_key = label_key or cfg.LABEL_KEY
    batch_key = batch_key or cfg.PRODUCTION_BATCH_KEY
    ensure_dirs(cfg.MODEL_OUTDIR, cfg.TABLE_OUTDIR, cfg.LATENT_OUTDIR)
    save_run_config(os.path.join(cfg.BASE_OUTDIR, "run_config_scanvi.json"), cfg)

    adata_filtered, class_summary = prepare_filtered_data(
        cfg,
        batch_key=batch_key,
        label_key=label_key,
    )
    split = _make_or_reuse_split(cfg, adata_filtered, label_key=label_key)
    adata_train = split["adata_train"]
    adata_val = split["adata_val"]
    adata_heldout = split["adata_heldout"]
    prepare_label_categories(adata_train, adata_val, adata_heldout, label_key=label_key, unlabeled=cfg.UNLABELED_CATEGORY)
    save_split_ids(
        cfg.TABLE_OUTDIR,
        train_names=split["train_names"],
        val_names=split["val_names"],
        heldout_names=split["heldout_names"],
    )
    class_summary.to_csv(os.path.join(cfg.TABLE_OUTDIR, "class_summary.csv"))

    sca.models.SCANVI.setup_anndata(
        adata_train,
        batch_key=batch_key,
        labels_key=label_key,
        unlabeled_category=cfg.UNLABELED_CATEGORY,
    )
    model = sca.models.SCANVI(
        adata_train,
        n_layers=cfg.N_LAYERS,
        n_hidden=cfg.N_HIDDEN,
        n_latent=cfg.N_LATENT,
        gene_likelihood=cfg.GENE_LIKELIHOOD,
    )
    model._training_plan_cls = WeightedSemiSupervisedTrainingPlan

    label_order = list(model.adata_manager.get_state_registry("labels").categorical_mapping)
    training_classes = set(label_order) - {cfg.UNLABELED_CATEGORY}
    weights = _make_class_weight_tensor(cfg, label_order, class_summary)

    model.train(
        max_epochs=cfg.MAX_EPOCHS,
        batch_size=cfg.BATCH_SIZE,
        plan_kwargs={"lr": cfg.LR, "class_weights": weights},
        check_val_every_n_epoch=10,
    )

    model_dir = os.path.join(cfg.MODEL_OUTDIR, f"scanvi_{_safe_name(label_key)}_{_safe_name(batch_key)}_model")
    model.save(model_dir, overwrite=True)

    eval_val = evaluate_scanvi_split(
        adata_val,
        split_name="VAL",
        model=model,
        label_key=label_key,
        training_classes=training_classes,
        unlabeled=cfg.UNLABELED_CATEGORY,
        min_class_eval=cfg.MIN_CLASS_EVAL,
    )
    eval_heldout = evaluate_scanvi_split(
        adata_heldout,
        split_name="ZERO-SHOT",
        model=model,
        label_key=label_key,
        training_classes=training_classes,
        unlabeled=cfg.UNLABELED_CATEGORY,
        min_class_eval=cfg.MIN_CLASS_EVAL,
        extend_categories=True,
    )

    full = _build_full_adata(adata_train, adata_val, adata_heldout, label_key=label_key, unlabeled=cfg.UNLABELED_CATEGORY)
    eval_full = evaluate_scanvi_split(
        full,
        split_name="FULL",
        model=model,
        label_key=label_key,
        training_classes=training_classes,
        unlabeled=cfg.UNLABELED_CATEGORY,
        min_class_eval=cfg.MIN_CLASS_EVAL,
        extend_categories=True,
    )

    z_full = model.get_latent_representation(eval_full["adata"])
    save_latent_npz(os.path.join(cfg.LATENT_OUTDIR, "scanvi_latents.npz"), X_SCANVI=z_full, obs_names=eval_full["adata"].obs_names.values)
    eval_full["adata"].obs.to_csv(os.path.join(cfg.TABLE_OUTDIR, "scanvi_full_obs_metadata.csv"))
    probability_summary(eval_full["proba"]).to_csv(os.path.join(cfg.TABLE_OUTDIR, "scanvi_full_prediction_summary.csv"))
    eval_full["proba"].to_csv(os.path.join(cfg.TABLE_OUTDIR, "scanvi_full_probabilities.csv"))
    pd.Series(sorted(training_classes), name="class").to_csv(os.path.join(model_dir, "training_classes.txt"), index=False, header=False)

    summary = pd.DataFrame(
        [
            {"split": "val", **_flat_metrics(eval_val["metrics"])},
            {"split": "zeroshot", **_flat_metrics(eval_heldout["metrics"])},
            {"split": "full", **_flat_metrics(eval_full["metrics"])},
        ]
    )
    summary.to_csv(os.path.join(cfg.TABLE_OUTDIR, "scanvi_metric_summary.csv"), index=False)

    print(f"[SAVE] SCANVI model: {model_dir}")
    return model, {"val": eval_val, "heldout": eval_heldout, "full": eval_full}


def _make_class_weight_tensor(cfg, label_order, class_summary):
    weights = np.ones(len(label_order), dtype=np.float32)
    for i, label in enumerate(label_order):
        if label == cfg.UNLABELED_CATEGORY:
            weights[i] = 0.0
        elif label in class_summary.index:
            weights[i] = float(class_summary.loc[label, "weight"])
        else:
            weights[i] = 1.0
        if label != cfg.UNLABELED_CATEGORY:
            weights[i] = min(cfg.WEIGHT_MAX, max(cfg.WEIGHT_MIN, weights[i]))
    return torch.tensor(weights, dtype=torch.float32)


def _build_full_adata(adata_train, adata_val, adata_heldout, *, label_key: str, unlabeled: str):
    train = adata_train.copy()
    val = adata_val.copy()
    held = adata_heldout.copy()
    train.obs["_split"] = "Train"
    val.obs["_split"] = "Val"
    held.obs["_split"] = "Held-out"
    full = sc.concat([train, val, held], join="outer")
    full.obs_names_make_unique()
    prepare_label_categories(full, label_key=label_key, unlabeled=unlabeled)
    return full


def _make_or_reuse_split(cfg, adata, *, label_key: str):
    split_source_dir = getattr(cfg, "SPLIT_ID_SOURCE_DIR", None)
    if not split_source_dir:
        return make_train_val_heldout_split(
            adata,
            dataset_key=cfg.DATASET_KEY,
            label_key=label_key,
            held_out_datasets=cfg.HELD_OUT_DATASETS,
            test_size=cfg.TRAIN_VAL_TEST_SIZE,
            seed=cfg.SEED,
        )

    print(f"[SPLIT] Reusing split IDs from: {split_source_dir}")
    train_names = _read_split_names(split_source_dir, "train_obs_names.txt", adata.obs_names)
    val_names = _read_split_names(split_source_dir, "val_obs_names.txt", adata.obs_names)
    heldout_names = _read_split_names(split_source_dir, "heldout_obs_names.txt", adata.obs_names)

    used = set(train_names) | set(val_names) | set(heldout_names)
    leftover = [name for name in adata.obs_names.astype(str) if name not in used]
    if leftover:
        print(f"[SPLIT] Warning: {len(leftover):,} filtered cells were not in reused split IDs; adding them to Train.")
        train_names = train_names + leftover

    print(
        "[SPLIT] reused counts: "
        f"train={len(train_names):,} | val={len(val_names):,} | held-out={len(heldout_names):,}"
    )
    return {
        "adata_train": adata[train_names].copy(),
        "adata_val": adata[val_names].copy(),
        "adata_heldout": adata[heldout_names].copy(),
        "train_names": train_names,
        "val_names": val_names,
        "heldout_names": heldout_names,
    }


def _read_split_names(split_source_dir: str, file_name: str, obs_names) -> list[str]:
    path = os.path.join(split_source_dir, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Split ID file not found: {path}")
    available = set(obs_names.astype(str))
    names = pd.read_csv(path, header=None)[0].astype(str).tolist()
    return [name for name in names if name in available]


def _flat_metrics(metrics):
    return {
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "dropped_unseen": ";".join(metrics["dropped_unseen"]),
        "dropped_rare": ";".join(metrics["dropped_rare"]),
    }


def _safe_name(value):
    out = str(value)
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        ("|", "_"),
        (":", "_"),
        ("'", ""),
        ('"', ""),
    ]:
        out = out.replace(old, new)
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_")
