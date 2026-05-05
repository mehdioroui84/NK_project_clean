#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scarches as sca

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.evaluate import probability_summary
from nk_project.io_utils import ensure_dirs, save_latent_npz, save_run_config
from nk_project.metrics import filtered_classification_metrics
from nk_project.workflows import prepare_filtered_data


DEFAULT_REF_OUTDIR_NAME = "refined_scanvi_v1"
DEFAULT_SURGERY_OUTDIR_NAME = "refined_scanvi_v1_surgery"


def main():
    args = parse_args()

    ref_outdir = args.ref_outdir or os.path.join(cfg.BASE_OUTDIR, DEFAULT_REF_OUTDIR_NAME)
    surgery_outdir = args.outdir or os.path.join(cfg.BASE_OUTDIR, DEFAULT_SURGERY_OUTDIR_NAME)
    input_h5ad = args.input_h5ad or os.path.join(
        cfg.BASE_OUTDIR,
        "refined_annotation_v1",
        "full_scvi_leiden_refined_v1.h5ad",
    )
    model_dir = args.model_dir or os.path.join(
        ref_outdir,
        "models",
        f"scanvi_{safe_name(cfg.REFINED_LABEL_KEY)}_{safe_name(cfg.PRODUCTION_BATCH_KEY)}_model",
    )
    split_dir = args.split_dir or os.path.join(ref_outdir, "tables")

    run_cfg = make_run_config(surgery_outdir, input_h5ad, args)
    ensure_dirs(
        run_cfg.BASE_OUTDIR,
        run_cfg.FIG_OUTDIR,
        run_cfg.MODEL_OUTDIR,
        run_cfg.TABLE_OUTDIR,
        run_cfg.LATENT_OUTDIR,
    )
    save_run_config(os.path.join(run_cfg.BASE_OUTDIR, "run_config_scanvi_surgery.json"), run_cfg)

    print(f"[INPUT] {input_h5ad}")
    print(f"[REFERENCE_MODEL] {model_dir}")
    print(f"[REFERENCE_OUTDIR] {ref_outdir}")
    print(f"[SPLIT_DIR] {split_dir}")
    print(f"[OUTDIR] {surgery_outdir}")
    print(f"[LABEL] {cfg.REFINED_LABEL_KEY}")
    print(f"[BATCH] {cfg.PRODUCTION_BATCH_KEY}")

    require_path(input_h5ad, "refined input h5ad")
    require_path(model_dir, "reference SCANVI model directory")
    require_path(split_dir, "reference split table directory")

    adata_filtered, _ = prepare_filtered_data(
        run_cfg,
        batch_key=cfg.PRODUCTION_BATCH_KEY,
        label_key=cfg.REFINED_LABEL_KEY,
    )
    train_names = read_split_names(split_dir, "train_obs_names.txt", adata_filtered.obs_names)
    heldout_names = read_split_names(split_dir, "heldout_obs_names.txt", adata_filtered.obs_names)
    if not heldout_names:
        raise ValueError("No held-out cells from the reference split were found after filtering.")

    train_assays = set(
        adata_filtered[train_names].obs[cfg.ASSAY_CLEAN_KEY].astype(str).dropna().unique().tolist()
    )
    heldout_assays = set(
        adata_filtered[heldout_names].obs[cfg.ASSAY_CLEAN_KEY].astype(str).dropna().unique().tolist()
    )
    known_heldout_assays = sorted(heldout_assays & train_assays)
    new_heldout_assays = sorted(heldout_assays - train_assays)

    print("\n[ASSAY ADAPTOR CHECK]")
    print("Held-out assays already present during reference training:")
    print("  " + (", ".join(known_heldout_assays) if known_heldout_assays else "None"))
    print("Held-out assays absent during reference training; surgery can learn new adaptor(s):")
    print("  " + (", ".join(new_heldout_assays) if new_heldout_assays else "None"))

    if args.new_assays_only:
        if not new_heldout_assays:
            raise ValueError(
                "--new-assays-only was requested, but no held-out assay_clean category is absent from training."
            )
        held = adata_filtered[heldout_names].obs[cfg.ASSAY_CLEAN_KEY].astype(str)
        heldout_names = held[held.isin(new_heldout_assays)].index.astype(str).tolist()
        print(
            "[QUERY FILTER] Restricting surgery comparison to held-out cells from new assay category/categories: "
            f"{', '.join(new_heldout_assays)}"
        )

    query = adata_filtered[heldout_names].copy()
    query.obs_names_make_unique()
    true_labels = query.obs[cfg.REFINED_LABEL_KEY].astype(str).copy()
    query.obs["NK_State_refined_true"] = true_labels
    query.obs["_split"] = "Held-out"

    # Important: surgery uses held-out cells as an unlabeled query. The true
    # labels are kept only in NK_State_refined_true for evaluation after fitting.
    query.obs[cfg.REFINED_LABEL_KEY] = cfg.UNLABELED_CATEGORY

    training_classes = read_training_classes(model_dir)
    print(f"[QUERY] {query.n_obs:,} held-out cells x {query.n_vars:,} genes")
    print(f"[TRAINING_CLASSES] {len(training_classes)} classes")

    if args.dry_run:
        print("[DRY-RUN] Configuration is valid; skipping surgery.")
        return

    print("[ZERO-SHOT] Loading held-out query into the frozen reference SCANVI model")
    zero_model = sca.models.SCANVI.load_query_data(
        query,
        model_dir,
        freeze_dropout=args.freeze_dropout,
    )
    zero_eval = evaluate_query(
        zero_model,
        query,
        true_labels,
        split_name="ZERO_SHOT_HELDOUT_SAME_QUERY",
        training_classes=training_classes,
        compute_latent=False,
    )
    save_query_outputs(
        run_cfg,
        query,
        zero_eval,
        prefix="scanvi_zeroshot_same_query",
        latent_key="X_SCANVI_ZERO_SHOT",
        model_save_dir=None,
    )

    print("[SURGERY] Loading held-out query into the reference SCANVI model")
    surgery_model = sca.models.SCANVI.load_query_data(
        query,
        model_dir,
        freeze_dropout=args.freeze_dropout,
    )

    print(f"[SURGERY] Training query model for {run_cfg.SURGERY_EPOCHS} epochs")
    surgery_model.train(
        max_epochs=run_cfg.SURGERY_EPOCHS,
        batch_size=run_cfg.BATCH_SIZE,
        plan_kwargs={"lr": run_cfg.SURGERY_LR},
        check_val_every_n_epoch=10,
    )

    surgery_eval = evaluate_query(
        surgery_model,
        query,
        true_labels,
        split_name="SURGERY_HELDOUT",
        training_classes=training_classes,
        compute_latent=True,
    )

    model_save_dir = os.path.join(run_cfg.MODEL_OUTDIR, "scanvi_surgery_heldout_model")
    surgery_model.save(model_save_dir, overwrite=True)
    save_query_outputs(
        run_cfg,
        query,
        surgery_eval,
        prefix="scanvi_surgery_heldout",
        latent_key="X_SCANVI_SURGERY",
        model_save_dir=None,
    )

    surgery_per_class = surgery_eval["metrics"]["per_class"].copy()
    if not surgery_per_class.empty:
        surgery_per_class["accuracy"] = surgery_per_class["recall"]
        surgery_per_class.to_csv(
            os.path.join(run_cfg.TABLE_OUTDIR, "scanvi_surgery_heldout_per_class_accuracy_f1.csv")
        )

    zero_per_class = zero_eval["metrics"]["per_class"].copy()
    if not zero_per_class.empty:
        zero_per_class["accuracy"] = zero_per_class["recall"]
        zero_per_class.to_csv(
            os.path.join(run_cfg.TABLE_OUTDIR, "scanvi_zeroshot_same_query_per_class_accuracy_f1.csv")
        )

    comparison = write_comparison_tables(
        run_cfg,
        zero_eval["metrics"],
        surgery_eval["metrics"],
        zero_per_class,
        surgery_per_class,
    )
    plot_comparison(run_cfg, comparison)

    print(f"[SAVE] Surgery model: {model_save_dir}")
    print("[DONE] SCANVI surgery held-out comparison complete.")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run SCANVI model surgery on the refined-v1 held-out query cells and "
            "compare surgery performance against the existing zero-shot baseline."
        )
    )
    parser.add_argument(
        "--input-h5ad",
        default=None,
        help="Default: outputs/refined_annotation_v1/full_scvi_leiden_refined_v1.h5ad",
    )
    parser.add_argument(
        "--ref-outdir",
        default=None,
        help="Default: outputs/refined_scanvi_v1, the existing zero-shot/reference run.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Default: ref-outdir/models/scanvi_NK_State_refined_assay_clean_model",
    )
    parser.add_argument(
        "--split-dir",
        default=None,
        help="Default: ref-outdir/tables; must contain heldout_obs_names.txt.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Default: outputs/refined_scanvi_v1_surgery",
    )
    parser.add_argument(
        "--surgery-epochs",
        type=int,
        default=None,
        help="Override cfg.SURGERY_EPOCHS.",
    )
    parser.add_argument(
        "--surgery-lr",
        type=float,
        default=None,
        help="Override cfg.SURGERY_LR.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override cfg.BATCH_SIZE.",
    )
    parser.add_argument(
        "--freeze-dropout",
        action="store_true",
        help="Pass freeze_dropout=True to SCANVI.load_query_data.",
    )
    parser.add_argument(
        "--new-assays-only",
        action="store_true",
        help=(
            "Restrict the held-out query to assay_clean categories absent from "
            "the reference training split. This tests whether surgery helps when "
            "there is a genuinely new assay adaptor to learn."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def make_run_config(outdir: str, input_h5ad: str, args):
    values = {name: getattr(cfg, name) for name in dir(cfg) if name.isupper()}
    values["BASE_OUTDIR"] = outdir
    values["FIG_OUTDIR"] = os.path.join(outdir, "figures")
    values["MODEL_OUTDIR"] = os.path.join(outdir, "models")
    values["TABLE_OUTDIR"] = os.path.join(outdir, "tables")
    values["LATENT_OUTDIR"] = os.path.join(outdir, "latents")
    values["MERGED_PATH"] = input_h5ad
    if args.surgery_epochs is not None:
        values["SURGERY_EPOCHS"] = int(args.surgery_epochs)
    if args.surgery_lr is not None:
        values["SURGERY_LR"] = float(args.surgery_lr)
    if args.batch_size is not None:
        values["BATCH_SIZE"] = int(args.batch_size)
    return SimpleNamespace(**values)


def require_path(path: str, label: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")


def read_split_names(split_dir: str, file_name: str, obs_names) -> list[str]:
    path = os.path.join(split_dir, file_name)
    require_path(path, "split file")
    available = set(pd.Index(obs_names).astype(str))
    names = pd.read_csv(path, header=None)[0].astype(str).tolist()
    names = [name for name in names if name in available]
    print(f"[SPLIT] {file_name}: {len(names):,} cells found after filtering")
    return names


def read_training_classes(model_dir: str) -> set[str]:
    path = os.path.join(model_dir, "training_classes.txt")
    require_path(path, "training class file")
    classes = pd.read_csv(path, header=None)[0].astype(str).tolist()
    return set(classes)


def evaluate_query(
    model,
    query,
    true_labels,
    *,
    split_name: str,
    training_classes: set[str],
    compute_latent: bool,
):
    proba = model.predict(query, soft=True)
    pred = proba.idxmax(axis=1).astype(str).values
    true = true_labels.loc[proba.index].astype(str).values
    metrics = filtered_classification_metrics(
        true,
        pred,
        split_name=split_name,
        training_classes=training_classes,
        unlabeled=cfg.UNLABELED_CATEGORY,
        min_class_eval=cfg.MIN_CLASS_EVAL,
    )
    z = model.get_latent_representation(query) if compute_latent else None
    summary = probability_summary(proba)
    summary["true_label"] = true
    summary["correct"] = summary["pred_label"].astype(str).values == summary["true_label"].astype(str).values
    summary["eval_included"] = metrics["kept_mask"]
    return {"proba": proba, "latent": z, "summary": summary, "metrics": metrics}


def save_query_outputs(run_cfg, query, eval_result, *, prefix: str, latent_key: str, model_save_dir: str | None):
    if eval_result["latent"] is not None:
        save_latent_npz(
            os.path.join(run_cfg.LATENT_OUTDIR, f"{prefix}_latents.npz"),
            **{latent_key: eval_result["latent"], "obs_names": query.obs_names.values},
        )
    query.obs.join(eval_result["summary"], how="left").to_csv(
        os.path.join(run_cfg.TABLE_OUTDIR, f"{prefix}_obs_metadata.csv")
    )
    eval_result["proba"].to_csv(os.path.join(run_cfg.TABLE_OUTDIR, f"{prefix}_probabilities.csv"))
    if model_save_dir:
        ensure_dirs(model_save_dir)


def write_comparison_tables(
    run_cfg,
    zero_metrics,
    surgery_metrics,
    zero_per_class: pd.DataFrame,
    surgery_per_class: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "method": "zero_shot_same_query",
            "macro_f1": zero_metrics["macro_f1"],
            "weighted_f1": zero_metrics["weighted_f1"],
            "dropped_unseen": ";".join(zero_metrics["dropped_unseen"]),
            "dropped_rare": ";".join(zero_metrics["dropped_rare"]),
        }
    ]
    rows.append(
        {
            "method": "surgery",
            "macro_f1": surgery_metrics["macro_f1"],
            "weighted_f1": surgery_metrics["weighted_f1"],
            "dropped_unseen": ";".join(surgery_metrics["dropped_unseen"]),
            "dropped_rare": ";".join(surgery_metrics["dropped_rare"]),
        }
    )
    comparison = pd.DataFrame(rows)
    comparison.to_csv(
        os.path.join(run_cfg.TABLE_OUTDIR, "scanvi_surgery_vs_zeroshot_metric_summary.csv"),
        index=False,
    )

    if not zero_per_class.empty and not surgery_per_class.empty:
        zero = zero_per_class.copy()
        zero = zero.rename(columns={"accuracy": "zero_shot_accuracy", "f1": "zero_shot_f1", "n_true": "zero_shot_n_true"})
        surg = surgery_per_class.rename(
            columns={
                "accuracy": "surgery_accuracy",
                "f1": "surgery_f1",
                "n_true": "surgery_n_true",
                "precision": "surgery_precision",
                "recall": "surgery_recall",
            }
        )
        joined = zero.join(surg, how="outer")
        joined["delta_f1_surgery_minus_zero_shot"] = joined["surgery_f1"] - joined["zero_shot_f1"]
        joined["delta_accuracy_surgery_minus_zero_shot"] = joined["surgery_accuracy"] - joined["zero_shot_accuracy"]
        joined.to_csv(
            os.path.join(run_cfg.TABLE_OUTDIR, "scanvi_surgery_vs_zeroshot_per_class.csv")
        )

    return comparison


def plot_comparison(run_cfg, comparison: pd.DataFrame):
    if comparison.empty:
        return

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    x = np.arange(len(comparison))
    width = 0.34
    ax.bar(x - width / 2, comparison["macro_f1"], width, label="Macro F1", color="#4c78a8")
    ax.bar(x + width / 2, comparison["weighted_f1"], width, label="Weighted F1", color="#f58518")
    ax.set_xticks(x)
    ax.set_xticklabels(comparison["method"].str.replace("_", " "), rotation=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1")
    ax.set_title("Held-out refined-v1 prediction: zero-shot vs surgery")
    ax.legend(frameon=False)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=8, padding=2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(run_cfg.FIG_OUTDIR, "scanvi_surgery_vs_zeroshot_summary.png"),
        dpi=180,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(run_cfg.FIG_OUTDIR, "scanvi_surgery_vs_zeroshot_summary.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)

    per_class_path = os.path.join(run_cfg.TABLE_OUTDIR, "scanvi_surgery_vs_zeroshot_per_class.csv")
    if not os.path.exists(per_class_path):
        return
    pc = pd.read_csv(per_class_path, index_col=0)
    needed = {"zero_shot_f1", "surgery_f1"}
    if not needed.issubset(pc.columns):
        return
    pc = pc.sort_values("zero_shot_f1", ascending=True)
    fig_h = max(4.5, 0.35 * len(pc))
    fig, ax = plt.subplots(figsize=(7.5, fig_h))
    y = np.arange(len(pc))
    ax.barh(y - 0.18, pc["zero_shot_f1"], height=0.34, label="Zero-shot", color="#9ecae9")
    ax.barh(y + 0.18, pc["surgery_f1"], height=0.34, label="Surgery", color="#31a354")
    ax.set_yticks(y)
    ax.set_yticklabels(pc.index)
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1")
    ax.set_title("Held-out per-class F1")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(
        os.path.join(run_cfg.FIG_OUTDIR, "scanvi_surgery_vs_zeroshot_per_class_f1.png"),
        dpi=180,
        bbox_inches="tight",
    )
    fig.savefig(
        os.path.join(run_cfg.FIG_OUTDIR, "scanvi_surgery_vs_zeroshot_per_class_f1.pdf"),
        bbox_inches="tight",
    )
    plt.close(fig)


def safe_name(value):
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


if __name__ == "__main__":
    main()
