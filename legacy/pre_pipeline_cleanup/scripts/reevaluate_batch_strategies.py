#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from experiments.adversarial_refiner import LatentAdversarialRefiner
from nk_project.io_utils import ensure_dirs, save_latent_npz
from nk_project.metrics import (
    compute_batch_asw_label_aware,
    compute_graph_connectivity,
    compute_knn_batch_accuracy,
    compute_knn_label_accuracy,
    compute_label_asw,
    subsample_for_metrics,
)
from nk_project.workflows import train_scanvi


STRATEGIES = {
    "dataset_only": cfg.DATASET_KEY,
    "assay_only": cfg.ASSAY_CLEAN_KEY,
    "composite_only": cfg.COMPOSITE_BATCH_KEY,
}

BATCH_MIXING_METRICS = [
    "dataset_asw_mixing",
    "assay_asw_mixing",
    "dataset_knn_mixing",
    "assay_knn_mixing",
]

BIOLOGY_PRESERVATION_METRICS = [
    "nk_state_asw",
    "knn_label_acc",
]


def main():
    args = parse_args()
    outdir = get_experiment_outdir(args)
    ensure_dirs(outdir)

    rows = []
    cached = {}
    for strategy in args.strategies:
        if strategy == "adversarial_refiner":
            continue
        row, payload = run_scanvi_strategy(strategy, outdir, args)
        rows.append(row)
        cached[strategy] = payload

    if "adversarial_refiner" in args.strategies:
        if "assay_only" not in cached:
            row, payload = run_scanvi_strategy("assay_only", outdir, args)
            rows.append(row)
            cached["assay_only"] = payload
        rows.append(run_adversarial_refiner(cached["assay_only"], outdir, args))

    summary = pd.DataFrame(rows).set_index("strategy")
    summary = add_normalized_scores(summary)

    raw_path = os.path.join(outdir, "batch_strategy_metrics_absolute_scores.csv")
    summary.to_csv(raw_path)
    print("\n" + "=" * 90)
    print("BATCH STRATEGY REEVALUATION")
    print("=" * 90)
    print(summary.round(4).to_string())
    print(f"[SAVE] {raw_path}")

    plot_path = os.path.join(outdir, "batch_strategy_absolute_scores.png")
    plot_absolute_scores(summary, plot_path)
    print(f"[SAVE] {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate SCANVI batch strategies and an optional adversarial refiner "
            "with normalized 0-1 metrics."
        )
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["dataset_only", "assay_only", "composite_only"],
        choices=["dataset_only", "assay_only", "composite_only", "adversarial_refiner"],
    )
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--metric-max-cells", type=int, default=50000)
    parser.add_argument("--refiner-epochs", type=int, default=80)
    parser.add_argument("--refiner-lr", type=float, default=1e-3)
    parser.add_argument("--refiner-batch-size", type=int, default=4096)
    parser.add_argument("--lambda-dataset", type=float, default=0.05)
    parser.add_argument("--lambda-assay", type=float, default=0.05)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain models/refiner even if cached metrics and latents already exist.",
    )
    return parser.parse_args()


def get_experiment_outdir(args):
    base = os.path.join(cfg.BASE_OUTDIR, "batch_strategy_reevaluation")
    is_default_training = args.max_epochs is None and args.refiner_epochs == 80
    if is_default_training:
        return base
    max_epochs = "default" if args.max_epochs is None else str(args.max_epochs)
    return f"{base}_epochs{max_epochs}_refiner{args.refiner_epochs}"


def run_scanvi_strategy(strategy, outdir, args):
    batch_key = STRATEGIES[strategy]
    run_cfg = make_strategy_cfg(strategy, outdir, args)
    cache = strategy_cache_paths(run_cfg, strategy)

    print("\n" + "#" * 90)
    print(f"[RUN] {strategy} | batch_key={batch_key}")
    print("#" * 90)

    if not args.force_retrain and scanvi_cache_exists(cache):
        print(f"[CACHE] Reusing saved strategy outputs for {strategy}")
        row, payload = load_scanvi_cache(cache)
        if cache_needs_metric_refresh(row):
            print(f"[CACHE] Refreshing metric cache for {strategy}; no retraining")
            row.update(compute_latent_comparison_metrics(payload["z"], payload["obs"], strategy, args))
            pd.DataFrame([row]).to_csv(cache["metrics"], index=False)
            print(f"[SAVE] {cache['metrics']}")
        return row, payload

    model, evals = train_scanvi(run_cfg, label_key=run_cfg.LABEL_KEY, batch_key=batch_key)
    full_adata = evals["full"]["adata"]
    z = model.get_latent_representation(full_adata).astype(np.float32)

    save_latent_npz(cache["latent"], X_SCANVI=z, obs_names=full_adata.obs_names.values)
    full_adata.obs.to_csv(cache["obs"])
    print(f"[SAVE] {cache['latent']}")
    print(f"[SAVE] {cache['obs']}")

    row = {
        "strategy": strategy,
        "batch_key": batch_key,
        "model_type": "SCANVI",
        "val_macro_f1": evals["val"]["metrics"]["macro_f1"],
        "zeroshot_macro_f1": evals["heldout"]["metrics"]["macro_f1"],
        "full_macro_f1": evals["full"]["metrics"]["macro_f1"],
        "val_weighted_f1": evals["val"]["metrics"]["weighted_f1"],
        "zeroshot_weighted_f1": evals["heldout"]["metrics"]["weighted_f1"],
        "full_weighted_f1": evals["full"]["metrics"]["weighted_f1"],
    }
    row.update(compute_latent_comparison_metrics(z, full_adata.obs, strategy, args))
    pd.DataFrame([row]).to_csv(cache["metrics"], index=False)
    print(f"[SAVE] {cache['metrics']}")
    return row, {"z": z, "obs": full_adata.obs.copy()}


def make_strategy_cfg(strategy, outdir, args):
    run_cfg = SimpleNamespace(**{k: getattr(cfg, k) for k in dir(cfg) if k.isupper()})
    run_cfg.BASE_OUTDIR = os.path.join(outdir, "runs", strategy)
    run_cfg.FIG_OUTDIR = os.path.join(run_cfg.BASE_OUTDIR, "figures")
    run_cfg.MODEL_OUTDIR = os.path.join(run_cfg.BASE_OUTDIR, "models")
    run_cfg.TABLE_OUTDIR = os.path.join(run_cfg.BASE_OUTDIR, "tables")
    run_cfg.LATENT_OUTDIR = os.path.join(run_cfg.BASE_OUTDIR, "latents")
    if args.max_epochs is not None:
        run_cfg.MAX_EPOCHS = args.max_epochs
    return run_cfg


def strategy_cache_paths(run_cfg, strategy):
    return {
        "metrics": os.path.join(run_cfg.BASE_OUTDIR, f"{strategy}_metrics.csv"),
        "latent": os.path.join(run_cfg.LATENT_OUTDIR, f"{strategy}_scanvi_latent.npz"),
        "obs": os.path.join(run_cfg.TABLE_OUTDIR, f"{strategy}_obs_metadata.csv"),
    }


def scanvi_cache_exists(cache):
    return all(os.path.exists(path) for path in cache.values())


def load_scanvi_cache(cache):
    row = pd.read_csv(cache["metrics"]).iloc[0].to_dict()
    z = np.load(cache["latent"])["X_SCANVI"].astype(np.float32)
    obs = pd.read_csv(cache["obs"], index_col=0, low_memory=False)
    return row, {"z": z, "obs": obs}


def cache_needs_metric_refresh(row):
    required = [
        "dataset_knn_baseline_acc",
        "assay_knn_baseline_acc",
        "dataset_asw_mixing",
        "assay_asw_mixing",
    ]
    return any(col not in row or pd.isna(row.get(col)) for col in required)


def compute_latent_comparison_metrics(z, obs, strategy, args):
    labels = obs[cfg.LABEL_KEY].astype(str).values
    dataset = obs[cfg.DATASET_KEY].astype(str).values
    assay = obs[cfg.ASSAY_CLEAN_KEY].astype(str).values

    labeled = labels != cfg.UNLABELED_CATEGORY
    z_lab, labels_lab, dataset_lab = subsample_for_metrics(
        z[labeled],
        labels[labeled],
        dataset[labeled],
        max_cells=args.metric_max_cells,
        seed=cfg.SEED,
    )
    _, _, assay_lab = subsample_for_metrics(
        z[labeled],
        labels[labeled],
        assay[labeled],
        max_cells=args.metric_max_cells,
        seed=cfg.SEED,
    )

    print(f"[{strategy}] Metric subsample: {z_lab.shape[0]:,} labeled cells")
    return {
        "n_cells_metric": int(z_lab.shape[0]),
        "n_datasets_metric": int(pd.Series(dataset_lab).nunique()),
        "n_assays_metric": int(pd.Series(assay_lab).nunique()),
        "nk_state_asw": compute_label_asw(z_lab, labels_lab),
        "graph_connectivity": compute_graph_connectivity(z_lab, labels_lab, n_neighbors=cfg.METRIC_KNN_K),
        "knn_label_acc": compute_knn_label_accuracy(z_lab, labels_lab, k=cfg.METRIC_KNN_K),
        "dataset_asw_mixing": compute_batch_asw_label_aware(z_lab, dataset_lab, labels_lab),
        "assay_asw_mixing": compute_batch_asw_label_aware(z_lab, assay_lab, labels_lab),
        "dataset_knn_batch_acc": compute_knn_batch_accuracy(
            z_lab, dataset_lab, within_labels=labels_lab, k=cfg.METRIC_KNN_K
        ),
        "assay_knn_batch_acc": compute_knn_batch_accuracy(
            z_lab, assay_lab, within_labels=labels_lab, k=cfg.METRIC_KNN_K
        ),
        "dataset_knn_baseline_acc": weighted_within_label_majority_baseline(dataset_lab, labels_lab),
        "assay_knn_baseline_acc": weighted_within_label_majority_baseline(assay_lab, labels_lab),
    }


def run_adversarial_refiner(assay_payload, outdir, args):
    print("\n" + "#" * 90)
    print("[RUN] adversarial_refiner | starting from assay_only SCANVI latent")
    print("#" * 90)

    cache = {
        "metrics": os.path.join(outdir, "adversarial_refiner_metrics.csv"),
        "latent": os.path.join(outdir, "adversarial_refiner_latent.npz"),
    }
    if not args.force_retrain and all(os.path.exists(path) for path in cache.values()):
        print("[CACHE] Reusing saved adversarial refiner outputs")
        row = pd.read_csv(cache["metrics"]).iloc[0].to_dict()
        if cache_needs_metric_refresh(row):
            print("[CACHE] Refreshing adversarial refiner metric cache; no retraining")
            obs = assay_payload["obs"].copy()
            z_refined = np.load(cache["latent"])["X_REFINED"].astype(np.float32)
            row.update(compute_latent_comparison_metrics(z_refined, obs, "adversarial_refiner", args))
            pd.DataFrame([row]).to_csv(cache["metrics"], index=False)
            print(f"[SAVE] {cache['metrics']}")
        return row

    obs = assay_payload["obs"].copy()
    z = assay_payload["z"].astype(np.float32)
    train_mask = obs["_split"].astype(str).values == "Train"

    label_encoder = LabelEncoder().fit(obs[cfg.LABEL_KEY].astype(str).values)
    dataset_encoder = LabelEncoder().fit(obs[cfg.DATASET_KEY].astype(str).values)
    assay_encoder = LabelEncoder().fit(obs[cfg.ASSAY_CLEAN_KEY].astype(str).values)

    y_label = label_encoder.transform(obs[cfg.LABEL_KEY].astype(str).values)
    y_dataset = dataset_encoder.transform(obs[cfg.DATASET_KEY].astype(str).values)
    y_assay = assay_encoder.transform(obs[cfg.ASSAY_CLEAN_KEY].astype(str).values)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LatentAdversarialRefiner(
        z_dim=z.shape[1],
        n_nk=len(label_encoder.classes_),
        n_dataset=len(dataset_encoder.classes_),
        n_assay=len(assay_encoder.classes_),
        hidden=64,
        lambda_dataset=args.lambda_dataset,
        lambda_assay=args.lambda_assay,
    ).to(device)

    loader = DataLoader(
        TensorDataset(
            torch.tensor(z[train_mask], dtype=torch.float32),
            torch.tensor(y_label[train_mask], dtype=torch.long),
            torch.tensor(y_dataset[train_mask], dtype=torch.long),
            torch.tensor(y_assay[train_mask], dtype=torch.long),
        ),
        batch_size=args.refiner_batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.refiner_lr, weight_decay=1e-4)

    for epoch in range(1, args.refiner_epochs + 1):
        model.train()
        total = 0.0
        n = 0
        for xb, yb_label, yb_dataset, yb_assay in loader:
            xb = xb.to(device)
            yb_label = yb_label.to(device)
            yb_dataset = yb_dataset.to(device)
            yb_assay = yb_assay.to(device)
            _, label_logits, dataset_logits, assay_logits = model(xb)
            loss = (
                F.cross_entropy(label_logits, yb_label)
                + F.cross_entropy(dataset_logits, yb_dataset)
                + F.cross_entropy(assay_logits, yb_assay)
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * xb.shape[0]
            n += xb.shape[0]
        if epoch == 1 or epoch % 10 == 0 or epoch == args.refiner_epochs:
            print(f"[REFINER] epoch={epoch:03d} loss={total / max(n, 1):.4f}")

    model.eval()
    with torch.no_grad():
        z_refined, label_logits, dataset_logits, assay_logits = model(
            torch.tensor(z, dtype=torch.float32).to(device)
        )
    z_refined = z_refined.cpu().numpy().astype(np.float32)
    pred_label = label_logits.argmax(dim=1).cpu().numpy()
    pred_dataset = dataset_logits.argmax(dim=1).cpu().numpy()
    pred_assay = assay_logits.argmax(dim=1).cpu().numpy()

    save_latent_npz(cache["latent"], X_REFINED=z_refined, obs_names=obs.index.astype(str).values)
    print(f"[SAVE] {cache['latent']}")

    val_mask = obs["_split"].astype(str).values == "Val"
    held_mask = obs["_split"].astype(str).values == "Held-out"
    row = {
        "strategy": "adversarial_refiner",
        "batch_key": "assay_only_latent + GRL refiner",
        "model_type": "posthoc_refiner",
        "val_macro_f1": macro_f1(y_label[val_mask], pred_label[val_mask]),
        "zeroshot_macro_f1": macro_f1(y_label[held_mask], pred_label[held_mask]),
        "full_macro_f1": macro_f1(y_label, pred_label),
        "val_weighted_f1": weighted_f1(y_label[val_mask], pred_label[val_mask]),
        "zeroshot_weighted_f1": weighted_f1(y_label[held_mask], pred_label[held_mask]),
        "full_weighted_f1": weighted_f1(y_label, pred_label),
        "probe_dataset_acc": float(np.mean(pred_dataset == y_dataset)),
        "probe_assay_acc": float(np.mean(pred_assay == y_assay)),
    }
    row.update(compute_latent_comparison_metrics(z_refined, obs, "adversarial_refiner", args))
    pd.DataFrame([row]).to_csv(cache["metrics"], index=False)
    print(f"[SAVE] {cache['metrics']}")
    return row


def macro_f1(y_true, y_pred):
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def weighted_f1(y_true, y_pred):
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))


def weighted_within_label_majority_baseline(batch_labels, labels):
    batch_labels = np.asarray(batch_labels).astype(str)
    labels = np.asarray(labels).astype(str)
    baselines = []
    weights = []
    for label in sorted(pd.unique(labels)):
        mask = labels == label
        if mask.sum() == 0:
            continue
        counts = pd.Series(batch_labels[mask]).value_counts()
        baselines.append(float(counts.iloc[0] / counts.sum()))
        weights.append(int(mask.sum()))
    return float(np.average(baselines, weights=weights)) if weights else np.nan


def batch_knn_mixing_score(knn_acc, baseline_acc):
    """Convert kNN batch predictability to 0-1 batch mixing success.

    1 means batch is no more predictable than the within-label majority
    baseline. 0 means batch is maximally predictable from neighbors.
    """
    if pd.isna(knn_acc) or pd.isna(baseline_acc):
        return np.nan
    if baseline_acc >= 1.0:
        return 1.0 if knn_acc <= baseline_acc else 0.0
    excess = max(0.0, float(knn_acc) - float(baseline_acc))
    return float(np.clip(1.0 - excess / (1.0 - float(baseline_acc)), 0.0, 1.0))


def add_normalized_scores(summary):
    out = summary.copy()
    for col in [
        "dataset_knn_batch_acc",
        "assay_knn_batch_acc",
        "dataset_knn_baseline_acc",
        "assay_knn_baseline_acc",
    ]:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out["dataset_knn_mixing"] = [
        batch_knn_mixing_score(acc, base)
        for acc, base in zip(out["dataset_knn_batch_acc"], out["dataset_knn_baseline_acc"])
    ]
    out["assay_knn_mixing"] = [
        batch_knn_mixing_score(acc, base)
        for acc, base in zip(out["assay_knn_batch_acc"], out["assay_knn_baseline_acc"])
    ]

    for col in BATCH_MIXING_METRICS + BIOLOGY_PRESERVATION_METRICS:
        if col in out:
            out[col] = pd.to_numeric(out[col], errors="coerce").clip(0.0, 1.0)

    return out.sort_index()


def plot_absolute_scores(summary, path):
    plot_cols = BATCH_MIXING_METRICS + BIOLOGY_PRESERVATION_METRICS
    plot_df = summary[plot_cols].copy()
    labels = [
        "dataset\nASW\nmixing",
        "assay\nASW\nmixing",
        "dataset\nkNN\nmixing",
        "assay\nkNN\nmixing",
        "NK_State\nASW",
        "NK_State\nkNN",
    ]

    fig, ax = plt.subplots(figsize=(14, max(4.5, 0.55 * len(plot_df))))
    im = ax.imshow(plot_df.values, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(plot_df.index)))
    ax.set_yticklabels(plot_df.index, fontsize=10)
    ax.set_title("Batch strategy comparison: direct metrics (0=bad, 1=good)")

    for i in range(plot_df.shape[0]):
        for j in range(plot_df.shape[1]):
            val = plot_df.iloc[i, j]
            ax.text(
                j,
                i,
                "" if pd.isna(val) else f"{val:.2f}",
                ha="center",
                va="center",
                color="white" if val < 0.55 else "black",
                fontsize=8,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("normalized score")
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
