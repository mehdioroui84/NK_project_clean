#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from math import ceil
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scarches as sca
import torch
import torch.nn as nn
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.io_utils import ensure_dirs


DEFAULT_REF_OUTDIR_NAME = "refined_scanvi_v1"
DEFAULT_MODEL_NAME = "scanvi_NK_State_refined_assay_clean_model"

DEFAULT_TARGET_STATES = [
    "Mature Cytotoxic",
    "Mature Cytotoxic TCF7+",
    "Lung Cytotoxic NK",
    "Lung DOCK4+ SLC8A1+ NK",
    "Transitional Cytotoxic Tissue-Resident",
    "Transitional Cytotoxic",
    "Cytokine-Stimulated CCR7+",
    "Cytokine-Stimulated Proliferative",
    "Proliferative",
    "Regulatory",
]
SANITY_TARGET_STATES = ["T", "B"]

BROAD_EXACT_GENES = {"MALAT1", "B2M", "TMSB4X", "HBB"}
BROAD_PREFIXES = ("MT-", "RPS", "RPL", "HBA")


class SCANVIClassifierWrapper(nn.Module):
    """Expose SCANVI as expression + batch_index -> classifier logits."""

    def __init__(self, scanvi_model, classifier_attr: str):
        super().__init__()
        self.module = scanvi_model.module
        self.classifier = getattr(self.module, classifier_attr)

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        inference_out = self.module.inference(
            x,
            batch_index=batch_index,
            n_samples=1,
        )
        qz = inference_out.get("qz")
        if hasattr(qz, "loc"):
            z = qz.loc
        elif "qz_m" in inference_out:
            z = inference_out["qz_m"]
        else:
            raise KeyError("Could not find deterministic q(z|x) mean in SCANVI inference output.")

        logits = self.classifier(z)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)

    ref_outdir = args.ref_outdir or os.path.join(cfg.BASE_OUTDIR, DEFAULT_REF_OUTDIR_NAME)
    model_dir = args.model_dir or os.path.join(ref_outdir, "models", DEFAULT_MODEL_NAME)
    input_h5ad = args.input_h5ad or os.path.join(
        cfg.BASE_OUTDIR,
        "refined_annotation_v1",
        "full_scvi_leiden_refined_v1.h5ad",
    )
    outdir = args.outdir or os.path.join(ref_outdir, "gene_attribution")
    if args.plot_existing and args.outdir is None:
        outdir = os.path.join(
            ref_outdir,
            "gene_attribution_filtered_biology" if args.filter_broad_genes else "gene_attribution_replot",
        )
    table_dir = os.path.join(outdir, "tables")
    fig_dir = os.path.join(outdir, "figures")
    ensure_dirs(outdir, table_dir, fig_dir)

    if args.plot_existing:
        existing_table = args.existing_table or os.path.join(
            ref_outdir,
            "gene_attribution",
            "tables",
            "gene_attribution_all_states_full.csv",
        )
        plot_existing_attribution_table(existing_table, outdir, table_dir, fig_dir, args)
        return

    obs_path = args.obs_csv or os.path.join(ref_outdir, "tables", "scanvi_full_obs_metadata.csv")
    proba_path = args.proba_csv or os.path.join(ref_outdir, "tables", "scanvi_full_probabilities.csv")
    train_names_path = args.train_names or os.path.join(ref_outdir, "tables", "train_obs_names.txt")

    print("=" * 80)
    print("Gene Attribution - Captum Integrated Gradients on refined SCANVI")
    print("=" * 80)
    print(f"[INPUT] {input_h5ad}")
    print(f"[MODEL] {model_dir}")
    print(f"[OBS] {obs_path}")
    print(f"[PROBA] {proba_path}")
    print(f"[TRAIN_NAMES] {train_names_path}")
    print(f"[OUTDIR] {outdir}")
    print(f"[LABEL] {args.label_key}")
    print(f"[BATCH] {args.batch_key}")

    obs, proba, adata = load_aligned_inputs(input_h5ad, obs_path, proba_path, train_names_path, args)

    model = load_scanvi_model(model_dir, adata, args)
    label_order = list(model.adata_manager.get_state_registry("labels").categorical_mapping)
    state_to_idx = {str(label): i for i, label in enumerate(label_order)}
    classifier_attr = detect_classifier_attr(model)
    wrapper = SCANVIClassifierWrapper(model, classifier_attr=classifier_attr)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    wrapper.to(device)
    wrapper.eval()
    print(f"[DEVICE] {device}")
    print(f"[CLASSIFIER] scanvi_model.module.{classifier_attr}")
    print("[LABEL_ORDER] " + ", ".join(f"{i}:{label}" for i, label in enumerate(label_order)))

    target_states = resolve_target_states(args, label_order, proba.columns)
    if not target_states:
        raise ValueError("No target states remain after filtering.")
    print("[TARGET_STATES] " + "; ".join(target_states))

    batch_indices = make_batch_indices(model, adata, args, device)
    method = choose_ig_method(args)
    if args.dry_run:
        print("[DRY-RUN] Eligible cells after correctness/confidence/split filters:")
        for state in target_states:
            selected = select_cells_for_state(obs, proba, state, args)
            n_used = min(len(selected), args.max_cells_per_state) if args.max_cells_per_state else len(selected)
            print(f"  {state:45s} eligible={len(selected):7,} used={n_used:7,}")
        print("[DRY-RUN] Configuration is valid; skipping attribution and output writing.")
        return

    baseline_vector = make_baseline_vector(adata, args).astype(np.float32)

    selected_rows = []
    ranked_tables = {}
    for state in target_states:
        if state not in state_to_idx:
            print(f"[SKIP] {state}: not present in model label order")
            continue
        if state not in proba.columns:
            print(f"[SKIP] {state}: not present in probability table")
            continue

        selected = select_cells_for_state(obs, proba, state, args)
        if selected.empty:
            print(f"[SKIP] {state}: no correctly predicted high-confidence cells")
            continue

        selected_positions = obs.index.get_indexer(selected.index)
        if args.max_cells_per_state and len(selected_positions) > args.max_cells_per_state:
            rng = np.random.default_rng(args.seed)
            selected_positions = np.sort(
                rng.choice(selected_positions, size=args.max_cells_per_state, replace=False)
            )
        selected = make_selected_cell_table(obs, proba, selected_positions, state, args)
        selected_rows.append(selected)

        print(
            f"[ATTR] {state}: target_idx={state_to_idx[state]} "
            f"cells={len(selected_positions):,} method={method}"
        )

        result = run_state_attribution(
            adata=adata,
            positions=selected_positions,
            batch_indices=batch_indices,
            baseline_vector=baseline_vector,
            wrapper=wrapper,
            target_idx=state_to_idx[state],
            method=method,
            args=args,
            device=device,
        )
        ranked = build_ranked_table(adata.var_names.astype(str).tolist(), result)
        ranked_tables[state] = ranked

        state_path = os.path.join(table_dir, f"gene_attribution_{safe_name(state)}.csv")
        ranked.to_csv(state_path, index=False)
        print(f"[SAVE] {state_path}")
        print(f"       top genes: {', '.join(ranked['gene'].head(args.top_n).tolist())}")

    if selected_rows:
        selected_df = pd.concat(selected_rows, axis=0)
        selected_df = selected_df[~selected_df.index.duplicated(keep="first")]
        selected_path = os.path.join(table_dir, "gene_attribution_selected_cells.csv")
        selected_df.to_csv(selected_path)
        print(f"[SAVE] {selected_path}")

    if not ranked_tables:
        raise ValueError("No attribution tables were produced. Relax filters or check target states.")

    plot_tables = select_plot_tables(ranked_tables, args)
    save_combined_tables(ranked_tables, table_dir, args.top_n)
    save_selected_plot_table(plot_tables, table_dir)
    plot_gene_selection_diagnostics(ranked_tables, plot_tables, fig_dir, args)
    plot_bar_per_state(plot_tables, fig_dir, None)
    plot_heatmap_and_dotplot(plot_tables, fig_dir, None, args)
    save_run_metadata(args, outdir, ref_outdir, model_dir, input_h5ad, obs_path, proba_path, train_names_path, target_states, method)
    print("[DONE] Gene attribution analysis complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute gene-level Integrated Gradients attribution for the refined-v1 "
            "SCANVI classifier head. Captum is used by default when available."
        )
    )
    parser.add_argument("--input-h5ad", default=None)
    parser.add_argument("--ref-outdir", default=None, help="Default: outputs/refined_scanvi_v1")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--obs-csv", default=None)
    parser.add_argument("--proba-csv", default=None)
    parser.add_argument("--train-names", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help="Skip model loading/IG and regenerate tables/figures from an existing full attribution CSV.",
    )
    parser.add_argument(
        "--existing-table",
        default=None,
        help="Default: outputs/refined_scanvi_v1/gene_attribution/tables/gene_attribution_all_states_full.csv",
    )
    parser.add_argument(
        "--filter-broad-genes",
        action="store_true",
        help="For plotting existing tables, remove broad/high-abundance genes before ranking/plotting.",
    )
    parser.add_argument("--label-key", default=cfg.REFINED_LABEL_KEY)
    parser.add_argument("--batch-key", default=cfg.PRODUCTION_BATCH_KEY)
    parser.add_argument(
        "--target-state",
        action="append",
        default=[],
        help="Target state to attribute. Can be repeated or comma-separated. Default: main NK refined labels.",
    )
    parser.add_argument(
        "--include-all-labels",
        action="store_true",
        help="Attribute every model label except the unlabeled category, unless --target-state is supplied.",
    )
    parser.add_argument(
        "--include-sanity-labels",
        action="store_true",
        help="Append T and B labels to the default target states as sanity-check classes.",
    )
    parser.add_argument(
        "--cell-split",
        default="all",
        choices=["all", "Train", "Val", "Held-out"],
        help="Which SCANVI split to attribute. Default: all evaluated cells.",
    )
    parser.add_argument("--min-proba", type=float, default=0.70)
    parser.add_argument("--max-cells-per-state", type=int, default=1000)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--ig-batch-size", type=int, default=128)
    parser.add_argument("--captum-internal-batch-size", type=int, default=256)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--gene-selection",
        choices=["top_n", "relative_to_top"],
        default="top_n",
        help=(
            "Genes to show in attribution figures. `top_n` preserves old behavior. "
            "`relative_to_top` keeps genes with mean_abs_attr >= fraction of the state's top gene."
        ),
    )
    parser.add_argument(
        "--relative-to-top-frac",
        type=float,
        default=0.01,
        help="Used with --gene-selection relative_to_top. Default: 0.01 = 1%% of top gene.",
    )
    parser.add_argument(
        "--min-genes-per-state",
        type=int,
        default=10,
        help="Minimum genes shown per state when using --gene-selection relative_to_top.",
    )
    parser.add_argument(
        "--max-genes-per-state",
        type=int,
        default=50,
        help="Maximum genes shown per state when using --gene-selection relative_to_top.",
    )
    parser.add_argument(
        "--heatmap-gene-order",
        choices=["input", "clustered", "max_state"],
        default="clustered",
        help=(
            "`input` preserves the selected gene order from each state's ranked list. "
            "`clustered` orders genes by hierarchical clustering. "
            "`max_state` groups genes by the state where they have maximum absolute attribution."
        ),
    )
    parser.add_argument("--baseline", choices=["zero", "gene_mean"], default="zero")
    parser.add_argument(
        "--include-unseen-batches",
        action="store_true",
        help=(
            "Do not filter out assay_clean categories absent from the SCANVI training split. "
            "The production reference model normally cannot load these categories directly."
        ),
    )
    parser.add_argument(
        "--fixed-batch-category",
        default=None,
        help=(
            "Optional sensitivity mode: force all cells to one assay_clean category. "
            "Default uses each cell's real assay_clean batch index."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["captum", "manual", "auto"],
        default="auto",
        help="Default auto uses Captum if installed, otherwise manual fallback.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load inputs and print selected cell counts without computing attribution.",
    )
    return parser.parse_args()


def plot_existing_attribution_table(
    existing_table: str,
    outdir: str,
    table_dir: str,
    fig_dir: str,
    args: argparse.Namespace,
) -> None:
    require_file(existing_table)
    print("=" * 80)
    print("Gene Attribution - replot existing table")
    print("=" * 80)
    print(f"[INPUT_TABLE] {existing_table}")
    print(f"[OUTDIR] {outdir}")
    print(f"[FILTER_BROAD_GENES] {args.filter_broad_genes}")

    full = pd.read_csv(existing_table)
    state_col = "NK_State_refined" if "NK_State_refined" in full.columns else "NK_state"
    required = {state_col, "gene", "mean_attr", "mean_abs_attr"}
    missing = sorted(required - set(full.columns))
    if missing:
        raise KeyError(f"Existing attribution table is missing required columns: {missing}")

    before = len(full)
    if args.filter_broad_genes:
        keep = ~full["gene"].astype(str).map(is_broad_gene)
        full = full.loc[keep].copy()
        removed = before - len(full)
        print(f"[FILTER] removed {removed:,} broad/high-abundance gene-state rows")

    if full.empty:
        raise ValueError("No rows remain after filtering.")

    ranked_tables = {}
    for state, sub in full.groupby(state_col, sort=False):
        sub = sub.sort_values("mean_abs_attr", ascending=False).reset_index(drop=True)
        sub["rank"] = np.arange(1, len(sub) + 1)
        sub["direction"] = np.where(sub["mean_attr"] >= 0, "positive", "negative")
        ranked_tables[str(state)] = sub.drop(columns=[state_col], errors="ignore")
    ranked_tables = reorder_existing_ranked_tables(ranked_tables, args)
    print("[STATE_ORDER] " + "; ".join(ranked_tables))

    plot_tables = select_plot_tables(ranked_tables, args)
    save_combined_tables(ranked_tables, table_dir, args.top_n)
    save_selected_plot_table(plot_tables, table_dir)
    plot_gene_selection_diagnostics(ranked_tables, plot_tables, fig_dir, args)
    plot_bar_per_state(plot_tables, fig_dir, None)
    plot_heatmap_and_dotplot(plot_tables, fig_dir, None, args)

    metadata = {
        "existing_table": existing_table,
        "filter_broad_genes": bool(args.filter_broad_genes),
        "broad_exact_genes": sorted(BROAD_EXACT_GENES),
        "broad_prefixes": list(BROAD_PREFIXES),
        "top_n": args.top_n,
        "gene_selection": args.gene_selection,
        "relative_to_top_frac": args.relative_to_top_frac,
        "min_genes_per_state": args.min_genes_per_state,
        "max_genes_per_state": args.max_genes_per_state,
        "heatmap_gene_order": args.heatmap_gene_order,
    }
    path = os.path.join(outdir, "gene_attribution_replot_config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[SAVE] {path}")
    print("[DONE] Existing attribution table replot complete.")


def reorder_existing_ranked_tables(
    ranked_tables: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> dict[str, pd.DataFrame]:
    """Apply the preferred display order when replotting a saved attribution table."""
    available = list(ranked_tables)
    if args.target_state:
        preferred = []
        for value in args.target_state:
            preferred.extend(part.strip() for part in value.split(",") if part.strip())
    else:
        preferred = DEFAULT_TARGET_STATES + SANITY_TARGET_STATES

    ordered = [state for state in preferred if state in ranked_tables]
    ordered.extend(state for state in available if state not in ordered)
    return {state: ranked_tables[state] for state in ordered}


def is_broad_gene(gene: str) -> bool:
    gene_upper = str(gene).upper()
    return gene_upper in BROAD_EXACT_GENES or gene_upper.startswith(BROAD_PREFIXES)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_aligned_inputs(
    input_h5ad: str,
    obs_path: str,
    proba_path: str,
    train_names_path: str,
    args: argparse.Namespace,
):
    require_file(input_h5ad)
    require_file(obs_path)
    require_file(proba_path)

    print(f"[LOAD] {obs_path}")
    obs = pd.read_csv(obs_path, index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)
    print(f"[LOAD] {proba_path}")
    proba = pd.read_csv(proba_path, index_col=0, low_memory=False)
    proba.index = proba.index.astype(str)

    if args.label_key not in obs.columns:
        raise KeyError(f"{args.label_key!r} is missing from {obs_path}")
    if args.batch_key not in obs.columns:
        raise KeyError(f"{args.batch_key!r} is missing from {obs_path}")

    common = obs.index.intersection(proba.index)
    if len(common) == 0:
        raise ValueError("No overlapping cells between obs metadata and probability table.")
    obs = obs.loc[common].copy()
    proba = proba.loc[common].copy()

    if not args.include_unseen_batches:
        require_file(train_names_path)
        print(f"[LOAD] {train_names_path}")
        train_names = pd.read_csv(train_names_path, header=None)[0].astype(str)
        train_common = obs.index.intersection(train_names)
        if len(train_common) == 0:
            raise ValueError("No overlapping training cells found in obs metadata and train_obs_names.txt.")
        known_batches = set(obs.loc[train_common, args.batch_key].astype(str).dropna())
        keep_known_batch = obs[args.batch_key].astype(str).isin(known_batches)
        before = len(obs)
        obs = obs.loc[keep_known_batch].copy()
        proba = proba.loc[obs.index].copy()
        print(
            "[FILTER] known training assay_clean categories only: "
            f"{before:,} -> {len(obs):,} cells"
        )
        print("[KNOWN_BATCHES] " + "; ".join(sorted(known_batches)))

    print(f"[LOAD] {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs_names_make_unique()
    missing = common.difference(adata.obs_names.astype(str))
    if len(missing) > 0:
        raise ValueError(f"{len(missing):,} evaluated cells are missing from {input_h5ad}")
    adata = adata[obs.index].copy()
    adata.obs = obs.copy()
    adata.obs[args.label_key] = adata.obs[args.label_key].astype("category")
    adata.obs[args.batch_key] = adata.obs[args.batch_key].astype(str).astype("category")
    print(f"[ALIGNED] {adata.n_obs:,} cells x {adata.n_vars:,} genes")
    return obs, proba, adata


def load_scanvi_model(model_dir: str, adata, args: argparse.Namespace):
    require_dir(model_dir)
    print(f"[LOAD] {model_dir}")
    model = sca.models.SCANVI.load(model_dir, adata=adata)
    model.module.eval()
    return model


def detect_classifier_attr(model) -> str:
    for attr in ["classifier", "classifier_cls", "_classifier", "cls_head"]:
        if hasattr(model.module, attr) and callable(getattr(model.module, attr)):
            return attr
    children = ", ".join(name for name, _ in model.module.named_children())
    raise RuntimeError(f"Could not find SCANVI classifier head. Module children: {children}")


def resolve_target_states(args: argparse.Namespace, label_order: list[str], proba_columns: Iterable[str]) -> list[str]:
    if args.target_state:
        raw = []
        for value in args.target_state:
            raw.extend(part.strip() for part in value.split(",") if part.strip())
        target_states = raw
    elif args.include_all_labels:
        target_states = [str(label) for label in label_order if str(label) != cfg.UNLABELED_CATEGORY]
    else:
        target_states = [state for state in DEFAULT_TARGET_STATES if state in label_order]
        if args.include_sanity_labels:
            for state in SANITY_TARGET_STATES:
                if state in label_order and state not in target_states:
                    target_states.append(state)

    available = set(map(str, label_order)).intersection(map(str, proba_columns))
    missing = [state for state in target_states if state not in available]
    if missing:
        print("[WARN] Missing target states will be skipped: " + "; ".join(missing))
    return [state for state in target_states if state in available]


def make_batch_indices(model, adata, args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    batch_state = model.adata_manager.get_state_registry("batch")
    batch_order = [str(x) for x in batch_state.categorical_mapping]
    batch_to_idx = {batch: i for i, batch in enumerate(batch_order)}
    print("[BATCH_ORDER] " + ", ".join(f"{i}:{batch}" for i, batch in enumerate(batch_order)))

    if args.fixed_batch_category:
        fixed = str(args.fixed_batch_category)
        if fixed not in batch_to_idx:
            raise ValueError(f"Fixed batch category {fixed!r} is not in model batch order: {batch_order}")
        indices = np.full(adata.n_obs, batch_to_idx[fixed], dtype=np.int64)
        print(f"[BATCH_MODE] fixed category: {fixed}")
    else:
        values = adata.obs[args.batch_key].astype(str)
        unknown = sorted(set(values) - set(batch_to_idx))
        if unknown:
            raise ValueError(
                f"Found assay/batch values not known to the SCANVI model: {unknown}. "
                "Use --fixed-batch-category only for a deliberate sensitivity run."
            )
        indices = values.map(batch_to_idx).to_numpy(dtype=np.int64)
        print("[BATCH_MODE] per-cell real batch indices from assay_clean")

    return torch.tensor(indices[:, None], dtype=torch.long, device=device)


def choose_ig_method(args: argparse.Namespace) -> str:
    if args.method == "manual":
        return "manual"
    try:
        import captum  # noqa: F401

        print("[METHOD] Captum IntegratedGradients")
        return "captum"
    except ImportError:
        if args.method == "captum":
            raise ImportError(
                "Captum is not installed. Install with `python -m pip install captum`, "
                "or rerun with --method manual."
            )
        print("[METHOD] Captum not installed; using manual Integrated Gradients fallback")
        return "manual"


def make_baseline_vector(adata, args: argparse.Namespace) -> np.ndarray:
    if args.baseline == "zero":
        return np.zeros(adata.n_vars, dtype=np.float32)
    mean = adata.X.mean(axis=0)
    if sparse.issparse(mean):
        mean = mean.toarray()
    return np.asarray(mean).ravel().astype(np.float32)


def select_cells_for_state(obs: pd.DataFrame, proba: pd.DataFrame, state: str, args: argparse.Namespace) -> pd.DataFrame:
    pred = proba.idxmax(axis=1).astype(str)
    confidence = proba.max(axis=1).astype(float)
    mask = (
        obs[args.label_key].astype(str).eq(state)
        & pred.eq(state)
        & confidence.ge(args.min_proba)
    )
    if args.cell_split != "all":
        if "_split" not in obs.columns:
            raise KeyError("Requested --cell-split but `_split` is missing from scanvi_full_obs_metadata.csv")
        mask &= obs["_split"].astype(str).eq(args.cell_split)

    out = obs.loc[mask].copy()
    if out.empty:
        return out
    out["target_state"] = state
    out["pred_label"] = pred.loc[out.index]
    out["confidence"] = confidence.loc[out.index]
    return out


def make_selected_cell_table(
    obs: pd.DataFrame,
    proba: pd.DataFrame,
    positions: np.ndarray,
    state: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    out = obs.iloc[positions].copy()
    out["target_state"] = state
    out["pred_label"] = proba.loc[out.index].idxmax(axis=1).astype(str)
    out["confidence"] = proba.loc[out.index].max(axis=1).astype(float)
    out["target_probability"] = proba.loc[out.index, state].astype(float)
    out["attribution_cell_split"] = args.cell_split
    return out


def run_state_attribution(
    *,
    adata,
    positions: np.ndarray,
    batch_indices: torch.Tensor,
    baseline_vector: np.ndarray,
    wrapper: SCANVIClassifierWrapper,
    target_idx: int,
    method: str,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, np.ndarray | int]:
    attr_sum = np.zeros(adata.n_vars, dtype=np.float64)
    abs_sum = np.zeros(adata.n_vars, dtype=np.float64)
    n_total = 0

    for start in range(0, len(positions), args.ig_batch_size):
        end = min(start + args.ig_batch_size, len(positions))
        chunk_pos = positions[start:end]
        x = dense_chunk(adata, chunk_pos)
        x_tensor = torch.tensor(x, dtype=torch.float32, device=device)
        batch_tensor = batch_indices[chunk_pos]
        baseline = torch.tensor(baseline_vector[None, :], dtype=torch.float32, device=device).expand_as(x_tensor)

        if method == "captum":
            attrs = integrated_gradients_captum(
                wrapper,
                x_tensor,
                batch_tensor,
                baseline,
                target_idx,
                args.ig_steps,
                args.captum_internal_batch_size,
            )
        else:
            attrs = integrated_gradients_manual(
                wrapper,
                x_tensor,
                batch_tensor,
                baseline,
                target_idx,
                args.ig_steps,
            )

        attrs_np = attrs.detach().cpu().numpy()
        attr_sum += attrs_np.sum(axis=0)
        abs_sum += np.abs(attrs_np).sum(axis=0)
        n_total += attrs_np.shape[0]
        print(f"       chunk {start:,}-{end:,} / {len(positions):,}", flush=True)

        del x_tensor, batch_tensor, baseline, attrs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "mean_attr": attr_sum / max(n_total, 1),
        "mean_abs_attr": abs_sum / max(n_total, 1),
        "n_cells": n_total,
    }


def integrated_gradients_captum(
    wrapper: SCANVIClassifierWrapper,
    x: torch.Tensor,
    batch_index: torch.Tensor,
    baseline: torch.Tensor,
    target_idx: int,
    n_steps: int,
    internal_batch_size: int,
) -> torch.Tensor:
    from captum.attr import IntegratedGradients

    ig = IntegratedGradients(wrapper)
    attrs = ig.attribute(
        inputs=x,
        baselines=baseline,
        target=target_idx,
        additional_forward_args=(batch_index,),
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )
    return attrs


def integrated_gradients_manual(
    wrapper: SCANVIClassifierWrapper,
    x: torch.Tensor,
    batch_index: torch.Tensor,
    baseline: torch.Tensor,
    target_idx: int,
    n_steps: int,
) -> torch.Tensor:
    alphas = torch.linspace(0.0, 1.0, n_steps, device=x.device)
    grad_sum = torch.zeros_like(x)
    for alpha in alphas:
        x_interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        logits = wrapper(x_interp, batch_index)
        target_logits = logits[:, target_idx].sum()
        wrapper.zero_grad(set_to_none=True)
        target_logits.backward()
        grad_sum += x_interp.grad.detach()
    return (x - baseline) * grad_sum / n_steps


def dense_chunk(adata, positions: np.ndarray) -> np.ndarray:
    x = adata.X[positions]
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def build_ranked_table(gene_names: list[str], result: dict[str, np.ndarray | int]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "gene": gene_names,
            "mean_attr": result["mean_attr"],
            "mean_abs_attr": result["mean_abs_attr"],
        }
    )
    df = df.sort_values("mean_abs_attr", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    df["n_cells"] = int(result["n_cells"])
    df["direction"] = np.where(df["mean_attr"] >= 0, "positive", "negative")
    return df


def save_combined_tables(ranked_tables: dict[str, pd.DataFrame], table_dir: str, top_n: int) -> None:
    full_rows = []
    top_rows = []
    wide = None

    for state, df in ranked_tables.items():
        state_df = df.copy()
        state_df.insert(0, "NK_State_refined", state)
        full_rows.append(state_df)

        top = df.head(top_n).copy()
        top.insert(0, "NK_State_refined", state)
        top_rows.append(top)

        cols = df[["gene", "mean_attr", "mean_abs_attr", "rank"]].rename(
            columns={
                "mean_attr": f"{safe_name(state)}_mean_attr",
                "mean_abs_attr": f"{safe_name(state)}_mean_abs_attr",
                "rank": f"{safe_name(state)}_rank",
            }
        )
        wide = cols if wide is None else wide.merge(cols, on="gene", how="outer")

    full = pd.concat(full_rows, ignore_index=True)
    top = pd.concat(top_rows, ignore_index=True)
    rank_cols = [col for col in wide.columns if col.endswith("_rank")]
    wide["mean_rank_across_states"] = wide[rank_cols].mean(axis=1)
    wide = wide.sort_values("mean_rank_across_states").reset_index(drop=True)

    for name, df in [
        ("gene_attribution_all_states_full.csv", full),
        (f"gene_attribution_top{top_n}_per_state.csv", top),
        ("gene_attribution_summary_wide.csv", wide),
    ]:
        path = os.path.join(table_dir, name)
        df.to_csv(path, index=False)
        print(f"[SAVE] {path}")


def select_plot_tables(ranked_tables: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    selected = {}
    for state, df in ranked_tables.items():
        ordered = df.sort_values("mean_abs_attr", ascending=False).reset_index(drop=True)
        if args.gene_selection == "relative_to_top":
            top_value = float(ordered["mean_abs_attr"].iloc[0]) if len(ordered) else 0.0
            threshold = args.relative_to_top_frac * top_value
            n_threshold = int((ordered["mean_abs_attr"] >= threshold).sum()) if top_value > 0 else 0
            n_select = max(args.min_genes_per_state, n_threshold)
            n_select = min(args.max_genes_per_state, n_select, len(ordered))
            chosen = ordered.head(n_select).copy()
            chosen["plot_selection_method"] = "relative_to_top"
            chosen["plot_selection_threshold"] = threshold
        else:
            chosen = ordered.head(args.top_n).copy()
            chosen["plot_selection_method"] = "top_n"
            chosen["plot_selection_threshold"] = np.nan
        chosen["plot_n_genes_for_state"] = len(chosen)
        selected[state] = chosen
        print(f"[PLOT_GENES] {state}: {len(chosen)} genes")
    return selected


def save_selected_plot_table(plot_tables: dict[str, pd.DataFrame], table_dir: str) -> None:
    rows = []
    for state, df in plot_tables.items():
        out = df.copy()
        out.insert(0, "NK_State_refined", state)
        rows.append(out)
    path = os.path.join(table_dir, "gene_attribution_selected_plot_genes.csv")
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)
    print(f"[SAVE] {path}")


def plot_gene_selection_diagnostics(
    ranked_tables: dict[str, pd.DataFrame],
    plot_tables: dict[str, pd.DataFrame],
    fig_dir: str,
    args: argparse.Namespace,
) -> None:
    n_states = len(ranked_tables)
    n_cols = min(3, n_states)
    n_rows = ceil(n_states / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    summary_rows = []
    for ax, (state, df) in zip(axes.ravel(), ranked_tables.items()):
        ax.axis("on")
        values = df.sort_values("mean_abs_attr", ascending=False)["mean_abs_attr"].to_numpy(dtype=float)
        ranks = np.arange(1, len(values) + 1)
        selected_n = len(plot_tables[state])
        top_value = values[0] if len(values) else 0.0
        threshold = args.relative_to_top_frac * top_value if args.gene_selection == "relative_to_top" else np.nan
        cumulative = np.cumsum(values) / values.sum() if values.sum() > 0 else np.zeros_like(values)
        selected_mass = cumulative[selected_n - 1] if selected_n > 0 and len(cumulative) else np.nan

        ax.plot(ranks, values, color="#4c78a8", linewidth=1.6)
        ax.axvline(selected_n, color="#b23a48", linestyle="--", linewidth=1.2)
        if args.gene_selection == "relative_to_top":
            ax.axhline(threshold, color="#777777", linestyle=":", linewidth=1.0)
        ax.scatter([selected_n], [values[selected_n - 1]], color="#b23a48", s=18, zorder=3)
        ax.set_title(f"{state}\nselected={selected_n}, mass={selected_mass:.2f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Gene rank by mean |attribution|", fontsize=8)
        ax.set_ylabel("Mean |attribution|", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)

        summary_rows.append(
            {
                "NK_State_refined": state,
                "n_available_genes": len(values),
                "n_selected_genes": selected_n,
                "top_mean_abs_attr": top_value,
                "selection_threshold": threshold,
                "selected_cumulative_attr_fraction": selected_mass,
                "selection_method": args.gene_selection,
            }
        )

    fig.suptitle(
        "Attribution gene-selection diagnostic: 1% of top gene, capped at 50",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, fig_dir, "gene_attribution_gene_selection_diagnostic")

    summary_path = os.path.join(os.path.dirname(fig_dir), "tables", "gene_attribution_gene_selection_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[SAVE] {summary_path}")


def plot_bar_per_state(ranked_tables: dict[str, pd.DataFrame], fig_dir: str, top_n: int | None) -> None:
    n_states = len(ranked_tables)
    n_cols = min(3, n_states)
    n_rows = ceil(n_states / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 4.2 * n_rows), squeeze=False)

    for ax in axes.ravel():
        ax.axis("off")

    for ax, (state, df) in zip(axes.ravel(), ranked_tables.items()):
        ax.axis("on")
        top = df if top_n is None else df.head(top_n)
        top = top.sort_values("mean_attr", ascending=True)
        colors = np.where(top["mean_attr"].values >= 0, "#b23a48", "#457b9d")
        ax.barh(top["gene"], top["mean_attr"], color=colors, alpha=0.88)
        ax.axvline(0, color="#222222", linewidth=0.8)
        ax.set_title(state, fontsize=10, fontweight="bold")
        ax.set_xlabel("Mean Integrated Gradients attribution")
        ax.tick_params(axis="y", labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)

    title_n = "selected" if top_n is None else f"top {top_n}"
    fig.suptitle(f"{title_n.capitalize()} classifier-attributed genes per refined NK state", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, fig_dir, "gene_attribution_bar_per_state")


def plot_heatmap_and_dotplot(
    ranked_tables: dict[str, pd.DataFrame],
    fig_dir: str,
    top_n: int | None,
    args: argparse.Namespace,
) -> None:
    union_genes = []
    for df in ranked_tables.values():
        genes = df["gene"].tolist() if top_n is None else df["gene"].head(top_n).tolist()
        union_genes.extend(genes)
    union_genes = list(dict.fromkeys(union_genes))
    states = list(ranked_tables)

    mean_attr = pd.DataFrame(index=states, columns=union_genes, dtype=float)
    mean_abs = pd.DataFrame(index=states, columns=union_genes, dtype=float)
    for state, df in ranked_tables.items():
        attr_map = dict(zip(df["gene"], df["mean_attr"]))
        abs_map = dict(zip(df["gene"], df["mean_abs_attr"]))
        mean_attr.loc[state] = [attr_map.get(gene, 0.0) for gene in union_genes]
        mean_abs.loc[state] = [abs_map.get(gene, 0.0) for gene in union_genes]

    mean_attr = order_heatmap_columns(mean_attr, mode=args.heatmap_gene_order)
    mean_abs = mean_abs[mean_attr.columns]
    vmax = float(np.nanmax(np.abs(mean_attr.values)))
    vmax = vmax if vmax > 0 else 1.0

    fig_w = max(12, 0.38 * len(mean_attr.columns))
    fig_h = max(4, 0.45 * len(states))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mean_attr.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.grid(False, which="both")
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticks(np.arange(len(mean_attr.columns)))
    ax.set_xticklabels(mean_attr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(states)))
    ax.set_yticklabels(states, fontsize=9, fontweight="bold")
    title_n = "selected" if top_n is None else f"top {top_n}"
    ax.set_title(f"Integrated Gradients heatmap: {title_n} genes per state", fontsize=12, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cbar.set_label("Mean attribution")
    fig.tight_layout()
    save_figure(fig, fig_dir, "gene_attribution_heatmap")

    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h + 0.8))
    max_mag = float(np.nanmax(mean_abs.values)) or 1.0
    for yi, state in enumerate(states):
        for xi, gene in enumerate(mean_attr.columns):
            attr = float(mean_attr.loc[state, gene])
            mag = float(mean_abs.loc[state, gene])
            size = 20 + 340 * mag / max_mag
            color = matplotlib.cm.RdBu_r((attr + vmax) / (2 * vmax))
            ax2.scatter(xi, yi, s=size, color=color, edgecolors="white", linewidths=0.3)
    ax2.set_xticks(np.arange(len(mean_attr.columns)))
    ax2.set_xticklabels(mean_attr.columns, rotation=45, ha="right", fontsize=7)
    ax2.set_yticks(np.arange(len(states)))
    ax2.set_yticklabels(states, fontsize=9, fontweight="bold")
    ax2.set_xlim(-0.5, len(mean_attr.columns) - 0.5)
    ax2.set_ylim(-0.5, len(states) - 0.5)
    ax2.invert_yaxis()
    ax2.grid(alpha=0.25, linewidth=0.3)
    ax2.spines[["top", "right", "bottom", "left"]].set_visible(False)
    sm = matplotlib.cm.ScalarMappable(cmap="RdBu_r", norm=matplotlib.colors.Normalize(vmin=-vmax, vmax=vmax))
    sm.set_array([])
    cbar2 = fig2.colorbar(sm, ax=ax2, fraction=0.015, pad=0.01)
    cbar2.set_label("Mean attribution")
    ax2.set_title(f"Integrated Gradients dot plot: {title_n} genes per state", fontsize=12, fontweight="bold")
    fig2.tight_layout()
    save_figure(fig2, fig_dir, "gene_attribution_dotplot")


def order_heatmap_columns(df: pd.DataFrame, mode: str = "clustered") -> pd.DataFrame:
    if df.shape[1] <= 2:
        return df
    if mode == "input":
        return df
    if mode == "max_state":
        abs_df = df.abs()
        state_order = {state: i for i, state in enumerate(df.index)}
        sort_rows = []
        for gene in df.columns:
            max_state = abs_df[gene].idxmax()
            sort_rows.append(
                {
                    "gene": gene,
                    "state_order": state_order[max_state],
                    "max_abs_attr": float(abs_df.loc[max_state, gene]),
                    "signed_attr": float(df.loc[max_state, gene]),
                }
            )
        order = (
            pd.DataFrame(sort_rows)
            .sort_values(["state_order", "max_abs_attr", "signed_attr", "gene"], ascending=[True, False, False, True])
            ["gene"]
            .tolist()
        )
        return df.loc[:, order]
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage

        order = leaves_list(linkage(df.values.T, method="ward", metric="euclidean"))
        return df.iloc[:, order]
    except Exception as exc:
        print(f"[WARN] Column clustering failed, preserving gene order: {exc}")
        return df


def save_figure(fig, fig_dir: str, stem: str) -> None:
    png = os.path.join(fig_dir, f"{stem}.png")
    pdf = os.path.join(fig_dir, f"{stem}.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")


def save_run_metadata(
    args: argparse.Namespace,
    outdir: str,
    ref_outdir: str,
    model_dir: str,
    input_h5ad: str,
    obs_path: str,
    proba_path: str,
    train_names_path: str,
    target_states: list[str],
    method: str,
) -> None:
    metadata = vars(args).copy()
    metadata.update(
        {
            "ref_outdir_resolved": ref_outdir,
            "model_dir_resolved": model_dir,
            "input_h5ad_resolved": input_h5ad,
            "obs_csv_resolved": obs_path,
            "proba_csv_resolved": proba_path,
            "train_names_resolved": train_names_path,
            "target_states_resolved": target_states,
            "ig_method_resolved": method,
        }
    )
    path = os.path.join(outdir, "gene_attribution_run_config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[SAVE] {path}")


def require_file(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)


def require_dir(path: str) -> None:
    if not os.path.isdir(path):
        raise FileNotFoundError(path)


def safe_name(value: str) -> str:
    out = str(value)
    for old, new in [
        (" ", "_"),
        ("/", "_"),
        ("\\", "_"),
        ("|", "_"),
        (":", "_"),
        ("'", ""),
        ('"', ""),
        ("+", "plus"),
    ]:
        out = out.replace(old, new)
    return out


if __name__ == "__main__":
    main()
