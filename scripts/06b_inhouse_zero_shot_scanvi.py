#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scanpy as sc
import scarches as sca
from scipy import sparse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.evaluation.scanvi_full_plots import (
    PREFERRED_STATE_COLORS,
    distinct_color_map,
)
from nk_project.io_utils import ensure_dirs, save_latent_npz
from nk_project.plot_style import (
    LEGEND_FONT_SIZE,
    SMALL_TICK_LABEL_SIZE,
    set_presentation_style,
    style_all_legends,
    style_axis,
    style_figure,
    style_legend,
)

set_presentation_style()


MAX_LEGEND_LABEL_CHARS = 42


def main() -> None:
    args = parse_args()
    table_dir = os.path.join(args.outdir, "tables")
    fig_dir = os.path.join(args.outdir, "figures")
    latent_dir = os.path.join(args.outdir, "latents")
    ensure_dirs(args.outdir, table_dir, fig_dir, latent_dir)

    if args.plot_existing:
        plot_existing_outputs(args, table_dir, fig_dir, latent_dir)
        print("[DONE] Existing in-house zero-shot plots regenerated.")
        return

    validate_inference_args(args)

    print(f"[INHOUSE] {args.inhouse_h5ad}")
    query_raw = load_inhouse_query(args)
    print(f"[QUERY] {query_raw.n_obs:,} cells x {query_raw.n_vars:,} genes")
    print("[QUERY CB COUNTS]")
    print(query_raw.obs[args.cb_key].astype(str).value_counts().to_string())

    print(f"[REF] {args.ref_h5ad}")
    ref = sc.read_h5ad(args.ref_h5ad)
    ref.obs_names_make_unique()
    ref.var_names_make_unique()
    ref_model = subset_reference_to_model_cells(ref, args)

    query = align_to_target_genes(query_raw, ref.var_names)
    prepare_query_obs(query, args)

    print(f"[MODEL] {args.model_dir}")
    if not args.query_only_umap:
        ref_plot, z_ref_plot = load_reference_latent_background(args)
    else:
        ref_plot = None
        z_ref_plot = None

    model = sca.models.SCANVI.load(args.model_dir, adata=ref_model)
    sca.models.SCANVI.prepare_query_anndata(query, model)
    new_manager = model.adata_manager.transfer_fields(query, extend_categories=True)
    model._register_manager_for_instance(new_manager)

    print("[PREDICT] in-house query")
    proba = model.predict(query, soft=True)
    summary = probability_summary(proba)
    for col in summary.columns:
        query.obs[col] = summary[col].reindex(query.obs_names).values
    z_query = model.get_latent_representation(query)

    save_latent_npz(
        os.path.join(latent_dir, f"{output_prefix(args)}_latents.npz"),
        X_SCANVI=z_query,
        obs_names=query.obs_names.astype(str).values,
    )
    query_h5ad = os.path.join(table_dir, f"{output_prefix(args)}_query_aligned.h5ad")
    query.write_h5ad(query_h5ad)
    print(f"[SAVE] {query_h5ad}")
    save_tables(query, proba, summary, args, table_dir)

    print("[PLOT] building query/reference UMAP panels")
    if args.query_only_umap:
        xy, plot_obs, is_query = build_query_only_umap(query, z_query, args)
    else:
        xy, plot_obs, is_query = build_reference_query_umap(ref_plot, query, z_ref_plot, z_query, args)
    save_umap_plot_payload(xy, plot_obs, is_query, args, table_dir, latent_dir)
    plot_panels(xy, plot_obs, is_query, args, fig_dir)
    plot_prediction_proportions(query.obs.copy(), args, fig_dir, table_dir)

    print("[DONE] In-house zero-shot complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply a trained SCANVI model to in-house samples excluding CB07 and make QC plots."
    )
    parser.add_argument("--ref-h5ad", default=None, help="Reference AnnData used to train the SCANVI model.")
    parser.add_argument("--model-dir", default=None, help="Trained SCANVI model directory.")
    parser.add_argument(
        "--inhouse-h5ad",
        default="/rsrch5/home/genomic_med/suorouji/projects/lsf_run/seurat_manual.h5ad",
        help="In-house AnnData. Default: old in-house seurat_manual.h5ad.",
    )
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--exclude-cb-id", default="CB07")
    parser.add_argument(
        "--include-excluded-cb",
        action="store_true",
        help=(
            "Include --exclude-cb-id in the in-house query. Default excludes it for "
            "true zero-shot; this option is for all-cord-blood comparison plots."
        ),
    )
    parser.add_argument("--cb-key", default="cb_id")
    parser.add_argument("--timepoint-key", default="timepoint")
    parser.add_argument("--sample-id-key", default="sample_id")
    parser.add_argument(
        "--quality-key",
        default=None,
        help=(
            "Deprecated placeholder for older sample-quality plotting. The current "
            "in-house panel uses --sample-id-key instead."
        ),
    )
    parser.add_argument("--true-label-key", default="functional_state")
    parser.add_argument("--model-label-key", default=cfg.REFINED_LABEL_KEY)
    parser.add_argument("--dataset-key", default=cfg.DATASET_KEY)
    parser.add_argument("--assay-clean-key", default=cfg.ASSAY_CLEAN_KEY)
    parser.add_argument("--unlabeled-category", default=cfg.UNLABELED_CATEGORY)
    parser.add_argument("--max-ref-cells", type=int, default=80000)
    parser.add_argument("--max-query-cells", type=int, default=None)
    parser.add_argument("--query-only-umap", action="store_true")
    parser.add_argument(
        "--plot-existing",
        action="store_true",
        help=(
            "Regenerate figures/tables from existing prediction outputs in --outdir. "
            "Uses cached UMAP plot coordinates when available; otherwise rebuilds UMAP "
            "from saved latents without rerunning SCANVI inference."
        ),
    )
    parser.add_argument("--seed", type=int, default=cfg.SEED)
    return parser.parse_args()


def validate_inference_args(args: argparse.Namespace) -> None:
    missing = []
    if not args.ref_h5ad:
        missing.append("--ref-h5ad")
    if not args.model_dir:
        missing.append("--model-dir")
    if missing:
        raise ValueError(f"Inference mode requires: {', '.join(missing)}")


def load_inhouse_query(args: argparse.Namespace) -> sc.AnnData:
    inhouse = sc.read_h5ad(args.inhouse_h5ad)
    inhouse.obs_names_make_unique()
    inhouse.var_names_make_unique()
    if args.cb_key not in inhouse.obs:
        raise KeyError(f"In-house AnnData is missing obs[{args.cb_key!r}].")
    if args.include_excluded_cb:
        mask = np.ones(inhouse.n_obs, dtype=bool)
        print(f"[QUERY_MODE] including all in-house cells, including {args.exclude_cb_id}")
    else:
        mask = inhouse.obs[args.cb_key].astype(str).values != str(args.exclude_cb_id)
        print(f"[QUERY_MODE] excluding {args.exclude_cb_id} for zero-shot query")
    query = inhouse[mask].copy()
    query.obs_names_make_unique()
    query.var_names_make_unique()
    return query


def model_run_outdir(model_dir: str) -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(model_dir)))


def subset_reference_to_model_cells(ref: sc.AnnData, args: argparse.Namespace) -> sc.AnnData:
    obs_path = os.path.join(model_run_outdir(args.model_dir), "tables", "scanvi_full_obs_metadata.csv")
    if not os.path.exists(obs_path):
        print(f"[WARN] Model obs metadata not found; loading model with full reference: {obs_path}")
        return remove_unused_categories(ref.copy())

    model_obs = pd.read_csv(obs_path, index_col=0, low_memory=False)
    model_obs.index = model_obs.index.astype(str)
    if "_split" in model_obs.columns:
        model_obs = model_obs.loc[model_obs["_split"].astype(str).eq("Train")].copy()
        print(f"[REF MODEL] loading model with Train split only: {len(model_obs):,} cells")
    else:
        print("[WARN] _split not found in model obs metadata; using all model metadata cells.")
    model_names = model_obs.index.astype(str).tolist()
    available = set(ref.obs_names.astype(str))
    keep = [name for name in model_names if name in available]
    if not keep:
        raise ValueError(
            "No overlap between model post-QC obs metadata and reference h5ad obs names:\n"
            f"  {obs_path}\n"
            f"  {args.ref_h5ad}"
        )
    if len(keep) < len(model_names):
        print(f"[WARN] Reference h5ad is missing {len(model_names) - len(keep):,} model cells.")
    out = ref[keep].copy()
    out.obs_names_make_unique()
    out.var_names_make_unique()
    out = remove_unused_categories(out)
    print(f"[REF MODEL] using {out.n_obs:,} post-QC model cells")
    return out


def remove_unused_categories(adata: sc.AnnData) -> sc.AnnData:
    for col in adata.obs.columns:
        if pd.api.types.is_categorical_dtype(adata.obs[col]):
            adata.obs[col] = adata.obs[col].cat.remove_unused_categories()
    return adata


def prepare_query_obs(query: sc.AnnData, args: argparse.Namespace) -> None:
    query.obs[args.dataset_key] = query.obs[args.cb_key].astype(str).values
    if args.assay_clean_key not in query.obs:
        if cfg.ASSAY_KEY in query.obs:
            query.obs[args.assay_clean_key] = query.obs[cfg.ASSAY_KEY].astype(str).values
        else:
            query.obs[args.assay_clean_key] = "inhouse"
    query.obs[args.assay_clean_key] = (
        query.obs[args.assay_clean_key].astype(str).replace({"nan": "inhouse", "None": "inhouse", "": "inhouse"})
    )
    if args.true_label_key in query.obs:
        query.obs["true_label"] = query.obs[args.true_label_key].astype(str).values
    else:
        query.obs["true_label"] = args.unlabeled_category
    query.obs[args.model_label_key] = args.unlabeled_category
    query.obs[args.model_label_key] = query.obs[args.model_label_key].astype("category")
    if args.unlabeled_category not in query.obs[args.model_label_key].cat.categories:
        query.obs[args.model_label_key] = query.obs[args.model_label_key].cat.add_categories(
            [args.unlabeled_category]
        )
    query.obs["_sample_group"] = sample_group_from_obs(query.obs, args.sample_id_key)
    print(f"[SAMPLE_GROUP] using last token of obs[{args.sample_id_key!r}] when available")


def find_quality_key(obs: pd.DataFrame, requested: str | None) -> str | None:
    if requested:
        if requested not in obs.columns:
            print(f"[WARN] Ignoring unknown --quality-key {requested!r}; current plot uses sample_id-derived group.")
            return None
        return requested
    candidates = []
    for col in obs.columns:
        low = str(col).lower()
        if ("good" in low and "bad" in low) or "quality" in low or "sample_status" in low:
            candidates.append(col)
    for col in candidates:
        values = set(obs[col].astype(str).str.lower().dropna().unique())
        if any("good" in v for v in values) or any("bad" in v for v in values):
            return col
    return candidates[0] if candidates else None


def sample_group_from_obs(obs: pd.DataFrame, sample_id_key: str) -> pd.Series:
    if sample_id_key in obs:
        sample_id = obs[sample_id_key].astype(str)
        group = sample_id.str.rsplit("_", n=1).str[-1]
        group = group.replace({"nan": "NA", "None": "NA", "": "NA"})
        return group
    if "genotype" in obs:
        return obs["genotype"].astype(str).replace({"nan": "NA", "None": "NA", "": "NA"})
    return pd.Series("NA", index=obs.index)


def align_to_target_genes(adata_in: sc.AnnData, target_genes: pd.Index) -> sc.AnnData:
    target_genes = pd.Index(target_genes.astype(str))
    target_upper = target_genes.str.upper()

    adata_in = adata_in.copy()
    adata_in.var_names = adata_in.var_names.astype(str)
    adata_in.var_names_make_unique()

    in_upper = pd.Index(adata_in.var_names.str.upper())
    in_map = pd.Series(np.arange(adata_in.n_vars), index=in_upper)
    idx = in_map.reindex(target_upper).fillna(-1).astype(np.int64).to_numpy()
    present = idx >= 0

    X = adata_in.X.tocsr() if sparse.issparse(adata_in.X) else sparse.csr_matrix(adata_in.X)
    X_present = X[:, idx[present]]
    out_pos = np.where(present)[0]
    projector = sparse.csr_matrix(
        (np.ones(len(out_pos), dtype=X_present.dtype), (np.arange(len(out_pos)), out_pos)),
        shape=(len(out_pos), len(target_genes)),
    )
    X_aligned = X_present @ projector

    out = ad.AnnData(
        X=X_aligned,
        obs=adata_in.obs.copy(),
        var=pd.DataFrame(index=target_genes),
    )
    out.obs_names_make_unique()
    out.var_names_make_unique()
    return out


def probability_summary(proba: pd.DataFrame) -> pd.DataFrame:
    p = proba.values + 1e-12
    confidence = p.max(axis=1)
    entropy = -(p * np.log(p)).sum(axis=1)
    certainty = 1.0 - entropy / np.log(p.shape[1])
    return pd.DataFrame(
        {
            "pred_label": proba.idxmax(axis=1).astype(str).values,
            "confidence": confidence,
            "certainty": certainty,
        },
        index=proba.index,
    )


def save_tables(
    query: sc.AnnData,
    proba: pd.DataFrame,
    summary: pd.DataFrame,
    args: argparse.Namespace,
    table_dir: str,
) -> None:
    obs_out = query.obs.copy()
    for col in summary.columns:
        obs_out[col] = summary[col].reindex(obs_out.index).values

    prefix = output_prefix(args)
    obs_path = os.path.join(table_dir, f"{prefix}_obs_predictions.csv")
    proba_path = os.path.join(table_dir, f"{prefix}_probabilities.csv")
    count_path = os.path.join(table_dir, f"{prefix}_pred_counts.csv")
    sample_path = os.path.join(table_dir, f"{prefix}_pred_counts_by_cb.csv")

    obs_out.to_csv(obs_path)
    proba.to_csv(proba_path)
    summary["pred_label"].value_counts().rename_axis("pred_label").reset_index(name="n_cells").to_csv(
        count_path,
        index=False,
    )
    (
        obs_out.groupby([args.cb_key, "pred_label"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
        .to_csv(sample_path, index=False)
    )

    print(f"[SAVE] {obs_path}")
    print(f"[SAVE] {proba_path}")
    print(f"[SAVE] {count_path}")
    print(f"[SAVE] {sample_path}")


def plot_existing_outputs(
    args: argparse.Namespace,
    table_dir: str,
    fig_dir: str,
    latent_dir: str,
) -> None:
    prefix = output_prefix(args)
    obs_path = os.path.join(table_dir, f"{prefix}_obs_predictions.csv")
    if not os.path.exists(obs_path):
        raise FileNotFoundError(
            f"Existing predictions not found: {obs_path}\n"
            "Run 06b once without --plot-existing to create prediction outputs."
        )
    print(f"[LOAD_EXISTING] {obs_path}")
    query_obs = pd.read_csv(obs_path, index_col=0, low_memory=False)
    query_obs.index = query_obs.index.astype(str)

    cached = load_umap_plot_payload(args, table_dir, latent_dir)
    if cached is not None:
        xy, plot_obs, is_query = cached
        print("[PLOT_EXISTING] using cached UMAP plot coordinates")
    else:
        print("[PLOT_EXISTING] cached UMAP coordinates not found; rebuilding UMAP from saved latents")
        xy, plot_obs, is_query = rebuild_umap_from_existing_latents(query_obs, args, table_dir, latent_dir)
        save_umap_plot_payload(xy, plot_obs, is_query, args, table_dir, latent_dir)

    plot_panels(xy, plot_obs, is_query, args, fig_dir)
    plot_prediction_proportions(query_obs, args, fig_dir, table_dir)


def load_umap_plot_payload(
    args: argparse.Namespace,
    table_dir: str,
    latent_dir: str,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray] | None:
    cache_prefix = umap_cache_prefix(args)
    payload_path = os.path.join(latent_dir, f"{cache_prefix}_umap_plot.npz")
    obs_path = os.path.join(table_dir, f"{cache_prefix}_umap_plot_obs.csv")
    if not os.path.exists(payload_path) or not os.path.exists(obs_path):
        return None
    payload = np.load(payload_path, allow_pickle=True)
    plot_obs = pd.read_csv(obs_path, index_col=0, low_memory=False)
    plot_obs.index = plot_obs.index.astype(str)
    xy = payload["X_umap"].astype(np.float32)
    is_query = payload["is_query"].astype(bool)
    if len(plot_obs) != xy.shape[0] or len(plot_obs) != len(is_query):
        print("[WARN] Cached UMAP payload shape mismatch; rebuilding from latents.")
        return None
    return xy, plot_obs, is_query


def save_umap_plot_payload(
    xy: np.ndarray,
    plot_obs: pd.DataFrame,
    is_query: np.ndarray,
    args: argparse.Namespace,
    table_dir: str,
    latent_dir: str,
) -> None:
    cache_prefix = umap_cache_prefix(args)
    payload_path = os.path.join(latent_dir, f"{cache_prefix}_umap_plot.npz")
    obs_path = os.path.join(table_dir, f"{cache_prefix}_umap_plot_obs.csv")
    np.savez_compressed(
        payload_path,
        X_umap=np.asarray(xy, dtype=np.float32),
        is_query=np.asarray(is_query, dtype=bool),
        obs_names=plot_obs.index.astype(str).values,
    )
    plot_obs.to_csv(obs_path)
    print(f"[SAVE] {payload_path}")
    print(f"[SAVE] {obs_path}")


def rebuild_umap_from_existing_latents(
    query_obs: pd.DataFrame,
    args: argparse.Namespace,
    table_dir: str,
    latent_dir: str,
) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    prefix = output_prefix(args)
    latent_path = os.path.join(latent_dir, f"{prefix}_latents.npz")
    if not os.path.exists(latent_path):
        raise FileNotFoundError(
            f"Existing query latents not found: {latent_path}\n"
            "Run 06b once without --plot-existing to create latents."
        )
    latent = np.load(latent_path, allow_pickle=True)
    z_query = latent["X_SCANVI"].astype(np.float32)
    obs_names = latent["obs_names"].astype(str)
    query_obs = align_existing_obs_to_latents(query_obs, obs_names)

    if args.query_only_umap:
        return build_query_only_umap_from_obs(query_obs, z_query, args)

    if not args.model_dir:
        raise ValueError(
            "--plot-existing needs --model-dir when cached UMAP is absent and "
            "reference background must be rebuilt. Re-run with --query-only-umap "
            "or provide --model-dir."
        )
    ref_plot, z_ref_plot = load_reference_latent_background(args)
    return build_reference_query_umap_from_obs(ref_plot, query_obs, z_ref_plot, z_query, args)


def align_existing_obs_to_latents(query_obs: pd.DataFrame, obs_names: np.ndarray) -> pd.DataFrame:
    obs_names = np.asarray(obs_names).astype(str)
    if set(obs_names).issubset(set(query_obs.index.astype(str))):
        return query_obs.loc[obs_names].copy()
    if len(query_obs) == len(obs_names):
        out = query_obs.copy()
        out.index = obs_names
        return out
    raise ValueError(
        "Cannot align existing prediction table with saved query latents: "
        f"{len(query_obs):,} rows vs {len(obs_names):,} latent cells."
    )


def plot_prediction_proportions(
    obs: pd.DataFrame,
    args: argparse.Namespace,
    fig_dir: str,
    table_dir: str,
) -> None:
    if "pred_label" not in obs:
        return

    prefix = output_prefix(args)
    label_order = obs["pred_label"].astype(str).value_counts().index.astype(str).tolist()
    label_colors = distinct_color_map(label_order, preferred=PREFERRED_STATE_COLORS)

    specs = []
    if args.timepoint_key in obs:
        time_values = obs[args.timepoint_key].astype(str)
        specs.append(
            {
                "name": "timepoint",
                "title": "Predicted cell-type proportions by timepoint",
                "values": time_values,
                "order": ordered_timepoints(time_values.values),
                "rotate": 0,
            }
        )
    if args.cb_key in obs:
        if "rank" in obs:
            cb_values = cb_labels_with_rank(obs, args.cb_key, "rank")
            cb_order = ordered_cb_labels_by_rank(obs, args.cb_key, "rank")
        else:
            cb_values = obs[args.cb_key].astype(str)
            cb_order = ordered_categories(cb_values.values)
        specs.append(
            {
                "name": "cb_sample",
                "title": "Predicted cell-type proportions by CB sample",
                "values": cb_values,
                "order": cb_order,
                "rotate": 25,
            }
        )
    if "_sample_group" in obs:
        sample_values = obs["_sample_group"].astype(str)
        specs.append(
            {
                "name": "sample_group",
                "title": "Predicted cell-type proportions by sample group",
                "values": sample_values,
                "order": ordered_categories(sample_values.values, ["NT", "CD27", "TROP2", "NA"]),
                "rotate": 0,
            }
        )

    if not specs:
        return

    fig, axes = plt.subplots(1, len(specs), figsize=(8.4 * len(specs), 6.2), squeeze=False)
    axes = axes.ravel()
    for ax, spec in zip(axes, specs):
        counts, proportions = prediction_proportion_tables(
            obs["pred_label"].astype(str),
            spec["values"],
            spec["order"],
            label_order,
        )
        counts_path = os.path.join(table_dir, f"{prefix}_pred_counts_by_{spec['name']}.csv")
        prop_path = os.path.join(table_dir, f"{prefix}_pred_proportions_by_{spec['name']}.csv")
        counts.to_csv(counts_path)
        proportions.to_csv(prop_path)
        print(f"[SAVE] {counts_path}")
        print(f"[SAVE] {prop_path}")
        draw_stacked_proportion_bars(
            ax,
            proportions,
            label_order,
            label_colors,
            title=spec["title"],
            rotate=spec["rotate"],
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="",
            markersize=10,
            markerfacecolor=label_colors.get(label, "#999999"),
            markeredgecolor="none",
            label=short_legend_label(label),
        )
        for label in label_order
    ]
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handletextpad=0.5,
    )
    fig.suptitle("In-house predicted cell-type proportions", fontsize=20, fontweight="bold")
    style_all_legends(fig)
    style_figure(fig, tick_size=SMALL_TICK_LABEL_SIZE, legend_size=LEGEND_FONT_SIZE)
    fig.tight_layout(rect=[0, 0, 0.86, 0.94])
    png = os.path.join(fig_dir, f"{prefix}_predicted_label_proportions.png")
    pdf = os.path.join(fig_dir, f"{prefix}_predicted_label_proportions.pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {png}")
    print(f"[SAVE] {pdf}")


def prediction_proportion_tables(
    labels: pd.Series,
    group_values: pd.Series,
    group_order: list[str],
    label_order: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.DataFrame(
        {
            "group": group_values.astype(str).values,
            "pred_label": labels.astype(str).values,
        }
    )
    counts = pd.crosstab(frame["group"], frame["pred_label"])
    counts = counts.reindex(index=group_order, fill_value=0)
    counts = counts.reindex(columns=label_order, fill_value=0)
    totals = counts.sum(axis=1).replace(0, np.nan)
    proportions = counts.div(totals, axis=0).fillna(0.0)
    return counts, proportions


def draw_stacked_proportion_bars(
    ax,
    proportions: pd.DataFrame,
    label_order: list[str],
    label_colors: dict[str, str],
    *,
    title: str,
    rotate: int,
) -> None:
    x = np.arange(proportions.shape[0])
    bottom = np.zeros(proportions.shape[0], dtype=float)
    for label in label_order:
        if label not in proportions:
            continue
        values = proportions[label].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            color=label_colors.get(label, "#999999"),
            edgecolor="white",
            linewidth=0.25,
            width=0.76,
        )
        bottom += values
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Proportion")
    ax.set_xticks(x)
    ax.set_xticklabels(proportions.index.astype(str), rotation=rotate, ha="right" if rotate else "center")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    style_axis(ax, tick_size=SMALL_TICK_LABEL_SIZE)


def build_query_only_umap(query: sc.AnnData, z_query: np.ndarray, args: argparse.Namespace):
    return build_query_only_umap_from_obs(query.obs.copy(), z_query, args)


def build_query_only_umap_from_obs(query_obs: pd.DataFrame, z_query: np.ndarray, args: argparse.Namespace):
    plot_obs = query_obs.copy()
    plot_obs["plot_split"] = "query"
    is_query = np.ones(len(plot_obs), dtype=bool)
    xy = compute_umap(z_query, args.seed)
    return xy, plot_obs, is_query


def load_reference_latent_background(args: argparse.Namespace):
    run_outdir = model_run_outdir(args.model_dir)
    latent_path = os.path.join(run_outdir, "latents", "scanvi_latents.npz")
    obs_path = os.path.join(run_outdir, "tables", "scanvi_full_obs_metadata.csv")
    if not os.path.exists(latent_path) or not os.path.exists(obs_path):
        raise FileNotFoundError(
            "Could not find saved reference SCANVI latents/metadata for background plotting:\n"
            f"  {latent_path}\n"
            f"  {obs_path}\n"
            "Use --query-only-umap to plot only the in-house query."
        )
    latent = np.load(latent_path, allow_pickle=True)
    z_ref_all = latent["X_SCANVI"].astype(np.float32)
    obs_names = latent["obs_names"].astype(str)
    ref_obs_all = pd.read_csv(obs_path, index_col=0, low_memory=False)
    ref_obs_all.index = ref_obs_all.index.astype(str)
    if set(obs_names).issubset(set(ref_obs_all.index)):
        ref_obs_all = ref_obs_all.loc[obs_names].copy()
    elif len(ref_obs_all) == len(obs_names):
        ref_obs_all.index = obs_names
    else:
        raise ValueError(
            "Cannot align reference latent obs_names with scanvi_full_obs_metadata.csv: "
            f"{len(obs_names):,} latent cells vs {len(ref_obs_all):,} metadata rows."
        )

    rng = np.random.default_rng(args.seed)
    ref_idx = np.arange(z_ref_all.shape[0])
    if args.max_ref_cells and z_ref_all.shape[0] > args.max_ref_cells:
        ref_idx = np.sort(rng.choice(ref_idx, size=args.max_ref_cells, replace=False))
    return ref_obs_all.iloc[ref_idx].copy(), z_ref_all[ref_idx]


def build_reference_query_umap(
    ref_plot: pd.DataFrame,
    query: sc.AnnData,
    z_ref: np.ndarray,
    z_query: np.ndarray,
    args: argparse.Namespace,
):
    return build_reference_query_umap_from_obs(ref_plot, query.obs.copy(), z_ref, z_query, args)


def build_reference_query_umap_from_obs(
    ref_plot: pd.DataFrame,
    query_obs_all: pd.DataFrame,
    z_ref: np.ndarray,
    z_query: np.ndarray,
    args: argparse.Namespace,
):
    rng = np.random.default_rng(args.seed)

    query_idx = np.arange(len(query_obs_all))
    if args.max_query_cells and len(query_obs_all) > args.max_query_cells:
        query_idx = np.sort(rng.choice(query_idx, size=args.max_query_cells, replace=False))
    query_obs = query_obs_all.iloc[query_idx].copy()
    z_query_plot = z_query[query_idx]

    ref_obs = ref_plot.copy()
    ref_obs["plot_split"] = "reference"
    query_obs["plot_split"] = "query"
    plot_obs = pd.concat([ref_obs, query_obs], axis=0)

    z = np.vstack([z_ref, z_query_plot]).astype(np.float32)
    xy = compute_umap(z, args.seed)
    is_query = plot_obs["plot_split"].astype(str).values == "query"
    return xy, plot_obs, is_query


def compute_umap(z: np.ndarray, seed: int) -> np.ndarray:
    adata = sc.AnnData(X=np.zeros((z.shape[0], 1), dtype=np.float32))
    adata.obsm["X_SCANVI"] = z.astype(np.float32)
    sc.pp.neighbors(adata, use_rep="X_SCANVI", n_neighbors=cfg.UMAP_N_NEIGHBORS, random_state=seed)
    sc.tl.umap(adata, min_dist=cfg.UMAP_MIN_DIST, random_state=seed)
    return np.asarray(adata.obsm["X_umap"])


def plot_panels(xy, obs: pd.DataFrame, is_query: np.ndarray, args: argparse.Namespace, fig_dir: str) -> None:
    pred = obs["pred_label"].astype(str).values if "pred_label" in obs else np.array(["reference"] * len(obs))
    certainty = obs["certainty"].astype(float).values if "certainty" in obs else np.full(len(obs), np.nan)

    query_obs = obs.loc[is_query]
    query_colors = distinct_color_map(query_obs["pred_label"].astype(str).values, preferred=PREFERRED_STATE_COLORS)

    fig, axes = plt.subplots(2, 3, figsize=(25, 14))
    axes = axes.ravel()
    fig.subplots_adjust(left=0.04, right=0.76, top=0.92, bottom=0.07, wspace=0.36, hspace=0.32)
    fig.suptitle(figure_title(args), fontsize=20, fontweight="bold")

    scatter_query_categories(
        axes[0],
        xy,
        is_query,
        pred,
        query_colors,
        title="1. Predicted label",
        legend=True,
    )
    scatter_query_continuous(axes[1], xy, is_query, certainty, fig, "2. Certainty")
    cb_values = obs[args.cb_key].astype(str).values if args.cb_key in obs else np.array(["NA"] * len(obs))
    cb_plot_values = cb_values
    cb_order = None
    if "rank" in obs:
        cb_plot_values = cb_labels_with_rank(obs, args.cb_key, "rank").values
        cb_order = ordered_cb_labels_by_rank(obs.loc[is_query], args.cb_key, "rank")
    cb_colors = cb_rank_color_map(cb_order) if cb_order else distinct_color_map(cb_plot_values[is_query])
    scatter_query_categories(
        axes[2],
        xy,
        is_query,
        cb_plot_values,
        cb_colors,
        title="3. CB sample",
        legend=True,
        category_order=cb_order,
    )

    if args.timepoint_key in obs:
        time_values = obs[args.timepoint_key].astype(str).values
    else:
        time_values = np.array(["NA"] * len(obs))
    time_order = ordered_timepoints(time_values[is_query])
    time_colors = distinct_color_map(time_order)
    scatter_query_categories(
        axes[3],
        xy,
        is_query,
        time_values,
        time_colors,
        title="4. Timepoint",
        legend=True,
        category_order=time_order,
    )

    sample_group_values = (
        obs["_sample_group"].astype(str).values if "_sample_group" in obs else np.array(["NA"] * len(obs))
    )
    sample_group_order = ordered_categories(sample_group_values[is_query], ["NT", "CD27", "TROP2", "NA"])
    sample_group_colors = distinct_color_map(sample_group_order)
    scatter_query_categories(
        axes[4],
        xy,
        is_query,
        sample_group_values,
        sample_group_colors,
        title=f"5. Sample group (from {args.sample_id_key})",
        legend=True,
        category_order=sample_group_order,
    )

    true_values = obs["true_label"].astype(str).values if "true_label" in obs else np.array(["NA"] * len(obs))
    true_colors = distinct_color_map(true_values[is_query], preferred=PREFERRED_STATE_COLORS)
    scatter_query_categories(
        axes[5],
        xy,
        is_query,
        true_values,
        true_colors,
        title=f"6. In-house reference label: {args.true_label_key}",
        legend=True,
    )

    png = os.path.join(fig_dir, f"{output_prefix(args)}_umap_panels.png")
    style_figure(fig, tick_size=SMALL_TICK_LABEL_SIZE, legend_size=LEGEND_FONT_SIZE)
    style_all_legends(fig)
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {png}")


def scatter_reference_background(ax, xy: np.ndarray, is_query: np.ndarray) -> None:
    if (~is_query).any():
        ax.scatter(
            xy[~is_query, 0],
            xy[~is_query, 1],
            s=0.12,
            alpha=0.12,
            c="#bdbdbd",
            linewidths=0,
            rasterized=True,
        )


def scatter_query_categories(
    ax,
    xy: np.ndarray,
    is_query: np.ndarray,
    values: np.ndarray,
    colors: dict[str, str],
    *,
    title: str,
    legend: bool,
    category_order: list[str] | None = None,
) -> None:
    scatter_reference_background(ax, xy, is_query)
    query_values = values[is_query].astype(str)
    xy_q = xy[is_query]
    categories = ordered_categories(query_values, category_order)
    for value in categories:
        mask = query_values == value
        if not mask.any():
            continue
        ax.scatter(
            xy_q[mask, 0],
            xy_q[mask, 1],
            s=0.25,
            alpha=0.65,
            c=[colors.get(value, "#999999")],
            linewidths=0,
            label=value,
            rasterized=True,
        )
    clean_ax(ax, title)
    if legend:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=10,
                markerfacecolor=colors.get(value, "#999999"),
                markeredgecolor="none",
                label=value,
            )
            for value in categories
        ]
        ax.legend(
            handles=handles,
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            fontsize=LEGEND_FONT_SIZE,
            handletextpad=0.4,
        )
        style_legend(ax.get_legend())


def scatter_query_continuous(ax, xy: np.ndarray, is_query: np.ndarray, values: np.ndarray, fig, title: str) -> None:
    scatter_reference_background(ax, xy, is_query)
    sc_plot = ax.scatter(
        xy[is_query, 0],
        xy[is_query, 1],
        c=values[is_query],
        cmap="RdBu",
        vmin=0,
        vmax=1,
        s=0.25,
        alpha=0.85,
        linewidths=0,
        rasterized=True,
    )
    clean_ax(ax, title)
    fig.colorbar(sc_plot, ax=ax, fraction=0.046, pad=0.02)


def cb_labels_with_rank(obs: pd.DataFrame, cb_key: str, rank_key: str) -> pd.Series:
    cb = obs[cb_key].astype(str) if cb_key in obs else pd.Series("NA", index=obs.index)
    rank_raw = obs[rank_key]
    rank_numeric = pd.to_numeric(rank_raw, errors="coerce")
    rank = rank_raw.astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    labels = cb.copy()
    has_numeric_rank = rank_numeric.notna()
    labels.loc[has_numeric_rank] = (
        cb.loc[has_numeric_rank]
        + " (rank "
        + rank_numeric.loc[has_numeric_rank].map(lambda value: f"{value:g}")
        + ")"
    )
    has_text_rank = rank.ne("") & ~has_numeric_rank
    labels.loc[has_text_rank] = cb.loc[has_text_rank] + " (rank " + rank.loc[has_text_rank] + ")"
    return labels


def ordered_cb_labels_by_rank(obs: pd.DataFrame, cb_key: str, rank_key: str) -> list[str]:
    if cb_key not in obs or rank_key not in obs:
        return []
    labels = cb_labels_with_rank(obs, cb_key, rank_key)
    frame = pd.DataFrame(
        {
            "cb": obs[cb_key].astype(str),
            "rank": pd.to_numeric(obs[rank_key], errors="coerce"),
            "label": labels.astype(str),
        }
    ).dropna(subset=["rank"])
    if frame.empty:
        return []
    ranked = (
        frame.groupby(["cb", "label"], sort=False)["rank"]
        .median()
        .reset_index()
        .sort_values("rank", ascending=True)
    )
    ordered_labels = ranked["label"].astype(str).tolist()
    extras = sorted(set(labels.astype(str)) - set(ordered_labels))
    ordered_labels.extend(extras)
    return ordered_labels


def cb_rank_color_map(labels: list[str]) -> dict[str, str]:
    palette = [
        "#08306b",
        "#2171b5",
        "#6baed6",
        "#fcae91",
        "#fb6a4a",
        "#a50f15",
    ]
    if not labels:
        return {}
    if len(labels) <= len(palette):
        colors = palette[: len(labels)]
    else:
        cmap = plt.get_cmap("RdBu_r")
        colors = [cmap(i / max(len(labels) - 1, 1)) for i in range(len(labels))]
    return {label: colors[i] for i, label in enumerate(labels)}


def output_prefix(args: argparse.Namespace) -> str:
    if args.include_excluded_cb:
        return "inhouse_all_cb_projection"
    return "inhouse_minus_excluded_cb_zero_shot"


def umap_cache_prefix(args: argparse.Namespace) -> str:
    mode = "query_only" if args.query_only_umap else "ref_query"
    return f"{output_prefix(args)}_{mode}"


def figure_title(args: argparse.Namespace) -> str:
    if args.include_excluded_cb:
        return f"In-house all cord blood projection including {args.exclude_cb_id}"
    return f"In-house zero-shot minus {args.exclude_cb_id}"


def clean_ax(ax, title: str) -> None:
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def short_legend_label(label: str, *, max_chars: int = MAX_LEGEND_LABEL_CHARS) -> str:
    text = str(label)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip("_- ") + "..."


def ordered_categories(values: np.ndarray, category_order: list[str] | None = None) -> list[str]:
    seen = set(map(str, values))
    if category_order:
        ordered = [str(v) for v in category_order if str(v) in seen]
        ordered.extend(sorted(seen - set(ordered)))
        return ordered
    return sorted(seen)


def ordered_timepoints(values: np.ndarray) -> list[str]:
    seen = set(map(str, values))
    ordered = [v for v in ["D0", "D6", "D15"] if v in seen]
    extras = sorted(seen - set(ordered), key=timepoint_sort_key)
    return ordered + extras


def timepoint_sort_key(value: str):
    text = str(value)
    digits = "".join(ch for ch in text if ch.isdigit())
    return (0, int(digits)) if digits else (1, text)


def ordered_quality_values(values: np.ndarray) -> list[str]:
    seen = set(map(str, values))
    preferred = ["good", "Good", "GOOD", "bad", "Bad", "BAD", "NA", "nan"]
    ordered = [v for v in preferred if v in seen]
    ordered.extend(sorted(seen - set(ordered)))
    return ordered


if __name__ == "__main__":
    main()
