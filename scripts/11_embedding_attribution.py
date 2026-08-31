#!/usr/bin/env python
"""
SIGnature-style gene attribution on an SCVI or SCANVI encoder embedding.

Reference: Gold et al., "Scoring gene importance by interpreting single-cell
foundation models," Nature Biotechnology (2026).
https://doi.org/10.1038/s41587-026-03112-5

HOW IT DIFFERS FROM 09_gene_attribution.py
-------------------------------------------
Script 09 attributes the SCANVI *classifier logit* for a specific NK state:
    target = logit[NK_state]
    question: "what genes drive classification into NK_state?"

This script attributes the *encoder embedding* magnitude (SIGnature approach):
    target = sum(|z_detached * z|)   z = encoder posterior mean (qz.loc)
    (weight = the cell's own detached latent vector, matching Genentech's
     src/SIGnature/models/scvi.py)
    question: "what genes shape this cell's position in latent space?"

Key practical difference:
  - Script 09 → one averaged score vector per NK state (requires labels+classifier)
  - This script → one score vector per cell, then aggregated by label for plotting
    (unsupervised per cell; labels only needed for aggregation and visualization)

IMPLEMENTATION
--------------
  This script now uses the released sc-signature package implementation for
  attribution computation. It calls sc-signature's SCimilarityWrapper.forward()
  and calculate_attributions() methods through a minimal adapter that makes this
  project's SCVI/SCANVI encoder look like an input-to-embedding model. Cells are
  grouped by their registered model batch so the correct batch covariate is
  retained. The official package returns absolute normalized scores only.

OUTPUTS
-------
  tables/embedding_attribution_per_cell.npz   sparse cells × genes absolute attribution
  tables/embedding_attribution_per_label.csv  mean attribution per label (long format)
  tables/embedding_attribution_top{N}.csv     top N genes per label
  figures/embedding_attribution_bar.png
  figures/embedding_attribution_heatmap.png
  figures/embedding_attribution_dotplot.png
  embedding_attribution_run_config.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from math import ceil
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import scarches as sca
import scvi
import torch
import torch.nn as nn
from scipy import sparse
from scipy.sparse import csr_matrix, vstack

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.annotation_agent.taxonomy_reference import load_taxonomy_entries
from nk_project.io_utils import ensure_dirs
from nk_project.plot_style import (
    LEGEND_FONT_SIZE,
    SMALL_TICK_LABEL_SIZE,
    set_presentation_style,
    style_axis,
    style_figure,
)

set_presentation_style()

DEFAULT_REF_OUTDIR_NAME = "refined_scanvi_v1"
DEFAULT_MODEL_NAME = "scanvi_NK_State_refined_assay_clean_model"
TAXONOMY_HIGHLIGHT_COLOR = "#E3B505"
ATTRIBUTION_LEGEND_FONT_SIZE = 17
ATTRIBUTION_BAR_N_COLS = 4
BROAD_EXACT_GENES = {"MALAT1", "B2M", "TMSB4X", "HBB"}
BROAD_PREFIXES = ("MT-", "RPS", "RPL", "HBA")


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class SCANVIEmbeddingWrapper(nn.Module):
    """
    Wraps SCANVI to expose gene expression → scalar embedding score.

    Target scalar (SIGnature equation, matching the Genentech reference code in
    src/SIGnature/models/scvi.py):
        S(x) = sum(|z_detached * z(x)|)

    where:
        z(x)       = encoder posterior mean (qz.loc, gradient flows through this)
        z_detached = the same latent mean, detached so it acts as a per-cell
                     constant weight

    In the default ``fixed_observed`` mode, each latent dimension is weighted by
    the cell's OWN observed coordinate and that weight stays fixed throughout the
    IG path, matching the released scVI SIGnature implementation. The optional
    ``path_recomputed_legacy`` mode preserves the previous local implementation,
    which recomputes the detached weight at every interpolation point.
    """

    def __init__(self, scanvi_model, latent_weight_mode: str = "fixed_observed"):
        super().__init__()
        self.module = scanvi_model.module
        if latent_weight_mode not in {"fixed_observed", "path_recomputed_legacy"}:
            raise ValueError(f"Unknown latent weight mode: {latent_weight_mode}")
        self.latent_weight_mode = latent_weight_mode

    @property
    def uses_fixed_observed_weights(self) -> bool:
        return self.latent_weight_mode == "fixed_observed"

    def observed_weights(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        """Calculate the observed cell's latent vector once for fixed-weight IG."""
        with torch.no_grad():
            inference_out = self.module.inference(x, batch_index=batch_index, n_samples=1)
            return _extract_z_mean(inference_out).detach()

    def forward(
        self,
        x: torch.Tensor,
        batch_index: torch.Tensor,
        fixed_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inference_out = self.module.inference(x, batch_index=batch_index, n_samples=1)
        z = _extract_z_mean(inference_out)
        if self.uses_fixed_observed_weights:
            if fixed_weights is None:
                raise ValueError("fixed_observed mode requires observed-cell latent weights")
            weights = fixed_weights
        else:
            # Preserve the original local implementation for direct comparisons.
            weights = z.detach()
        return torch.sum(torch.abs(weights * z), dim=1)  # [batch]


class ContrastiveCentroidWrapper(nn.Module):
    """
    Contrastive centroid attribution with two modes:

    simple (other_centroids=None):
        S(x) = cos(z, c_target)
        "Does this gene push the cell toward this state?"
        Limitation: genes shared across all states score high even if not specific.

    contrastive (other_centroids provided):
        S(x) = cos(z, c_target) - mean( cos(z, c_j) for j != target )
        "Does this gene push the cell toward this state MORE than toward other states?"
        Addresses centroid-proximity: only state-specific genes score high.
    """

    def __init__(self, scanvi_model, target_centroid: torch.Tensor,
                 other_centroids: torch.Tensor | None = None):
        super().__init__()
        self.module = scanvi_model.module
        self.register_buffer("target_centroid", target_centroid / (target_centroid.norm() + 1e-8))
        if other_centroids is not None:
            norms = other_centroids.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.register_buffer("other_centroids", other_centroids / norms)
        else:
            self.other_centroids = None

    def forward(self, x: torch.Tensor, batch_index: torch.Tensor) -> torch.Tensor:
        inference_out = self.module.inference(x, batch_index=batch_index, n_samples=1)
        z = _extract_z_mean(inference_out)
        z_norm = z / (z.norm(dim=1, keepdim=True).clamp(min=1e-8))
        sim_target = (z_norm * self.target_centroid).sum(dim=1)
        if self.other_centroids is not None:
            sim_others = (z_norm @ self.other_centroids.T).mean(dim=1)
            return sim_target - sim_others
        return sim_target


def _extract_z_mean(inference_out: dict[str, Any]) -> torch.Tensor:
    qz = inference_out.get("qz")
    if hasattr(qz, "loc"):
        return qz.loc
    if "qz_m" in inference_out:
        return inference_out["qz_m"]
    raise KeyError("Cannot find z posterior mean in SCANVI inference output.")


# ---------------------------------------------------------------------------
# Attribution computation
# ---------------------------------------------------------------------------

def run_embedding_attribution(
    *,
    adata,
    cell_positions: np.ndarray,
    batch_indices: torch.Tensor,
    wrapper: nn.Module,
    baseline: np.ndarray,
    ig_steps: int,
    ig_batch_size: int,
    captum_internal_batch_size: int,
    method: str,
    device: torch.device,
) -> tuple[csr_matrix, csr_matrix]:
    """
    Compute per-cell, per-gene attribution scores.

    Returns:
        abs_matrix : sparse (n_cells, n_genes) — |attribution|, each row normalized to 1000
        signed_matrix : sparse (n_cells, n_genes) — signed attribution, same normalization scale
                        (sign = direction of gene's influence on embedding magnitude)
    """
    abs_list = []
    signed_list = []
    baseline_t = torch.tensor(baseline[None, :], dtype=torch.float32, device=device)

    for start in range(0, len(cell_positions), ig_batch_size):
        end = min(start + ig_batch_size, len(cell_positions))
        chunk = cell_positions[start:end]

        x = _dense_chunk(adata, chunk)
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
        b_t = batch_indices[chunk]
        bl = baseline_t.expand_as(x_t)
        fixed_weights = None
        if isinstance(wrapper, SCANVIEmbeddingWrapper) and wrapper.uses_fixed_observed_weights:
            # SIGnature: f_k(X) is evaluated once on the observed cell and held
            # constant while f_k(.) follows the baseline-to-observation IG path.
            fixed_weights = wrapper.observed_weights(x_t, b_t)

        if method == "captum":
            attrs = _ig_captum(
                wrapper, x_t, b_t, bl, ig_steps, captum_internal_batch_size, fixed_weights
            )
        else:
            attrs = _ig_manual(wrapper, x_t, b_t, bl, ig_steps, fixed_weights)

        attrs_np = attrs.detach().cpu().numpy().astype(np.float32)
        abs_list.append(csr_matrix(np.abs(attrs_np)))
        signed_list.append(csr_matrix(attrs_np))

        del x_t, b_t, bl, attrs, fixed_weights
        if device.type == "cuda":
            torch.cuda.empty_cache()

        print(f"  cells {start:,}–{end:,} / {len(cell_positions):,}", flush=True)

    abs_stacked = vstack(abs_list)
    signed_stacked = vstack(signed_list)
    # Normalize abs to 1000; apply same per-row scale to signed so magnitudes match
    row_sums = np.asarray(abs_stacked.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    scale = 1000.0 / row_sums
    diag = sparse.diags(scale)
    return (diag @ abs_stacked).tocsr(), (diag @ signed_stacked).tocsr()


class _SCVIEncoderForSIGnature(nn.Module):
    """Expose a fixed-batch SCVI/SCANVI posterior mean as ``inputs -> embedding``."""

    def __init__(self, model_module: nn.Module, batch_code: int):
        super().__init__()
        self.model_module = model_module
        self.batch_code = int(batch_code)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_index = torch.full(
            (inputs.shape[0], 1),
            self.batch_code,
            dtype=torch.long,
            device=inputs.device,
        )
        inference_out = self.model_module.inference(
            inputs, batch_index=batch_index, n_samples=1
        )
        return _extract_z_mean(inference_out)


def _make_signature_package_wrapper(model, batch_code: int, device: torch.device):
    """Construct an SCVI/SCANVI-backed instance of the released SIGnature wrapper."""
    try:
        from SIGnature.models.scimilarity import SCimilarityWrapper
    except ImportError as exc:
        raise ImportError(
            "--implementation signature_package requires sc-signature==1.0.0. "
            "Install it with: pip install sc-signature==1.0.0"
        ) from exc

    class _ProjectSIGnaturePackageWrapper(SCimilarityWrapper):
        # Do not call SCimilarityWrapper.__init__: that constructor loads the
        # authors' pretrained SCimilarity model, which is intentionally not used.
        def __init__(self, encoder: nn.Module):
            nn.Module.__init__(self)
            self.model = encoder
            self.model.eval()
            self.eval()

    encoder = _SCVIEncoderForSIGnature(model.module, batch_code).to(device)
    return _ProjectSIGnaturePackageWrapper(encoder).to(device)


def run_signature_package_attribution(
    *,
    adata,
    cell_positions: np.ndarray,
    batch_indices: torch.Tensor,
    model,
    ig_batch_size: int,
    device: torch.device,
) -> csr_matrix:
    """Run the released sc-signature attribution code on an SCVI/SCANVI encoder.

    SIGnature's generic calculation does not accept a per-cell batch covariate,
    so cells are evaluated in registered-batch groups. Each group uses an encoder
    adapter with the corresponding fixed SCANVI batch code; rows are then restored
    to the original selected-cell order.
    """
    selected_batches = (
        batch_indices[cell_positions].detach().cpu().numpy().reshape(-1)
    )
    group_matrices: list[csr_matrix] = []
    grouped_row_order: list[np.ndarray] = []

    for batch_code in np.unique(selected_batches):
        local_rows = np.flatnonzero(selected_batches == batch_code)
        positions = cell_positions[local_rows]
        X_group = adata.X[positions]
        if sparse.issparse(X_group):
            X_group = csr_matrix(X_group)
        else:
            X_group = np.asarray(X_group, dtype=np.float32)

        print(
            f"[SIGNATURE_PACKAGE] batch_code={int(batch_code)} "
            f"cells={len(local_rows):,}",
            flush=True,
        )
        wrapper = _make_signature_package_wrapper(model, int(batch_code), device)
        # This is the released sc-signature==1.0.0 implementation. For IG it
        # uses fixed observed weights, a zero baseline, 50 Captum steps, absolute
        # values, and per-cell normalization to target_sum=1000.
        attrs = wrapper.calculate_attributions(
            X_group,
            method="ig",
            batch_size=ig_batch_size,
            multiply_by_inputs=True,
            target_sum=1e3,
        )
        group_matrices.append(csr_matrix(attrs))
        grouped_row_order.append(local_rows)

        del wrapper, attrs, X_group
        if device.type == "cuda":
            torch.cuda.empty_cache()

    grouped = vstack(group_matrices).tocsr()
    row_order = np.concatenate(grouped_row_order)
    restore_order = np.argsort(row_order)
    return grouped[restore_order].tocsr()




def _ig_captum(
    wrapper, x, batch_index, baseline, n_steps, internal_batch_size, fixed_weights=None
):
    from captum.attr import IntegratedGradients
    ig = IntegratedGradients(wrapper)
    additional_forward_args = (
        (batch_index, fixed_weights) if fixed_weights is not None else (batch_index,)
    )
    return ig.attribute(
        inputs=x,
        baselines=baseline,
        additional_forward_args=additional_forward_args,
        n_steps=n_steps,
        internal_batch_size=internal_batch_size,
    )


def _ig_manual(wrapper, x, batch_index, baseline, n_steps, fixed_weights=None):
    alphas = torch.linspace(0.0, 1.0, n_steps, device=x.device)
    grad_sum = torch.zeros_like(x)
    for alpha in alphas:
        x_interp = (baseline + alpha * (x - baseline)).detach().requires_grad_(True)
        if fixed_weights is None:
            score = wrapper(x_interp, batch_index).sum()
        else:
            score = wrapper(x_interp, batch_index, fixed_weights).sum()
        wrapper.zero_grad(set_to_none=True)
        score.backward()
        grad_sum += x_interp.grad.detach()
    return (x - baseline) * grad_sum / n_steps


def _dense_chunk(adata, positions: np.ndarray) -> np.ndarray:
    x = adata.X[positions]
    if sparse.issparse(x):
        x = x.toarray()
    return np.asarray(x, dtype=np.float32)


def get_latent_for_positions(
    *,
    model,
    adata,
    positions: np.ndarray,
    batch_indices: torch.Tensor,
    device: torch.device,
    batch_size: int = 1024,
) -> np.ndarray:
    """Get deterministic z means without asking scvi-tools to revalidate sliced query AnnData."""
    z_chunks = []
    module = model.module.to(device)
    module.eval()
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = positions[start:start + batch_size]
            x = _dense_chunk(adata, chunk)
            x_t = torch.tensor(x, dtype=torch.float32, device=device)
            b_t = batch_indices[chunk]
            inference_out = module.inference(x_t, batch_index=b_t, n_samples=1)
            z = _extract_z_mean(inference_out)
            z_chunks.append(z.detach().cpu().numpy().astype(np.float32))
    return np.vstack(z_chunks)


# ---------------------------------------------------------------------------
# Aggregation and query
# ---------------------------------------------------------------------------

def aggregate_by_label(
    abs_matrix: csr_matrix,
    signed_matrix: csr_matrix | None,
    gene_names: list[str],
    labels: np.ndarray,
) -> pd.DataFrame:
    """
    Average per-cell attribution matrices by label.

    Returns long-format DataFrame:
        label, gene, mean_abs_attr, mean_attr (signed), direction, rank, n_cells

    mean_attr sign interpretation:
        positive → gene expression increases the embedding norm (drives latent identity)
        negative → gene expression decreases the embedding norm
    """
    rows = []
    for label in np.unique(labels):
        mask = labels == label
        if mask.sum() == 0:
            continue
        mean_abs = np.asarray(abs_matrix[mask].mean(axis=0)).ravel()
        if signed_matrix is None:
            mean_signed = np.full(len(gene_names), np.nan, dtype=np.float32)
        else:
            mean_signed = np.asarray(signed_matrix[mask].mean(axis=0)).ravel()
        df = pd.DataFrame({
            "gene": gene_names,
            "mean_abs_attr": mean_abs,
            "mean_attr": mean_signed,
        })
        if signed_matrix is None:
            df["direction"] = "not_available"
        else:
            df["direction"] = np.where(df["mean_attr"] >= 0, "positive", "negative")
        df = df.sort_values("mean_abs_attr", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
        df["label"] = str(label)
        df["n_cells"] = int(mask.sum())
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def query_gene_set(
    agg_df: pd.DataFrame,
    gene_set: list[str],
) -> pd.DataFrame:
    """
    For each label, sum the mean_abs_attr of genes in gene_set.
    Returns a DataFrame sorted by gene-set attribution score (descending).
    """
    gene_set_upper = {g.upper() for g in gene_set}
    hits = agg_df[agg_df["gene"].str.upper().isin(gene_set_upper)].copy()
    scores = (
        hits.groupby("label")["mean_abs_attr"]
        .sum()
        .rename("gene_set_attribution")
        .reset_index()
        .sort_values("gene_set_attribution", ascending=False)
    )
    scores["n_genes_found"] = hits.groupby("label")["gene"].count().reindex(scores["label"]).values
    return scores


def select_genes_by_attribution_mass(
    agg_df: pd.DataFrame,
    *,
    target_mass: float,
    min_genes: int,
    max_genes: int,
    relative_to_top_frac: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a cluster-specific number of genes from ranked SIGnature scores.

    The smallest prefix reaching ``target_mass`` is selected, subject to the
    minimum/maximum gene bounds and a relative-to-top attribution floor. The
    minimum gene count takes precedence when fewer genes pass the relative
    floor. If the floor or maximum prevents reaching the requested mass, the
    achieved mass is recorded explicitly in the summary.
    """
    agg_df = agg_df.copy()
    agg_df["label"] = agg_df["label"].astype(str)
    selected_rows = []
    summary_rows = []
    for label in ordered_labels(agg_df):
        ordered = (
            agg_df.loc[agg_df["label"].astype(str).eq(str(label))]
            .sort_values(["mean_abs_attr", "gene"], ascending=[False, True])
            .reset_index(drop=True)
            .copy()
        )
        values = ordered["mean_abs_attr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
        total = float(values.sum())
        cumulative = np.cumsum(values) / total if total > 0 else np.zeros_like(values)
        top_value = float(values[0]) if len(values) else 0.0
        relative_floor = relative_to_top_frac * top_value

        if total > 0:
            mass_n = int(np.searchsorted(cumulative, target_mass, side="left") + 1)
            floor_n = int(np.sum(values >= relative_floor)) if top_value > 0 else 0
        else:
            mass_n = 0
            floor_n = 0

        allowed_n = min(max_genes, floor_n, len(ordered))
        selected_n = max(min_genes, min(mass_n, allowed_n))
        selected_n = min(selected_n, max_genes, len(ordered))
        achieved_mass = float(cumulative[selected_n - 1]) if selected_n else 0.0

        chosen = ordered.head(selected_n).copy()
        chosen["mass_selection_rank"] = np.arange(1, selected_n + 1)
        chosen["mass_selection_target"] = target_mass
        chosen["mass_selection_achieved"] = achieved_mass
        chosen["mass_selection_relative_floor"] = relative_floor
        chosen["mass_selection_reached_target"] = achieved_mass >= target_mass
        selected_rows.append(chosen)

        limiting_reason = "target_mass"
        if achieved_mass < target_mass:
            limiting_reason = "max_genes" if max_genes <= floor_n else "relative_floor"
        if selected_n == min_genes and mass_n < min_genes:
            limiting_reason = "minimum_genes"
        summary_rows.append(
            {
                "label": label,
                "n_available_genes": len(ordered),
                "n_selected_genes": selected_n,
                "target_mass": target_mass,
                "achieved_mass": achieved_mass,
                "reached_target_mass": achieved_mass >= target_mass,
                "top_mean_abs_attr": top_value,
                "relative_to_top_frac": relative_to_top_frac,
                "relative_attribution_floor": relative_floor,
                "n_genes_above_relative_floor": floor_n,
                "n_genes_needed_for_target_mass": mass_n,
                "selection_limit": limiting_reason,
            }
        )

    selected = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    return selected, pd.DataFrame(summary_rows)


def save_and_plot_mass_selection(
    agg_df: pd.DataFrame,
    table_dir: str,
    fig_dir: str,
    args,
) -> None:
    selected, summary = select_genes_by_attribution_mass(
        agg_df,
        target_mass=args.selection_target_mass,
        min_genes=args.selection_min_genes,
        max_genes=args.selection_max_genes,
        relative_to_top_frac=args.selection_relative_to_top_frac,
    )
    output_prefix = (
        args.selection_output_prefix
        if args.selection_output_prefix
        else "embedding_attribution_mass"
    )
    selected_path = os.path.join(table_dir, f"{output_prefix}_selected_genes.csv")
    summary_path = os.path.join(table_dir, f"{output_prefix}_selection_summary.csv")
    selected.to_csv(selected_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[SAVE] {selected_path}")
    print(f"[SAVE] {summary_path}")
    if not summary.empty:
        print(
            "[MASS_SELECTION] "
            f"median={summary['n_selected_genes'].median():g} genes; "
            f"target reached in {int(summary['reached_target_mass'].sum())}/{len(summary)} labels"
        )
    plot_mass_selection_diagnostic(agg_df, summary, fig_dir, args)
    plot_ranked_attribution_diagnostic(agg_df, summary, fig_dir, args)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_inputs(input_h5ad: str, obs_path: str, label_key: str, batch_key: str, args):
    print(f"[LOAD] {obs_path}")
    obs = pd.read_csv(obs_path, index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)

    print(f"[LOAD] {input_h5ad}")
    adata = sc.read_h5ad(input_h5ad)
    adata.obs_names = adata.obs_names.astype(str)
    adata.obs_names_make_unique()
    h5ad_obs = adata.obs.copy()
    h5ad_obs.index = h5ad_obs.index.astype(str)

    common = obs.index.intersection(adata.obs_names)
    if len(common) == 0:
        raise ValueError("No overlapping cells between obs CSV and h5ad.")
    obs = obs.loc[common]
    for key in [label_key, batch_key]:
        if key not in obs.columns and key in h5ad_obs.columns:
            obs[key] = h5ad_obs.loc[common, key].values
            print(f"[OBS_MERGE] Added {key!r} from input h5ad obs.")
    adata = adata[common].copy()
    adata.obs = obs.copy()

    if label_key not in adata.obs.columns:
        raise KeyError(f"{label_key!r} not in obs columns.")
    if batch_key not in adata.obs.columns:
        raise KeyError(f"{batch_key!r} not in obs columns.")

    leiden_cols = [c for c in adata.obs.columns if str(c).startswith("leiden_")]
    for key in leiden_cols:
        adata.obs[key] = adata.obs[key].astype(str).astype("category")
    if leiden_cols:
        print(f"[OBS_CAST] Converted Leiden columns to string categories: {', '.join(leiden_cols)}")

    adata.obs[label_key] = adata.obs[label_key].astype(str).astype("category")
    adata.obs[batch_key] = adata.obs[batch_key].astype(str).astype("category")
    print(f"[ALIGNED] {adata.n_obs:,} cells × {adata.n_vars:,} genes")
    return adata


def load_model_for_attribution(model_dir: str, adata, args):
    """
    Load SCVI or SCANVI for either reference or query attribution.

    Reference attribution can load directly with the attributed AnnData. Query
    attribution needs the reference AnnData first so scvi-tools sees the original
    category registry, then the query is registered with extend_categories=True.
    """
    model_class = scvi.model.SCVI if args.model_type == "scvi" else sca.models.SCANVI

    if not args.query_model_reference_h5ad:
        return model_class.load(model_dir, adata=adata)

    print(f"[QUERY_MODEL_REFERENCE] {args.query_model_reference_h5ad}")
    ref = sc.read_h5ad(args.query_model_reference_h5ad)
    ref.obs_names = ref.obs_names.astype(str)
    ref.obs_names_make_unique()
    ref.var_names_make_unique()

    model = model_class.load(model_dir, adata=ref)
    model_class.prepare_query_anndata(adata, model)
    query_manager = model.adata_manager.transfer_fields(adata, extend_categories=True)
    model._register_manager_for_instance(query_manager)
    print("[QUERY_MODEL] Registered query AnnData with extended categories.")
    return model


def make_batch_indices(model, adata, batch_key: str, device: torch.device) -> torch.Tensor:
    if "_scvi_batch" in adata.obs:
        indices = np.asarray(adata.obs["_scvi_batch"], dtype=np.int64)
        print("[BATCH] Using scvi-tools encoded obs['_scvi_batch'].")
        return torch.tensor(indices[:, None], dtype=torch.long, device=device)

    batch_state = model.adata_manager.get_state_registry("batch")
    batch_order = [str(x) for x in batch_state.categorical_mapping]
    batch_to_idx = {b: i for i, b in enumerate(batch_order)}
    values = adata.obs[batch_key].astype(str)
    unknown = sorted(set(values) - set(batch_to_idx))
    if unknown:
        raise ValueError(f"Batch values unknown to model: {unknown}")
    indices = values.map(batch_to_idx).to_numpy(dtype=np.int64)
    return torch.tensor(indices[:, None], dtype=torch.long, device=device)


def load_taxonomy_genes() -> set[str]:
    genes: set[str] = set()
    for entry in load_taxonomy_entries():
        for marker_list in entry.markers.values():
            genes.update(str(g).upper() for g in marker_list)
    return genes


def load_taxonomy_gene_tiers() -> dict[str, str]:
    """
    Returns a dict mapping gene (uppercase) → highest tier across all taxonomy entries.
    Priority: core > support > context > negative_expected_low
    """
    tier_rank = {"core": 4, "support": 3, "context": 2, "negative_expected_low": 1}
    result: dict[str, int] = {}
    for entry in load_taxonomy_entries():
        for tier, marker_list in entry.markers.items():
            rank = tier_rank.get(tier, 0)
            for gene in marker_list:
                g = str(gene).upper()
                if result.get(g, 0) < rank:
                    result[g] = rank
    rank_to_tier = {4: "core", 3: "support", 2: "context", 1: "negative_expected_low"}
    return {g: rank_to_tier[r] for g, r in result.items()}


TIER_COLORS = {
    "core": "#E3B505",               # saturated gold
    "support": "#AEB7C2",            # cool silver
    "context": "#CD7F32",            # bronze
    "negative_expected_low": "#4E79A7",  # steel blue
}
UNCLASSIFIED_COLOR = "#2A9D8F"       # teal: discovered by attribution, not in taxonomy


def style_taxonomy_tick_labels(axis, taxonomy_gene_tiers: dict[str, str]) -> None:
    for tick in axis.get_ticklabels():
        tick.set_fontweight("bold")
        tier = taxonomy_gene_tiers.get(tick.get_text().upper())
        if tier:
            tick.set_color(TIER_COLORS.get(tier, TAXONOMY_HIGHLIGHT_COLOR))


def is_broad(gene: str) -> bool:
    g = str(gene).upper()
    return g in BROAD_EXACT_GENES or g.startswith(BROAD_PREFIXES)


def natural_label_sort_key(label: str) -> tuple[int, Any]:
    text = str(label)
    try:
        return (0, int(text))
    except ValueError:
        parts = re.split(r"(\d+)", text)
        return (1, tuple(int(p) if p.isdigit() else p.lower() for p in parts))


def ordered_labels(df: pd.DataFrame) -> list[str]:
    return sorted(df["label"].astype(str).unique().tolist(), key=natural_label_sort_key)


def infer_plot_label_key(args, labels: list[str]) -> str:
    label_key = str(getattr(args, "label_key", "") or "")
    outdir = str(getattr(args, "outdir", "") or "")

    if label_key and label_key != cfg.REFINED_LABEL_KEY:
        return label_key

    for pattern, inferred in [
        (r"(?:^|/)leiden_0_1(?:/|$)", "leiden_0_1"),
        (r"(?:^|/)leiden_0_4(?:/|$)", "leiden_0_4"),
        (r"(?:^|/)resolution_0_1(?:/|$)", "leiden_0_1"),
        (r"(?:^|/)resolution_0_4(?:/|$)", "leiden_0_4"),
    ]:
        if re.search(pattern, outdir):
            return inferred

    if labels and all(str(x).isdigit() for x in labels):
        return "cluster"
    return label_key or "label"


def format_panel_label(label: str, plot_label_key: str) -> str:
    if plot_label_key.startswith("leiden_"):
        return f"{plot_label_key} cluster {label}"
    if plot_label_key == "cluster":
        return f"cluster {label}"
    return str(label)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_mass_selection_diagnostic(
    agg_df: pd.DataFrame,
    summary: pd.DataFrame,
    fig_dir: str,
    args,
) -> None:
    labels = ordered_labels(agg_df)
    plot_label_key = infer_plot_label_key(args, labels)
    n_cols = min(ATTRIBUTION_BAR_N_COLS, len(labels))
    n_rows = ceil(len(labels) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.1 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    summary_by_label = summary.assign(_label=summary["label"].astype(str)).set_index("_label")
    for ax in axes.ravel():
        ax.axis("off")

    for ax, label in zip(axes.ravel(), labels):
        ax.axis("on")
        ordered = (
            agg_df.loc[agg_df["label"].astype(str).eq(str(label))]
            .sort_values(["mean_abs_attr", "gene"], ascending=[False, True])
            .reset_index(drop=True)
        )
        values = ordered["mean_abs_attr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
        ranks = np.arange(1, len(values) + 1)
        row = summary_by_label.loc[str(label)]
        selected_n = int(row["n_selected_genes"])
        target_mass = float(row["target_mass"])
        total = float(values.sum())
        cumulative_mass = (
            np.cumsum(values) / total if total > 0 else np.zeros_like(values)
        )

        ax.plot(
            ranks,
            100 * cumulative_mass,
            color="#4c78a8",
            linewidth=2.0,
        )
        if selected_n:
            ax.fill_between(
                ranks[:selected_n],
                0,
                100 * cumulative_mass[:selected_n],
                color="#4c78a8",
                alpha=0.12,
            )
        ax.axhline(
            100 * target_mass,
            color="#b23a48",
            linestyle="--",
            linewidth=1.3,
        )
        ax.axvline(
            selected_n,
            color="#b23a48",
            linestyle=":",
            linewidth=1.3,
        )
        if selected_n and len(values):
            ax.scatter(
                [selected_n],
                [100 * cumulative_mass[selected_n - 1]],
                color="#b23a48",
                s=28,
                zorder=4,
            )
        ax.set_title(
            format_panel_label(label, plot_label_key),
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Number of top-ranked genes included", fontsize=11, fontweight="bold")
        ax.set_ylabel("Cumulative attribution mass (%)", fontsize=11, fontweight="bold")
        ax.set_ylim(0, 102)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.97,
            0.08,
            f"Selected genes: {selected_n}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            color="black",
        )

    fig.suptitle(
        "SIGnature cumulative attribution-mass selection: "
        f"smallest ranked gene set reaching {100 * args.selection_target_mass:g}% mass",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, fig_dir, "embedding_attribution_gene_selection_diagnostic")


def plot_ranked_attribution_diagnostic(
    agg_df: pd.DataFrame,
    summary: pd.DataFrame,
    fig_dir: str,
    args,
) -> None:
    """Plot individual gene attribution values from highest to lowest."""
    labels = ordered_labels(agg_df)
    plot_label_key = infer_plot_label_key(args, labels)
    n_cols = min(ATTRIBUTION_BAR_N_COLS, len(labels))
    n_rows = ceil(len(labels) / n_cols)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.1 * n_cols, 4.5 * n_rows),
        squeeze=False,
    )
    summary_by_label = summary.assign(
        _label=summary["label"].astype(str)
    ).set_index("_label")
    for ax in axes.ravel():
        ax.axis("off")

    for ax, label in zip(axes.ravel(), labels):
        ax.axis("on")
        ordered = (
            agg_df.loc[agg_df["label"].astype(str).eq(str(label))]
            .sort_values(["mean_abs_attr", "gene"], ascending=[False, True])
            .reset_index(drop=True)
        )
        values = ordered["mean_abs_attr"].fillna(0).clip(lower=0).to_numpy(dtype=float)
        ranks = np.arange(1, len(values) + 1)
        selected_n = int(summary_by_label.loc[str(label), "n_selected_genes"])

        ax.plot(ranks, values, color="#4c78a8", linewidth=2.0)
        if selected_n:
            ax.fill_between(
                ranks[:selected_n],
                0,
                values[:selected_n],
                color="#4c78a8",
                alpha=0.12,
            )
            ax.axvline(
                selected_n,
                color="#b23a48",
                linestyle="--",
                linewidth=1.3,
            )
            ax.scatter(
                [selected_n],
                [values[selected_n - 1]],
                color="#b23a48",
                s=28,
                zorder=4,
            )

        ax.set_title(
            format_panel_label(label, plot_label_key),
            fontsize=12,
            fontweight="bold",
        )
        ax.set_xlabel("Gene rank", fontsize=11, fontweight="bold")
        ax.set_ylabel("Mean |attribution|", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(
            0.97,
            0.92,
            f"Selected genes: {selected_n}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=11,
            fontweight="bold",
            color="black",
        )

    fig.suptitle(
        "SIGnature individual gene attribution ranked from highest to lowest",
        fontsize=18,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    _save_fig(fig, fig_dir, "embedding_attribution_ranked_gene_attribution_diagnostic")


def plot_bar(agg_df: pd.DataFrame, fig_dir: str, top_n: int, args) -> None:
    """
    Horizontal bar plot ranked by mean |attribution|.

    Bar color encodes taxonomy tier of gene:
        ● Gold   — core marker
        ● Silver — support marker
        ● Bronze — context marker
        ● Blue   — negative/expected-low marker
        ● Teal   — not in taxonomy

    Genes are selected/ranked by mean |attribution|. Bars show attribution
    magnitude only because this non-contrastive embedding target is not a
    state-direction objective.
    """
    tier_map = load_taxonomy_gene_tiers()
    DEFAULT_COLOR = UNCLASSIFIED_COLOR

    labels = ordered_labels(agg_df)
    plot_label_key = infer_plot_label_key(args, labels)
    n_cols = min(ATTRIBUTION_BAR_N_COLS, len(labels))
    n_rows = ceil(len(labels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.0 * n_cols, 5.5 * n_rows), squeeze=False)

    shared_max = 0.0
    for label in labels:
        sub = agg_df.loc[agg_df["label"].astype(str).eq(str(label))].nlargest(top_n, "mean_abs_attr")
        if len(sub):
            shared_max = max(shared_max, float(sub["mean_abs_attr"].max()))
    shared_max = shared_max if shared_max > 0 else 1.0

    for ax in axes.ravel():
        ax.axis("off")

    for ax, label in zip(axes.ravel(), labels):
        ax.axis("on")
        sub = agg_df.loc[agg_df["label"].astype(str).eq(str(label))].nlargest(top_n, "mean_abs_attr")
        sub = sub.sort_values("mean_abs_attr", ascending=True).reset_index(drop=True)

        colors = [
            TIER_COLORS.get(tier_map.get(str(g).upper(), ""), DEFAULT_COLOR)
            for g in sub["gene"]
        ]
        ax.barh(np.arange(len(sub)), sub["mean_abs_attr"], color=colors, alpha=0.92)
        ax.set_xlim(0, shared_max * 1.06)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["gene"].astype(str), fontsize=10.5, fontweight="bold")
        ax.set_title(format_panel_label(label, plot_label_key), fontsize=13, fontweight="bold")
        ax.set_xlabel("Mean |attribution| (norm. to 1000)", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TIER_COLORS["core"],                  label="Core marker"),
        Patch(facecolor=TIER_COLORS["support"],               label="Support marker"),
        Patch(facecolor=TIER_COLORS["context"],               label="Context marker"),
        Patch(facecolor=TIER_COLORS["negative_expected_low"], label="Negative/expected-low"),
        Patch(facecolor=UNCLASSIFIED_COLOR, label="Not in taxonomy"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        fontsize=ATTRIBUTION_LEGEND_FONT_SIZE,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.2,
        handleheight=1.4,
        borderpad=0.8,
        labelspacing=0.8,
        columnspacing=1.6,
    )
    if plot_label_key.startswith("leiden_"):
        title = f"SIGnature attribution: top {top_n} genes by {plot_label_key}"
    else:
        title = f"Embedding attribution: top {top_n} genes per {plot_label_key}"
    fig.suptitle(title, fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.055, 1, 0.97])
    _save_fig(fig, fig_dir, "embedding_attribution_bar")


def run_contrastive_all_labels(
    *,
    adata,
    label_cell_positions: dict[str, np.ndarray],
    batch_indices: torch.Tensor,
    model,
    baseline: np.ndarray,
    ig_steps: int,
    ig_batch_size: int,
    captum_internal_batch_size: int,
    method: str,
    device: torch.device,
    gene_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each label, run IG in both contrastive modes.

    Returns two long-format DataFrames:
        simple_df   — target = cos(z, c_label)
        vs_others_df — target = cos(z, c_label) - mean(cos(z, c_others))
    """
    print("[CONTRASTIVE] Computing z centroids for all labels...")
    centroids: dict[str, torch.Tensor] = {}
    for label, positions in label_cell_positions.items():
        z = get_latent_for_positions(
            model=model,
            adata=adata,
            positions=positions,
            batch_indices=batch_indices,
            device=device,
        )
        centroids[label] = torch.tensor(z.mean(axis=0), dtype=torch.float32, device=device)
        print(f"  {label}: centroid computed ({len(positions):,} cells)")

    all_labels = list(centroids.keys())
    simple_rows, vs_others_rows = [], []

    for label, positions in label_cell_positions.items():
        target_centroid = centroids[label]
        other_centroids = torch.stack([centroids[l] for l in all_labels if l != label])

        for mode, other_c in [("simple", None), ("vs_others", other_centroids)]:
            print(f"\n[CONTRASTIVE/{mode}] '{label}' ({len(positions):,} cells)")
            wrapper = ContrastiveCentroidWrapper(model, target_centroid, other_c).to(device)
            wrapper.eval()

            abs_mat, signed_mat = run_embedding_attribution(
                adata=adata,
                cell_positions=positions,
                batch_indices=batch_indices,
                wrapper=wrapper,
                baseline=baseline,
                ig_steps=ig_steps,
                ig_batch_size=ig_batch_size,
                captum_internal_batch_size=captum_internal_batch_size,
                method=method,
                device=device,
            )

            mean_signed = np.asarray(signed_mat.mean(axis=0)).ravel()
            mean_abs = np.asarray(abs_mat.mean(axis=0)).ravel()
            df = pd.DataFrame({"gene": gene_names, "mean_attr": mean_signed, "mean_abs_attr": mean_abs})
            df = df.sort_values("mean_abs_attr", ascending=False).reset_index(drop=True)
            df["rank"] = np.arange(1, len(df) + 1)
            df["label"] = label
            df["n_cells"] = len(positions)

            if mode == "simple":
                simple_rows.append(df)
            else:
                vs_others_rows.append(df)

    return pd.concat(simple_rows, ignore_index=True), pd.concat(vs_others_rows, ignore_index=True)


def plot_bar_diverging(contrastive_df: pd.DataFrame, fig_dir: str, top_n: int,
                       fname: str = "embedding_attribution_contrastive_bar",
                       title: str = "Contrastive attribution: top {n} genes per label",
                       args=None) -> None:
    """
    Diverging horizontal bar chart for contrastive mode.

    Bar goes RIGHT  → higher gene expression pushes cells TOWARD this state.
    Bar goes LEFT   → higher gene expression pushes cells AWAY from this state.
    Bars are ranked by mean |attribution| so the most influential genes are at the top.
    Colors encode taxonomy tier.
    """
    tier_map = load_taxonomy_gene_tiers()
    DEFAULT_COLOR = UNCLASSIFIED_COLOR

    labels = ordered_labels(contrastive_df)
    plot_label_key = infer_plot_label_key(args, labels) if args is not None else "label"
    n_cols = min(ATTRIBUTION_BAR_N_COLS, len(labels))
    n_rows = ceil(len(labels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7.2 * n_cols, 5.7 * n_rows), squeeze=False)

    shared_abs = 0.0
    for label in labels:
        sub = contrastive_df.loc[
            contrastive_df["label"].astype(str).eq(str(label))
        ].nlargest(top_n, "mean_abs_attr")
        if len(sub):
            shared_abs = max(shared_abs, float(np.abs(sub["mean_attr"]).max()))
    shared_abs = shared_abs if shared_abs > 0 else 1.0

    for ax in axes.ravel():
        ax.axis("off")

    for ax, label in zip(axes.ravel(), labels):
        ax.axis("on")
        sub = (contrastive_df.loc[contrastive_df["label"].astype(str).eq(str(label))]
               .nlargest(top_n, "mean_abs_attr")
               .assign(_plot_abs_signed_attr=lambda x: x["mean_attr"].abs())
               .sort_values("_plot_abs_signed_attr", ascending=True)
               .reset_index(drop=True))

        colors = [
            TIER_COLORS.get(tier_map.get(str(g).upper(), ""), DEFAULT_COLOR)
            for g in sub["gene"]
        ]
        ax.barh(np.arange(len(sub)), sub["mean_attr"], color=colors, alpha=0.92)
        ax.axvline(0, color="black", linewidth=0.9)
        ax.set_xlim(-shared_abs * 1.08, shared_abs * 1.08)
        ax.set_yticks(np.arange(len(sub)))
        ax.set_yticklabels(sub["gene"].astype(str), fontsize=10.5, fontweight="bold")
        ax.set_title(format_panel_label(label, plot_label_key), fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        ax.tick_params(axis="x", labelsize=10)
        ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=TIER_COLORS["core"],                  label="Core marker"),
        Patch(facecolor=TIER_COLORS["support"],               label="Support marker"),
        Patch(facecolor=TIER_COLORS["context"],               label="Context marker"),
        Patch(facecolor=TIER_COLORS["negative_expected_low"], label="Negative/expected-low"),
        Patch(facecolor=UNCLASSIFIED_COLOR, label="Not in taxonomy"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=5,
        fontsize=ATTRIBUTION_LEGEND_FONT_SIZE,
        frameon=True,
        bbox_to_anchor=(0.5, 0.005),
        handlelength=2.2,
        handleheight=1.4,
        borderpad=0.8,
        labelspacing=0.8,
        columnspacing=1.6,
    )
    plot_title = title.replace("{n}", str(top_n))
    if plot_label_key.startswith("leiden_"):
        plot_title = plot_title.replace("per label", f"by {plot_label_key}")
    fig.suptitle(plot_title, fontsize=15, fontweight="bold")
    fig.tight_layout(rect=[0, 0.055, 1, 0.96])
    _save_fig(fig, fig_dir, fname)


def plot_heatmap_and_dotplot(agg_df: pd.DataFrame, fig_dir: str, top_n: int, args) -> None:
    taxonomy_genes = load_taxonomy_genes()
    taxonomy_gene_tiers = load_taxonomy_gene_tiers()
    labels = ordered_labels(agg_df)
    plot_label_key = infer_plot_label_key(args, labels)

    # union of top-N genes across all labels
    union_genes: list[str] = []
    for label in labels:
        sub = agg_df.loc[agg_df["label"].astype(str).eq(str(label))].nlargest(top_n, "mean_abs_attr")
        union_genes.extend(sub["gene"].tolist())
    union_genes = list(dict.fromkeys(union_genes))

    mat = pd.DataFrame(index=labels, columns=union_genes, dtype=float)
    for label in labels:
        sub = agg_df.loc[agg_df["label"].astype(str).eq(str(label))]
        attr_map = dict(zip(sub["gene"], sub["mean_abs_attr"]))
        mat.loc[label] = [attr_map.get(g, 0.0) for g in union_genes]

    # cluster columns
    try:
        from scipy.cluster.hierarchy import leaves_list, linkage
        order = leaves_list(linkage(mat.values.T, method="ward", metric="euclidean"))
        mat = mat.iloc[:, order]
    except Exception as exc:
        print(f"[WARN] Column clustering failed: {exc}")

    vmax = float(np.nanmax(mat.values)) or 1.0
    fig_w = max(14, 0.46 * len(mat.columns))
    fig_h = max(5, 0.56 * len(labels))

    # heatmap
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(mat.values, cmap="Reds", vmin=0, vmax=vmax, aspect="auto")
    ax.grid(False)
    ax.tick_params(which="both", length=0)
    ax.set_xticks(np.arange(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=SMALL_TICK_LABEL_SIZE + 1)
    style_taxonomy_tick_labels(ax.xaxis, taxonomy_gene_tiers)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(
        [format_panel_label(label, plot_label_key) for label in labels],
        fontsize=11,
        fontweight="bold",
    )
    heatmap_title = (
        f"SIGnature attribution heatmap: top {top_n} genes by {plot_label_key}"
        if plot_label_key.startswith("leiden_")
        else f"Embedding attribution heatmap: top {top_n} genes"
    )
    ax.set_title(heatmap_title, fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01).set_label("Mean |attribution|")
    fig.tight_layout()
    _save_fig(fig, fig_dir, "embedding_attribution_heatmap")

    # dotplot
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h + 0.8))
    for yi, label in enumerate(labels):
        for xi, gene in enumerate(mat.columns):
            mag = float(mat.loc[label, gene])
            size = 20 + 340 * mag / vmax if vmax > 0 else 20
            color = matplotlib.cm.Reds(mag / vmax if vmax > 0 else 0)
            ec = TAXONOMY_HIGHLIGHT_COLOR if str(gene).upper() in taxonomy_genes else "white"
            ax2.scatter(xi, yi, s=size, color=color, edgecolors=ec, linewidths=0.9 if ec != "white" else 0.3)
    ax2.set_xticks(np.arange(len(mat.columns)))
    ax2.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=SMALL_TICK_LABEL_SIZE + 1)
    style_taxonomy_tick_labels(ax2.xaxis, taxonomy_gene_tiers)
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_yticklabels(
        [format_panel_label(label, plot_label_key) for label in labels],
        fontsize=11,
        fontweight="bold",
    )
    ax2.set_xlim(-0.5, len(mat.columns) - 0.5)
    ax2.set_ylim(-0.5, len(labels) - 0.5)
    ax2.invert_yaxis()
    ax2.grid(alpha=0.25, linewidth=0.3)
    ax2.spines[["top", "right", "bottom", "left"]].set_visible(False)
    sm = matplotlib.cm.ScalarMappable(cmap="Reds", norm=matplotlib.colors.Normalize(0, vmax))
    sm.set_array([])
    fig2.colorbar(sm, ax=ax2, fraction=0.015, pad=0.01).set_label("Mean |attribution|")
    dotplot_title = (
        f"SIGnature attribution dot plot: top {top_n} genes by {plot_label_key}"
        if plot_label_key.startswith("leiden_")
        else f"Embedding attribution dot plot: top {top_n} genes"
    )
    ax2.set_title(dotplot_title, fontsize=14, fontweight="bold")
    fig2.tight_layout()
    _save_fig(fig2, fig_dir, "embedding_attribution_dotplot")


def _save_fig(fig, fig_dir: str, stem: str) -> None:
    path = os.path.join(fig_dir, f"{stem}.png")
    style_figure(fig, tick_size=SMALL_TICK_LABEL_SIZE, legend_size=ATTRIBUTION_LEGEND_FONT_SIZE)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[SAVE] {path}")


def plot_existing_outputs(table_dir: str, fig_dir: str, args) -> None:
    plotted = 0
    agg_path = (
        args.existing_attribution_csv
        if args.existing_attribution_csv
        else os.path.join(table_dir, "embedding_attribution_per_label.csv")
    )
    if os.path.exists(agg_path):
        print(f"[LOAD] {agg_path}")
        agg_df = pd.read_csv(agg_path, dtype={"label": str})
        agg_df["label"] = agg_df["label"].astype(str)
        save_and_plot_mass_selection(agg_df, table_dir, fig_dir, args)
        plot_bar(agg_df, fig_dir, args.top_n, args)
        plot_heatmap_and_dotplot(agg_df, fig_dir, args.top_n, args)
        plotted += 1
    else:
        print(f"[SKIP] Existing non-contrastive table not found: {agg_path}")

    for csv_name, fig_name, title in [
        (
            "embedding_attribution_contrastive_simple_per_label.csv",
            "embedding_attribution_contrastive_simple_bar",
            "Contrastive attribution (vs own centroid): top {n} genes per label",
        ),
        (
            "embedding_attribution_contrastive_vs_others_per_label.csv",
            "embedding_attribution_contrastive_vs_others_bar",
            "Contrastive attribution (vs all other states): top {n} genes per label",
        ),
    ]:
        path = os.path.join(table_dir, csv_name)
        if os.path.exists(path):
            print(f"[LOAD] {path}")
            contrastive_df = pd.read_csv(path, dtype={"label": str})
            contrastive_df["label"] = contrastive_df["label"].astype(str)
            plot_bar_diverging(
                contrastive_df,
                fig_dir,
                args.top_n,
                fname=fig_name,
                title=title,
                args=args,
            )
            plotted += 1
        else:
            print(f"[SKIP] Existing contrastive table not found: {path}")

    if plotted == 0:
        raise FileNotFoundError(f"No existing attribution tables found in: {table_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "SIGnature-style gene attribution: attributes an SCVI/SCANVI encoder embedding "
            "sum(|z_detached*z|) per cell, following Gold et al. Nat Biotech 2026."
        )
    )
    p.add_argument("--input-h5ad", default=None)
    p.add_argument("--ref-outdir", default=None, help="Default: outputs/refined_scanvi_v1")
    p.add_argument("--model-dir", default=None)
    p.add_argument(
        "--model-type",
        choices=["scanvi", "scvi"],
        default="scanvi",
        help=(
            "Encoder model type. Use 'scvi' to run released SIGnature attribution "
            "on the original unsupervised SCVI embedding."
        ),
    )
    p.add_argument(
        "--query-model-reference-h5ad",
        default=None,
        help=(
            "Reference AnnData used to load the trained SCVI/SCANVI model before registering "
            "the input h5ad as query data with extended categories. Use for in-house/query attribution."
        ),
    )
    p.add_argument("--obs-csv", default=None)
    p.add_argument("--outdir", default=None)
    p.add_argument("--label-key", default=cfg.REFINED_LABEL_KEY)
    p.add_argument("--batch-key", default=cfg.PRODUCTION_BATCH_KEY)
    p.add_argument(
        "--exclude-labels",
        action="append",
        default=[],
        help="Labels to exclude from aggregation plots (e.g. Non-NK, Unknown). Repeatable.",
    )
    p.add_argument(
        "--max-cells-per-label",
        type=int,
        default=None,
        help="Random subsample cells per label before attribution (None = all).",
    )
    p.add_argument(
        "--max-cells-total",
        type=int,
        default=None,
        help="Cap total cells attributed (random sample across all labels).",
    )
    p.add_argument("--ig-steps", type=int, default=50)
    p.add_argument("--ig-batch-size", type=int, default=64)
    p.add_argument("--captum-internal-batch-size", type=int, default=128)
    p.add_argument(
        "--implementation",
        choices=["signature_package"],
        default="signature_package",
        help=(
            "Attribution implementation. Script 11 is restricted to the released "
            "sc-signature package implementation."
        ),
    )
    p.add_argument(
        "--latent-weight-mode",
        choices=["fixed_observed", "path_recomputed_legacy"],
        default="fixed_observed",
        help=(
            "Kept for run-config compatibility. The released sc-signature package "
            "requires fixed_observed."
        ),
    )
    p.add_argument("--top-n", type=int, default=20, help="Top genes per label for plots/tables.")
    p.add_argument(
        "--selection-target-mass",
        type=float,
        default=0.50,
        help="Cumulative attribution mass target for adaptive per-label gene selection. Default 0.50.",
    )
    p.add_argument(
        "--selection-min-genes",
        type=int,
        default=10,
        help="Minimum genes selected per label by the mass diagnostic. Default 10.",
    )
    p.add_argument(
        "--selection-max-genes",
        type=int,
        default=100,
        help="Maximum genes selected per label by the mass diagnostic. Default 100.",
    )
    p.add_argument(
        "--selection-relative-to-top-frac",
        type=float,
        default=0.01,
        help="Minimum attribution relative to each label's top gene. Default 0.01 (1%%).",
    )
    p.add_argument(
        "--selection-output-prefix",
        default=None,
        help=(
            "Optional basename for mass-selection CSVs, for example "
            "embedding_attribution_mass60. The default preserves the existing "
            "embedding_attribution_mass_* filenames."
        ),
    )
    p.add_argument(
        "--filter-broad-genes",
        action="store_true",
        help="Remove MT-, RPS, RPL, HBB etc. before aggregation and plotting.",
    )
    p.add_argument(
        "--contrastive-label",
        default=None,
        help=(
            "Deprecated in script 11. Contrastive centroid attribution was a custom "
            "local method and is no longer used here."
        ),
    )
    p.add_argument(
        "--contrastive-all-labels",
        action="store_true",
        help=(
            "Deprecated in script 11. Contrastive centroid attribution was a custom "
            "local method and is no longer used here."
        ),
    )
    p.add_argument(
        "--query-genes",
        default=None,
        help="Comma-separated gene set to query after attribution (e.g. MKI67,TOP2A,TYMS).",
    )
    p.add_argument(
        "--method",
        choices=["captum", "manual", "auto"],
        default="auto",
    )
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=cfg.SEED)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load inputs and print cell counts per label without running attribution.",
    )
    p.add_argument(
        "--plot-existing",
        action="store_true",
        help="Regenerate figures from existing attribution CSVs in --outdir without rerunning IG.",
    )
    p.add_argument(
        "--existing-attribution-csv",
        default=None,
        help=(
            "Optional existing embedding_attribution_per_label.csv to read while "
            "writing newly selected genes and figures under --outdir. This allows "
            "multiple cumulative-mass selections without copying or overwriting "
            "the original all-gene attribution table."
        ),
    )
    p.add_argument(
        "--keep-old-figures",
        action="store_true",
        help="Keep any pre-existing PNGs in the figures dir. By default the figures dir is "
             "cleared at the start of each run so it only holds the current run's output.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _set_seed(args.seed)

    if not 0 < args.selection_target_mass <= 1:
        raise ValueError("--selection-target-mass must be in (0, 1].")
    if args.selection_min_genes < 1:
        raise ValueError("--selection-min-genes must be at least 1.")
    if args.selection_max_genes < args.selection_min_genes:
        raise ValueError("--selection-max-genes must be >= --selection-min-genes.")
    if not 0 <= args.selection_relative_to_top_frac <= 1:
        raise ValueError("--selection-relative-to-top-frac must be in [0, 1].")

    ref_outdir = args.ref_outdir or os.path.join(cfg.BASE_OUTDIR, DEFAULT_REF_OUTDIR_NAME)
    model_dir = args.model_dir or os.path.join(ref_outdir, "models", DEFAULT_MODEL_NAME)
    input_h5ad = args.input_h5ad or os.path.join(
        cfg.BASE_OUTDIR, "refined_annotation_v1", "full_scvi_leiden_refined_v1.h5ad"
    )
    obs_path = args.obs_csv or os.path.join(ref_outdir, "tables", "scanvi_full_obs_metadata.csv")
    outdir = args.outdir or os.path.join(ref_outdir, "embedding_attribution")
    table_dir = os.path.join(outdir, "tables")
    fig_dir = os.path.join(outdir, "figures")
    ensure_dirs(outdir, table_dir, fig_dir)

    # Clear stale figures so the dir only holds this run's output (avoids
    # accumulating outdated PNGs from earlier runs with different stems).
    if not args.keep_old_figures:
        import glob
        stale = glob.glob(os.path.join(fig_dir, "*.png"))
        for f in stale:
            os.remove(f)
        if stale:
            print(f"[CLEAN] Removed {len(stale)} old figure(s) from {fig_dir}")

    print("=" * 80)
    print("Embedding Attribution — SIGnature-style (Gold et al. Nat Biotech 2026)")
    print("=" * 80)
    print(f"[INPUT]   {input_h5ad}")
    print(f"[MODEL]   {model_dir}")
    print(f"[MODEL_TYPE] {args.model_type}")
    print(f"[OBS]     {obs_path}")
    print(f"[OUTDIR]  {outdir}")
    print(f"[LABEL]   {args.label_key}")
    print(f"[BATCH]   {args.batch_key}")
    print(f"[IMPLEMENTATION] {args.implementation}")

    if args.plot_existing:
        plot_existing_outputs(table_dir, fig_dir, args)
        print("[DONE] Existing embedding-attribution plots regenerated.")
        return

    if args.implementation == "signature_package":
        if args.contrastive_label or args.contrastive_all_labels:
            raise ValueError(
                "Script 11 is now restricted to released sc-signature attribution. "
                "Do not pass --contrastive-label or --contrastive-all-labels here."
            )
        if args.method == "manual":
            raise ValueError(
                "--implementation signature_package requires Captum; "
                "--method manual is not supported."
            )
        if args.ig_steps != 50:
            raise ValueError(
                "sc-signature==1.0.0 does not expose n_steps and uses Captum's "
                "default of 50. Set --ig-steps 50 for an exact package run."
            )
        if args.latent_weight_mode != "fixed_observed":
            raise ValueError(
                "The released sc-signature implementation always uses fixed observed "
                "weights. Set --latent-weight-mode fixed_observed."
            )

    if args.model_type == "scvi" and args.implementation != "signature_package":
        raise ValueError(
            "--model-type scvi is intentionally restricted to "
            "--implementation signature_package so this control uses only the "
            "released SIGnature attribution implementation."
        )
    if args.model_type == "scvi" and (
        args.contrastive_label or args.contrastive_all_labels
    ):
        raise ValueError(
            "The SCVI control is standard released SIGnature attribution only; "
            "do not pass contrastive options."
        )

    adata = load_inputs(input_h5ad, obs_path, args.label_key, args.batch_key, args)
    gene_names = adata.var_names.astype(str).tolist()

    model = load_model_for_attribution(model_dir, adata, args)
    model.module.eval()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    print(f"[DEVICE]  {device}")

    # Choose attribution target
    if args.contrastive_label:
        target_centroid = _compute_z_centroid(model, adata, args.contrastive_label, args.batch_key, device)
        all_label_vals = adata.obs[args.label_key].astype(str).unique().tolist()
        other_centroids = torch.stack([
            _compute_z_centroid(model, adata, l, args.batch_key, device)
            for l in all_label_vals if l != args.contrastive_label
        ])
        wrapper = ContrastiveCentroidWrapper(model, target_centroid, other_centroids).to(device)
        wrapper.eval()
        print(f"[MODE] Contrastive: cos(z, c_{args.contrastive_label}) - mean(cos(z, c_others))")
    elif args.implementation == "signature_package":
        wrapper = None
        print(
            "[MODE] Released sc-signature attribution with this project's "
            f"{args.model_type.upper()} encoder (fixed observed weights)"
        )
    else:
        wrapper = SCANVIEmbeddingWrapper(
            model, latent_weight_mode=args.latent_weight_mode
        ).to(device)
        wrapper.eval()
        if args.latent_weight_mode == "fixed_observed":
            print("[MODE] SIGnature fixed observed weights: sum(|z_observed * z(path)|)")
        else:
            print("[MODE] Legacy path-recomputed weights: sum(|z(path).detach() * z(path)|)")

    batch_indices = make_batch_indices(model, adata, args.batch_key, device)

    # Select cells
    labels = adata.obs[args.label_key].astype(str).values
    exclude = set(args.exclude_labels)
    rng = np.random.default_rng(args.seed)

    label_cell_positions: dict[str, np.ndarray] = {}
    for label in np.unique(labels):
        if label in exclude:
            continue
        pos = np.where(labels == label)[0]
        if args.max_cells_per_label and len(pos) > args.max_cells_per_label:
            pos = np.sort(rng.choice(pos, size=args.max_cells_per_label, replace=False))
        label_cell_positions[label] = pos

    all_positions = np.concatenate(list(label_cell_positions.values()))
    if args.max_cells_total and len(all_positions) > args.max_cells_total:
        all_positions = np.sort(rng.choice(all_positions, size=args.max_cells_total, replace=False))

    print(f"[CELLS] {len(all_positions):,} cells across {len(label_cell_positions)} labels")
    for label, pos in label_cell_positions.items():
        kept = len(np.intersect1d(pos, all_positions))
        print(f"  {label}: {len(pos):,} → {kept:,} after total cap")

    if args.dry_run:
        print("[DRY-RUN] Done. Skipping attribution.")
        return

    method = _choose_method(args)
    baseline = np.zeros(adata.n_vars, dtype=np.float32)

    print(f"[IG] steps={args.ig_steps}, batch={args.ig_batch_size}, method={method}")
    if args.implementation == "signature_package" and not args.contrastive_label:
        abs_matrix = run_signature_package_attribution(
            adata=adata,
            cell_positions=all_positions,
            batch_indices=batch_indices,
            model=model,
            ig_batch_size=args.ig_batch_size,
            device=device,
        )
        signed_matrix = None
    else:
        abs_matrix, signed_matrix = run_embedding_attribution(
            adata=adata,
            cell_positions=all_positions,
            batch_indices=batch_indices,
            wrapper=wrapper,
            baseline=baseline,
            ig_steps=args.ig_steps,
            ig_batch_size=args.ig_batch_size,
            captum_internal_batch_size=args.captum_internal_batch_size,
            method=method,
            device=device,
        )

    # Save per-cell sparse matrices
    cell_names = adata.obs_names[all_positions].tolist()
    npz_path = os.path.join(table_dir, "embedding_attribution_per_cell.npz")
    npz_signed_path = os.path.join(table_dir, "embedding_attribution_per_cell_signed.npz")
    sparse.save_npz(npz_path, abs_matrix)
    if signed_matrix is not None:
        sparse.save_npz(npz_signed_path, signed_matrix)
    elif os.path.exists(npz_signed_path):
        os.remove(npz_signed_path)
    pd.Series(cell_names).to_csv(
        os.path.join(table_dir, "embedding_attribution_cell_names.txt"),
        index=False, header=False,
    )
    pd.Series(gene_names).to_csv(
        os.path.join(table_dir, "embedding_attribution_gene_names.txt"),
        index=False, header=False,
    )
    print(f"[SAVE] {npz_path}  shape={abs_matrix.shape}")
    if signed_matrix is not None:
        print(f"[SAVE] {npz_signed_path}")
    else:
        print("[SIGNED] Not emitted: released sc-signature returns absolute scores only.")

    # Optionally filter broad genes before aggregation
    kept_gene_mask = np.ones(len(gene_names), dtype=bool)
    if args.filter_broad_genes:
        kept_gene_mask = np.array([not is_broad(g) for g in gene_names])
        print(f"[FILTER] Removed {(~kept_gene_mask).sum()} broad/housekeeping genes")
        abs_matrix = abs_matrix[:, kept_gene_mask]
        if signed_matrix is not None:
            signed_matrix = signed_matrix[:, kept_gene_mask]

    kept_genes = [g for g, k in zip(gene_names, kept_gene_mask) if k]
    attributed_labels = labels[all_positions]

    agg_df = aggregate_by_label(abs_matrix, signed_matrix, kept_genes, attributed_labels)

    agg_path = os.path.join(table_dir, "embedding_attribution_per_label.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"[SAVE] {agg_path}")

    top_df = agg_df.groupby("label", group_keys=False).apply(
        lambda x: x.nlargest(args.top_n, "mean_abs_attr")
    )
    top_path = os.path.join(table_dir, f"embedding_attribution_top{args.top_n}.csv")
    top_df.to_csv(top_path, index=False)
    print(f"[SAVE] {top_path}")

    save_and_plot_mass_selection(agg_df, table_dir, fig_dir, args)

    # Gene set query
    if args.query_genes:
        gene_set = [g.strip() for g in args.query_genes.split(",") if g.strip()]
        query_result = query_gene_set(agg_df, gene_set)
        query_path = os.path.join(table_dir, "embedding_attribution_gene_set_query.csv")
        query_result.to_csv(query_path, index=False)
        print(f"\n[QUERY] Gene set: {gene_set}")
        print(query_result.to_string(index=False))
        print(f"[SAVE] {query_path}")

    # Plots — normal embedding-norm mode
    plot_bar(agg_df, fig_dir, args.top_n, args)
    plot_heatmap_and_dotplot(agg_df, fig_dir, args.top_n, args)

    # Contrastive mode: per-label diverging bar chart
    if args.contrastive_all_labels:
        print("\n" + "=" * 60)
        print("CONTRASTIVE MODE — running per-label centroid attribution")
        print("=" * 60)
        simple_df, vs_others_df = run_contrastive_all_labels(
            adata=adata,
            label_cell_positions=label_cell_positions,
            batch_indices=batch_indices,
            model=model,
            baseline=baseline,
            ig_steps=args.ig_steps,
            ig_batch_size=args.ig_batch_size,
            captum_internal_batch_size=args.captum_internal_batch_size,
            method=method,
            device=device,
            gene_names=gene_names,
        )
        if args.filter_broad_genes:
            simple_df = simple_df[~simple_df["gene"].apply(is_broad)].copy()
            vs_others_df = vs_others_df[~vs_others_df["gene"].apply(is_broad)].copy()

        simple_path = os.path.join(table_dir, "embedding_attribution_contrastive_simple_per_label.csv")
        vs_others_path = os.path.join(table_dir, "embedding_attribution_contrastive_vs_others_per_label.csv")
        simple_df.to_csv(simple_path, index=False)
        vs_others_df.to_csv(vs_others_path, index=False)
        print(f"[SAVE] {simple_path}")
        print(f"[SAVE] {vs_others_path}")

        plot_bar_diverging(simple_df, fig_dir, args.top_n,
                           fname="embedding_attribution_contrastive_simple_bar",
                           title="Contrastive attribution (vs own centroid): top {n} genes per label",
                           args=args)
        plot_bar_diverging(vs_others_df, fig_dir, args.top_n,
                           fname="embedding_attribution_contrastive_vs_others_bar",
                           title="Contrastive attribution (vs all other states): top {n} genes per label",
                           args=args)

    # Save run config
    config = {
        **vars(args),
        "input_h5ad_resolved": input_h5ad,
        "model_dir_resolved": model_dir,
        "model_type": args.model_type,
        "query_model_reference_h5ad": args.query_model_reference_h5ad,
        "obs_csv_resolved": obs_path,
        "n_cells_attributed": int(len(all_positions)),
        "n_genes": int(adata.n_vars),
        "target_scalar": (
            f"cosine_similarity_to_{args.contrastive_label}_centroid"
            if args.contrastive_label
            else (
                "sum(|z_observed*z(path)|)"
                if args.latent_weight_mode == "fixed_observed"
                else "sum(|z(path).detach()*z(path)|)"
            )
        ),
        "baseline": "zero",
        "normalization": "target_sum=1000 per cell",
        "signed_attribution_available": signed_matrix is not None,
    }
    if args.implementation == "signature_package":
        from importlib.metadata import version

        config["sc_signature_version"] = version("sc-signature")
    with open(os.path.join(outdir, "embedding_attribution_run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("[DONE] Embedding attribution complete.")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _compute_z_centroid(
    model, adata, label: str, batch_key: str, device: torch.device
) -> torch.Tensor:
    """Compute mean z for all cells with the given label."""
    import scarches as sca  # noqa: F811

    label_key = adata.obs.columns[
        [c for c in adata.obs.columns if "State" in c or "label" in c.lower()][-1]
    ]
    mask = adata.obs[label_key].astype(str) == label
    if mask.sum() == 0:
        raise ValueError(f"No cells found with label '{label}'")
    sub = adata[mask].copy()
    print(f"[CENTROID] Computing z centroid for '{label}': {mask.sum()} cells")
    z = model.get_latent_representation(sub)
    centroid = torch.tensor(z.mean(axis=0), dtype=torch.float32, device=device)
    return centroid


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _choose_method(args) -> str:
    if args.method == "manual":
        return "manual"
    try:
        import captum  # noqa: F401
        print("[METHOD] Captum IntegratedGradients")
        return "captum"
    except ImportError:
        if args.method == "captum":
            raise ImportError("Captum not installed. Run: pip install captum")
        print("[METHOD] Captum not available; using manual IG fallback")
        return "manual"


if __name__ == "__main__":
    main()
