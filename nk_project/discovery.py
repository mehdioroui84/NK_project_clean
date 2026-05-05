from __future__ import annotations

import os

import numpy as np
import pandas as pd
import scanpy as sc


def run_leiden_grid(
    adata,
    *,
    latent_key: str = "X_scVI",
    resolutions: list[float] | None = None,
    n_neighbors: int = 30,
    seed: int = 0,
    outdir: str | None = None,
    label_key: str = "NK_State",
    dataset_key: str = "dataset_id",
    assay_key: str = "assay_clean",
):
    """Run Leiden clustering across resolutions and save crosstabs."""
    if resolutions is None:
        resolutions = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]

    ad = adata.copy()
    if latent_key not in ad.obsm:
        raise KeyError(f"{latent_key!r} not found in adata.obsm")

    sc.pp.neighbors(ad, use_rep=latent_key, n_neighbors=n_neighbors, random_state=seed)
    sc.tl.umap(ad, min_dist=0.3, random_state=seed)

    summaries = []
    for resolution in resolutions:
        key = f"leiden_{str(resolution).replace('.', '_')}"
        sc.tl.leiden(
            ad,
            resolution=resolution,
            key_added=key,
            flavor="igraph",
            n_iterations=2,
            directed=False,
        )
        clusters = ad.obs[key].astype(str)
        counts = clusters.value_counts().sort_index()
        summaries.append(
            {
                "resolution": resolution,
                "n_clusters": int(counts.size),
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
            }
        )

        if outdir is not None:
            os.makedirs(outdir, exist_ok=True)
            counts.rename("n_cells").to_csv(os.path.join(outdir, f"{key}_cluster_sizes.csv"))
            _save_crosstab(ad, key, label_key, outdir)
            _save_crosstab(ad, key, dataset_key, outdir)
            _save_crosstab(ad, key, assay_key, outdir)

    summary_df = pd.DataFrame(summaries)
    if outdir is not None:
        summary_df.to_csv(os.path.join(outdir, "leiden_resolution_summary.csv"), index=False)
        ad.obs.to_csv(os.path.join(outdir, "obs_with_leiden.csv"))
        np.save(os.path.join(outdir, "X_umap.npy"), ad.obsm["X_umap"])

    return ad, summary_df


def _save_crosstab(adata, cluster_key: str, obs_key: str, outdir: str) -> None:
    if obs_key not in adata.obs:
        return
    tab = pd.crosstab(adata.obs[cluster_key].astype(str), adata.obs[obs_key].astype(str))
    safe_obs_key = obs_key.replace("/", "_")
    tab.to_csv(os.path.join(outdir, f"{cluster_key}_by_{safe_obs_key}.csv"))
