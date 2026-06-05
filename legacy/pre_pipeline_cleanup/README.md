# NK Project

Modular SCVI/SCANVI project for NK-cell reference modeling, batch-key evaluation,
and latent-space discovery.

The current production decision is:

- Use `assay_clean` as the main batch key.
- Use SCVI for unsupervised latent discovery and Leiden clustering.
- Use SCANVI for supervised label transfer/classification.
- Keep dataset/composite/refiner comparisons as experiments, not the default path.

## Folder Layout

```text
NK_project/
  configs/
    default_config.py
  nk_project/
    preprocessing.py
    qc.py
    training_plan.py
    splits.py
    evaluate.py
    metrics.py
    plotting.py
    discovery.py
    workflows.py
    io_utils.py
  scripts/
    profile_batches.py
    train_scvi.py
    train_scanvi.py
    run_leiden_discovery.py
    run_batch_key_comparison.py
  logs/
    .gitkeep
  experiments/
    batch_key_comparison.py
    adversarial_refiner.py
  slurm/
    train_scvi.sbatch
    train_scanvi.sbatch
    leiden_discovery.sbatch
```

## Install

Use the environment available on Kubernetes/HPC. The important packages are listed
in `requirements.txt`, but GPU/CUDA-compatible versions should match your cluster.

## Run From Terminal

From the project folder:

```bash
mkdir -p logs

python scripts/profile_batches.py 2>&1 | tee logs/profile_batches.log
python scripts/train_scvi.py 2>&1 | tee logs/scvi_train.log
python scripts/train_scanvi.py 2>&1 | tee logs/scanvi_train.log
python scripts/plot_scanvi_results.py 2>&1 | tee logs/scanvi_plot.log
python scripts/plot_latent_metrics.py 2>&1 | tee logs/latent_metrics.log
python scripts/run_leiden_validation.py 2>&1 | tee logs/leiden_validation.log
python scripts/run_leiden_discovery.py 2>&1 | tee logs/leiden_discovery.log
```

Optional historical comparison:

```bash
python scripts/run_batch_key_comparison.py 2>&1 | tee logs/batch_key_comparison.log
```

## Main Workflow

1. Train final SCVI with `batch_key = assay_clean`.
2. Save SCVI model, latent embeddings, metadata, UMAP, and clustering inputs.
3. Run Leiden clustering on SCVI latent space over multiple resolutions.
4. Inspect cluster biology, assay balance, dataset balance, and markers.
5. Create a refined label column only after biological validation.
6. Train SCANVI with original `NK_State` as the baseline classifier.
7. Later train SCANVI with a refined label column such as `NK_State_refined`.

See `PROJECT_PLAN.md` for the living analysis plan and progress checklist.

## Important Paths

The default config keeps the original HPC data paths:

```text
/rsrch5/home/genomic_med/suorouji/projects/lsf_run/cellxgene_plus_CB07.h5ad
```

Outputs default to:

```text
/rsrch5/home/genomic_med/suorouji/projects/lsf_run/NK_project_outputs
```

Change these in `configs/default_config.py` if needed.
