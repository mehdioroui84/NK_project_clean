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
    01_train_scvi.py
    02_run_leiden_discovery.py
    03_run_marker_analysis.py
    04_apply_refined_v1_labels.py
    05_train_scanvi_refined_v1.py
    06_evaluate_scanvi_refined_v1.py
    07_run_scanvi_surgery.py
    08_compare_batch_strategies.py
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

python scripts/01_train_scvi.py 2>&1 | tee logs/01_train_scvi.log
python scripts/02_run_leiden_discovery.py 2>&1 | tee logs/02_run_leiden_discovery.log
python scripts/03_run_marker_analysis.py 2>&1 | tee logs/03_run_marker_analysis.log
python scripts/04_apply_refined_v1_labels.py 2>&1 | tee logs/04_apply_refined_v1_labels.log
python scripts/05_train_scanvi_refined_v1.py 2>&1 | tee logs/05_train_scanvi_refined_v1.log
python scripts/06_evaluate_scanvi_refined_v1.py 2>&1 | tee logs/06_evaluate_scanvi_refined_v1.log
```

Optional comparisons:

```bash
python scripts/07_run_scanvi_surgery.py --new-assays-only 2>&1 | tee logs/07_run_scanvi_surgery.log
python scripts/08_compare_batch_strategies.py 2>&1 | tee logs/08_compare_batch_strategies.log
```

## Main Workflow

1. Train final SCVI with `batch_key = assay_clean`.
2. Run full-data Leiden clustering at resolutions `0.2`, `0.3`, and `0.4`.
3. Inspect cluster biology and marker evidence, then use `leiden_0_4`.
4. Apply the refined-v1 labels from the `leiden_0_4` cluster map.
5. Train SCANVI with `NK_State_refined`.
6. Evaluate full-data, held-out zero-shot, and by-dataset performance.
7. Optionally compare SCANVI surgery and batch-key strategies.

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
