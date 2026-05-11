# NK Project Pipeline Organization Plan

This file records the cleanup plan for moving the project from exploratory scripts
to a small, ordered v1 pipeline. The goal is to make the active code easy to run
without losing the history of how the analysis was developed.

## Guiding Rules

1. Keep the current scientific outputs and paths stable unless there is a clear
   reason to change them.
2. Preserve the old working code before removing or combining scripts.
3. Combine scripts one workflow stage at a time, then test that stage on HPC.
4. Do not delete large output folders during code cleanup.
5. Use numbered script names while the workflow is still being stabilized.

## Preservation Snapshot

The pre-cleanup code-side project state is preserved here:

```text
legacy/pre_pipeline_cleanup/
```

This snapshot contains the previous `scripts/`, `configs/`, `nk_project/`,
`slurm/`, `experiments/`, `reports/`, and project documentation files. It does
not intentionally preserve generated `outputs/`.

## Canonical v1 Pipeline Scripts

The active `scripts/` folder should converge to this small set:

```text
scripts/01_train_scvi.py
scripts/02_run_leiden_discovery.py
scripts/03_run_marker_analysis.py
scripts/04_apply_refined_v1_labels.py
scripts/05_train_scanvi_refined_v1.py
scripts/06_evaluate_scanvi_refined_v1.py
scripts/07_run_scanvi_surgery.py
scripts/08_compare_batch_strategies.py
```

## Current Cleanup Status

Completed:

- Created `legacy/pre_pipeline_cleanup/`.
- Removed obsolete validation/subsampled scripts from active `scripts/`.
- Combined full marker DE and curated marker plotting into:

```text
scripts/03_run_marker_analysis.py
```

- Combined full Leiden clustering, resolution plots, single-resolution overview,
  and annotation worksheets into:

```text
scripts/02_run_leiden_discovery.py
```

- Created a single refined-v1 SCANVI evaluation entrypoint that runs full plots,
  held-out zero-shot plots, and held-out by-dataset summaries:

```text
scripts/06_evaluate_scanvi_refined_v1.py
```

- Moved the refined-v1 SCANVI evaluation helper code out of active `scripts/`
  and into:

```text
nk_project/evaluation/
```

- Renamed the remaining active user-facing scripts into the numbered pipeline
  order:

```text
scripts/01_train_scvi.py
scripts/04_apply_refined_v1_labels.py
scripts/05_train_scanvi_refined_v1.py
scripts/07_run_scanvi_surgery.py
scripts/08_compare_batch_strategies.py
```

Still to do:

- Run lightweight HPC checks for `04`, `05 --dry-run`, `07 --dry-run`, and
  `08` with cached/smoke-test options as needed.
- Commit the cleanup after the HPC checks pass.

## Active Output Folders

Keep these as canonical v1 outputs:

```text
outputs/leiden_discovery/
outputs/markers/full/leiden_0_4/
outputs/refined_annotation_v1/
outputs/refined_scanvi_v1/
outputs/refined_scanvi_v1_surgery_new_assays_only_e10_lr1e4/
```

Keep these as provenance/older analysis for now:

```text
outputs/leiden_validation/
outputs/markers/validation/
outputs/batch_strategy_reevaluation*/
outputs/refined_scanvi_v1_surgery*/
```

Do not reorganize generated output folders until the code pipeline is stable.

## HPC Testing Order

After each combined script is copied to HPC, test that stage before continuing:

1. `02_run_leiden_discovery.py --skip-clustering`
2. `03_run_marker_analysis.py --skip-rank-genes`
3. `03_run_marker_analysis.py`
4. `05_train_scanvi_refined_v1.py --dry-run`
5. `06_evaluate_scanvi_refined_v1.py`
6. `07_run_scanvi_surgery.py --new-assays-only --surgery-epochs 10 --surgery-lr 1e-4`

The full model-training steps should only be rerun when we intentionally want to
regenerate model outputs.
