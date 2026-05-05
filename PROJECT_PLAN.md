# NK Project Living Plan

This document is the working plan for the NK-cell SCVI/SCANVI project. It is meant
to be updated as results come in. At each checkpoint, we decide whether to keep the
plan, revise the plan, or go back to an earlier step.

## Current Scientific Goal

Build a reliable NK-cell latent space and use it to refine NK-state annotations.

The current working direction is:

1. Use `assay_clean` as the production batch key.
2. Use SCVI latent space for unsupervised structure discovery.
3. Use Leiden clustering to identify candidate biological substructure.
4. Use differential expression and marker biology to interpret clusters.
5. Create a refined annotation column.
6. Retrain SCANVI using the refined annotations.

## Completed Work

- [x] Modularized notebook code into a terminal-first project.
- [x] Preserved HPC input path for the merged AnnData file.
- [x] Moved project outputs into `NK_project/outputs`.
- [x] Added `logs/` folder and logging commands.
- [x] Created `assay_clean` preprocessing.
- [x] Trained baseline SCANVI using `batch_key = assay_clean`.
- [x] Saved SCANVI model, predictions, probabilities, metadata, and latents.
- [x] Generated SCANVI UMAP/prediction/confidence/certainty figures.
- [x] Trained SCVI using `batch_key = assay_clean`.
- [x] Saved SCVI model and latent space.
- [x] Ran validation-set Leiden clustering on SCVI latent space.
- [x] Switched Leiden backend to explicit Scanpy `flavor="igraph"`.
- [x] Generated validation Leiden overview figures.
- [x] Compared Leiden resolutions using focused radar metrics: ARI, NMI, silhouette, and size balance.
- [x] Selected `leiden_0_4` as the primary working resolution.
- [x] Generated initial cluster composition summaries for `leiden_0_4` and `leiden_0_8`.
- [x] Ran first-pass differential expression for validation-set `leiden_0_4` clusters.
- [x] Generated top-marker and curated-marker dotplots/matrixplots for `leiden_0_4`.
- [x] Decided that intentional B/T reference compartments should not be called contamination.
- [x] Added an interpretation worksheet builder for `leiden_0_4`.
- [x] Added an interpretation UMAP plotting script for draft compartment/state/review labels.
- [x] Added a focused pairwise DE script for `Cytokine-Stimulated_c9` vs `Cytokine-Stimulated_c12`.
- [x] Added curated draft refined labels that remove `candidate` wording and preserve key Leiden-supported splits.
- [x] Ran priority pairwise DE checks for cytokine-stimulated, tissue-resident transitional, lung-enriched, regulatory, and unconventional clusters.
- [x] Locked the current validation-set refined annotation mapping based on marker and pairwise DE evidence.

## Current Decision

Primary clustering resolution:

```text
leiden_0_4
```

Reason:

- It has the best balance of agreement with current `NK_State` and clustering quality.
- It avoids the very coarse imbalance of lower resolutions.
- It is less fragmented than higher resolutions.

Working interpretation:

- `leiden_0_4` is the main resolution for marker analysis and possible refined labels.
- `leiden_0_6` or `leiden_0_8` may still be useful later for substructure checks.

## Next Major Step: Differential Expression

Goal:

Identify marker genes for `leiden_0_4` clusters and decide whether clusters represent:

- existing NK_State labels
- meaningful subtypes within existing labels
- tissue-associated states
- dataset/assay artifacts
- transitional or ambiguous states
- candidate novel NK states

### Step 1: Cluster Summary

For each `leiden_0_4` cluster, summarize:

- number of cells
- dominant current `NK_State`
- dominant tissue
- dominant dataset
- dominant assay
- purity for each of these categories

Status:

- [x] Regenerate final `leiden_0_4` cluster summary table after the `igraph` Leiden update.
- [x] Review clusters for obvious dataset/assay domination.
- [ ] Decide which clusters are safe to interpret biologically.

Checkpoint:

If many clusters are strongly dataset- or assay-specific, pause before assigning
biological names and consider whether clustering should be repeated or interpreted
more conservatively.

### Step 2: Marker Discovery

First-pass method:

```python
sc.tl.rank_genes_groups(
    adata,
    groupby="leiden_0_4",
    method="wilcoxon",
)
```

Outputs to generate:

- top marker table for each cluster
- dotplot of top markers
- heatmap/matrixplot of selected markers
- cluster-vs-rest marker table

Status:

- [x] Run cluster-vs-rest marker analysis.
- [x] Save marker tables.
- [x] Generate marker dotplots.
- [x] Generate marker heatmaps/matrixplots.

Checkpoint:

If cluster markers are dominated by technical genes, stress genes, mitochondrial
genes, ribosomal genes, or dataset-specific artifacts, do not assign a new
biological label yet.

### Step 3: Biological Interpretation

Marker families to inspect:

```text
Cytotoxicity:
NKG7, GNLY, GZMB, GZMH, PRF1, CST7

Proliferation / cell cycle:
MKI67, TOP2A, STMN1, TYMS, HMGB2

Cytokine / interferon response:
ISG15, IFIT1, IFIT2, IFIT3, MX1, STAT1, IRF7

Tissue or activation context:
CXCR6, XCL1, XCL2, CCL3, CCL4, CCL5, ITGAE, ZNF683
```

Status:

- [ ] Review top markers cluster by cluster.
- [ ] Compare marker patterns to current `NK_State`.
- [ ] Identify clusters that may be subtypes.
- [ ] Identify clusters that should stay under existing labels.
- [ ] Identify clusters that are suspicious/technical.
- [ ] Build and review a cluster interpretation worksheet with separate `broad_compartment` and `refined_NK_state` draft calls.
- [ ] Generate interpretation UMAPs for draft broad compartment, refined state, and review priority.

Important convention:

- B-cell and T-cell clusters are intentional immune reference compartments, not
  contamination by default.
- Myeloid-like, epithelial-like, and erythroid-like clusters should be reviewed
  cautiously as possible non-NK signal, ambient RNA, doublets, or expected
  reference context depending on the source dataset.
- Refined NK labels should only be assigned to clusters whose broad compartment
  is NK.

Checkpoint:

Only create a new refined label when the cluster has both:

- coherent marker biology
- acceptable dataset/assay/tissue context

### Step 4: Refined Annotation Mapping

Create a mapping table:

```text
leiden_0_4 cluster -> refined label
```

Possible decisions for each cluster:

```text
keep existing label
split existing label into subtype
merge with nearby label
candidate novel state
technical/suspicious
unresolved
```

Status:

- [x] Create cluster-to-label mapping table.
- [ ] Add refined label column to AnnData.
- [ ] Save AnnData with refined labels.

Proposed refined label column:

```text
NK_State_refined
```

Checkpoint:

Before retraining SCANVI, review the refined label distribution. Avoid labels with
too few cells unless they are biologically essential and clearly marker-supported.

### Step 5: Retrain SCANVI

Train a new SCANVI model using:

```text
labels_key = NK_State_refined
batch_key = assay_clean
```

Status:

- [ ] Train refined-label SCANVI.
- [ ] Save model and outputs.
- [ ] Compare against original SCANVI.

Comparison metrics:

- validation macro F1
- validation weighted F1
- zero-shot macro F1
- full-dataset macro F1
- confusion between refined states
- confidence/certainty maps

Checkpoint:

If refined-label SCANVI performs poorly or collapses new labels, revisit the
annotation mapping and marker interpretation.

## Active Open Questions

- Are `leiden_0_4` clusters mostly biological or partly dataset/tissue artifacts?
- Which clusters split large labels such as Mature Cytotoxic and Transitional Cytotoxic?
- Are lung unknown clusters coherent biological states or dataset-specific groups?
- Should rare/held-out labels like `Unknown_Lung_3` be retained, merged, or renamed?
- Which marker genes best support any new refined labels?
- What distinguishes the two CB07/Flex-enriched Cytokine-Stimulated clusters
  (`leiden_0_4` clusters 9 and 12)?

## Rules For Updating This Plan

At every major result, update one of these sections:

- `Completed Work`
- `Current Decision`
- `Next Major Step`
- `Active Open Questions`

When results contradict the current plan:

1. State what changed.
2. Decide whether to revise the plan.
3. Record the new decision.
4. Continue from the most appropriate step instead of blindly moving forward.

This plan is intentionally flexible. The goal is not to force clusters into labels,
but to use clustering, markers, metadata, and model behavior together to refine the
NK annotation responsibly.
