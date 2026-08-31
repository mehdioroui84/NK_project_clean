# Cluster-level mitochondrial and dataset evidence

These tables are preserved inputs for annotation-agent quality and context
assessment at Leiden resolutions 0.1 and 0.5.

## Files

- `leiden_0_1_dataset_mt_percentage_summary.tsv`
- `leiden_0_5_dataset_mt_percentage_summary.tsv`

## Row definition

Each row represents one Leiden cluster and dataset combination.

## Columns

- `leiden_resolution`: clustering resolution used for the cluster identifier.
- `cluster`: cluster identifier stored as a string.
- `dataset_id`: contributing dataset identifier.
- `n_cells`: cells from this dataset in this cluster.
- `cluster_percentage`: percentage of the cluster contributed by this dataset;
  percentages sum to approximately 100 within each cluster.
- `median_mt_percentage`: median mitochondrial percentage for cells in this
  cluster-dataset stratum.

## Coverage and interpretation

Both tables cover 284,545 cells. This matches the CellxGene-matched full-gene
source and does not cover the 26,926 excluded in-house CB07/no-suffix cells in
the 311,471-cell model AnnData. The annotation agent must report this coverage
limitation and must not treat missing mitochondrial evidence for those cells as
normal mitochondrial evidence.

These tables should be used to evaluate dataset dominance and mitochondrial
patterns together. Neither high dataset concentration nor high mitochondrial
percentage alone proves a technical artifact.
