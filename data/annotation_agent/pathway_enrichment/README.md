# GO Biological Process enrichment evidence

These files are preserved as received for annotation-agent evidence at Leiden
resolutions 0.1 and 0.5.

## Files

- `go_bp_enrichment_leiden01.csv`: results for all 11 Leiden 0.1 clusters.
- `go_bp_enrichment_all_leiden05.csv`: results for all 26 Leiden 0.5 clusters.

Each row represents one cluster and GO Biological Process term. The files
contain the selected and GO-mapped gene counts, enrichment background and term
sizes, overlap counts and genes, gene ratios, raw and adjusted p-values, and an
FDR < 0.05 indicator. Empty `overlap_genes` values are expected for terms with
no selected-gene overlap.

## Agent usage

The annotation agent should receive a compact, deterministic summary rather
than these complete tables. Retain significant terms, require sufficient gene
overlap, reduce redundant GO terms, and always preserve the overlap genes that
support each retained pathway. Lack of a significant pathway is absence of
pathway evidence, not proof that a cluster lacks biological function.

Pathways should support functional interpretation and naming but should not by
themselves define cell lineage. Curated project marker lists are intentionally
excluded from the initial data-driven annotation workflow.

## Background-universe check required

Both received files report `background_size = 18000`. The current attribution
analysis selects genes from the 2,007 model genes. Before these enrichment
statistics are used as final production evidence, verify or recompute the
over-representation analysis using the genes that were actually eligible for
attribution selection (the GO-mapped subset of the 2,007 model genes). Using a
broader genome-scale background may overstate enrichment significance.
