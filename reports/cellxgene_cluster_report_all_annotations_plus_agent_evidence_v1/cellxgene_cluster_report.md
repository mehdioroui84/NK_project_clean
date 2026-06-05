# CellXGene Cluster Annotation Report

This report summarizes CellXGene annotation composition, Yuntao annotation composition, current agent labels, agent rationale, and cluster-specific positive/negative DE markers.

## Figures

- `figures/marker_dotplot.png`

## Cluster Summary

|   leiden_0_4 |   n_cells | top_tissue   |   top_tissue_percent |   cellxgene_pct_NK |   cellxgene_pct_T |   cellxgene_pct_B | top_yuntao_annotation   |   top_yuntao_annotation_percent | final_structured_label                  | free_label                                                                               | needs_human_review   |
|-------------:|----------:|:-------------|---------------------:|-------------------:|------------------:|------------------:|:------------------------|--------------------------------:|:----------------------------------------|:-----------------------------------------------------------------------------------------|:---------------------|
|            0 |     49483 | blood        |                94.63 |              99.82 |              0.09 |              0.09 | Mature Cytotoxic        |                           81.5  | NK1_Cytotoxic_activated                 | NK1_cytotoxic_activated_possible_myeloid_contamination                                   | True                 |
|            1 |      8720 | blood        |                97.1  |              99.87 |              0.13 |              0    | Proliferative           |                           91.42 | cNK_Proliferating                       | cNK_Proliferating_cycling_cytotoxic_NK                                                   | True                 |
|            2 |     47134 | blood        |                93.23 |              99.79 |              0.21 |              0    | Mature Cytotoxic        |                           89.56 | NK1_Cytotoxic_activated                 | NK1_Cytotoxic_activated_FGFBP2+_GZMH+_blood                                              | True                 |
|            3 |     12339 | spleen       |                39.65 |              87.45 |             12.53 |              0.02 | T                       |                           72.45 | Non-NK                                  | CD3+_T_cells_with_IFNG_GZMK_inflammatory_and_ER_stress_signature_likely_T_lineage_not_NK | False                |
|            4 |     26628 | decidua      |                95.61 |              99.61 |              0.39 |              0    | Transitional Cytotoxic  |                           92.99 | NK2_Chemokine_inflammatory              | NK2_Chemokine_inflammatory_decidual_chemokine-producing_NK                               | True                 |
|            5 |     10058 | cord blood   |                82.44 |              99.47 |              0.08 |              0.45 | Cytokine-Stimulated     |                           56.59 | L6_Developmental_immature_Proliferating | Developmental_immature_Proliferating_cord_blood                                          | True                 |
|            6 |      9557 | lung         |                47.17 |              99.91 |              0.06 |              0.03 | Mature Cytotoxic        |                           33.6  | NK2_Chemokine_inflammatory              | NK2_Chemokine_inflammatory_AREG+_cytotoxic_mito_stress                                   | True                 |
|            7 |      1361 | kidney       |                99.12 |              99.78 |              0    |              0.22 | Unknown_Kidney          |                           98.31 | cNK_ER_stress_UPR                       | Stressed_kidney_epithelial_likely_tubular_cells                                          | True                 |
|            8 |      9537 | spleen       |                55.75 |              99.44 |              0.49 |              0.06 | T                       |                           45.28 | NK2_Chemokine_inflammatory              | NK2_chemokine-rich_low-cytotoxic_tissue-associated                                       | True                 |
|            9 |     24357 | blood        |                97.44 |              99.78 |              0.16 |              0.06 | Transitional Cytotoxic  |                           88.43 | NK1_Checkpoint_exhausted                | Myeloid-like_S100A8_S100A9_LYZ_with_high_mitochondrial_gene_expression                   | True                 |
|           10 |     38400 | lung         |                95.79 |              99.71 |              0.28 |              0.02 | Unknown_Lung_6          |                           69.84 | NK1_Chemokine_inflammatory              | NK1_Chemokine_inflammatory_CX3CR1+_FCGR3A+_cytotoxic_CCL4-high                           | True                 |
|           11 |      3056 | cord blood   |                93.88 |              99.97 |              0.03 |              0    | Mature Cytotoxic        |                           86.62 | NK2_Homeostatic_quiescent               | NK2_cord_blood_IL2RB+_ZBTB16+_KLRF1-low_-_CD56bright-like_quiescent                      | True                 |
|           12 |     15756 | cord blood   |                92.32 |              99.98 |              0.02 |              0    | Cytokine-Stimulated     |                           74.19 | NK1_Proliferating                       | Proliferating_cytotoxic_NK_NK1                                                           | True                 |
|           13 |      2421 | decidua      |                96.49 |              99.92 |              0.08 |              0    | Transitional Cytotoxic  |                           95.62 | NK2_Chemokine_inflammatory              | NK2_Chemokine_inflammatory_decidual_XCL+_IFN_tissue-resident                             | True                 |
|           14 |      5263 | blood        |                55.39 |              99.96 |              0.04 |              0    | Mature Cytotoxic        |                           56.26 | NK2_CIMP_cytokine_primed_memory_like    | NK2_cytokine-primed_memory-like_TCF7+_SELL+_GZMK+_with_XCL1_2_chemokine_program          | True                 |
|           15 |     17939 | lung         |                93.11 |              97.24 |              2.76 |              0.01 | T                       |                           26.03 | NK2_Chemokine_inflammatory              | NK2_Chemokine_inflammatory_lung_ER-stress_cytokine_high                                  | True                 |
|           16 |     14707 | blood        |                40.93 |              34.11 |              0.06 |             65.83 | B                       |                           65.96 | Non-NK                                  | B_cells_mature_B-lineage                                                                 | False                |
|           17 |      6316 | blood        |                87.27 |              99.97 |              0    |              0.03 | Mature Cytotoxic        |                           77.55 | NK2_Checkpoint_exhausted                | NK2_Checkpoint_exhausted_IKZF2+_low_cytotoxic_program                                    | True                 |
|           18 |      5730 | decidua      |                99.9  |              99.37 |              0.63 |              0    | Regulatory              |                           99.16 | Non-NK                                  | Decidual_stromal_epithelial-like_Non-NK                                                  | False                |
|           19 |      1395 | spleen       |                95.27 |              97.06 |              0    |              2.94 | B                       |                           83.01 | NK2_ER_stress_UPR                       | B_cells_spleen                                                                           | True                 |
|           20 |      1719 | lung         |                70.74 |               1.63 |              0.12 |             98.25 | B                       |                           98.25 | Non-NK                                  | B_cells_MS4A1+_CD79A+_IGKC+                                                              | False                |
|           21 |      8193 | bone marrow  |                95.26 |             100    |              0    |              0    | Unknown_BM_1            |                           56.6  | cNK_Cytotoxic_activated                 | cNK_Cytotoxic_activated_erythroid_contamination                                          | True                 |
|           22 |       929 | blood        |                95.8  |              99.78 |              0.22 |              0    | Proliferative           |                           95.37 | Non-NK                                  | Myeloid_monocyte_macrophage_-_Non-NK                                                     | False                |
|           23 |      4071 | lung         |                99.98 |             100    |              0    |              0    | Unknown_Lung_4          |                           94.96 | NK1_Cytotoxic_activated                 | NK1_Cytotoxic_activated_lung_with_epithelial_signal                                      | True                 |
|           24 |      1124 | lung         |                79.63 |              99.91 |              0    |              0.09 | Unknown_Lung_1          |                           71.8  | Non-NK                                  | Non-NK_lung_stromal_epithelial-like_KRT19_KRT8_with_macrophage-associated_genes          | True                 |

## Cluster Details

### Cluster 0

- n cells: 49483
- Top tissue: blood (94.63%)
- CellXGene composition: NK 99.82%, T 0.09%, B 0.09%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK1_Cytotoxic_activated
- Free agent label: NK1_cytotoxic_activated_possible_myeloid_contamination
- Agent rationale: Strong NK cytotoxic signature (high GNLY, NKG7, FGFBP2, GZMH) and taxonomy match to NK1; state match Cytotoxic_activated. However, strong myeloid markers S100A8/S100A9/LYZ and many mitochondrial genes indicate possible contamination or stressed cells. Pairwise split audit: Pairwise DE and cluster markers support NK1 Cytotoxic_activated identity (high GNLY, NKG7, FGFBP2, GZMH) while showing myeloid marker signal (S100A8, S100A9, LYZ) and elevated mitochondrial/stress genes. Pairwise comparison to cluster 23 preserves NK cytotoxic program but highlights myeloid/stress features unique to cluster 0, consistent with possible myeloid contamination or stressed NK cells.
- Human review reason: Mixed signals: clear NK cytotoxic program vs strong myeloid marker expression and depleted EOMES/NCAM1 — require human curation to resolve contamination vs true biology.; Epithelial markers are highly enriched in cluster 23 and could reflect ambient contamination, doublets, or true tissue-resident NK transcriptional interaction; recommend manual QC (doublet detection / ambient RNA assessment) and contextual review.
- Top markers: 
- Marker details: S100A8 logFC=3.11 pct=0.43 ref_pct=0.10; B2M logFC=0.95 pct=1.00 ref_pct=0.99; S100A9 logFC=2.77 pct=0.42 ref_pct=0.12; IFITM1 logFC=2.52 pct=0.83 ref_pct=0.56; S100A4 logFC=1.89 pct=0.93 ref_pct=0.74; NKG7 logFC=1.58 pct=0.99 ref_pct=0.88; MT-CO2 logFC=2.88 pct=1.00 ref_pct=0.66; HLA-B logFC=1.49 pct=0.98 ref_pct=0.86; FGFBP2 logFC=2.21 pct=0.79 ref_pct=0.45; MT-ND4L logFC=2.27 pct=0.78 ref_pct=0.44

### Cluster 1

- n cells: 8720
- Top tissue: blood (97.1%)
- CellXGene composition: NK 99.87%, T 0.13%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: cNK_Proliferating
- Free agent label: cNK_Proliferating_cycling_cytotoxic_NK
- Agent rationale: Strong, specific proliferation program (MKI67, TOP2A, PCNA, TYMS, RRM2, etc.) and taxonomy state match = Proliferating; pan-NK markers (GNLY, NKG7, GZMA/B, PRF1, FCGR3A, KLRF1) are broadly detected and NK cytotoxic module is high, and cluster is blood-enriched, consistent with circulating NKs undergoing cell cycle.
- Human review reason: T-lineage marker positivity (CD3 genes) despite strong NK marker expression and proliferation signature warrants inspection for doublets/annotation ambiguity.
- Top markers: 
- Marker details: STMN1 logFC=7.49 pct=0.98 ref_pct=0.15; RRM2 logFC=5.52 pct=0.73 ref_pct=0.07; CLSPN logFC=4.99 pct=0.66 ref_pct=0.06; TYMS logFC=5.42 pct=0.87 ref_pct=0.10; PCLAF logFC=4.89 pct=0.66 ref_pct=0.07; TK1 logFC=4.71 pct=0.69 ref_pct=0.07; PCNA logFC=4.72 pct=0.84 ref_pct=0.14; CENPU logFC=4.27 pct=0.62 ref_pct=0.07; CENPM logFC=4.19 pct=0.67 ref_pct=0.08; ASF1B logFC=4.11 pct=0.59 ref_pct=0.07

### Cluster 2

- n cells: 47134
- Top tissue: blood (93.23%)
- CellXGene composition: NK 99.79%, T 0.21%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK1_Cytotoxic_activated
- Free agent label: NK1_Cytotoxic_activated_FGFBP2+_GZMH+_blood
- Agent rationale: Strong positive DE of NK cytotoxic program genes (FGFBP2, GZMH, NKG7, GNLY) and taxonomy matching (NK1 strong). Top DE and curated marker means show high NK_cytotoxic and individual cytotoxic genes; cluster is blood-enriched supporting circulating cytotoxic NKs. Pairwise split audit: Pairwise DE and cluster summaries support circulating cytotoxic NKs: strong NK cytotoxic program (FGFBP2, GZMH, NKG7, GNLY; curated NK_cytotoxic mean 3.69), top tissue = blood (93% of cells). The pairwise comparison shows absence of lung-specific epithelial markers that define cluster 23; mitochondrial differences dominate some DE but reflect tissue/batch differences rather than loss of NK identity.
- Human review reason: TRBC1 positivity and pervasive mitochondrial upregulation raise the possibility of T-cell contamination/doublets or technical stress; recommend manual review to confirm true NK lineage and exclude artifacts.; Pairwise DE supports a distinct free label.
- Top markers: 
- Marker details: IFITM1 logFC=3.02 pct=0.93 ref_pct=0.54; FGFBP2 logFC=2.64 pct=0.85 ref_pct=0.44; MT-ND5 logFC=2.35 pct=0.90 ref_pct=0.54; MT-ND2 logFC=2.51 pct=0.95 ref_pct=0.56; MT-CO1 logFC=3.03 pct=1.00 ref_pct=0.67; NKG7 logFC=1.55 pct=1.00 ref_pct=0.88; MT-ND1 logFC=2.40 pct=0.94 ref_pct=0.55; MT-ATP6 logFC=2.54 pct=0.98 ref_pct=0.63; KLF2 logFC=2.05 pct=0.66 ref_pct=0.38; CRIP1 logFC=1.86 pct=0.74 ref_pct=0.43

### Cluster 3

- n cells: 12339
- Top tissue: spleen (39.65%)
- CellXGene composition: NK 87.45%, T 12.53%, B 0.02%, Other %
- Top CellXGene annotations: 
- Structured agent label: Non-NK
- Free agent label: CD3+_T_cells_with_IFNG_GZMK_inflammatory_and_ER_stress_signature_likely_T_lineage_not_NK
- Agent rationale: Strong T-lineage markers (CD3D, CD3G, TRBC1, CD3E, TRAC) are highly upregulated (logFC >1). Canonical pan-NK cytotoxic markers are strongly depleted (GNLY, GZMB, PRF1, NCAM1, KLRF1). Cluster also shows upregulation of IFNG and GZMK and heat-shock/UPR genes (HSPA1A/B, DNAJB1) consistent with an activated/inflammatory T cell with stress signature. Taxonomy hits suggesting NK states are moderate but contradicted by clear T marker enrichment; no pairwise DE available.
- Top markers: 
- Marker details: HSPA1B logFC=5.56 pct=0.63 ref_pct=0.10; CD3D logFC=4.27 pct=0.68 ref_pct=0.12; HSPA6 logFC=4.95 pct=0.41 ref_pct=0.05; GZMK logFC=3.52 pct=0.67 ref_pct=0.17; HSPA1A logFC=4.47 pct=0.57 ref_pct=0.15; DNAJB1 logFC=4.22 pct=0.76 ref_pct=0.30; HBG2 logFC=5.04 pct=0.12 ref_pct=0.00; H2AC19 logFC=3.98 pct=0.19 ref_pct=0.02; CD3G logFC=2.98 pct=0.40 ref_pct=0.07; HSPD1 logFC=3.17 pct=0.66 ref_pct=0.24

### Cluster 4

- n cells: 26628
- Top tissue: decidua (95.61%)
- CellXGene composition: NK 99.61%, T 0.39%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK2_Chemokine_inflammatory
- Free agent label: NK2_Chemokine_inflammatory_decidual_chemokine-producing_NK
- Agent rationale: Strong enrichment of chemokine/regulatory NK markers (XCL1, XCL2, KLRC1) and taxonomy top-match NK2; marker program hits include XCL1/XCL2/CCL3/CCL4. Pan-NK summary shows broadly preserved NK program (median pan-NK logFC positive, high pct_nz). Negative DE for cytotoxic NK genes (FGFBP2, FCGR3A) supports a non-cytotoxic, regulatory/chemokine NK phenotype in tissue.
- Human review reason: Mixed lineage signals: although NK2/chemokine program is strong (XCL1/XCL2/KLRC1), sizable stromal/epithelial marker enrichment (LUM, DCN, COL1A1) and T marker presence raise concern for doublets or mis-clustering; recommend manual inspection (UMAP, protein/ADTs if available) and filtering.
- Top markers: 
- Marker details: KRT81 logFC=5.68 pct=0.54 ref_pct=0.02; SPINK2 logFC=5.94 pct=0.45 ref_pct=0.02; IGFBP2 logFC=4.94 pct=0.43 ref_pct=0.02; CAPG logFC=4.68 pct=0.76 ref_pct=0.12; COTL1 logFC=5.15 pct=0.94 ref_pct=0.29; KRT86 logFC=4.48 pct=0.52 ref_pct=0.05; LINC00996 logFC=4.24 pct=0.47 ref_pct=0.04; TNFRSF18 logFC=4.12 pct=0.75 ref_pct=0.15; KLRC1 logFC=4.14 pct=0.88 ref_pct=0.29; XCL1 logFC=4.51 pct=0.73 ref_pct=0.18

### Cluster 5

- n cells: 10058
- Top tissue: cord blood (82.44%)
- CellXGene composition: NK 99.47%, T 0.08%, B 0.45%, Other %
- Top CellXGene annotations: 
- Structured agent label: L6_Developmental_immature_Proliferating
- Free agent label: Developmental_immature_Proliferating_cord_blood
- Agent rationale: Strong, specific proliferation program (MKI67, TOP2A, TYMS, CDK1, MCM7) with immature/lymphoid/activation markers (IL2RA, CCR7, IRF4, TOX2) consistent with proliferating developmental/immature NKs in cord blood. Weak NK2 support via CCR7 and MYC context; pan-NK cytotoxic markers KLRF1 and NKG7 are depleted but many pan-NK markers remain detected, so best call is developmental/immature NK in a proliferative state, with possible T-lineage signal needing review.
- Human review reason: T-cell markers (TRAC, CD3E) are positive while key NK cytotoxic markers are depleted; confirm whether this cluster represents proliferating immature NKs vs proliferating T cells or doublets in cord blood.
- Top markers: 
- Marker details: IL2RA logFC=5.94 pct=0.62 ref_pct=0.03; IRF4 logFC=6.09 pct=0.72 ref_pct=0.06; CCR7 logFC=6.07 pct=0.61 ref_pct=0.04; H1-5 logFC=6.73 pct=0.81 ref_pct=0.07; TOX2 logFC=5.36 pct=0.68 ref_pct=0.06; HELLS logFC=5.74 pct=0.81 ref_pct=0.09; TYMS logFC=6.10 pct=0.83 ref_pct=0.10; TK1 logFC=5.44 pct=0.71 ref_pct=0.07; YBX3 logFC=5.14 pct=0.72 ref_pct=0.07; IL12RB2 logFC=5.20 pct=0.75 ref_pct=0.08

### Cluster 6

- n cells: 9557
- Top tissue: lung (47.17%)
- CellXGene composition: NK 99.91%, T 0.06%, B 0.03%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK2_Chemokine_inflammatory
- Free agent label: NK2_Chemokine_inflammatory_AREG+_cytotoxic_mito_stress
- Agent rationale: Taxonomy hits (IL7R core, AREG support) point to NK2; state hit CCL4 supports Chemokine_inflammatory. Strong NK cytotoxic program present in curated means (NKG7, GNLY, GZMB, PRF1) despite modest median pan-NK logFC. Top DE shows AREG, FKBP5, CCL4 and elevated mitochondrial genes, consistent with activated/stressed chemokine-producing NK subset. However, erythroid markers (HBB, HBA1/2) are also enriched, suggesting contamination or mixed signal. Pairwise split audit: Pairwise DE (6_vs_13) shows strong enrichment of canonical circulating/cytotoxic NK markers (FGFBP2, FCGR3A, KLRF1) and AREG/LINC‑PINT in cluster 6 vs cluster 13, together with elevated mitochondrial genes and FKBP5/CCL4 in top DE and curated means. These signatures indicate a chemokine-producing NK2-like population with retained cytotoxic program and mitochondrial stress, distinct from the XCL1/XCL2+ IFN/tissue-resident profile of cluster 13. Note: erythroid markers (HBB, HBA1/2) are detected and suggest possible contamination that should be considered.
- Human review reason: Erythroid gene co-expression and strong mitochondrial/stress signature could reflect doublets, ambient RNA, or stressed NK cells; NK2/Chemokine_inflammatory assignment is plausible but uncertain and should be validated.; Cluster 6 shows erythroid marker signal; cluster 13 shows CD3E/T signal — review for doublets/contamination and confirm tissue-residency assignment with orthogonal markers.
- Top markers: 
- Marker details: ENSG00000270240.2 logFC=5.40 pct=0.08 ref_pct=0.00; LINC-PINT logFC=3.45 pct=0.47 ref_pct=0.12; FKBP5 logFC=3.15 pct=0.52 ref_pct=0.16; AREG logFC=3.37 pct=0.68 ref_pct=0.28; ENSG00000227240.2 logFC=4.90 pct=0.07 ref_pct=0.00; LINGO2 logFC=3.53 pct=0.14 ref_pct=0.02; FOSL2-AS1 logFC=3.64 pct=0.09 ref_pct=0.01; ENSG00000276241.1 logFC=3.78 pct=0.10 ref_pct=0.01; TEX14 logFC=3.53 pct=0.13 ref_pct=0.01; PLCB1 logFC=2.64 pct=0.32 ref_pct=0.08

### Cluster 7

- n cells: 1361
- Top tissue: kidney (99.12%)
- CellXGene composition: NK 99.78%, T 0.0%, B 0.22%, Other %
- Top CellXGene annotations: 
- Structured agent label: cNK_ER_stress_UPR
- Free agent label: Stressed_kidney_epithelial_likely_tubular_cells
- Agent rationale: Top positive DE genes are epithelial/kidney-associated and stress/mitochondrial: FXYD2, GSTA1, GPX3, PDZK1IP1, APOE and strong mitochondrial genes (MT-*) and heat-shock genes (HSPA1A/HSPA1B/DNAJB1). Non-NK marker summary matched Stromal_Epithelial (KRT8, KRT19). Pan-NK markers are overall depleted (strong depletion of NCAM1, EOMES, PRF1), and taxonomy hit indicates ER_stress_UPR support. Together these indicate an epithelial, stressed cell population rather than an NK subtype.
- Human review reason: Conflicting signals: robust epithelial/stress DE and KRT8/KRT19 vs elevated curated GNLY/NKG7 means. Recommend review for doublets, ambient cytotoxic transcript contamination, and validation of epithelial marker expression at single-cell level.
- Top markers: 
- Marker details: GSTA1 logFC=9.35 pct=0.37 ref_pct=0.00; MT1H logFC=8.34 pct=0.40 ref_pct=0.00; MT1G logFC=8.65 pct=0.74 ref_pct=0.01; FXYD2 logFC=8.07 pct=0.60 ref_pct=0.01; RBP5 logFC=7.05 pct=0.24 ref_pct=0.00; PDZK1IP1 logFC=6.40 pct=0.31 ref_pct=0.01; DPYS logFC=7.68 pct=0.08 ref_pct=0.00; GPX3 logFC=5.64 pct=0.48 ref_pct=0.02; ADIRF logFC=5.95 pct=0.20 ref_pct=0.00; HSPA1A logFC=6.16 pct=0.91 ref_pct=0.16

### Cluster 8

- n cells: 9537
- Top tissue: spleen (55.75%)
- CellXGene composition: NK 99.44%, T 0.49%, B 0.06%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK2_Chemokine_inflammatory
- Free agent label: NK2_chemokine-rich_low-cytotoxic_tissue-associated
- Agent rationale: Cluster shows strong positive expression of chemokine/tissue-associated NK markers (CXCR6, XCL1, XCL2, KLRB1) and GZMK, while classic cytotoxic pan-NK genes are depleted (GNLY, GZMB, PRF1). Taxonomy matching: NK2 (strong) and Chemokine_inflammatory (moderate). No strong non-NK lineage signal.
- Human review reason: Ambiguity between NK2 and tissue-resident (trNK) identity due to strong CXCR6 and tissue enrichment; recommend orthogonal marker checks (e.g., CD49a, CD69, KIRs) or protein data to confirm.
- Top markers: 
- Marker details: CXCR6 logFC=4.81 pct=0.36 ref_pct=0.03; GZMK logFC=3.89 pct=0.68 ref_pct=0.18; CEBPD logFC=3.17 pct=0.66 ref_pct=0.24; CD160 logFC=3.09 pct=0.61 ref_pct=0.21; CMC1 logFC=3.16 pct=0.88 ref_pct=0.53; KLRB1 logFC=2.58 pct=0.94 ref_pct=0.66; MT-ND4 logFC=2.63 pct=0.95 ref_pct=0.66; MT-ND2 logFC=2.58 pct=0.94 ref_pct=0.61; MT-ND3 logFC=2.70 pct=0.94 ref_pct=0.62; MT-CO3 logFC=2.76 pct=0.98 ref_pct=0.70

### Cluster 9

- n cells: 24357
- Top tissue: blood (97.44%)
- CellXGene composition: NK 99.78%, T 0.16%, B 0.06%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK1_Checkpoint_exhausted
- Free agent label: Myeloid-like_S100A8_S100A9_LYZ_with_high_mitochondrial_gene_expression
- Agent rationale: Cluster shows clear positive myeloid marker signal (S100A8, S100A9, LYZ; median positive logFC ~1.59) and multiple myeloid markers passing logFC>=1. Pan-NK markers are strongly depleted (GZMA, NKG7, EOMES, KLRF1; median pan-NK logFC = -1.02). The top positive genes are dominated by mitochondrial genes (MT-CO1, MT-CO2, MT-CYB, MT-ND*), consistent with a high-mitochondrial / stress signature rather than an NK subtype program. Taxonomy matches (CX3CR1, ZEB2) are weak and do not overcome the strong myeloid signal and pan-NK depletion.
- Human review reason: Confirm that myeloid marker expression is not driven solely by ambient RNA or dying-cell mitochondrial reads (check percent mito per cell and QC); validate LYZ/S100A8/S100A9 expression distribution and rule out mixed NK/myeloid doublets or annotation artifacts.
- Top markers: 
- Marker details: MT-ND4L logFC=3.80 pct=0.91 ref_pct=0.46; MT-CO1 logFC=3.37 pct=1.00 ref_pct=0.70; MT-CO2 logFC=3.23 pct=0.99 ref_pct=0.69; MT-ND5 logFC=2.79 pct=0.91 ref_pct=0.56; SYNE1 logFC=2.83 pct=0.70 ref_pct=0.33; MT-CYB logFC=2.92 pct=0.96 ref_pct=0.65; SYNE2 logFC=2.63 pct=0.74 ref_pct=0.41; MT-ATP8 logFC=2.70 pct=0.67 ref_pct=0.29; MT-CO3 logFC=2.62 pct=0.97 ref_pct=0.68; MT-ATP6 logFC=2.55 pct=0.95 ref_pct=0.66

### Cluster 10

- n cells: 38400
- Top tissue: lung (95.79%)
- CellXGene composition: NK 99.71%, T 0.28%, B 0.02%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK1_Chemokine_inflammatory
- Free agent label: NK1_Chemokine_inflammatory_CX3CR1+_FCGR3A+_cytotoxic_CCL4-high
- Agent rationale: Strong positive DE of CCL4, FGFBP2, SPON2 and taxonomy core hits for NK1 (CX3CR1, FCGR3A, FGFBP2, GZMB, SPON2). Pan-NK markers (NKG7, GNLY, GZMB, PRF1) are highly expressed. Chemokine_inflammatory state supported by high CCL4/CCL5/CCL3.
- Human review reason: Resolve presence of T-cell and epithelial marker signals (TRBC1, CD3G, SFTPC, SCGB1A1) that could indicate doublets/contamination; confirm NCAM1 depletion is biological or technical.
- Top markers: 
- Marker details: CCL4 logFC=3.44 pct=0.90 ref_pct=0.55; SFTPC logFC=3.87 pct=0.25 ref_pct=0.03; B2M logFC=1.05 pct=1.00 ref_pct=0.99; FGFBP2 logFC=3.04 pct=0.86 ref_pct=0.45; APOC1 logFC=3.63 pct=0.24 ref_pct=0.04; SCGB1A1 logFC=3.78 pct=0.17 ref_pct=0.02; KLRF1 logFC=1.89 pct=0.73 ref_pct=0.49; FCGR3A logFC=1.89 pct=0.77 ref_pct=0.52; FABP4 logFC=3.80 pct=0.11 ref_pct=0.01; GZMB logFC=1.74 pct=0.94 ref_pct=0.74

### Cluster 11

- n cells: 3056
- Top tissue: cord blood (93.88%)
- CellXGene composition: NK 99.97%, T 0.03%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK2_Homeostatic_quiescent
- Free agent label: NK2_cord_blood_IL2RB+_ZBTB16+_KLRF1-low_-_CD56bright-like_quiescent
- Agent rationale: Taxonomy matches weakly to NK2 driven by IL2RB (core hit) and IL18RAP; ZBTB16 (PLZF) and cytokine receptor IL2RB are upregulated. Pan-NK program (GNLY, NKG7, PRF1, GZMB) remains broadly detected, while KLRF1 is strongly depleted, consistent with a CD56bright-like/less cytotoxic NK2 state in cord blood.
- Human review reason: Mixed signals: IL2RB/ZBTB16 and KLRF1-low support NK2 CD56bright-like identity, but strong cytotoxic gene expression and T-marker positivity warrant manual check for doublets, gating, or subclustering.
- Top markers: 
- Marker details: ASTL logFC=7.16 pct=0.35 ref_pct=0.01; TAMALIN logFC=4.69 pct=0.49 ref_pct=0.06; TSPOAP1 logFC=4.73 pct=0.62 ref_pct=0.10; IER5 logFC=5.05 pct=0.80 ref_pct=0.22; APOBR logFC=4.62 pct=0.72 ref_pct=0.16; CSRNP1 logFC=4.80 pct=0.76 ref_pct=0.20; XIST logFC=4.84 pct=0.91 ref_pct=0.31; JUND logFC=5.49 pct=0.95 ref_pct=0.46; ZC3H12A logFC=4.47 pct=0.64 ref_pct=0.14; PLXNA4 logFC=4.73 pct=0.21 ref_pct=0.01

### Cluster 12

- n cells: 15756
- Top tissue: cord blood (92.32%)
- CellXGene composition: NK 99.98%, T 0.02%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK1_Proliferating
- Free agent label: Proliferating_cytotoxic_NK_NK1
- Agent rationale: Strong proliferation program (MKI67, TYMS, CDK1, TOP2A) and high cytotoxic NK transcripts (GNLY, GZMA, NKG7, PRF1) with broad pan-NK detection (median pan-NK pct_nz ~0.94) support a proliferating cytotoxic NK identity; KLRF1 is depleted and some T markers (CD3E, TRAC, TRBC1) are present, so ambiguity remains.
- Human review reason: T-lineage marker expression and KLRF1 depletion create ambiguity between proliferating NK vs proliferating T cells or doublets; recommend checking NCAM1/EOMES protein/UMI-level expression and doublet metrics to confirm NK identity.
- Top markers: 
- Marker details: GPR15 logFC=7.40 pct=0.50 ref_pct=0.01; H1-5 logFC=7.67 pct=0.81 ref_pct=0.06; MKI67 logFC=6.33 pct=0.80 ref_pct=0.07; TYMS logFC=6.48 pct=0.82 ref_pct=0.08; DUSP4 logFC=5.90 pct=0.71 ref_pct=0.06; TPX2 logFC=5.49 pct=0.61 ref_pct=0.05; NUSAP1 logFC=5.35 pct=0.63 ref_pct=0.06; CDK1 logFC=5.37 pct=0.57 ref_pct=0.05; CENPM logFC=5.20 pct=0.67 ref_pct=0.07; ADAM19 logFC=5.05 pct=0.55 ref_pct=0.04

### Cluster 13

- n cells: 2421
- Top tissue: decidua (96.49%)
- CellXGene composition: NK 99.92%, T 0.08%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK2_Chemokine_inflammatory
- Free agent label: NK2_Chemokine_inflammatory_decidual_XCL+_IFN_tissue-resident
- Agent rationale: Strong upregulation of chemokines XCL1 and XCL2 and NK2 taxonomy support (core hits XCL1/XCL2/KLRC1). Concurrent IFN-stimulated genes (MX1, IFI44L, IRF7) present. Typical circulatory cytotoxic NK markers are depleted (FGFBP2, FCGR3A, PRF1, KLRF1), consistent with a tissue-adapted chemokine/IFN-responsive NK2-like decidual population. Some tissue-residency hits (ZNF683, ITGA1, KIR2DL4) further support a tissue NK phenotype. Pairwise split audit: Pairwise DE shows cluster 13 is overwhelmingly decidual (96%) with strong XCL1/XCL2 chemokine signal plus IFN-stimulated genes (MX1, IFI44L, IRF7) and tissue-residency hits (ZNF683, ITGA1, KIR2DL4). Canonical circulating cytotoxic NK markers (FGFBP2, FCGR3A, PRF1) are strongly depleted, consistent with a tissue-adapted, IFN-responsive NK2 chemokine population in decidua. Pairwise split audit: Pairwise DE (13_vs_6) shows strong upregulation of XCL1/XCL2 and IFN-stimulated genes (MX1, IFI44L, IRF7) and tissue-residency/support hits (ZNF683, ITGA1, KIR2DL4) while canonical circulating cytotoxic markers (FGFBP2, FCGR3A, PRF1, KLRF1) are strongly depleted. Composition is overwhelmingly decidua. These differences support a decidual, IFN-responsive, tissue-adapted NK2 chemokine population distinct from the cytotoxic/AREG+ NK2 in cluster 6.
- Human review reason: CD3E expression and strong depletion of canonical cytotoxic NK markers warrant inspection for T/NKT contamination, doublets, or decidua-specific program confirmation.; Cluster 13 shows very high mitochondrial gene expression and low-level T-marker signal in both clusters (CD3E/CD3D) — recommend manual check to exclude technical artifacts or doublet contamination while accepting the biological split.; Cluster 6 shows erythroid marker signal; cluster 13 shows CD3E/T signal — review for doublets/contamination and confirm tissue-residency assignment with orthogonal markers.
- Top markers: 
- Marker details: KRT81 logFC=4.96 pct=0.65 ref_pct=0.06; IGFBP2 logFC=5.07 pct=0.59 ref_pct=0.05; KRT86 logFC=4.75 pct=0.68 ref_pct=0.08; H2AC19 logFC=4.34 pct=0.32 ref_pct=0.02; XCL1 logFC=5.78 pct=0.91 ref_pct=0.22; ITGAD logFC=4.41 pct=0.56 ref_pct=0.08; IFI44L logFC=4.09 pct=0.68 ref_pct=0.10; MT-ATP8 logFC=5.32 pct=0.99 ref_pct=0.32; BGLAP logFC=5.05 pct=0.30 ref_pct=0.03; CAPG logFC=4.31 pct=0.78 ref_pct=0.17

### Cluster 14

- n cells: 5263
- Top tissue: blood (55.39%)
- CellXGene composition: NK 99.96%, T 0.04%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK2_CIMP_cytokine_primed_memory_like
- Free agent label: NK2_cytokine-primed_memory-like_TCF7+_SELL+_GZMK+_with_XCL1_2_chemokine_program
- Agent rationale: Top DE and taxonomy strongly support NK2: high SELL, TCF7, IL7R, GZMK, XCL1/XCL2, KLRC1 and explicit NK2 taxonomy match. State support for cytokine-primed/memory-like comes from core hits GZMK and TCF7 and support hits (CD44, LTB) and absence/low expression of terminal cytotoxic markers (FGFBP2, FCGR3A, depleted GZMB).
- Human review reason: Erythroid markers (AHSP, HBB) and some T-cell marker signal present despite strong NK2 signature; confirm NK identity and exclude contamination/doublets.
- Top markers: 
- Marker details: SELL logFC=4.99 pct=0.90 ref_pct=0.22; SPTSSB logFC=4.33 pct=0.46 ref_pct=0.05; IL7R logFC=4.16 pct=0.55 ref_pct=0.09; GZMK logFC=4.08 pct=0.75 ref_pct=0.18; IGFBP4 logFC=3.78 pct=0.29 ref_pct=0.03; XCL1 logFC=3.70 pct=0.78 ref_pct=0.21; TCF7 logFC=3.23 pct=0.50 ref_pct=0.11; XCL2 logFC=3.31 pct=0.83 ref_pct=0.31; FXYD7 logFC=3.69 pct=0.17 ref_pct=0.02; IFITM3 logFC=2.94 pct=0.78 ref_pct=0.33

### Cluster 15

- n cells: 17939
- Top tissue: lung (93.11%)
- CellXGene composition: NK 97.24%, T 2.76%, B 0.01%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK2_Chemokine_inflammatory
- Free agent label: NK2_Chemokine_inflammatory_lung_ER-stress_cytokine_high
- Agent rationale: Strong positive DE of chemokine/inflammatory genes (XCL1, XCL2, CCL3/CCL4, IFNG) and taxonomy support for NK2 (core hits GZMK, XCL1/XCL2, KLRC1). ER-stress/UPR signature present (HSPA1A, HSPA1B, DNAJB1, DUSP1). NK cytotoxic markers are relatively depleted (GNLY, GZMB, FGFBP2), and pan-NK markers overall show modest depletion but remain broadly detected, consistent with an NK2 inflammatory state with attenuated cytotoxic program. Pairwise split audit: Pairwise DE shows cluster 15 is predominantly lung (93%) and while it shares the NK2 chemokine core (XCL1/XCL2, CCL3/CCL4), it is distinguished by an ER-stress/UPR signature (HSPA1A/HSPA1B, DNAJB1, DUSP1) and increased cytokine-related hits including IFNG and GZMK. Pan-NK cytotoxic metrics are relatively higher (and differentially expressed) compared with cluster 13, indicating a distinct lung/ER-stress-associated NK2 inflammatory state.
- Human review reason: T-cell marker positivity and extreme mitochondrial gene depletion could indicate doublets, contamination, or technical artifacts affecting marker signals; recommend manual inspection of cell-level QC and cluster purity before finalizing annotation.; Cluster 13 shows very high mitochondrial gene expression and low-level T-marker signal in both clusters (CD3E/CD3D) — recommend manual check to exclude technical artifacts or doublet contamination while accepting the biological split.
- Top markers: 
- Marker details: HSPA1B logFC=3.88 pct=0.45 ref_pct=0.10; DNAJB1 logFC=3.00 pct=0.61 ref_pct=0.30; HSP90AA1 logFC=1.94 pct=0.89 ref_pct=0.75; DUSP2 logFC=2.59 pct=0.76 ref_pct=0.46; BTG1 logFC=1.48 pct=0.96 ref_pct=0.86; XCL1 logFC=2.55 pct=0.52 ref_pct=0.20; HSPA1A logFC=2.84 pct=0.42 ref_pct=0.15; XCL2 logFC=2.38 pct=0.62 ref_pct=0.30; CD69 logFC=1.77 pct=0.79 ref_pct=0.60; HSP90AB1 logFC=1.67 pct=0.76 ref_pct=0.59

### Cluster 16

- n cells: 14707
- Top tissue: blood (40.93%)
- CellXGene composition: NK 34.11%, T 0.06%, B 65.83%, Other %
- Top CellXGene annotations: 
- Structured agent label: Non-NK
- Free agent label: B_cells_mature_B-lineage
- Agent rationale: Strong positive B-cell markers (MS4A1, CD79A, PAX5, BANK1, BLK, FCRL1) with very large logFC (~7-9) and high pct_in_cluster; concurrent strong depletion of pan-NK cytotoxic markers (GNLY, PRF1, NKG7, GZMB, GZMA, EOMES, NCAM1) with median pan-NK logFC ~ -9.34. Curated B_cell program mean is high (2.70) while NK cytotoxic program is very low (0.055), supporting a B-cell identity.
- Top markers: 
- Marker details: MS4A1 logFC=8.94 pct=0.94 ref_pct=0.02; CD79A logFC=8.47 pct=0.94 ref_pct=0.03; BLK logFC=8.45 pct=0.68 ref_pct=0.01; LINC00926 logFC=7.82 pct=0.69 ref_pct=0.01; ARHGAP24 logFC=8.30 pct=0.68 ref_pct=0.01; BANK1 logFC=8.23 pct=0.86 ref_pct=0.03; FCRL1 logFC=8.34 pct=0.56 ref_pct=0.00; NIBAN3 logFC=7.87 pct=0.59 ref_pct=0.01; AFF3 logFC=7.63 pct=0.68 ref_pct=0.02; PAX5 logFC=7.67 pct=0.51 ref_pct=0.01

### Cluster 17

- n cells: 6316
- Top tissue: blood (87.27%)
- CellXGene composition: NK 99.97%, T 0.0%, B 0.03%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK2_Checkpoint_exhausted
- Free agent label: NK2_Checkpoint_exhausted_IKZF2+_low_cytotoxic_program
- Agent rationale: Cluster shows weak NK2 taxonomy support driven by IKZF2 and positive DE of RAP1GAP2/PLCB1/AOAH/ZBTB20 while canonical NK cytotoxic markers (GZMA, GZMB, NKG7, GNLY) are strongly depleted; non-NK lineage markers are not enriched, consistent with a non-cytotoxic/exhausted NK2-like blood population.
- Human review reason: Weak taxonomy support and strong depletion of canonical NK cytotoxic markers make subtype/state assignment uncertain; review recommended to confirm NK identity versus an atypical/non-cytotoxic population.
- Top markers: 
- Marker details: LINC02899 logFC=7.43 pct=0.31 ref_pct=0.01; PLCB1 logFC=7.28 pct=0.81 ref_pct=0.08; MIR646HG logFC=6.25 pct=0.41 ref_pct=0.02; RAP1GAP2 logFC=6.75 pct=0.87 ref_pct=0.12; PZP logFC=5.89 pct=0.23 ref_pct=0.01; A2M logFC=5.32 pct=0.34 ref_pct=0.02; FNDC3B logFC=5.74 pct=0.84 ref_pct=0.17; BNC2 logFC=5.44 pct=0.50 ref_pct=0.05; PPM1L logFC=5.40 pct=0.58 ref_pct=0.07; LINC01505 logFC=5.89 pct=0.23 ref_pct=0.01

### Cluster 18

- n cells: 5730
- Top tissue: decidua (99.9%)
- CellXGene composition: NK 99.37%, T 0.63%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: Non-NK
- Free agent label: Decidual_stromal_epithelial-like_Non-NK
- Agent rationale: Strong enrichment of stromal/epithelial markers (KRT19, COL1A1, DCN, LUM, KRT8) and ECM/secreted programs (SPP1, FN1, TIMP3) with large positive logFCs; pan-NK cytotoxic markers are strongly depleted (PRF1, GZMA, GZMB, NKG7, EOMES, KLRF1), and taxonomy-level summary flagged Stromal_Epithelial as strongest non-NK match. Cluster composition is almost entirely decidua.
- Top markers: 
- Marker details: GPX3 logFC=10.85 pct=0.94 ref_pct=0.01; NNMT logFC=9.75 pct=0.74 ref_pct=0.01; CP logFC=8.98 pct=0.53 ref_pct=0.00; TM4SF1 logFC=8.55 pct=0.52 ref_pct=0.01; SPP1 logFC=9.74 pct=0.98 ref_pct=0.05; CRYAB logFC=8.26 pct=0.46 ref_pct=0.01; GDF15 logFC=8.45 pct=0.38 ref_pct=0.00; TIMP3 logFC=7.77 pct=0.61 ref_pct=0.01; DEPP1 logFC=7.93 pct=0.38 ref_pct=0.00; NUPR1 logFC=7.27 pct=0.56 ref_pct=0.01

### Cluster 19

- n cells: 1395
- Top tissue: spleen (95.27%)
- CellXGene composition: NK 97.06%, T 0.0%, B 2.94%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK2_ER_stress_UPR
- Free agent label: B_cells_spleen
- Agent rationale: Top DE genes are canonical B cell markers and immunoglobulins (IGKC, IGHM, IGHA1, IGLC2/3, CD79A, CD79B, MS4A1, JCHAIN) with large logFC; taxonomy and lineage sanity check indicate strong B signature and pan-NK markers are depleted (NCAM1, GNLY, GZMA, PRF1), supporting a non-NK B-lineage identity.
- Human review reason: High mitochondrial gene expression noted (stress_mito score and several MT genes high) which may reflect stress or technical effects; Presence of substantial immunoglobulin expression along with MS4A1—verify B vs plasmablast/plasma differentiation (not resolved here); Minor signals for T-group markers in taxonomy summary suggest checking for potential doublets or mixed population
- Top markers: 
- Marker details: IGLC3 logFC=7.53 pct=0.89 ref_pct=0.05; IGHM logFC=7.63 pct=0.95 ref_pct=0.07; IGLC2 logFC=7.20 pct=0.92 ref_pct=0.07; JCHAIN logFC=6.38 pct=0.85 ref_pct=0.06; IGHA1 logFC=6.40 pct=0.90 ref_pct=0.09; IGHG1 logFC=5.14 pct=0.56 ref_pct=0.04; TCL1A logFC=5.01 pct=0.43 ref_pct=0.02; IGKC logFC=8.38 pct=0.97 ref_pct=0.16; IGHG2 logFC=5.17 pct=0.34 ref_pct=0.02; VPREB3 logFC=4.91 pct=0.46 ref_pct=0.03

### Cluster 20

- n cells: 1719
- Top tissue: lung (70.74%)
- CellXGene composition: NK 1.63%, T 0.12%, B 98.25%, Other %
- Top CellXGene annotations: 
- Structured agent label: Non-NK
- Free agent label: B_cells_MS4A1+_CD79A+_IGKC+
- Agent rationale: Cluster shows strong upregulation of B cell markers (CD79A, MS4A1, IGKC, BANK1, CD79B; logFCs ~>2-7 and high pct_in_cluster) and marker-program hit for B cell. Pan-NK markers are strongly and consistently depleted (PRF1, NKG7, GNLY, GZMA, GZMB, KLRF1, NCAM1, EOMES; median pan-NK logFC ~ -8), and taxonomy/non-NK summaries flag this as B-lineage. Curated marker set means support high B_cell score (2.30) vs low NK_cytotoxic (0.063).
- Top markers: 
- Marker details: VPREB3 logFC=6.26 pct=0.58 ref_pct=0.03; MS4A1 logFC=6.69 pct=0.85 ref_pct=0.06; CD79A logFC=6.92 pct=0.88 ref_pct=0.07; HLA-DQA1 logFC=5.91 pct=0.83 ref_pct=0.11; HLA-DQB1 logFC=6.10 pct=0.89 ref_pct=0.14; TNFRSF13C logFC=5.22 pct=0.48 ref_pct=0.04; BANK1 logFC=4.84 pct=0.61 ref_pct=0.07; IGKC logFC=6.70 pct=0.86 ref_pct=0.16; HLA-DRA logFC=7.30 pct=1.00 ref_pct=0.28; IGLC2 logFC=5.41 pct=0.60 ref_pct=0.07

### Cluster 21

- n cells: 8193
- Top tissue: bone marrow (95.26%)
- CellXGene composition: NK 100.0%, T 0.0%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: cNK_Cytotoxic_activated
- Free agent label: cNK_Cytotoxic_activated_erythroid_contamination
- Agent rationale: Strong hemoglobin/erythroid DE (HBB, HBA1, HBA2; high logFC ~5) in a bone marrow-dominated cluster, but pan-NK cytotoxic program is retained (KLRF1 present; high GNLY, NKG7, GZMB means; NK_cytotoxic set mean=3.61). EOMES and NCAM1 are depleted, suggesting partial loss of some canonical NK markers; overall pattern best fits conventional cytotoxic NK cells with substantial erythroid contamination or RBC doublets.
- Human review reason: High hemoglobin gene expression indicates likely erythroid contamination or RBC doublets; confirm NK vs erythroid identity (e.g., inspect single-cell QC, doublet rates, protein/ADTs if available) and verify biological vs technical depletion of EOMES/NCAM1 before final use.
- Top markers: 
- Marker details: HBB logFC=5.80 pct=0.60 ref_pct=0.07; HBD logFC=6.88 pct=0.15 ref_pct=0.00; HBA2 logFC=5.11 pct=0.46 ref_pct=0.05; HBA1 logFC=4.58 pct=0.43 ref_pct=0.05; MALAT1 logFC=2.19 pct=1.00 ref_pct=0.91; AHSP logFC=5.44 pct=0.09 ref_pct=0.00; B2M logFC=1.14 pct=1.00 ref_pct=0.99; HBM logFC=5.16 pct=0.07 ref_pct=0.00; CA1 logFC=5.04 pct=0.07 ref_pct=0.00; TMSB4X logFC=1.21 pct=1.00 ref_pct=0.97

### Cluster 22

- n cells: 929
- Top tissue: blood (95.8%)
- CellXGene composition: NK 99.78%, T 0.22%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: Non-NK
- Free agent label: Myeloid_monocyte_macrophage_-_Non-NK
- Agent rationale: Strong myeloid marker expression (LST1, LYZ, S100A8, S100A9, CST3, MS4A7, MAFB, C5AR1, CLEC7A, FCN1) and high myeloid program mean. Concurrent strong depletion of pan-NK/cytotoxic markers (PRF1, GZMA, GZMB, GNLY, NKG7, NCAM1, EOMES, KLRF1) supports a non-NK myeloid identity. Cluster is blood-dominated and matches myeloid marker programs in taxonomy and curated summaries.
- Top markers: 
- Marker details: PELATON logFC=9.05 pct=0.83 ref_pct=0.01; LINC00877 logFC=8.18 pct=0.43 ref_pct=0.00; CLEC7A logFC=8.35 pct=0.88 ref_pct=0.01; C5AR1 logFC=8.20 pct=0.82 ref_pct=0.01; CFP logFC=7.76 pct=0.71 ref_pct=0.01; MAFB logFC=8.13 pct=0.83 ref_pct=0.02; CD302 logFC=7.51 pct=0.66 ref_pct=0.01; LMO2 logFC=7.47 pct=0.51 ref_pct=0.01; MS4A7 logFC=8.37 pct=0.90 ref_pct=0.02; IGSF6 logFC=7.36 pct=0.69 ref_pct=0.01

### Cluster 23

- n cells: 4071
- Top tissue: lung (99.98%)
- CellXGene composition: NK 100.0%, T 0.0%, B 0.0%, Other %
- Top CellXGene annotations: 
- Structured agent label: NK1_Cytotoxic_activated
- Free agent label: NK1_Cytotoxic_activated_lung_with_epithelial_signal
- Agent rationale: Strong NK cytotoxic program (high NKG7, GNLY, GZMB, PRF1, FGFBP2, FCGR3A; curated NK_cytotoxic score highest). Taxonomy marker matching strongly supports NK1 (core hits FCGR3A, FGFBP2, GZMB). State-level taxonomy and marker hits support Cytotoxic_activated. However, cluster shows high lung epithelial markers (SCGB3A1, SCGB1A1, SFTPC) and some non-NK signals (HBA1), suggesting possible ambient/contaminating epithelial/erythroid signal. Pairwise split audit: Pairwise DE shows coherent lung-associated signal in cluster 23 (SCGB3A1 logFC ~9.06, SCGB1A1 ~8.76, SFTPC ~8.03 with ~36% cells expressing these genes vs ~0.3% in cluster 2), and curated epithelial_lung mean is elevated in 23 (0.611 vs 0.019 in 2). NK cytotoxic markers remain high (FGFBP2, FCGR3A, GZMB, NKG7), indicating NK identity plus substantial ambient/contaminating epithelial (and some erythroid/myeloid) signal — thus label should note lung association/epithelial signal. Pairwise split audit: Pairwise DE shows a coherent and strong lung epithelial program in cluster 23 (SCGB3A1 logFC ~9.06, SCGB1A1 ~8.76, SFTPC ~8.03; ~36–52% cells expressing these vs ~0.3% in cluster 0) and elevated epithelial_lung curated score (0.611 vs 0.017). NK cytotoxic markers (FGFBP2, FCGR3A, GZMB, NKG7) remain high, indicating NK identity with substantial epithelial (and some erythroid/myeloid) signal — therefore label notes lung/epithelial association.
- Human review reason: Epithelial (SCGB3A1/SCGB1A1/SFTPC) and erythroid (HBA1) signals suggest possible ambient contamination or doublets; verify co-expression of NK markers at single-cell level and review QC/doublet filtering.; Pairwise DE supports a distinct free label.; Epithelial markers are highly enriched in cluster 23 and could reflect ambient contamination, doublets, or true tissue-resident NK transcriptional interaction; recommend manual QC (doublet detection / ambient RNA assessment) and contextual review.
- Top markers: 
- Marker details: SCGB3A1 logFC=5.94 pct=0.52 ref_pct=0.02; SCGB1A1 logFC=5.42 pct=0.50 ref_pct=0.03; SFTPC logFC=3.67 pct=0.36 ref_pct=0.05; FABP4 logFC=3.91 pct=0.22 ref_pct=0.02; MALAT1 logFC=2.25 pct=1.00 ref_pct=0.91; CCL20 logFC=4.26 pct=0.13 ref_pct=0.01; GZMB logFC=2.43 pct=0.99 ref_pct=0.76; B2M logFC=1.03 pct=1.00 ref_pct=0.99; ID2 logFC=2.38 pct=0.93 ref_pct=0.64; MYOM2 logFC=2.82 pct=0.48 ref_pct=0.14

### Cluster 24

- n cells: 1124
- Top tissue: lung (79.63%)
- CellXGene composition: NK 99.91%, T 0.0%, B 0.09%, Other %
- Top CellXGene annotations: 
- Structured agent label: Non-NK
- Free agent label: Non-NK_lung_stromal_epithelial-like_KRT19_KRT8_with_macrophage-associated_genes
- Agent rationale: Strong depletion of pan-NK cytotoxic markers (NKG7, GZMA, GZMB, GNLY; median pan-NK logFC ≈ -6.09). Positive DE signal for stromal/epithelial markers (KRT19, KRT8; taxonomy_non_nk strongest = Stromal_Epithelial) and a lung/stromal-like marker program (DOCK4, SLC8A1, FMN1, PLXDC2). Additional macrophage-associated markers (CD163, MRC1) are also upregulated, consistent with a non-NK lung stromal/myeloid identity rather than NK.
- Human review reason: Mixed lineage signals (stromal/epithelial KRTs plus macrophage markers and some B-cell markers) warrant inspection for doublets, ambient RNA, or annotation errors despite clear NK depletion.
- Top markers: 
- Marker details: DOCK4 logFC=8.44 pct=0.49 ref_pct=0.01; SLC8A1 logFC=8.71 pct=0.52 ref_pct=0.01; FMN1 logFC=8.10 pct=0.45 ref_pct=0.01; ARHGEF10L logFC=7.68 pct=0.25 ref_pct=0.00; SLC1A3 logFC=7.87 pct=0.25 ref_pct=0.00; NHSL1 logFC=7.33 pct=0.34 ref_pct=0.00; SIRPB2 logFC=7.64 pct=0.17 ref_pct=0.00; SHTN1 logFC=6.79 pct=0.32 ref_pct=0.01; TNS1 logFC=7.08 pct=0.21 ref_pct=0.00; PLXDC2 logFC=7.40 pct=0.58 ref_pct=0.02
