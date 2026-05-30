# Final Unified NK Cell Taxonomy (DEG-integrated)

> Genes are tiered by literature support: CORE (>=4 papers), SUPPORT (2-3), CONTEXT (1).

> RNA publication DEG evidence integrated from local data: Crinier2019, Kim2020, Vivier2024, Zhang2020.
> RNA papers listed but not yet locally integrated by DEG files: Dogra2020, Gao2021, Netskar2024, Peng2019, Wang2021.

> Gene tags: `(D#)` = number of RNA publications (from local DEG datasets) that support this gene by differential expression.

---
## LAYER 1 - NK Cell Subtypes

###  NK2 (Vivier 2024)

**🔴 CORE** (10 genes)  
`GZMK(D4)  ·  IL2RB(D4)  ·  SELL(D4)  ·  XCL1(D4)  ·  XCL2(D4)  ·  CCR7(D3)  ·  NCAM1(D3)  ·  IL7R(D2)  ·  KLRC1(D2)  ·  NCR1(D2)`

**🟡 SUPPORT** (14 genes)  
`CXCR3(D3)  ·  SPINK2(D3)  ·  AREG(D2)  ·  GPR183(D2)  ·  IKZF2(D2)  ·  KIT(D2)  ·  KLRK1(D2)  ·  LTB(D2)  ·  NCR2(D2)  ·  TCF7(D2)  ·  IL18R1(D1)  ·  IL18RAP(D1)  ·  PTGDS(D1)  ·  FLT3LG`

**🟢 CONTEXT** (4 genes)  
`EEF1A1(D2)  ·  TNFRSF18(D2)  ·  TPT1(D2)  ·  S1PR1`


### NK1 (Vivier 2024)

**🔴 CORE** (10 genes)  
`CST7(D4)  ·  CX3CR1(D4)  ·  FCGR3A(D4)  ·  FGFBP2(D4)  ·  GZMB(D4)  ·  NKG7(D4)  ·  PRF1(D4)  ·  SPON2(D4)  ·  GNLY(D3)  ·  KLRD1(D3)`

**🟡 SUPPORT** (21 genes)  
`ADGRG1(D4)  ·  GZMH(D4)  ·  GZMM(D4)  ·  S1PR5(D4)  ·  ZEB2(D4)  ·  ACTB(D3)  ·  FCER1G(D3)  ·  HAVCR2(D3)  ·  KLRG1(D3)  ·  PRDM1(D3)  ·  TBX21(D3)  ·  CD160(D2)  ·  CHST2(D2)  ·  CLIC3(D2)  ·  KLF2(D2)  ·  LAIR2(D2)  ·  SIGLEC7(D2)  ·  CFL1(D1)  ·  CTSW(D1)  ·  GZMA(D1)  ·  RAC2(D1)`


### adaptive_NK_CMV

**🔴 CORE** (7 genes)  
`KLRC2(D3)  ·  B3GAT1(D1)  ·  LILRB1(D3)  ·  PRDM1(D3)  ·  IL32(D2)  ·  ZBTB38(D2)  ·  FCGR3A(D4)`

**🟡 SUPPORT** (13 genes)  
`GZMH(D4)  ·  CCL5(D2)  ·  CX3CR1(D4)  ·  TBX21(D3)  ·  ZEB2(D4)  ·  KLRD1(D3)  ·  FGFBP2(D4)  ·  SPON2(D4)  ·  IFNG(D1)  ·  CD244(D1)  ·  CD2  ·  FCGR2C  ·  ITGAM`

**🟢 CONTEXT** (5 genes)  
`ARID5B  ·  JAKMIP1  ·  PATL2  ·  KIR2DL1  ·  KIR3DL1`

**⚪ CMV-adaptive negative-defining features (expected low/absent)**  
`FCER1G  ·  SYK  ·  ZBTB16  ·  KLRC1  ·  KLRB1  ·  CCR7  ·  IL2RB  ·  TYROBP  ·  SIGLEC7`


### adaptive_NK_nonCMV

**🔴 CORE** (6 genes)  
`PRDM1(D3)  ·  IL32(D2)  ·  BATF(D2)  ·  MAF  ·  CCL5(D2)  ·  FCGR3A(D4)`

**🟡 SUPPORT** (9 genes)  
`GZMH(D4)  ·  CD52(D2)  ·  ZBTB32  ·  ZBTB38(D2)  ·  KIR2DL2(D1)  ·  KIR2DL1  ·  KIR3DL1  ·  LILRB1(D3)  ·  TMSB4X(D2)`

**🟢 CONTEXT** (4 genes)  
`CD3E(D2)  ·  CX3CR1(D4)  ·  TBX21(D3)  ·  KLRC2(D3)`

**⚪ Non-CMV-adaptive interpretation note**  
`Use when adaptive-like program is present but canonical CMV-adaptive negative axis (FCER1G/SYK/ZBTB16-loss) is weak or absent.`


### trNK

**🔴 CORE** (4 genes)  
`CD69(D3)  ·  EOMES(D3)  ·  CXCR6(D2)  ·  ITGA1(D2)`

**🟡 SUPPORT** (9 genes)  
`TBX21(D3)  ·  AREG(D2)  ·  ITGB1(D2)  ·  CCL4L2(D1)  ·  HOPX(D1)  ·  IKZF3(D1)  ·  ITGAE(D1)  ·  RGS1(D1)  ·  ZNF683(D1)`

**🟢 CONTEXT** (5 genes)  
`PDCD1(D1)  ·  PSMA2(D1)  ·  SCGB1A1(D1)  ·  SLC5A3(D1)  ·  CLN3`


### cNK

**🔴 CORE** (1 genes)  
`CX3CR1(D4)`

**🟡 SUPPORT** (5 genes)  
`FCGR3A(D4)  ·  S1PR5(D4)  ·  KLRG1(D3)  ·  KLF2(D2)  ·  B3GAT1(D1)`

**🟢 CONTEXT** (1 genes)  
`IKZF1(D1)`


### L6_Developmental_immature --> not sure yet if add to final nomenclature, try on a dataset to see

**🟡 SUPPORT** (11 genes)  
`NCAM1(D3)  ·  SPINK2(D3)  ·  GATA3(D2)  ·  KIT(D2)  ·  TCF7(D2)  ·  ID2(D1)  ·  MYC(D1)  ·  CD34  ·  FLT3  ·  LEF1  ·  RUNX2`

**🟢 CONTEXT** (2 genes)  
`IL2RA  ·  IL3RA`


---
## LAYER 2 - NK Cell States

### Chemokine_inflammatory

**🔴 CORE** (6 genes)  
`XCL1(D4)  ·  XCL2(D4)  ·  CCL4(D2)  ·  CCL5(D2)  ·  CCL3(D1)  ·  IFNG(D1)`

**🟡 SUPPORT** (5 genes)  
`STAT1(D3)  ·  CCL4L2(D1)  ·  GZMA(D1)  ·  CXCL10  ·  CXCL9`


### Checkpoint_exhausted

**🔴 CORE** (8 genes)  
`TIGIT(D2)  ·  HAVCR2(D3)  ·  TOX(D2)  ·  NR4A2(D3)  ·  NR4A1(D2)  ·  LAG3(D2)  ·  PDCD1(D1)  ·  ENTPD1(D1)`

**🟡 SUPPORT** (26 genes)  
`KLRC1(D2)  ·  CD96(D1)  ·  TOX2(D1)  ·  BATF(D2)  ·  LILRB1(D3)  ·  SIGLEC7(D2)  ·  KIR2DL1  ·  KIR3DL1  ·  NR4A3  ·  CTLA4  ·  CD274  ·  LAYN  ·  CISH  ·  SOCS1  ·  SOCS3  ·  NFATC1  ·  NFATC2  ·  BHLHE40  ·  IRF4  ·  EZH2  ·  CD38  ·  NT5E  ·  RELB  ·  NFKB2  ·  RXRA  ·  TNFRSF9`

**🟢 CONTEXT** (16 genes)  
`PRDM1(D3)  ·  CD244(D1)  ·  CD160(D2)  ·  KLRG1(D3)  ·  CX3CR1(D4)  ·  EOMES(D3)  ·  IKZF2(D2)  ·  ZEB2(D4)  ·  MAF  ·  MAFB  ·  EGR2(D1)  ·  EGR3  ·  RGS1(D1)  ·  BATF3  ·  CDKN1A  ·  HIF1A`


### ER_stress_UPR

**🟡 SUPPORT** (12 genes)  
`CD44(D2)  ·  CD74(D2)  ·  CXCR4(D2)  ·  MAFF(D2)  ·  ATF3(D1)  ·  DDIT3(D1)  ·  DNAJB1(D1)  ·  EGR2(D1)  ·  HSPA1A(D1)  ·  HSPA1B(D1)  ·  XBP1(D1)  ·  EGR3`

**🟢 CONTEXT** (3 genes)  
`DUSP1(D2)  ·  GADD45B(D1)  ·  BAG3`


### Metabolic_stress_hypoxia

**🟡 SUPPORT** (11 genes)  
`LDHA(D1)  ·  MYC(D1)  ·  NFE2L2(D1)  ·  TXNIP(D1)  ·  BNIP3  ·  HIF1A  ·  HK2  ·  MTOR  ·  PDK1  ·  PFKP  ·  SLC2A1`

**🟢 CONTEXT** (3 genes)  
`PRDX1(D1)  ·  GCLC  ·  SOD2`


### Proliferating

**🔴 CORE** (1 genes)  
`MKI67(D3)`

**🟡 SUPPORT** (10 genes)  
`AURKB(D3)  ·  CCNB1(D3)  ·  CDK1(D3)  ·  PCNA(D3)  ·  TOP2A(D3)  ·  TYMS(D3)  ·  UBE2C(D3)  ·  BIRC5(D1)  ·  CCNB2(D1)  ·  STMN1(D1)`


### IFN_stimulated

**🔴 CORE** (3 genes)  
`ISG15(D3)  ·  MX1(D3)  ·  MX2(D2)`

**🟡 SUPPORT** (8 genes)  
`BST2(D3)  ·  IFI44L(D3)  ·  IFIT1(D3)  ·  IRF7(D3)  ·  OAS1(D3)  ·  STAT1(D3)  ·  IFIT2(D2)  ·  IFIT3(D2)`

**🟢 CONTEXT** (6 genes)  
`RSAD2(D3)  ·  STAT2(D2)  ·  HERC5(D1)  ·  OAS3(D1)  ·  DDX58  ·  OAS2`


### Cytotoxic_activated

**🔴 CORE** (5 genes)  
`FCGR3A(D4)  ·  GZMB(D4)  ·  NKG7(D4)  ·  PRF1(D4)  ·  GNLY(D3)`

**🟡 SUPPORT** (6 genes)  
`CD160(D2)  ·  KLRK1(D2)  ·  NCR1(D2)  ·  CTSW(D1)  ·  GZMA(D1)  ·  NCR3(D1)`


### Homeostatic_quiescent

**🔴 CORE** (1 genes)  
`CST7(D4)`

**🟡 SUPPORT** (6 genes)  
`GNLY(D3)  ·  KLRD1(D3)  ·  CD244(D1)  ·  CHST12(D1)  ·  IL18RAP(D1)  ·  KLRF1(D1)`


### CIML_cytokine_preactivated

**🟡 SUPPORT** (8 genes)  
`GZMB(D4)  ·  IL2RB(D4)  ·  PRF1(D4)  ·  KLRC2(D3)  ·  IL12RB2(D2)  ·  IFNG(D1)  ·  IL12RB1(D1)  ·  STAT4(D1)`

**🟢 CONTEXT** (3 genes)  
`CD44(D2)  ·  CD122  ·  IL15RA`


### CIMP_cytokine_primed_memory_like

**🔴 CORE** (11 genes)  
`IL2RA  ·  IFNG(D1)  ·  IL12RB1(D1)  ·  IL12RB2(D2)  ·  STAT4(D1)  ·  PRF1(D4)  ·  GZMB(D4)  ·  GZMK(D4)  ·  NCR2(D2)  ·  NCR3(D1)  ·  TCF7(D2)`

**🟡 SUPPORT** (18 genes)  
`TNFRSF9  ·  RUNX3  ·  ID2(D1)  ·  CD44(D2)  ·  NFIL3  ·  JAK2  ·  STAT5A  ·  STAT5B  ·  TNFSF14  ·  LTB(D2)  ·  LTA  ·  CSF2  ·  IFNGR2  ·  BCL2  ·  ZBTB32  ·  IRF8  ·  TNF  ·  CD96(D1)`

**🟢 CONTEXT** (2 genes)  
`OSM  ·  LIF`

**⚪ CIMP negative-trend genes (expected low)**  
`CXCR4(D2)  ·  S1PR5(D4)  ·  NR4A1(D2)  ·  NR4A2(D3)  ·  NR4A3`
