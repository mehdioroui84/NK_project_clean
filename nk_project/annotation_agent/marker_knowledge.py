from __future__ import annotations

MARKER_PROGRAMS: dict[str, list[str]] = {
    "NK cytotoxic": [
        "NKG7",
        "GNLY",
        "PRF1",
        "GZMB",
        "GZMH",
        "GZMA",
        "CST7",
        "FGFBP2",
        "KLRF1",
        "FCGR3A",
    ],
    "NK tissue/regulatory/chemokine": [
        "XCL1",
        "XCL2",
        "KLRC1",
        "KLRC2",
        "KLRB1",
        "CXCR6",
        "ITGAE",
        "ZNF683",
        "CCL3",
        "CCL4",
        "CCL5",
    ],
    "proliferation": [
        "MKI67",
        "TOP2A",
        "STMN1",
        "TYMS",
        "RRM2",
        "TK1",
        "PCNA",
        "PCLAF",
        "NUSAP1",
    ],
    "interferon/cytokine": [
        "ISG15",
        "IFIT1",
        "IFIT2",
        "IFIT3",
        "IFI44L",
        "MX1",
        "STAT1",
        "IRF7",
        "IL2RA",
        "IL7R",
        "CCR7",
        "IRF4",
    ],
    "T cell": [
        "CD3D",
        "CD3E",
        "CD3G",
        "TRAC",
        "TRBC1",
        "TRBC2",
        "TCF7",
        "SELL",
        "LEF1",
    ],
    "B cell": [
        "MS4A1",
        "CD79A",
        "CD79B",
        "BANK1",
        "BLK",
        "FCRL1",
        "IGHM",
        "IGKC",
    ],
    "myeloid": [
        "LYZ",
        "LST1",
        "S100A8",
        "S100A9",
        "C5AR1",
        "CLEC7A",
        "MS4A7",
        "MAFB",
    ],
    "erythroid": [
        "HBB",
        "HBA1",
        "HBA2",
        "HBD",
        "HBM",
        "AHSP",
    ],
    "lung/stromal-like": [
        "DOCK4",
        "SLC8A1",
        "FMN1",
        "PLXDC2",
        "SLC1A3",
        "NHSL1",
        "LHFPL2",
        "LRP1",
        "NRP1",
    ],
}


KNOWN_REFINED_LABELS: list[str] = [
    "Mature Cytotoxic",
    "Mature Cytotoxic TCF7+",
    "Lung Cytotoxic NK",
    "Lung DOCK4+ SLC8A1+ NK",
    "Transitional Cytotoxic Tissue-Resident",
    "Transitional Cytotoxic",
    "Cytokine-Stimulated CCR7+",
    "Cytokine-Stimulated Proliferative",
    "Proliferative",
    "Regulatory",
    "T",
    "B",
    "Unknown_Kidney",
    "Unknown_BM_1 Erythroid-like",
    "Myeloid-like",
]


def marker_program_hits(genes: list[str]) -> dict[str, list[str]]:
    gene_set = {str(gene).upper() for gene in genes}
    hits: dict[str, list[str]] = {}
    for program, markers in MARKER_PROGRAMS.items():
        present = [gene for gene in markers if gene.upper() in gene_set]
        if present:
            hits[program] = present
    return hits

