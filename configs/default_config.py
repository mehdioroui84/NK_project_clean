"""Default configuration for NK_project.

This is a plain Python config rather than YAML so it is easy to edit/debug on
Kubernetes or HPC. Keep data paths as HPC paths.
"""

from __future__ import annotations

import os

SEED = 0

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MERGED_PATH = "/rsrch5/home/genomic_med/suorouji/projects/lsf_run/cellxgene_plus_CB07.h5ad"
BASE_OUTDIR = os.environ.get(
    "NK_PROJECT_OUTDIR",
    os.path.join(PROJECT_ROOT, "outputs"),
)

FIG_OUTDIR = os.path.join(BASE_OUTDIR, "figures")
MODEL_OUTDIR = os.path.join(BASE_OUTDIR, "models")
TABLE_OUTDIR = os.path.join(BASE_OUTDIR, "tables")
LATENT_OUTDIR = os.path.join(BASE_OUTDIR, "latents")

LABEL_KEY = "NK_State"
REFINED_LABEL_KEY = "NK_State_refined"
DATASET_KEY = "dataset_id"
ASSAY_KEY = "assay"
ASSAY_CLEAN_KEY = "assay_clean"
PRODUCTION_BATCH_KEY = "assay_clean"
COMPOSITE_BATCH_KEY = "batch_composite"
UNLABELED_CATEGORY = "Unknown"
PROTECTED_DATASET = "CB07"

FLEX_ASSAY_FILL = "Flex Gene Expression"
COMPOSITE_MERGE_THRESHOLD = 100

HELD_OUT_DATASETS = [
    "350237e0-9f48-4cbd-9140-3b44495549f3",
    "30cd5311-6c09-46c9-94f1-71fe4b91813c",
    "9f222629-9e39-47d0-b83f-e08d610c7479",
    "e84f2780-51e8-4cfa-8aa0-13bbfef677c7",
    "2c820d53-cbd7-4e0a-be7a-a0ad1989a98f",
]

QC_LOW_CUT = 200
QC_MAX_COUNTS = 10000
MIN_CLASS_SIZE = 780
MIN_BATCH_SIZE = 100
CAP_CLASSES = {"T": 20000, "B": 20000}
MAJOR_CLASS = "Mature Cytotoxic"
MAJOR_RATIO = 0.50
WEIGHT_MODE = "inv_percent"
WEIGHT_CLIP = (0.1, 10.0)

N_LAYERS = 2
N_HIDDEN = 128
N_LATENT = 10
GENE_LIKELIHOOD = "zinb"

MAX_EPOCHS = 50
SURGERY_EPOCHS = 30
BATCH_SIZE = 1024
LR = 5e-4
SURGERY_LR = 1e-3
WEIGHT_MIN = 0.25
WEIGHT_MAX = 1.00

TRAIN_VAL_TEST_SIZE = 0.20
MIN_CLASS_EVAL = 30

UMAP_N_NEIGHBORS = 30
UMAP_MIN_DIST = 0.3
UMAP_SEED = 0
PLOT_MAX_POINTS = None
MARKER_SIZE = 0.5

LEIDEN_RESOLUTIONS = [0.3, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
DISCOVERY_N_NEIGHBORS = 30

METRIC_MAX_CELLS = 50000
METRIC_KNN_K = 30
LEIDEN_RES = 1.0
