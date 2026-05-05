from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dirs(*paths: str) -> None:
    for path in paths:
        os.makedirs(path, exist_ok=True)


def save_split_ids(outdir: str, *, train_names, val_names, heldout_names) -> None:
    ensure_dirs(outdir)
    pd.Series(train_names, name="obs_name").to_csv(os.path.join(outdir, "train_obs_names.txt"), index=False, header=False)
    pd.Series(val_names, name="obs_name").to_csv(os.path.join(outdir, "val_obs_names.txt"), index=False, header=False)
    pd.Series(heldout_names, name="obs_name").to_csv(os.path.join(outdir, "heldout_obs_names.txt"), index=False, header=False)


def save_latent_npz(path: str, **arrays) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_run_config(path: str, config_module) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        name: getattr(config_module, name)
        for name in dir(config_module)
        if name.isupper() and _jsonable(getattr(config_module, name))
    }
    with open(path, "w") as handle:
        json.dump(cfg, handle, indent=2, sort_keys=True)


def _jsonable(value) -> bool:
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False
