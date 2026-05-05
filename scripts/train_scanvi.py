#!/usr/bin/env python
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from configs import default_config as cfg
from nk_project.workflows import train_scanvi


def main():
    train_scanvi(cfg, label_key=cfg.LABEL_KEY, batch_key=cfg.PRODUCTION_BATCH_KEY)


if __name__ == "__main__":
    main()
