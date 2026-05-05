#!/usr/bin/env python
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from experiments.batch_key_comparison import main


if __name__ == "__main__":
    main()
