#!/usr/bin/env python3
"""Generate charts from this train's artifacts. Same as `aq plot`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

kernel = os.environ.get("AQ_KERNEL")
if kernel:
    sys.path.insert(0, kernel)

from plot import do_plot  # noqa: E402

train = Path(os.environ.get("AQ_TRAIN", ".")).resolve()
kind = sys.argv[1] if len(sys.argv) > 1 else "all"
for line in do_plot(train, {"kind": kind}):
    print(line)
