#!/usr/bin/env python3
"""python scripts/replay.py --from-cache RUN_ID | --video PATH | --script PATH | --tune

Thin CLI entry point (IMPLEMENTATION_PLAN.md 5.9); the implementation is
harness/replay.py, shared with tests. ``--tune`` is the P2.6 sweep on the val split only
(harness/tune.py); P2.7 adds --split test on top of this same module in a later session.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `python scripts/replay.py` works

from harness.replay import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
