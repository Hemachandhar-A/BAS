#!/usr/bin/env python3
"""python scripts/replay.py --from-cache RUN_ID | --video PATH | --script PATH

Thin CLI entry point (IMPLEMENTATION_PLAN.md 5.9); the implementation is
harness/replay.py, shared with tests. P2.6/P2.7 add --tune and
--split test on top of this same module in later sessions.
"""

from __future__ import annotations

from harness.replay import main

if __name__ == "__main__":
    raise SystemExit(main())
