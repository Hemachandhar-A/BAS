"""outputs/logger.py -- JsonlLogger (F9; IMPLEMENTATION_PLAN.md 5.5).

One line per *event*, never per frame; JSON Lines, UTF-8, append-only,
never rewritten. ``seq`` is a per-run counter owned by the caller (the
runtime router) and starts at 0 -- the logger only asserts it is
contiguous, raising ``ContractViolation`` on a gap. ``t_wall`` is stamped
here from the injected ``wall_clock`` -- the only wall-clock read in the
system (AGENTS.md rule 8) -- overwriting whatever placeholder the caller
supplied at construction (``LogEntry.t_wall`` is optional precisely so a
caller without a clock can build one; ISSUES.md, 2026-09-28 CONTRACT).
Each line is flushed and ``fsync``ed before ``write()`` returns, so a
mid-run kill leaves every already-written line complete and parseable.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from contracts import ContractViolation, LogEntry


class JsonlLogger:
    """Implements ``contracts.LogSink``. One instance per run/path."""

    def __init__(self, path: str | Path, wall_clock: Callable[[], datetime]) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._wall_clock = wall_clock
        self._next_seq = 0

    def write(self, entry: LogEntry) -> None:
        if entry.seq != self._next_seq:
            raise ContractViolation(
                f"LogEntry.seq out of order: expected {self._next_seq}, got {entry.seq}"
            )
        now = self._wall_clock()
        if now.tzinfo is None:
            raise ContractViolation("wall_clock() must return a timezone-aware datetime")
        self._next_seq += 1
        stamped = entry.model_copy(update={"t_wall": now})
        line = stamped.model_dump_json() + "\n"
        with self._path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
