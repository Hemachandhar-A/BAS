"""Strict loader and per-class summary for the Lead's ``bad_boxes.csv``
review sheets (P1.2 checkpoint 1c). Pure: no model, no image I/O.

Columns: run_id,frame_id,class,reason,visibility,note. ``visibility`` is
required when ``reason`` is ``missing`` and must be empty otherwise. Rows
with reason=missing and visibility=hidden are not counted as bad (the object
is not visible, so no box is correct); they are reported separately.
"""

from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CLASSES = ("outer_box", "tray", "start_button", "red_box", "yellow_box")
REASONS = ("wrong_box", "missing", "duplicate", "wrong_class")
VISIBILITIES = ("visible", "partial", "hidden")
HEADER = ["run_id", "frame_id", "class", "reason", "visibility", "note"]


class ReviewCsvError(ValueError):
    """Raised with every problem found, one per line."""


@dataclass(frozen=True)
class ReviewRow:
    run_id: str
    frame_id: int
    cls: str
    reason: str
    visibility: str
    note: str

    @property
    def counts_as_bad(self) -> bool:
        return not (self.reason == "missing" and self.visibility == "hidden")


def load_bad_boxes(path: Path, valid_frames: set[tuple[str, int]]) -> list[ReviewRow]:
    problems: list[str] = []
    rows: list[ReviewRow] = []
    seen: set[tuple[str, int, str]] = set()
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != HEADER:
            raise ReviewCsvError(f"header must be {','.join(HEADER)}, got {header}")
        for line_no, rec in enumerate(reader, start=2):
            if not rec:
                continue
            if len(rec) != len(HEADER):
                problems.append(f"line {line_no}: expected {len(HEADER)} fields, got {len(rec)}")
                continue
            run_id, frame_s, cls, reason, vis, note = rec
            bad_ws = [v for v in rec[:5] if v != v.strip()]
            if bad_ws:
                problems.append(f"line {line_no}: stray whitespace in {bad_ws!r}")
                continue
            try:
                frame_id = int(frame_s)
            except ValueError:
                problems.append(f"line {line_no}: frame_id {frame_s!r} is not an integer")
                continue
            if cls not in CLASSES:
                problems.append(f"line {line_no}: unknown class {cls!r}")
            if reason not in REASONS:
                problems.append(f"line {line_no}: unknown reason {reason!r}")
            if reason == "missing":
                if vis not in VISIBILITIES:
                    problems.append(
                        f"line {line_no}: reason=missing needs visibility in {VISIBILITIES}, "
                        f"got {vis!r}"
                    )
            elif vis:
                problems.append(f"line {line_no}: visibility {vis!r} only allowed with missing")
            if (run_id, frame_id) not in valid_frames:
                problems.append(f"line {line_no}: ({run_id}, {frame_id}) is not in the cache")
            key = (run_id, frame_id, cls)
            if key in seen:
                problems.append(f"line {line_no}: duplicate row for {key}")
            seen.add(key)
            rows.append(ReviewRow(run_id, frame_id, cls, reason, vis, note))
    if problems:
        raise ReviewCsvError("\n".join(problems))
    return rows


def summarize(rows: list[ReviewRow], n_frames: int) -> dict[str, dict]:
    """Per class: ``bad`` count, ``bad_fraction`` of ``n_frames``, and
    ``hidden_missing`` (reported separately, not counted as bad)."""
    bad: Counter[str] = Counter()
    hidden: Counter[str] = Counter()
    for r in rows:
        (bad if r.counts_as_bad else hidden)[r.cls] += 1
    return {
        c: {"bad": bad[c], "bad_fraction": bad[c] / n_frames, "hidden_missing": hidden[c]}
        for c in CLASSES
    }
