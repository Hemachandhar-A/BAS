"""Per-class bad counts for P1.2 checkpoint 3. Pure: reads nothing but the
rows it is given (plus a CSV loader). The v3 verdict for a (run, frame, class)
cell is the changed-cell review's verdict when the cell is in it, else the
v2 review's (an unchanged cell's box is the same, so its verdict carries
over). Hidden-and-missing is not counted as bad."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from training.spikes.review_csv import CLASSES, ReviewRow

Cell = tuple[str, int, str]
CHANGED_HEADER = ["run_id", "frame_id", "class", "verdict", "reason", "visibility", "note"]


@dataclass(frozen=True)
class ChangedRow:
    run_id: str
    frame_id: int
    cls: str
    verdict: str  # "ok" | "bad"
    visibility: str


def load_changed_review(path: Path) -> list[ChangedRow]:
    rows: list[ChangedRow] = []
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != CHANGED_HEADER:
            raise ValueError(f"header must be {','.join(CHANGED_HEADER)}, got {header}")
        for line_no, rec in enumerate(reader, start=2):
            if not rec:
                continue
            run_id, frame_s, cls, verdict, _reason, vis, _note = rec
            if verdict not in ("ok", "bad"):
                raise ValueError(f"line {line_no}: verdict {verdict!r} must be ok or bad")
            if cls not in CLASSES:
                raise ValueError(f"line {line_no}: unknown class {cls!r}")
            rows.append(ChangedRow(run_id, int(frame_s), cls, verdict, vis))
    return rows


def v3_bad_cells(v2_rows: list[ReviewRow], changed: list[ChangedRow]) -> set[Cell]:
    verdict: dict[Cell, bool] = {(r.run_id, r.frame_id, r.cls): r.counts_as_bad for r in v2_rows}
    for c in changed:
        verdict[(c.run_id, c.frame_id, c.cls)] = c.verdict == "bad"
    return {cell for cell, bad in verdict.items() if bad}


def bad_counts(cells: set[Cell], n_frames: int) -> dict[str, dict]:
    out = {c: {"bad": 0, "bad_fraction": 0.0} for c in CLASSES}
    for _run, _fid, cls in cells:
        out[cls]["bad"] += 1
    for c in CLASSES:
        out[c]["bad_fraction"] = out[c]["bad"] / n_frames
    return out
