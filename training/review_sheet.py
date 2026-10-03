"""F14 stage 4: the human-review material for the auto-labels.

  python -m training.review_sheet --sample   # draw the TRAIN review sample
  python -m training.review_sheet            # gate sheets + pre-filled data/label_review.csv
  python -m training.review_sheet --static   # one frame per run (all runs), static boxes only,
                                             # + pre-filled data/review/static_review.csv
  python -m training.review_sheet --outcome  # review outcome (gate numbers) -> reports/dataset.json
  python -m training.review_sheet --fresh-sample [--n 60]
                                             # Stage 4b: fresh TRAIN sample from the BUILT dataset
                                             # (overlay applied): fresh_sheet_NN.jpg,
                                             # fresh_sample.json, prefilled fresh_review.csv
  python -m training.review_sheet --fresh-outcome [--record]
                                             # strict-load fresh_review.csv, print the gate;
                                             # --record adds label_review.fresh_after_repair
  python -m training.review_sheet --edit --split <s> [--gold [N]] [--static] [--only-sample]
                                             # the label editor page data/review/editor_<s>.html

The gate sample is drawn from TRAIN runs only (never val or test, AGENTS.md rule 12): a
seeded stratified draw (at least ``PER_RUN_MIN`` frames per run) plus up to ``N_FLAGGED``
frames with the largest change from the previous kept frame (a motion/blur proxy: the
hard cases rules v3 lost were moving, blurred or hand-covered containers). The static
sheet shows every run, once, because one static box applies to every frame of its run.

The CSV (``CSV_HEADER``) is Plan 5.8's five columns plus ``visibility`` and ``note``.
Sheets: 12 frames each, 5 classes in fixed colours with a legend, the frame tag
``<run_id>#<frame_id>`` under every frame.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

REVIEW_DIR = Path("data/review")
LABEL_REVIEW_PATH = Path("data/label_review.csv")
STATIC_REVIEW_PATH = REVIEW_DIR / "static_review.csv"
SAMPLE_PATH = REVIEW_DIR / "sample.json"
FRESH_SAMPLE_PATH = REVIEW_DIR / "fresh_sample.json"
FRESH_REVIEW_PATH = REVIEW_DIR / "fresh_review.csv"
BUILT_TRAIN_DIR = Path("data/dataset/train")
CORRECTIONS_TRAIN_PATH = Path("data/corrections/train.json")
FRAME_LABELS_PATH = Path("data/labels/frame_labels.json")

SEED = 20261002
FRESH_SEED = 20261003  # a new recorded seed for the Stage 4b fresh sample
N_FRESH = 60
N_STRATIFIED = 60
PER_RUN_MIN = 2
N_FLAGGED = 20
FLAGGED_MAX_PER_RUN = 2
PER_SHEET = 12
GATE_MAX_BAD_FRACTION = 0.10

CSV_HEADER = ["run_id", "frame_id", "class", "verdict", "reason", "visibility", "note"]
VERDICTS = ("ok", "bad")
REASONS = ("wrong_box", "missing", "duplicate", "wrong_class")
VISIBILITIES = ("visible", "partial", "hidden")
NO_BOX_NOTE = "no box drawn (auto-labeler found none)"

# BGR, fixed per class (same values as the P1.2 contact sheets)
COLORS = {
    "outer_box": (255, 128, 0),
    "tray": (128, 128, 128),
    "red_box": (0, 0, 255),
    "yellow_box": (0, 200, 200),
    "start_button": (255, 0, 255),
}
STATIC_CLASSES = ("outer_box", "tray", "start_button")

TILE_W, TILE_H, TAG_H, COLS, LEGEND_H = 400, 225, 38, 4, 56


class ReviewCsvError(ValueError):
    """Raised with every problem found, one per line."""


# --- the sample (pure) ----------------------------------------------------------------


def draw_review_sample(
    index: dict[str, dict], n: int, per_run_min: int, seed: int
) -> list[tuple[str, int]]:
    """Seeded draw of ``n`` (run_id, frame_id) cells from TRAIN runs: ``per_run_min``
    frames of every train run, each picked at random (``random.Random(seed).sample``,
    not the first ones), then the rest at random from the other train frames. Sorted."""
    runs = sorted(r for r, info in index.items() if info["split"] == "train")
    if n < per_run_min * len(runs):
        raise ValueError(f"n={n} is below {per_run_min} frames x {len(runs)} runs")
    rng = random.Random(seed)
    chosen: list[tuple[str, int]] = []
    for run in runs:
        ids = sorted(index[run]["frame_ids"])
        chosen += [(run, f) for f in rng.sample(ids, min(per_run_min, len(ids)))]
    taken = set(chosen)
    pool = [(r, f) for r in runs for f in sorted(index[r]["frame_ids"]) if (r, f) not in taken]
    extra = n - len(chosen)
    if extra > len(pool):
        raise ValueError(f"only {len(chosen) + len(pool)} train frames, need {n}")
    chosen += rng.sample(pool, extra)
    return sorted(chosen)


def draw_fresh_sample(
    frames_by_run: dict[str, list[int]],
    exclude: set[tuple[str, int]],
    n: int,
    per_run_min: int,
    seed: int,
) -> list[tuple[str, int]]:
    """Fresh draw of ``n`` cells from the built train frames, none in ``exclude`` (the earlier
    review sample), at least ``per_run_min`` per run where the run has that many left
    (``draw_review_sample`` on the remaining frames). Sorted."""
    left = {
        run: {"split": "train", "frame_ids": [f for f in ids if (run, f) not in exclude]}
        for run, ids in frames_by_run.items()
    }
    return draw_review_sample(left, n, per_run_min, seed)


def coco_boxes_by_frame(coco: dict) -> dict[tuple[str, int], dict[str, tuple[float, ...]]]:
    """(run, frame) -> {class: xyxy} from a COCO dict (xywh boxes), classes in category order."""
    name_of = {c["id"]: c["name"] for c in coco["categories"]}
    order = [c["name"] for c in coco["categories"]]
    cell_of = {}
    for img in coco["images"]:
        run, _, fid = img["file_name"][:-4].rpartition("_")
        cell_of[img["id"]] = (run, int(fid))
    out: dict[tuple[str, int], dict[str, tuple[float, ...]]] = {c: {} for c in cell_of.values()}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        out[cell_of[a["image_id"]]][name_of[a["category_id"]]] = (x, y, x + w, y + h)
    return {cell: {c: boxes[c] for c in order if c in boxes} for cell, boxes in out.items()}


def pick_flagged(
    scores: dict[tuple[str, int], float],
    exclude: set[tuple[str, int]],
    n: int,
    max_per_run: int,
) -> list[tuple[str, int]]:
    """The ``n`` highest-scoring cells not in ``exclude``, at most
    ``max_per_run`` per run; ties broken by (run, frame) so the pick is stable."""
    out: list[tuple[str, int]] = []
    per_run: dict[str, int] = {}
    for cell in sorted(scores, key=lambda c: (-scores[c], c[0], c[1])):
        if cell in exclude or per_run.get(cell[0], 0) >= max_per_run:
            continue
        out.append(cell)
        per_run[cell[0]] = per_run.get(cell[0], 0) + 1
        if len(out) == n:
            break
    return out


# --- sheets (pure given images) ------------------------------------------------------------


def _legend(width: int, classes: list[str]) -> np.ndarray:
    bar = np.full((LEGEND_H, width, 3), 20, dtype=np.uint8)
    x = 14
    for cls in classes:
        cv2.rectangle(bar, (x, 14), (x + 30, 42), COLORS[cls], -1)
        cv2.putText(bar, cls, (x + 38, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        x += 38 + 17 * len(cls) + 36
    return bar


def _draw_tile(
    image_bgr: np.ndarray, boxes: dict[str, tuple[float, float, float, float]], tag: str
) -> np.ndarray:
    h, w = image_bgr.shape[:2]
    thumb = cv2.resize(image_bgr, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
    sx, sy = TILE_W / w, TILE_H / h
    for cls, (x1, y1, x2, y2) in boxes.items():
        color = COLORS[cls]
        p1, p2 = (
            (int(round(x1 * sx)), int(round(y1 * sy))),
            (int(round(x2 * sx)), int(round(y2 * sy))),
        )
        cv2.rectangle(thumb, p1, p2, color, 2)
        label = cls
        org = (p1[0] + 2, max(16, p1[1] + 16))
        cv2.putText(thumb, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
        cv2.putText(thumb, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    tile = np.full((TILE_H + TAG_H, TILE_W, 3), 40, dtype=np.uint8)
    tile[:TILE_H] = thumb
    cv2.putText(tile, tag, (8, TILE_H + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return tile


def render_sheets(
    tiles: list[tuple[str, np.ndarray, dict]], classes: list[str], per_sheet: int = PER_SHEET
) -> list[np.ndarray]:
    """``tiles`` = (tag, BGR image, {class: xyxy box in image pixels}). Returns
    one image per ``per_sheet`` tiles, all the same size (a short last sheet is
    padded)."""
    rows = (per_sheet + COLS - 1) // COLS
    width = COLS * TILE_W
    sheets = []
    for start in range(0, len(tiles), per_sheet):
        sheet = np.full((LEGEND_H + rows * (TILE_H + TAG_H), width, 3), 40, dtype=np.uint8)
        sheet[:LEGEND_H] = _legend(width, classes)
        for i, (tag, img, boxes) in enumerate(tiles[start : start + per_sheet]):
            r, c = divmod(i, COLS)
            y0 = LEGEND_H + r * (TILE_H + TAG_H)
            sheet[y0 : y0 + TILE_H + TAG_H, c * TILE_W : (c + 1) * TILE_W] = _draw_tile(
                img, boxes, tag
            )
        sheets.append(sheet)
    return sheets


# --- the review CSVs ---------------------------------------------------------------------------


def prefill_rows(
    cells: list[tuple[str, int]],
    classes: list[str],
    boxes_found: dict[tuple[str, int], set[str]],
) -> list[list]:
    """One verdict=ok row per (cell, class). A class with no box drawn gets a
    note so the reviewer checks it (ok is only right if the object is not there)."""
    rows = []
    for run, fid in cells:
        found = boxes_found.get((run, fid))
        for cls in classes:
            note = NO_BOX_NOTE if found is not None and cls not in found else ""
            rows.append([run, fid, cls, "ok", "", "", note])
    return rows


def write_review_csv(path: Path, rows: list[list]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(rows)


@dataclass(frozen=True)
class ReviewRow:
    run_id: str
    frame_id: int
    cls: str
    verdict: str
    reason: str
    visibility: str
    note: str

    @property
    def hidden_missing(self) -> bool:
        return self.verdict == "bad" and self.reason == "missing" and self.visibility == "hidden"

    @property
    def counts_as_bad(self) -> bool:
        return self.verdict == "bad" and not self.hidden_missing


def load_label_review(
    path: Path, classes: list[str], valid_keys: set[tuple[str, int, str]]
) -> list[ReviewRow]:
    """Strict loader for ``label_review.csv`` and ``static_review.csv``.
    Class names come from ``classes`` (config/experiment.json); ``verdict`` ok|bad;
    ``reason`` empty when ok and one of ``REASONS`` when bad; ``visibility`` required
    when the reason is ``missing`` (optional otherwise, as in the P1.2 reviews);
    no stray whitespace around the first six fields; every key must be in
    ``valid_keys``; no duplicate keys. Raises ``ReviewCsvError`` listing every problem."""
    problems: list[str] = []
    rows: list[ReviewRow] = []
    seen: set[tuple[str, int, str]] = set()
    with Path(path).open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header != CSV_HEADER:
            raise ReviewCsvError(f"header must be {','.join(CSV_HEADER)}, got {header}")
        for line_no, rec in enumerate(reader, start=2):
            if not rec:
                continue
            if len(rec) != len(CSV_HEADER):
                problems.append(
                    f"line {line_no}: expected {len(CSV_HEADER)} fields, got {len(rec)}"
                )
                continue
            run_id, frame_s, cls, verdict, reason, vis, note = rec
            if [v for v in rec[:6] if v != v.strip()]:
                problems.append(f"line {line_no}: stray whitespace in {rec[:6]!r}")
                continue
            try:
                frame_id = int(frame_s)
            except ValueError:
                problems.append(f"line {line_no}: frame_id {frame_s!r} is not an integer")
                continue
            if cls not in classes:
                problems.append(f"line {line_no}: unknown class {cls!r}")
            if verdict not in VERDICTS:
                problems.append(f"line {line_no}: verdict {verdict!r} must be ok or bad")
            elif verdict == "ok" and reason:
                problems.append(
                    f"line {line_no}: reason {reason!r} must be empty when verdict is ok"
                )
            elif verdict == "bad" and reason not in REASONS:
                problems.append(
                    f"line {line_no}: verdict bad needs a reason in {REASONS}, got {reason!r}"
                )
            if vis and vis not in VISIBILITIES:
                problems.append(f"line {line_no}: visibility {vis!r} must be one of {VISIBILITIES}")
            if reason == "missing" and vis not in VISIBILITIES:
                problems.append(
                    f"line {line_no}: reason missing needs a visibility in {VISIBILITIES}"
                )
            key = (run_id, frame_id, cls)
            if key not in valid_keys:
                problems.append(f"line {line_no}: {key} is not in the sample")
            if key in seen:
                problems.append(f"line {line_no}: duplicate row for {key}")
            seen.add(key)
            rows.append(ReviewRow(run_id, frame_id, cls, verdict, reason, vis, note))
    if problems:
        raise ReviewCsvError("\n".join(problems))
    return rows


def bad_fractions(
    rows: list[ReviewRow],
    classes: list[str],
    n_frames: int,
    max_fraction: float = GATE_MAX_BAD_FRACTION,
) -> dict[str, dict]:
    """Per class: ``bad`` rows (hidden-and-missing is not bad; reported as
    ``hidden_missing``), ``bad_fraction`` of ``n_frames`` and the gate
    (``passes`` = fraction <= ``max_fraction``)."""
    out = {}
    for cls in classes:
        bad = sum(1 for r in rows if r.cls == cls and r.counts_as_bad)
        hidden = sum(1 for r in rows if r.cls == cls and r.hidden_missing)
        frac = bad / n_frames
        out[cls] = {
            "bad": bad,
            "bad_fraction": frac,
            "hidden_missing": hidden,
            "passes": frac <= max_fraction,
        }
    return out


def _gate(per_class: dict[str, dict]) -> bool:
    return all(v["passes"] for v in per_class.values())


def fresh_outcome(rows: list[ReviewRow], classes: list[str], cells: list[tuple[str, int]]) -> dict:
    """The Stage 4b gate on the fresh sample: ``bad_fractions`` (hidden-and-missing not
    counted, limit ``GATE_MAX_BAD_FRACTION`` per class) over the sampled frames."""
    in_sample = set(cells)
    per_class = bad_fractions(
        [r for r in rows if (r.run_id, r.frame_id) in in_sample], classes, n_frames=len(cells)
    )
    return {"n_frames": len(cells), "per_class": per_class, "gate_passes": _gate(per_class)}


def fresh_block(
    outcome: dict, rows: list[ReviewRow], meta: dict, csv_sha256: str, recorded_at: str
) -> dict:
    """The ``label_review.fresh_after_repair`` block of ``reports/dataset.json``: the fresh
    outcome plus where it came from (seed, CSV hash) and every bad cell, so a frame to
    exclude can be found without opening the CSV."""
    cells = {(r.run_id, r.frame_id) for r in rows}
    bad = [
        {
            "frame": f"{r.run_id}_{r.frame_id}.jpg",
            "class": r.cls,
            "reason": r.reason,
            "visibility": r.visibility,
        }
        for r in sorted(rows, key=lambda r: (r.run_id, r.frame_id, r.cls))
        if r.counts_as_bad and (r.run_id, r.frame_id) in cells
    ]
    return {
        "status": "recorded",
        "recorded_at": recorded_at,
        "source": str(FRESH_REVIEW_PATH).replace("\\", "/"),
        "csv_sha256": csv_sha256,
        "sample_seed": meta["seed"],
        "gate_max_bad_fraction": GATE_MAX_BAD_FRACTION,
        **outcome,
        "bad_cells": bad,
    }


def unexcluded_bad_frames(block: dict, overlay: dict) -> list[str]:
    """Frames with a bad cell in the fresh review that the overlay does not exclude (rule: every
    bad row in a fresh review excludes its frame from training)."""
    entries = overlay.get("frames", {})
    frames = sorted({c["frame"] for c in block["bad_cells"]})
    return [f for f in frames if not entries.get(f, {}).get("excluded")]


def record_fresh(report: dict, block: dict) -> dict:
    """A copy of ``report`` with ``block`` under ``label_review.fresh_after_repair``; the rest
    of ``label_review`` (the earlier 80-frame block) is kept as it is."""
    out = dict(report)
    lr = out.get("label_review")
    out["label_review"] = {**(lr if isinstance(lr, dict) else {}), "fresh_after_repair": block}
    return out


def _kept_gate(
    rows: list[ReviewRow],
    classes: list[str],
    cells: list[tuple[str, int]],
    labels: dict,
) -> dict:
    kept = {c for c in cells if labels[c[0]][str(c[1])]["status"] == "ok"}
    kept_rows = [r for r in rows if (r.run_id, r.frame_id) in kept]
    per_class = bad_fractions(kept_rows, classes, n_frames=len(kept)) if kept else {}
    return {"n_kept": len(kept), "per_class": per_class, "gate_passes": _gate(per_class)}


def review_outcome(
    rows: list[ReviewRow],
    classes: list[str],
    stratified: list[tuple[str, int]],
    flagged: list[tuple[str, int]],
    labels: dict,
) -> dict:
    """The review outcome recorded in ``reports/dataset.json``: the plan's gate on the
    stratified frames, the flagged frames reported separately, the same gate over the
    frames the labeler kept (status ``ok``; excluded frames never become training
    labels), and the check that the reviewer's ``missing`` cells equal the labeler's
    ``no box drawn`` notes and its own missing classes. Raises ``ValueError`` if a
    sampled frame is not labeled yet or has no review row for a class."""
    cells = stratified + flagged
    have = {(r.run_id, r.frame_id, r.cls) for r in rows}
    for run, fid in cells:
        lab = labels.get(run, {}).get(str(fid))
        if lab is None or lab["status"] == "pending":
            raise ValueError(f"{run}#{fid} is not labeled yet")
        for cls in classes:
            if (run, fid, cls) not in have:
                raise ValueError(f"no review row for {run}#{fid} {cls}")
    in_sample = set(cells)
    rows = [r for r in rows if (r.run_id, r.frame_id) in in_sample]
    strat_set, flag_set = set(stratified), set(flagged)
    strat_rows = [r for r in rows if (r.run_id, r.frame_id) in strat_set]
    flag_rows = [r for r in rows if (r.run_id, r.frame_id) in flag_set]
    strat = bad_fractions(strat_rows, classes, n_frames=len(stratified))
    flag = bad_fractions(flag_rows, classes, n_frames=len(flagged))

    reviewer = {(r.run_id, r.frame_id, r.cls) for r in rows if r.reason == "missing"}
    notes = {(r.run_id, r.frame_id, r.cls) for r in rows if r.note.startswith(NO_BOX_NOTE)}
    labeler = {
        (run, fid, cls)
        for run, fid in cells
        if labels[run][str(fid)]["status"] == "excluded"
        for cls in labels[run][str(fid)]["missing"]
    }

    def _cells(s: set) -> list[list]:
        return [[r, f, c] for r, f, c in sorted(s)]

    return {
        "gate_max_bad_fraction": GATE_MAX_BAD_FRACTION,
        "stratified": {
            "n_frames": len(stratified),
            "per_class": strat,
            "gate_passes": _gate(strat),
        },
        "flagged": {"n_frames": len(flagged), "per_class": flag, "gate_passes": _gate(flag)},
        "kept_frames": _kept_gate(rows, classes, cells, labels),
        "kept_frames_stratified": _kept_gate(rows, classes, stratified, labels),
        "labeler_excluded_frames": sum(
            1 for run, fid in cells if labels[run][str(fid)]["status"] == "excluded"
        ),
        "missing_cells_check": {
            "reviewer_missing": len(reviewer),
            "labeler_no_box_notes": len(notes),
            "labeler_missing": len(labeler),
            "equal": reviewer == notes == labeler,
            "only_reviewer": _cells(reviewer - notes),
            "only_notes": _cells(notes - reviewer),
            "only_labeler": _cells(labeler - reviewer),
        },
    }


# --- I/O ---------------------------------------------------------------------------


def _classes() -> list[str]:
    from training.autolabel import load_classes

    return load_classes()


def cmd_sample() -> None:
    from training.autolabel import frame_path, load_index
    from training.sample_frames import thumb_diff, thumbnail

    index = load_index()
    stratified = draw_review_sample(index, N_STRATIFIED, PER_RUN_MIN, SEED)
    scores: dict[tuple[str, int], float] = {}
    for run, info in index.items():
        if info["split"] != "train":
            continue
        prev = None
        for fid in sorted(info["frame_ids"]):
            img = cv2.imread(str(frame_path(index, run, fid)))
            t = thumbnail(img)
            scores[(run, fid)] = 0.0 if prev is None else thumb_diff(t, prev)
            prev = t
    flagged = pick_flagged(scores, set(stratified), N_FLAGGED, FLAGGED_MAX_PER_RUN)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_stratified": N_STRATIFIED,
                "per_run_min": PER_RUN_MIN,
                "flag_rule": f"largest 32x32 thumbnail change from the previous kept frame, "
                f"max {FLAGGED_MAX_PER_RUN} per run (motion/blur proxy, no model)",
                "stratified": [list(c) for c in stratified],
                "flagged": [[r, f, round(scores[(r, f)], 4)] for r, f in flagged],
                "cells": [list(c) for c in stratified] + [[r, f] for r, f in flagged],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"sample: {len(stratified)} stratified + {len(flagged)} flagged -> {SAMPLE_PATH}")


def _labels() -> dict:
    return json.loads(FRAME_LABELS_PATH.read_text(encoding="utf-8"))


def cmd_sheets() -> None:
    from training.autolabel import frame_path, load_index

    index, classes = load_index(), _classes()
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    cells = [(r, int(f)) for r, f in sample["cells"]]
    assert all(index[r]["split"] == "train" for r, _ in cells), "gate sheets are TRAIN only"
    labels = _labels()
    tiles, found = [], {}
    for run, fid in cells:
        lab = labels[run][str(fid)]
        if lab["status"] == "pending":
            raise SystemExit(f"{run}#{fid} is not labeled yet; run training.autolabel first")
        boxes = {c: tuple(b) for c, b in lab["boxes"].items()}
        found[(run, fid)] = set(boxes)
        tiles.append((f"{run}#{fid}", cv2.imread(str(frame_path(index, run, fid))), boxes))
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sheets = render_sheets(tiles, classes)
    for i, sheet in enumerate(sheets, start=1):
        cv2.imwrite(str(REVIEW_DIR / f"review_sheet_{i:02d}.jpg"), sheet)
    print(f"{len(tiles)} frames -> {len(sheets)} sheets in {REVIEW_DIR}")
    if LABEL_REVIEW_PATH.exists():
        print(f"{LABEL_REVIEW_PATH} exists; not overwritten")
    else:
        write_review_csv(LABEL_REVIEW_PATH, prefill_rows(cells, classes, found))
        print(f"wrote {LABEL_REVIEW_PATH} ({len(cells) * len(classes)} rows, all verdict=ok)")


def cmd_static() -> None:
    from training.autolabel import frame_path, load_index

    index, classes = load_index(), _classes()
    labels = _labels()
    order = sorted(index, key=lambda r: (["train", "val", "test"].index(index[r]["split"]), r))
    tiles, cells = [], []
    for run in order:
        fid = min(index[run]["frame_ids"])
        lab = labels[run][str(fid)]
        boxes = {c: tuple(lab["boxes"][c]) for c in STATIC_CLASSES if c in lab["boxes"]}
        cells.append((run, fid))
        tiles.append((f"{run}#{fid}", cv2.imread(str(frame_path(index, run, fid))), boxes))
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    sheets = render_sheets(tiles, [c for c in classes if c in STATIC_CLASSES])
    for i, sheet in enumerate(sheets, start=1):
        cv2.imwrite(str(REVIEW_DIR / f"static_sheet_{i:02d}.jpg"), sheet)
    print(f"{len(tiles)} runs -> {len(sheets)} static sheets in {REVIEW_DIR}")
    if STATIC_REVIEW_PATH.exists():
        print(f"{STATIC_REVIEW_PATH} exists; not overwritten")
    else:
        static = [c for c in classes if c in STATIC_CLASSES]
        found = {
            cell: {c for c in static if c in labels[cell[0]][str(cell[1])]["boxes"]}
            for cell in cells
        }
        write_review_csv(STATIC_REVIEW_PATH, prefill_rows(cells, static, found))
        print(f"wrote {STATIC_REVIEW_PATH} ({len(cells) * len(static)} rows, all verdict=ok)")


def compute_outcome() -> dict:
    """Load both review CSVs with the strict loader and compute the review outcome
    (fails loudly on any CSV problem or a frame the labeler has not finished)."""
    from training.autolabel import load_index

    index, classes = load_index(), _classes()
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    stratified = [(r, int(f)) for r, f in sample["stratified"]]
    flagged = [(r, int(f)) for r, f, *_ in sample["flagged"]]
    cells = stratified + flagged
    keys = {(r, f, c) for r, f in cells for c in classes}
    rows = load_label_review(LABEL_REVIEW_PATH, classes, keys)
    static_keys = {(r, min(i["frame_ids"]), c) for r, i in index.items() for c in STATIC_CLASSES}
    static_rows = load_label_review(STATIC_REVIEW_PATH, classes, static_keys)
    out = review_outcome(rows, classes, stratified, flagged, _labels())
    out["static_review"] = {
        "runs": len(index),
        "rows": len(static_rows),
        "bad_rows": sum(1 for r in static_rows if r.counts_as_bad),
    }
    return out


def cmd_outcome() -> None:
    from datetime import datetime

    out = compute_outcome()
    out["generated_at"] = datetime.now().isoformat(timespec="seconds")
    path = Path("reports/dataset.json")
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"version": 1}
    report["label_review"] = out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


def cmd_fresh_sample(n: int) -> None:
    coco = json.loads((BUILT_TRAIN_DIR / "_annotations.coco.json").read_text(encoding="utf-8"))
    boxes = coco_boxes_by_frame(coco)
    frames_by_run: dict[str, list[int]] = {}
    for run, fid in sorted(boxes):
        frames_by_run.setdefault(run, []).append(fid)
    old = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    exclude = {(r, int(f)) for r, f in old["cells"]}
    cells = draw_fresh_sample(frames_by_run, exclude, n, PER_RUN_MIN, FRESH_SEED)
    overlay_names: set[str] = set()
    if CORRECTIONS_TRAIN_PATH.exists():
        entries = json.loads(CORRECTIONS_TRAIN_PATH.read_text(encoding="utf-8")).get("frames", {})
        overlay_names = {k for k, e in entries.items() if not e.get("excluded")}
    corrected = [c for c in cells if f"{c[0]}_{c[1]}.jpg" in overlay_names]
    classes = [c["name"] for c in coco["categories"]]
    tiles = [
        (f"{run}#{fid}", cv2.imread(str(BUILT_TRAIN_DIR / f"{run}_{fid}.jpg")), boxes[(run, fid)])
        for run, fid in cells
    ]
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for stale in REVIEW_DIR.glob("fresh_sheet_*.jpg"):
        stale.unlink()
    sheets = render_sheets(tiles, classes)
    for i, sheet in enumerate(sheets, start=1):
        cv2.imwrite(str(REVIEW_DIR / f"fresh_sheet_{i:02d}.jpg"), sheet)
    per_run = {r: sum(1 for c in cells if c[0] == r) for r in sorted({c[0] for c in cells})}
    FRESH_SAMPLE_PATH.write_text(
        json.dumps(
            {
                "seed": FRESH_SEED,
                "n": len(cells),
                "per_run_min": PER_RUN_MIN,
                "source": "data/dataset/train (built, overlay applied)",
                "excluded_earlier_sample": len(exclude),
                "n_overlay_corrected": len(corrected),
                "overlay_corrected": [list(c) for c in corrected],
                "frames_per_run": per_run,
                "cells": [list(c) for c in cells],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(
        f"fresh sample: {len(cells)} frames ({len(corrected)} overlay-corrected), "
        f"{len(per_run)} runs, min {min(per_run.values())} max {max(per_run.values())} per run "
        f"-> {len(sheets)} sheets in {REVIEW_DIR}"
    )
    if FRESH_REVIEW_PATH.exists():
        print(f"{FRESH_REVIEW_PATH} exists; not overwritten")
    else:
        found = {c: set(boxes[c]) for c in cells}
        write_review_csv(FRESH_REVIEW_PATH, prefill_rows(cells, classes, found))
        print(f"wrote {FRESH_REVIEW_PATH} ({len(cells) * len(classes)} rows, all verdict=ok)")


def cmd_fresh_outcome(record: bool = False) -> None:
    meta = json.loads(FRESH_SAMPLE_PATH.read_text(encoding="utf-8"))
    cells = [(r, int(f)) for r, f in meta["cells"]]
    classes = _classes()
    keys = {(r, f, c) for r, f in cells for c in classes}
    rows = load_label_review(FRESH_REVIEW_PATH, classes, keys)
    missing = keys - {(r.run_id, r.frame_id, r.cls) for r in rows}
    if missing:
        raise SystemExit(f"{len(missing)} rows are missing, e.g. {sorted(missing)[:3]}")
    outcome = fresh_outcome(rows, classes, cells)
    print(json.dumps(outcome, indent=1))
    if not record:
        return
    import hashlib
    from datetime import datetime

    in_sample = set(cells)
    block = fresh_block(
        outcome,
        [r for r in rows if (r.run_id, r.frame_id) in in_sample],
        meta,
        hashlib.sha256(FRESH_REVIEW_PATH.read_bytes()).hexdigest(),
        datetime.now().isoformat(timespec="seconds"),
    )
    overlay = (
        json.loads(CORRECTIONS_TRAIN_PATH.read_text(encoding="utf-8"))
        if CORRECTIONS_TRAIN_PATH.exists()
        else {}
    )
    block["bad_frames_not_excluded"] = unexcluded_bad_frames(block, overlay)
    path = Path("reports/dataset.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    path.write_bytes(json.dumps(record_fresh(report, block), indent=1).encode("utf-8"))
    print(f"recorded label_review.fresh_after_repair in {path}")
    if block["bad_frames_not_excluded"]:
        print("NOT EXCLUDED yet:", block["bad_frames_not_excluded"])


def cmd_edit(split: str, gold: int | None, static: bool, only_sample: bool) -> None:
    from training.label_editor.build import generate_editor

    path, info = generate_editor(split, gold, static, only_sample)
    print(
        f"{path}: {info['n_excluded']} excluded frames ({info['missing_cells']} missing cells), "
        f"{info['gold_frames']} gold, {info['n_static']} static check frames, "
        f"{info['pending_skipped']} pending frames skipped"
    )
    for cls, counts in info["proposals_per_missing_class"].items():
        if counts:
            print(f"  {cls}: {len(counts)} missing, proposals per cell {counts}")
    cov = info["coverage"]
    if cov["missing_cells"]:
        print(
            f"  missing cells {cov['missing_cells']}: with any cached candidate "
            f"{cov['cells_with_any_candidate']}, with a container-sized non-static candidate "
            f"{cov['cells_with_plausible_candidate']} (a proxy; only looking tells)"
        )
    print(
        "open it from file:// in Chrome or Edge; download the overlay to "
        f"data/corrections/{split}.json"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--sample", action="store_true", help="draw the TRAIN review sample")
    g.add_argument(
        "--outcome", action="store_true", help="record the review outcome in reports/dataset.json"
    )
    g.add_argument("--edit", action="store_true", help="generate the label editor page")
    g.add_argument(
        "--fresh-sample", action="store_true", help="Stage 4b: fresh review sample, built dataset"
    )
    g.add_argument(
        "--fresh-outcome", action="store_true", help="gate numbers of data/review/fresh_review.csv"
    )
    ap.add_argument(
        "--record",
        action="store_true",
        help="with --fresh-outcome: write the block into reports/dataset.json (label_review)",
    )
    ap.add_argument("--n", type=int, default=N_FRESH, help="with --fresh-sample")
    ap.add_argument(
        "--static",
        action="store_true",
        help="one frame per run, static boxes only (with --edit: add the check frames)",
    )
    ap.add_argument("--split", choices=("train", "val", "test"), help="with --edit")
    ap.add_argument(
        "--gold",
        nargs="?",
        type=int,
        const=6,
        default=None,
        metavar="N",
        help="with --edit: also N seeded frames per run to verify (default 6 when given)",
    )
    ap.add_argument(
        "--only-sample",
        action="store_true",
        help="with --edit: only the frames of data/review/sample.json (the reviewed frames)",
    )
    args = ap.parse_args(argv)
    if args.edit:
        if not args.split:
            ap.error("--edit needs --split")
        cmd_edit(args.split, args.gold, args.static, args.only_sample)
    elif args.fresh_sample:
        cmd_fresh_sample(args.n)
    elif args.fresh_outcome:
        cmd_fresh_outcome(args.record)
    elif args.static and (args.sample or args.outcome):
        ap.error("--static combines only with --edit")
    elif args.sample:
        cmd_sample()
    elif args.static:
        cmd_static()
    elif args.outcome:
        cmd_outcome()
    else:
        cmd_sheets()


if __name__ == "__main__":
    main()
