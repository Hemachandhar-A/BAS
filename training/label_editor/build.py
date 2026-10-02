"""F14 stage 4b: generate the label-editor page (a static, offline HTML file).

  python -m training.review_sheet --edit --split <s> [--gold N] [--static] [--only-sample]

writes ``data/review/editor_<split>.html``. The frame list, the auto boxes and the
proposals (every cached raw candidate for a class the labeler found no box for, with
its score, area fraction and the rule that rejected it) are INLINED as
``window.EDITOR_DATA``; images are referenced by relative path. No server, no network.

Frames shown, in this order: (a) frames the labeler EXCLUDED, grouped by run; (b) with
``gold`` a seeded draw of N frames per run (stratified over the run's timeline) for
verification; (c) with ``static`` one frame per run for the static boxes (an edit there
becomes a ``static_overrides`` entry). Frames still ``pending`` are never shown.

This module is pure given its inputs except ``generate_editor`` (the I/O wrapper).
Images are BGR in Python; the page shows the JPEGs as they are.
"""

from __future__ import annotations

import json
import os
import random
from collections.abc import Callable
from pathlib import Path

from training.autolabel_rules import MOVABLE_CLASSES, STATIC_CLASSES, StaticResult
from training.spikes.postprocess import Box, RawDetection, box_area_frac
from training.spikes.select_v3 import exclusion_boxes
from training.spikes.select_v4 import HueSat, select_container_v4

HERE = Path(__file__).parent
SEED = 20261003
DEFAULT_GOLD = 6
EDITOR_DIR = Path("data/review")


class PageError(ValueError):
    """The page cannot be generated as asked."""


# --- small pure helpers --------------------------------------------------------------


def css_color(bgr: tuple[int, int, int]) -> str:
    """BGR triple (the Python side) -> ``#rrggbb`` (CSS is RGB)."""
    b, g, r = bgr
    return f"#{r:02x}{g:02x}{b:02x}"


def explain(code: str, box: Box, width: int, height: int, params: dict, cls: str) -> str:
    """Human text for a rejection code (select_v4's codes plus the static ones), with
    the numbers behind it. ``params`` = ``container`` rules (small_floor, area_ceiling,
    container_band, width_max, height_max)."""
    frac = box_area_frac(box, width, height)
    bw, bh = box[2] - box[0], box[3] - box[1]
    lo, hi = params["small_floor"], params["area_ceiling"]
    if code == "out_of_band":
        side = "too small" if frac < lo else "too large"
        return f"area {frac:.4f} outside {lo:.4f} to {hi:.4f} ({side})"
    if code == "too_wide":
        return f"width {bw:.0f}px over the {params['width_max']:.0f}px cap"
    if code == "too_tall":
        return f"height {bh:.0f}px over the {params['height_max']:.0f}px cap"
    if code == "small_wrong_colour":
        return (
            f"area {frac:.4f} below {params['container_band'][0]:.4f} and the centre is not "
            f"{cls}-coloured"
        )
    if code.startswith("overlaps_"):
        return f"overlaps the run's {code[len('overlaps_') :]} (IoU over 0.5)"
    if code == "not_highest_score":
        return "passed the rules but a higher score won"
    if code == "not_run_consensus":
        return "not the run's consensus box for this class"
    return code


def proposals_for_class(
    cls: str,
    cands: list[dict],
    width: int,
    height: int,
    rules: dict,
    statics: StaticResult | None,
    measure_hs: Callable[[Box], HueSat | None],
) -> list[dict]:
    """Every cached candidate for ``cls`` in cache order, numbered from 1:
    ``{n, box, score, area_frac, reason_code, reason}``. Movable classes get the exact
    code select_container_v4 gives (the rules ran on them already); static classes get
    ``out_of_band`` (outside the run-level band) or ``not_run_consensus``."""
    dets = [RawDetection(phrase="", score=c["score"], box=tuple(c["box"])) for c in cands]
    codes: dict[int, str] = {}
    params = rules["container"]
    if cls in MOVABLE_CLASSES:
        excl = exclusion_boxes(dict(statics.boxes), statics.start_button_white) if statics else {}
        sel = select_container_v4(dets, params, cls, width, height, excl, measure_hs)
        codes = {id(d): reason for d, reason in sel.rejected}
        if sel.chosen is not None:
            codes[id(sel.chosen)] = "chosen"
    else:
        lo, hi = rules["static"]["static_bands"][cls]
        params = {**params, "small_floor": lo, "area_ceiling": hi}
        for d in dets:
            frac = box_area_frac(d.box, width, height)
            codes[id(d)] = "out_of_band" if not lo <= frac <= hi else "not_run_consensus"
    out = []
    for n, d in enumerate(dets, start=1):
        code = codes.get(id(d), "unknown")
        out.append(
            {
                "n": n,
                "box": [float(v) for v in d.box],
                "score": float(d.score),
                "area_frac": box_area_frac(d.box, width, height),
                "reason_code": code,
                "reason": explain(code, d.box, width, height, params, cls),
            }
        )
    return out


def pick_gold(
    index: dict[str, dict], split: str, n: int, seed: int = SEED
) -> list[tuple[str, int]]:
    """``n`` frames per run of ``split``: the run's sorted frames are cut into ``n`` equal
    consecutive chunks and one frame is drawn from each (seeded), so a pick spreads over
    the run. A run with ``n`` frames or fewer contributes all of them."""
    rng = random.Random(seed)
    out: list[tuple[str, int]] = []
    for run in sorted(r for r, i in index.items() if i["split"] == split):
        ids = sorted(index[run]["frame_ids"])
        if len(ids) <= n:
            out += [(run, f) for f in ids]
            continue
        bounds = [round(i * len(ids) / n) for i in range(n + 1)]
        out += [(run, rng.choice(ids[bounds[i] : bounds[i + 1]])) for i in range(n)]
    return out


# --- the frame list ----------------------------------------------------------------------


def _records_by_cell(records: list[dict]) -> tuple[dict[tuple[int, str], list[dict]], tuple]:
    cells: dict[tuple[int, str], list[dict]] = {}
    dims = None
    for r in records:
        cells.setdefault((r["frame_id"], r["class"]), []).extend(r["candidates"])
        dims = dims or (r["width"], r["height"])
    return cells, dims


def build_frames(
    *,
    split: str,
    classes: list[str],
    index: dict[str, dict],
    labels: dict[str, dict[str, dict]],
    raw_by_run: dict[str, list[dict]],
    rules: dict,
    statics_by_run: dict[str, StaticResult | None],
    measure_factory: Callable[[str, int], Callable[[Box], HueSat | None]],
    img_dir: str,
    gold: int | None = None,
    static: bool = False,
    only: set[tuple[str, int]] | None = None,
    dims: dict[str, tuple[int, int]] | None = None,
    seed: int = SEED,
) -> tuple[list[dict], dict]:
    """The ordered frame entries of the page and a small ``info`` dict (counts)."""
    runs = sorted(r for r, i in index.items() if i["split"] == split)
    cells_by_run = {r: _records_by_cell(raw_by_run.get(r, [])) for r in runs}
    pending = 0

    def label_of(run: str, fid: int) -> dict | None:
        nonlocal pending
        lab = labels.get(run, {}).get(str(fid))
        if lab is None or lab["status"] == "pending":
            pending += 1
            return None
        return lab

    def entry(kind: str, run: str, fid: int, lab: dict, only_classes=None) -> dict:
        cells, rdims = cells_by_run[run]
        w, h = rdims or (dims or {}).get(run) or (None, None)
        if w is None:
            raise PageError(f"{run}: image size unknown (no raw record and no dims)")
        wanted = [c for c in classes if only_classes is None or c in only_classes]
        boxes = {c: [float(v) for v in lab["boxes"][c]] for c in wanted if c in lab["boxes"]}
        missing = [c for c in wanted if c not in boxes]
        proposals = {
            c: proposals_for_class(
                c,
                cells.get((fid, c), []),
                w,
                h,
                rules,
                statics_by_run.get(run),
                measure_factory(run, fid),
            )
            for c in missing
        }
        return {
            "key": f"{kind}|{run}_{fid}.jpg",
            "kind": kind,
            "run": run,
            "frame": fid,
            "file": f"{run}_{fid}.jpg",
            "img": f"{img_dir}/{run}_{fid}.jpg",
            "width": w,
            "height": h,
            "auto_boxes": boxes,
            "missing": missing,
            "reason": lab.get("reason", ""),
            "proposals": proposals,
            "gold": False,
        }

    def allowed(run: str, fid: int) -> bool:
        return only is None or (run, fid) in only

    frames: list[dict] = []
    seen: dict[tuple[str, int], dict] = {}
    for run in runs:  # (a) excluded, by run
        for fid in sorted(index[run]["frame_ids"]):
            if not allowed(run, fid):
                continue
            lab = label_of(run, fid)
            if lab is not None and lab["status"] == "excluded":
                e = entry("excluded", run, fid, lab)
                frames.append(e)
                seen[(run, fid)] = e
    gold_cells = [c for c in pick_gold(index, split, gold, seed) if allowed(*c)] if gold else []
    n_gold = 0
    for run, fid in gold_cells:  # (b) gold
        lab = label_of(run, fid)
        if lab is None:
            continue
        n_gold += 1
        if (run, fid) in seen:
            seen[(run, fid)]["gold"] = True
            continue
        e = entry("gold", run, fid, lab)
        e["gold"] = True
        frames.append(e)
    if static:  # (c) one frame per run, static boxes only
        for run in runs:
            fid = min(index[run]["frame_ids"])
            # a static check ignores ``only`` (it is per run) and needs only the run's static
            # boxes, which exist before the frame's movable calls are done
            lab = labels.get(run, {}).get(str(fid))
            if lab is None or (lab["status"] == "pending" and not lab["boxes"]):
                pending += 1
                continue
            frames.append(entry("static", run, fid, lab, only_classes=STATIC_CLASSES))
    info = {
        "split": split,
        "n_excluded": sum(1 for f in frames if f["kind"] == "excluded"),
        "gold_frames": n_gold,
        "n_static": sum(1 for f in frames if f["kind"] == "static"),
        "pending_skipped": pending,
        "missing_cells": sum(len(f["missing"]) for f in frames if f["kind"] == "excluded"),
        "proposals_per_missing_class": {
            c: [
                len(f["proposals"][c])
                for f in frames
                if f["kind"] == "excluded" and c in f["proposals"]
            ]
            for c in classes
        },
    }
    return frames, info


# --- the page ------------------------------------------------------------------------------


def _inline_json(obj) -> str:
    """JSON safe inside a <script>: '</' cannot end the tag, U+2028/9 cannot end a line."""
    text = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    return text.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def render_page(split: str, classes: list[str], frames: list[dict], info: dict) -> str:
    from training.review_sheet import COLORS

    for c in classes:
        if c not in COLORS:
            raise PageError(f"class {c!r} has no colour in review_sheet.COLORS")
    data = {
        "version": 1,
        "split": split,
        "classes": list(classes),
        "static_classes": [c for c in classes if c in STATIC_CLASSES],
        "colors": {c: css_color(COLORS[c]) for c in classes},
        "frames": frames,
        "info": info,
    }
    tpl = (HERE / "editor.html").read_text(encoding="utf-8")
    parts = {
        "%%CSS%%": (HERE / "editor.css").read_text(encoding="utf-8"),
        "%%DATA%%": _inline_json(data),
        "%%CORE_JS%%": (HERE / "editor_core.js").read_text(encoding="utf-8"),
        "%%UI_JS%%": (HERE / "editor_ui.js").read_text(encoding="utf-8"),
        "%%TITLE%%": f"label editor - {split}",
    }
    for key, val in parts.items():
        tpl = tpl.replace(key, val)
    return tpl


# --- I/O wrapper -------------------------------------------------------------------------------


def generate_editor(
    split: str,
    gold: int | None,
    static: bool,
    only_sample: bool,
    out_dir: Path = EDITOR_DIR,
) -> tuple[Path, dict]:
    """Read the raw cache and frozen rules, recompute the labels for the split's runs
    (offline, no model), build the entries and write the page."""
    import cv2

    from training.autolabel import (
        FRAMES_DIR,
        RAW_DIR,
        RULES_PATH,
        _detections,
        build_run_labels,
        frame_path,
        load_classes,
        load_index,
        load_priority_cells,
        read_raw,
    )
    from training.autolabel_rules import select_statics, static_frame_ids
    from training.spikes.select_v3 import mean_sat_val_in_box
    from training.spikes.select_v4 import patch_hue_sat

    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    classes = load_classes()
    index = load_index()
    runs = sorted(r for r, i in index.items() if i["split"] == split)
    only = set(load_priority_cells()) if only_sample else None
    if only is not None:
        only = {c for c in only if index[c[0]]["split"] == split}
        runs = sorted({r for r, _ in only})
    out_dir = Path(out_dir)
    img_dir = os.path.relpath(FRAMES_DIR / split, out_dir).replace("\\", "/")

    raw_by_run = {r: read_raw(RAW_DIR / f"{r}.jsonl") for r in runs}

    def sv(run, fid, box):
        img = cv2.imread(str(frame_path(index, run, fid)))
        return None if img is None else mean_sat_val_in_box(img, box)

    def hs(run, fid, box):
        img = cv2.imread(str(frame_path(index, run, fid)))
        return None if img is None else patch_hue_sat(img, box)

    labels: dict[str, dict] = {}
    statics_by_run: dict[str, StaticResult | None] = {}
    for run in runs:
        fids = sorted(index[run]["frame_ids"])

        recs = raw_by_run[run]
        labels[run] = {
            str(f): lab
            for f, lab in build_run_labels(
                run,
                fids,
                recs,
                rules,
                lambda f, b, run=run: sv(run, f, b),
                lambda f, b, run=run: hs(run, f, b),
            ).items()
        }
        statics_by_run[run] = None
        asked = {(r["frame_id"], r["class"]) for r in recs}
        sf = static_frame_ids(fids)
        if recs and all((f, c) in asked for f in sf for c in STATIC_CLASSES):
            dets = _detections(recs, run)
            statics_by_run[run] = select_statics(
                sf,
                {(f, c): dets.get((f, c), []) for f in sf for c in STATIC_CLASSES},
                recs[0]["width"],
                recs[0]["height"],
                rules,
                lambda fid, box, run=run: sv(run, fid, box),
            )

    def measure_factory(run, fid):
        return lambda box: hs(run, fid, box)

    frames, info = build_frames(
        split=split,
        classes=classes,
        index={r: index[r] for r in runs},
        labels=labels,
        raw_by_run=raw_by_run,
        rules=rules,
        statics_by_run=statics_by_run,
        measure_factory=measure_factory,
        img_dir=img_dir,
        gold=gold,
        static=static,
        only=only,
    )
    page = render_page(split, classes, frames, info)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"editor_{split}.html"
    path.write_text(page, encoding="utf-8")
    return path, info
