"""P1.2 checkpoint 1c: selection v3 on the 60 cached frames (offline, no
model), thresholds derived and FROZEN, changed-vs-v2 table, v3 contact
sheets, validation of the Lead's v2 review CSV. Time-boxed, exploratory, like
run_checkpoint1b.py (orchestration, no tests); the tested pure logic is
select_v3.py and review_csv.py. ``select_v3_frames`` is shared with the
hold-out script (run_holdout.py), which loads the frozen parameters this
script writes instead of re-deriving anything.

Run: python -m training.spikes.run_checkpoint1c
"""

from __future__ import annotations

import csv
import json
import logging
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np

from training.spikes.postprocess import Box, RawDetection, box_area_frac, iou
from training.spikes.review_csv import load_bad_boxes, summarize
from training.spikes.run_checkpoint1 import RUNS_DIR, _read_frame
from training.spikes.run_checkpoint1b import (
    CONTAINER_CLASSES,
    RAW_CACHE_PATH,
    STATIC_CLASSES,
    _draw_v2_boxes,
    frame_dims,
    frames_by_run,
    index_by_run_frame_class,
    load_raw_cache,
    write_contact_sheets_v2,
)
from training.spikes.select_v2 import select_container, single_clean_candidate, static_consensus
from training.spikes.select_v3 import (
    exclusion_boxes,
    is_white_paper,
    mean_sat_val_in_box,
    static_consensus_filtered,
    status_change,
    tight_band,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p1.2-checkpoint1c")

V2_DIR = Path("data/spikes/v2")
V3_DIR = Path("data/spikes/v3")
FROZEN_PATH = V3_DIR / "frozen_params.json"
LEAD_CSV = V2_DIR / "bad_boxes.csv"
EXCLUDED_CARD_RUNS = ("x034", "x044")  # the two runs whose v2 start_button was the red container

FrameLoader = Callable[[str, int], "np.ndarray | None"]


# --- the v3 rule, applied to one run ------------------------------------------


def select_v3_frames(
    run_id: str,
    frame_ids: list[int],
    by_rfc: dict[tuple[str, int, str], list[RawDetection]],
    dims: dict[tuple[str, int], tuple[int, int]],
    params: dict,
    load_frame: FrameLoader,
) -> dict:
    """Frozen v3 rule for one run. Returns ``boxes[frame_id][class]`` (a box
    or None = missing) plus per-frame container rejection reasons and the
    start_button consensus's own white-paper measurement."""
    width, height = dims[(run_id, frame_ids[0])]
    bands = {k: tuple(v) for k, v in params["static_bands"].items()}
    container_band = tuple(params["container_band"])
    sat_max, val_min = params["sat_max"], params["val_min"]

    measured: dict[tuple[int, Box], tuple[float, float] | None] = {}

    def measure(fid: int, box: Box) -> tuple[float, float] | None:
        key = (fid, box)
        if key not in measured:
            img = load_frame(run_id, fid)
            measured[key] = None if img is None else mean_sat_val_in_box(img, box)
        return measured[key]

    def accept(fid: int, det: RawDetection) -> bool:
        sv = measure(fid, det.box)
        return sv is not None and is_white_paper(sv[0], sv[1], sat_max, val_min)

    static = {}
    for cls in ("outer_box", "tray"):
        cands = {fid: by_rfc.get((run_id, fid, cls), []) for fid in frame_ids}
        static[cls] = static_consensus(cands, bands[cls], width, height)
    cands = {fid: by_rfc.get((run_id, fid, "start_button"), []) for fid in frame_ids}
    static["start_button"] = static_consensus_filtered(
        cands, bands["start_button"], width, height, accept
    )

    # the consensus excludes containers only if it passed the white-paper test itself
    sb = static["start_button"]
    sb_white: bool | None = None
    sb_meas: tuple[float, float] | None = None
    if sb.consensus_box is not None:
        ms = [
            m
            for fid in frame_ids
            if sb.per_frame_box[fid] is not None
            and (m := measure(fid, sb.consensus_box)) is not None
        ]
        if ms:
            sb_meas = (float(np.median([m[0] for m in ms])), float(np.median([m[1] for m in ms])))
            sb_white = is_white_paper(sb_meas[0], sb_meas[1], sat_max, val_min)
    exclude = exclusion_boxes({c: static[c].consensus_box for c in STATIC_CLASSES}, sb_white)

    boxes: dict[int, dict[str, Box | None]] = {fid: {} for fid in frame_ids}
    chosen: dict[int, dict[str, RawDetection | None]] = {fid: {} for fid in frame_ids}
    rejected: dict[int, dict[str, list]] = {fid: {} for fid in frame_ids}
    for fid in frame_ids:
        for cls in STATIC_CLASSES:
            boxes[fid][cls] = static[cls].per_frame_box[fid]
        for cls in CONTAINER_CLASSES:
            sel = select_container(
                by_rfc.get((run_id, fid, cls), []), container_band, width, height, exclude
            )
            boxes[fid][cls] = sel.chosen.box if sel.chosen else None
            chosen[fid][cls] = sel.chosen
            rejected[fid][cls] = [(d.box, r) for d, r in sel.rejected]
    return {
        "boxes": boxes,
        "chosen": chosen,
        "rejected": rejected,
        "static": static,
        "start_button_white": sb_white,
        "start_button_measure": sb_meas,
        "exclude": exclude,
    }


def winning_phrases(
    run_boxes: dict[int, dict[str, Box | None]],
    run_id: str,
    by_rfc: dict[tuple[str, int, str], list[RawDetection]],
    tally: dict[str, Counter],
) -> None:
    """Per frame and class with a final box: the phrasing of the
    highest-scoring candidate whose box agrees (IoU >= 0.5) with it."""
    for fid, per_class in run_boxes.items():
        for cls, box in per_class.items():
            if box is None:
                continue
            agree = [d for d in by_rfc.get((run_id, fid, cls), []) if iou(d.box, box) >= 0.5]
            if agree:
                tally[cls][max(agree, key=lambda d: d.score).phrase] += 1


# --- threshold derivation (item 2a, 2d) ---------------------------------------


def _pct(xs: list[float]) -> dict:
    a = np.asarray(xs)
    return {
        "n": len(xs),
        "min": float(a.min()),
        "p05": float(np.percentile(a, 5)),
        "p10": float(np.percentile(a, 10)),
        "median": float(np.median(a)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "max": float(a.max()),
    }


def derive_params(
    by_rfc, by_run_frames, dims, v2_report: dict, lead_rows, load_frame: FrameLoader
) -> tuple[dict, dict]:
    """Derive and return (frozen_params, derivation_report)."""
    bad_card = {(r.run_id, r.frame_id) for r in lead_rows if r.cls == "start_button"}
    v2_pf = {(e["run_id"], e["frame_id"]): e["classes"] for e in v2_report["per_frame"]}

    # card: v2 start_button box in a frame the Lead did not flag, outside x034/x044
    card = []
    for (run_id, fid), classes in sorted(v2_pf.items()):
        box = classes["start_button"]["box"]
        if run_id in EXCLUDED_CARD_RUNS or box is None or (run_id, fid) in bad_card:
            continue
        img = load_frame(run_id, fid)
        sv = None if img is None else mean_sat_val_in_box(img, tuple(box))
        if sv is not None:
            card.append((run_id, fid, *sv))

    # containers: single clean candidates (v2 definition), over all 12 runs
    v2_bands = {k: tuple(v) for k, v in v2_report["area_bands"].items()}
    cont = {c: [] for c in CONTAINER_CLASSES}
    for run_id in sorted(by_run_frames):
        outer = v2_pf[(run_id, by_run_frames[run_id][0])]["outer_box"]["box"]
        w, h = dims[(run_id, by_run_frames[run_id][0])]
        for cls in CONTAINER_CLASSES:
            per_frame = {f: by_rfc.get((run_id, f, cls), []) for f in by_run_frames[run_id]}
            clean = single_clean_candidate(
                per_frame, v2_bands[cls], tuple(outer) if outer else None, w, h
            )
            for fid, det in clean.items():
                img = load_frame(run_id, fid)
                sv = None if img is None else mean_sat_val_in_box(img, det.box)
                if sv is not None:
                    cont[cls].append((run_id, fid, *sv, box_area_frac(det.box, w, h)))

    card_sat = [c[2] for c in card]
    card_val = [c[3] for c in card]
    cont_sat = [c[2] for cls in CONTAINER_CLASSES for c in cont[cls]]
    # rule: sat_max = midpoint of (card sat p95, lowest container sat); val_min = 0.9 * card min val
    sat_max = round((float(np.percentile(card_sat, 95)) + min(cont_sat)) / 2.0)
    val_min = float(int(0.9 * min(card_val)))
    areas = [c[4] for cls in CONTAINER_CLASSES for c in cont[cls]]
    band = tight_band(areas)

    params = {
        "sat_max": float(sat_max),
        "val_min": val_min,
        "container_band": [band[0], band[1]],
        "static_bands": {c: list(v2_bands[c]) for c in STATIC_CLASSES},
    }
    report = {
        "card": {
            "sat": _pct(card_sat),
            "val": _pct(card_val),
            "frames": [f"{r}#{f}" for r, f, *_ in card],
        },
        "containers": {
            cls: {
                "sat": _pct([c[2] for c in cont[cls]]),
                "val": _pct([c[3] for c in cont[cls]]),
                "area_frac": _pct([c[4] for c in cont[cls]]),
                "frames": [f"{r}#{f}" for r, f, *_ in cont[cls]],
            }
            for cls in CONTAINER_CLASSES
        },
        "pooled_container_area_frac": _pct(areas),
        "pooled_container_sat_min": float(min(cont_sat)),
    }
    return params, report


def _load_frame_cached() -> FrameLoader:
    cache: dict[tuple[str, int], np.ndarray | None] = {}

    def load(run_id: str, fid: int):
        key = (run_id, fid)
        if key not in cache:
            cache[key] = _read_frame(RUNS_DIR / run_id / "video.mp4", fid)
        return cache[key]

    return load


def v2_boxes_by_frame(v2_report: dict) -> dict[tuple[str, int], dict[str, Box | None]]:
    out = {}
    for e in v2_report["per_frame"]:
        out[(e["run_id"], e["frame_id"])] = {
            c: (tuple(v["box"]) if v["box"] else None) for c, v in e["classes"].items()
        }
    return out


def main() -> None:
    V3_DIR.mkdir(parents=True, exist_ok=True)
    records = load_raw_cache(RAW_CACHE_PATH)
    by_rfc = index_by_run_frame_class(records)
    by_run_frames = frames_by_run(records)
    dims = frame_dims(records)
    run_ids = sorted(by_run_frames)
    v2_report = json.loads((V2_DIR / "report_v2.json").read_text())
    load_frame = _load_frame_cached()

    # 1. Lead's review, validated
    rows = load_bad_boxes(LEAD_CSV, set(dims))
    review = summarize(rows, n_frames=len(dims))
    log.info("Lead v2 review (validated, %d rows): %s", len(rows), review)

    # 2a / 2d. thresholds
    params, deriv = derive_params(by_rfc, by_run_frames, dims, v2_report, rows, load_frame)
    FROZEN_PATH.write_text(json.dumps(params, indent=2))
    log.info("Frozen params: %s", params)
    log.info("Derivation: %s", json.dumps(deriv, indent=1)[:4000])

    # 2c/e/f. selection v3 on the 60 cached frames
    v3 = {
        r: select_v3_frames(r, by_run_frames[r], by_rfc, dims, params, load_frame) for r in run_ids
    }
    v2_boxes = v2_boxes_by_frame(v2_report)

    changed = []
    v3_boxes: dict[tuple[str, int], dict[str, Box | None]] = {}
    for r in run_ids:
        for fid in by_run_frames[r]:
            v3_boxes[(r, fid)] = v3[r]["boxes"][fid]
            for cls in (*STATIC_CLASSES, *CONTAINER_CLASSES):
                ch = status_change(v2_boxes[(r, fid)][cls], v3_boxes[(r, fid)][cls])
                if ch:
                    changed.append((r, fid, cls, *ch))
    with (V3_DIR / "changed_vs_v2.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "frame_id", "class", "v2_status", "v3_status"])
        w.writerows(changed)
    log.info("%d (frame,class) cells changed vs v2 across %d frames",
             len(changed), len({(c[0], c[1]) for c in changed}))

    # in-sample view of the Lead's 24 rows (mechanical; visual check is separate)
    insample = []
    for rw in rows:
        v2b = v2_boxes[(rw.run_id, rw.frame_id)][rw.cls]
        v3b = v3_boxes[(rw.run_id, rw.frame_id)][rw.cls]
        insample.append({
            "run": rw.run_id, "frame": rw.frame_id, "class": rw.cls, "reason": rw.reason,
            "visibility": rw.visibility,
            "v2": [round(x) for x in v2b] if v2b else None,
            "v3": [round(x) for x in v3b] if v3b else None,
            "iou_v2_v3": round(iou(v2b, v3b), 3) if v2b and v3b else None,
        })

    # phrasing tallies (cache has both phrasings per class)
    tally = {c: Counter() for c in (*STATIC_CLASSES, *CONTAINER_CLASSES)}
    for r in run_ids:
        winning_phrases(v3[r]["boxes"], r, by_rfc, tally)

    per_run = {
        r: {
            "start_button_consensus": v3[r]["static"]["start_button"].consensus_box,
            "start_button_white": v3[r]["start_button_white"],
            "start_button_measure": v3[r]["start_button_measure"],
            "missing": {
                cls: sum(1 for f in by_run_frames[r] if v3[r]["boxes"][f][cls] is None)
                for cls in (*STATIC_CLASSES, *CONTAINER_CLASSES)
            },
        }
        for r in run_ids
    }
    # cross-run position of the card consensus (for the (b) decision)
    card_boxes = {r: v3[r]["static"]["start_button"].consensus_box for r in run_ids}

    report = {
        "lead_v2_review": review,
        "params": params,
        "derivation": deriv,
        "changed_cells": len(changed),
        "changed_frames": len({(c[0], c[1]) for c in changed}),
        "in_sample_rows": insample,
        "phrase_tally": {c: dict(v) for c, v in tally.items()},
        "per_run": per_run,
        "card_consensus_by_run": card_boxes,
        "missing_totals": {
            cls: sum(1 for k in v3_boxes if v3_boxes[k][cls] is None)
            for cls in (*STATIC_CLASSES, *CONTAINER_CLASSES)
        },
    }
    (V3_DIR / "report_v3.json").write_text(json.dumps(report, indent=2, default=str))

    tiles = []
    for r in run_ids:
        for fid in by_run_frames[r]:
            frame = load_frame(r, fid)
            if frame is not None:
                tiles.append((f"{r}#{fid}", _draw_v2_boxes(frame, v3_boxes[(r, fid)])))
    sheets = write_contact_sheets_v2(tiles, V3_DIR, prefix="contact_sheet_v3_")
    log.info("Wrote sheets %s", sheets)
    log.info("Missing totals: %s", report["missing_totals"])
    log.info("Phrase tally: %s", report["phrase_tally"])


if __name__ == "__main__":
    main()
