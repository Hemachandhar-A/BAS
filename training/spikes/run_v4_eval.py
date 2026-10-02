"""P1.4 rules v4: derive the parameters from the CLEAN container cells of the
80 reviewed frames (60 + 20 hold-out, all TRAIN runs), evaluate v4 against the
Lead's three review CSVs on the cached raw candidates (offline, no model), apply
the safeguard, and with ``--freeze`` write ``data/labels/frozen_rules.json``.

Clean cell = a container (red_box / yellow_box) box that v3 produced and the
Lead did not mark bad. Safeguard: if v4 changes any box in a cell the Lead did
not mark bad by more than IoU 0.9, or any reviewed cell gets worse, keep v3.

  python -m training.spikes.run_v4_eval            # evaluate, write the report
  python -m training.spikes.run_v4_eval --freeze   # also write frozen_rules.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import cv2
import numpy as np

from training.autolabel_rules import (
    MOVABLE_CLASSES,
    select_movable,
    select_statics,
    v3_equivalent_container_params,
)
from training.spikes.bad_counts import load_changed_review, v3_bad_cells
from training.spikes.postprocess import Box, box_area_frac
from training.spikes.review_csv import load_bad_boxes
from training.spikes.run_checkpoint1b import (
    frame_dims,
    frames_by_run,
    index_by_run_frame_class,
    load_raw_cache,
)
from training.spikes.run_checkpoint1c import _load_frame_cached, select_v3_frames
from training.spikes.select_v3 import mean_sat_val_in_box
from training.spikes.select_v4 import (
    CleanCell,
    compare_to_v3,
    derive_v4,
    in_colour_range,
    patch_hue_sat,
    reintroduced_wrong,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("p1.4-v4-eval")

V2 = Path("data/spikes/v2")
HO = Path("data/spikes/holdout")
V3_PARAMS = Path("data/spikes/v3/frozen_params.json")
OUT = Path("data/labels")
FROZEN = OUT / "frozen_rules.json"
EVAL = OUT / "rules_eval.json"
COLORS = {"v3": (0, 200, 0), "v4": (255, 0, 255)}


def _load():
    v3p = json.loads(V3_PARAMS.read_text())
    recs = {
        "v2": load_raw_cache(V2 / "raw_candidates.jsonl"),
        "ho": load_raw_cache(HO / "raw_candidates.jsonl"),
    }
    bad = v3_bad_cells(
        load_bad_boxes(V2 / "bad_boxes.csv", set(frame_dims(recs["v2"]))),
        load_changed_review(Path("data/spikes/v3/changed_review.csv")),
    )
    bad |= {
        (r.run_id, r.frame_id, r.cls)
        for r in load_bad_boxes(HO / "bad_boxes.csv", set(frame_dims(recs["ho"])))
        if r.counts_as_bad
    }
    return v3p, recs, bad


def _v3_boxes(v3p, recs, load):
    out: dict[tuple[str, int, str], Box | None] = {}
    for recs_ in recs.values():
        by, fr, dims = index_by_run_frame_class(recs_), frames_by_run(recs_), frame_dims(recs_)
        for run in sorted(fr):
            res = select_v3_frames(run, fr[run], by, dims, v3p, load)
            for fid in fr[run]:
                for cls in MOVABLE_CLASSES:
                    out[(run, fid, cls)] = res["boxes"][fid][cls]
    return out


def _v4_boxes(rules, recs, load):
    out: dict[tuple[str, int, str], Box | None] = {}
    detail: dict[tuple[str, int, str], dict] = {}
    for recs_ in recs.values():
        by, fr, dims = index_by_run_frame_class(recs_), frames_by_run(recs_), frame_dims(recs_)
        for run in sorted(fr):
            frames = fr[run]
            w, h = dims[(run, frames[0])]
            cands = {
                (f, c): by.get((run, f, c), [])
                for f in frames
                for c in ("outer_box", "tray", "start_button")
            }

            def sv(fid, box, run=run):
                img = load(run, fid)
                return None if img is None else mean_sat_val_in_box(img, box)

            st = select_statics(frames, cands, w, h, rules, sv)
            for fid in frames:
                img = load(run, fid)
                sel = select_movable(
                    {c: by.get((run, fid, c), []) for c in MOVABLE_CLASSES},
                    w,
                    h,
                    rules,
                    st,
                    lambda box, img=img: patch_hue_sat(img, box),
                )
                for cls in MOVABLE_CLASSES:
                    out[(run, fid, cls)] = sel[cls].chosen.box if sel[cls].chosen else None
                    detail[(run, fid, cls)] = {
                        "chosen_score": sel[cls].chosen.score if sel[cls].chosen else None,
                        "rejected": [[list(d.box), r] for d, r in sel[cls].rejected],
                    }
    return out, detail


def _clean_cells(v3_boxes, bad, load) -> list[CleanCell]:
    cells = []
    for (run, fid, cls), box in sorted(v3_boxes.items()):
        if box is None or (run, fid, cls) in bad:
            continue
        hs = patch_hue_sat(load(run, fid), box)
        if hs is None:
            continue
        cells.append(
            CleanCell(cls, box_area_frac(box, 848, 480), box[2] - box[0], box[3] - box[1], *hs)
        )
    return cells


def _small_band_report(params, recs, load) -> dict:
    """Cached container candidates between the small floor and v3's floor:
    how many pass the colour test (the skin-fragment question)."""
    lo3, floor = params["container_band"][0], params["small_floor"]
    rows = []
    for recs_ in recs.values():
        for rec in recs_:
            if rec["class"] not in MOVABLE_CLASSES:
                continue
            img = load(rec["run_id"], rec["frame_id"])
            for c in rec["candidates"]:
                a = box_area_frac(tuple(c["box"]), rec["width"], rec["height"])
                if floor <= a < lo3:
                    hs = patch_hue_sat(img, tuple(c["box"]))
                    ok = hs is not None and in_colour_range(
                        hs[0], hs[1], params["colour"][rec["class"]]
                    )
                    rows.append(
                        {
                            "cell": f"{rec['run_id']}#{rec['frame_id']}/{rec['class']}",
                            "area": round(a, 4),
                            "score": round(c["score"], 2),
                            "hue": None if hs is None else round(hs[0], 1),
                            "sat": None if hs is None else round(hs[1], 1),
                            "passes_colour": bool(ok),
                            "box": [round(v) for v in c["box"]],
                        }
                    )
    return {
        "n_candidates": len(rows),
        "n_pass_colour": sum(r["passes_colour"] for r in rows),
        "rows": rows,
    }


def _sheet(cells, v3_boxes, v4_boxes, load, path: Path) -> None:
    tiles = []
    for run, fid, cls in cells:
        img = load(run, fid).copy()
        for name, boxes in (("v3", v3_boxes), ("v4", v4_boxes)):
            b = boxes.get((run, fid, cls))
            if b:
                x1, y1, x2, y2 = (int(round(v)) for v in b)
                cv2.rectangle(img, (x1, y1), (x2, y2), COLORS[name], 3)
        cv2.putText(
            img,
            f"{run}#{fid} {cls}  green=v3 magenta=v4",
            (6, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        tiles.append(cv2.resize(img, (636, 360)))
    rows = [
        np.hstack(tiles[i : i + 2] + [np.zeros_like(tiles[0])] * (2 - len(tiles[i : i + 2])))
        for i in range(0, len(tiles), 2)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.vstack(rows))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args(argv)

    load = _load_frame_cached()
    v3p, recs, bad = _load()
    v3_boxes = _v3_boxes(v3p, recs, load)
    cells = _clean_cells(v3_boxes, bad, load)
    params = derive_v4(cells, tuple(v3p["container_band"]))
    rules_v4 = {
        "static": {k: v3p[k] for k in ("sat_max", "val_min", "static_bands")},
        "container": params,
    }
    v4_boxes, detail = _v4_boxes(rules_v4, recs, load)
    # v3 through the same new code path must reproduce v3 (sanity of the refactor)
    rules_v3 = {
        "static": rules_v4["static"],
        "container": v3_equivalent_container_params(v3p["container_band"]),
    }
    v3_again, _ = _v4_boxes(rules_v3, recs, load)
    path_diff = compare_to_v3(v3_boxes, v3_again, set())

    cmp_ = compare_to_v3(v3_boxes, v4_boxes, bad)
    small = _small_band_report(params, recs, load)
    log.info(
        "clean cells: %d; ceiling %.4f; width_max %.1f; height_max %.1f",
        len(cells),
        params["area_ceiling"],
        params["width_max"],
        params["height_max"],
    )
    log.info("v3 via new code path differs from select_v3_frames in: %s", path_diff)
    log.info("changed in Lead-bad cells: %s", cmp_["changed_in_bad_cells"])
    log.info("changed in UNMARKED cells: %s", cmp_["changed_in_unmarked_cells"])
    log.info(
        "small-band candidates: %d, pass colour %d", small["n_candidates"], small["n_pass_colour"]
    )
    changed = cmp_["changed_in_bad_cells"] + cmp_["changed_in_unmarked_cells"]
    for cell in changed:
        log.info(
            "  %s v3=%s v4=%s %s",
            cell,
            None if v3_boxes[cell] is None else [round(v) for v in v3_boxes[cell]],
            None if v4_boxes[cell] is None else [round(v) for v in v4_boxes[cell]],
            detail[cell]["rejected"][:4],
        )
    if changed:
        _sheet(changed, v3_boxes, v4_boxes, load, OUT / "eval" / "changed_cells.jpg")

    v2_boxes = {
        (e["run_id"], e["frame_id"], c): (tuple(v["box"]) if v["box"] else None)
        for e in json.loads((V2 / "report_v2.json").read_text())["per_frame"]
        for c, v in e["classes"].items()
    }
    lead_wrong: dict[tuple[str, int, str], list] = {}
    for r in load_bad_boxes(V2 / "bad_boxes.csv", {(k[0], k[1]) for k in v2_boxes}):
        cell = (r.run_id, r.frame_id, r.cls)
        if r.reason == "wrong_box" and v2_boxes.get(cell):
            lead_wrong.setdefault(cell, []).append(v2_boxes[cell])
    worse = reintroduced_wrong(v4_boxes, cmp_["changed_in_bad_cells"], lead_wrong)
    log.info("v4 reintroduces a box the Lead marked wrong_box in: %s", worse)
    safeguard_tripped = (
        bool(cmp_["changed_in_unmarked_cells"])
        or bool(worse)
        or bool(path_diff["changed_in_bad_cells"] or path_diff["changed_in_unmarked_cells"])
    )
    report = {
        "n_frames": len({(r, f) for r, f, _ in v3_boxes}),
        "n_cells": len(v3_boxes),
        "n_bad_cells_marked": len(bad),
        "bad_cells": sorted(map(list, bad)),
        "v4_params": params,
        "changed_in_bad_cells": cmp_["changed_in_bad_cells"],
        "changed_in_unmarked_cells": cmp_["changed_in_unmarked_cells"],
        "v3_new_path_matches_select_v3_frames": not (
            path_diff["changed_in_bad_cells"] or path_diff["changed_in_unmarked_cells"]
        ),
        "reintroduced_wrong_box": worse,
        "small_band": small,
        "safeguard_tripped": safeguard_tripped,
        "v3_boxes": {f"{r}#{f}/{c}": b for (r, f, c), b in sorted(v3_boxes.items())},
        "v4_boxes": {f"{r}#{f}/{c}": b for (r, f, c), b in sorted(v4_boxes.items())},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    EVAL.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    log.info("safeguard tripped: %s (report %s)", safeguard_tripped, EVAL)

    if args.freeze:
        keep = "v3" if safeguard_tripped else "v4"
        frozen = {
            "rules_version": keep,
            "static": rules_v4["static"],
            "container": rules_v3["container"] if keep == "v3" else params,
            "v4_candidate": params,
            "evidence": {
                "clean_cells": len(cells),
                "reviewed_frames": report["n_frames"],
                "safeguard_tripped": safeguard_tripped,
                "changed_in_bad_cells": cmp_["changed_in_bad_cells"],
                "changed_in_unmarked_cells": cmp_["changed_in_unmarked_cells"],
                "reintroduced_wrong_box": worse,
            },
        }
        FROZEN.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
        log.info("froze rules_version=%s -> %s", keep, FROZEN)


if __name__ == "__main__":
    main()
