"""F14 stage 3: zero-shot auto-labeling with Grounding DINO tiny (P1.2 DECISION),
resumable, with the box rules applied OFFLINE to a raw-candidate cache.

  python -m training.autolabel --detach     # start the labeling queue in the background
  python -m training.autolabel --resume     # run the queue in this terminal (resumes)
  python -m training.autolabel --status     # done/total, seconds per call, estimated finish
  python -m training.autolabel --build      # raw cache -> boxes -> COCO per split (offline)
  python -m training.autolabel --hands      # hand boxes for every sampled frame (hard-case counts)

Model work: one call per (frame, class), ONE phrasing per class
(``training/prompts.yaml``). Static classes (outer_box, tray, start_button) are
asked on about 5 frames spread across each run; movable classes (red_box,
yellow_box) on every frame. Every call's candidates (up to 10, score >= 0.20)
are appended to ``data/labels/raw/<run_id>.jsonl`` as one JSON line, so the
rules (``data/labels/frozen_rules.json``) can change without touching the
model. Run with ``HF_HUB_OFFLINE=1`` (set here if unset). Images are BGR;
RGB only at the model boundary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from training.autolabel_rules import (  # noqa: E402
    MOVABLE_CLASSES,
    STATIC_CLASSES,
    select_movable,
    select_statics,
    static_frame_ids,
)
from training.spikes.postprocess import Box, RawDetection  # noqa: E402
from training.spikes.select_v3 import mean_sat_val_in_box  # noqa: E402
from training.spikes.select_v4 import patch_hue_sat  # noqa: E402

FRAMES_DIR = Path("data/frames")
LABELS_DIR = Path("data/labels")
RAW_DIR = LABELS_DIR / "raw"
HANDS_DIR = LABELS_DIR / "hands"
COCO_DIR = LABELS_DIR / "coco"
RULES_PATH = LABELS_DIR / "frozen_rules.json"
PROMPTS_PATH = Path("training/prompts.yaml")
SAMPLE_PATH = Path("data/review/sample.json")
PID_PATH = LABELS_DIR / "autolabel.pid"
LOG_PATH = LABELS_DIR / "autolabel.log"
EXPERIMENT_PATH = Path("config/experiment.json")

SPLIT_ORDER = ("train", "val", "test")
BOX_THRESHOLD = TEXT_THRESHOLD = SCORE_FLOOR = 0.20
MAX_CANDIDATES = 10


@dataclass(frozen=True, order=True)
class Job:
    run_id: str
    frame_id: int
    cls: str


# --- raw cache (pure) -----------------------------------------------------------


def raw_record(
    run_id: str,
    frame_id: int,
    cls: str,
    phrase: str,
    width: int,
    height: int,
    candidates: list[tuple[Box, float]],
    seconds: float,
    thresholds: tuple[float, float, float] = (BOX_THRESHOLD, TEXT_THRESHOLD, SCORE_FLOOR),
) -> dict:
    """One cache line, same shape as the spike caches (data/spikes/v2)."""
    return {
        "run_id": run_id,
        "frame_id": frame_id,
        "class": cls,
        "phrase": phrase,
        "width": width,
        "height": height,
        "box_threshold": thresholds[0],
        "text_threshold": thresholds[1],
        "score_floor": thresholds[2],
        "seconds": seconds,
        "candidates": [{"box": list(b), "score": s} for b, s in candidates],
    }


def read_raw(path: Path) -> list[dict]:
    """Records of one raw file. A truncated LAST line (a killed process) is
    ignored; a malformed line anywhere else is an error."""
    path = Path(path)
    if not path.exists():
        return []
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out = []
    for i, line in enumerate(lines):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as e:
            if i == len(lines) - 1:
                break
            raise ValueError(f"{path}: line {i + 1} is not valid JSON") from e
    return out


def done_keys(records: list[dict]) -> set[tuple[str, int, str]]:
    return {(r["run_id"], r["frame_id"], r["class"]) for r in records}


# --- job planning (pure) -----------------------------------------------------------


def plan_jobs(index: dict[str, dict], priority_cells: list[tuple[str, int]]) -> list[Job]:
    """The whole queue, in order: (1) the static-class calls of every run on
    ``static_frame_ids`` (train runs first), (2) the movable-class calls of the
    ``priority_cells`` (the review sample), (3) the movable-class calls of
    every other frame (train, val, test; by run, then frame). No duplicates."""
    runs = sorted(index, key=lambda r: (SPLIT_ORDER.index(index[r]["split"]), r))
    jobs: list[Job] = []
    for run in runs:
        for fid in static_frame_ids(index[run]["frame_ids"]):
            jobs += [Job(run, fid, c) for c in STATIC_CLASSES]
    seen: set[tuple[str, int]] = set()
    for run, fid in priority_cells:
        if (run, fid) not in seen:
            seen.add((run, fid))
            jobs += [Job(run, fid, c) for c in MOVABLE_CLASSES]
    for run in runs:
        for fid in sorted(index[run]["frame_ids"]):
            if (run, fid) not in seen:
                jobs += [Job(run, fid, c) for c in MOVABLE_CLASSES]
    return jobs


def status_summary(done: int, total: int, recent_seconds: list[float]) -> dict:
    mean = float(np.mean(recent_seconds)) if recent_seconds else None
    remaining = total - done
    return {
        "done": done,
        "total": total,
        "remaining": remaining,
        "mean_seconds_per_call": mean,
        "eta_seconds": None if mean is None else remaining * mean,
    }


# --- applying the rules to a run's raw records (pure) ----------------------------------


def _detections(records: list[dict], run_id: str) -> dict[tuple[int, str], list[RawDetection]]:
    out: dict[tuple[int, str], list[RawDetection]] = {}
    for r in records:
        if r["run_id"] != run_id:
            continue
        out.setdefault((r["frame_id"], r["class"]), []).extend(
            RawDetection(phrase=r["phrase"], score=c["score"], box=tuple(c["box"]))
            for c in r["candidates"]
        )
    return out


def build_run_labels(
    run_id: str,
    frame_ids: list[int],
    records: list[dict],
    rules: dict,
    measure_sv,
    measure_hs,
    static_ids: list[int] | None = None,
) -> dict[int, dict]:
    """Boxes for every frame of one run from its raw records.

    Per frame: ``status`` ``ok`` (all classes have a box), ``excluded`` (a
    class is missing; ``missing`` and ``reason`` say which) or ``pending``
    (calls not finished); ``boxes`` (class -> box found); ``candidates``
    (class -> every raw candidate box, for the hard-case count).
    ``measure_sv(frame_id, box)`` and ``measure_hs(frame_id, box)`` read pixels
    only when the rules need them."""
    static_ids = list(static_ids) if static_ids is not None else static_frame_ids(frame_ids)
    dets = _detections(records, run_id)
    # an empty call still leaves a record: "asked" is the key set, not the candidates
    asked = {(r["frame_id"], r["class"]) for r in records if r["run_id"] == run_id}
    static_complete = all((f, c) in asked for f in static_ids for c in STATIC_CLASSES)

    width = height = None
    for r in records:
        if r["run_id"] == run_id:
            width, height = r["width"], r["height"]
            break

    out: dict[int, dict] = {}
    statics = None
    if static_complete and width is not None:
        statics = select_statics(
            static_ids,
            {(f, c): dets.get((f, c), []) for f in static_ids for c in STATIC_CLASSES},
            width,
            height,
            rules,
            measure_sv,
        )
    for fid in sorted(frame_ids):
        label = {"status": "pending", "boxes": {}, "missing": [], "reason": "", "candidates": {}}
        out[fid] = label
        if statics is None:
            continue
        for cls in STATIC_CLASSES:
            if statics.boxes[cls] is not None:
                label["boxes"][cls] = statics.boxes[cls]
        for cls in MOVABLE_CLASSES:
            label["candidates"][cls] = [d.box for d in dets.get((fid, cls), [])]
        static_missing = [c for c in STATIC_CLASSES if statics.boxes[c] is None]
        if static_missing:
            label["status"] = "excluded"
            label["missing"] = static_missing
            label["reason"] = "static_missing:" + ",".join(static_missing)
            continue
        if not all((fid, c) in asked for c in MOVABLE_CLASSES):
            continue
        sel = select_movable(
            {c: dets.get((fid, c), []) for c in MOVABLE_CLASSES},
            width,
            height,
            rules,
            statics,
            lambda box, fid=fid: measure_hs(fid, box),
        )
        missing = []
        for cls in MOVABLE_CLASSES:
            if sel[cls].chosen is None:
                missing.append(cls)
            else:
                label["boxes"][cls] = sel[cls].chosen.box
        if missing:
            label["status"] = "excluded"
            label["missing"] = missing
            label["reason"] = "missing:" + ",".join(missing)
        else:
            label["status"] = "ok"
    return out


# --- hard cases (pure) ----------------------------------------------------------------


def _intersects(a: Box, b: Box) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def hand_overlaps_class(
    hand_boxes: list[Box], candidate_boxes: list[Box], nearest_box: Box | None
) -> bool:
    """A hand box intersects a raw candidate of the class in this frame, or the
    box the class has in the nearest frame of the run that has one (a proxy
    for where the container is when it has no box here)."""
    targets = list(candidate_boxes) + ([nearest_box] if nearest_box is not None else [])
    return any(_intersects(h, t) for h in hand_boxes for t in targets)


def _nearest_box(run_labels: dict[int, dict], fid: int, cls: str) -> Box | None:
    have = [(abs(f - fid), f) for f, lab in run_labels.items() if f != fid and cls in lab["boxes"]]
    if not have:
        return None
    return run_labels[min(have)[1]]["boxes"][cls]


def hard_case_stats(
    labels: dict[str, dict[int, dict]], hands: dict[tuple[str, int], list[Box]]
) -> dict:
    """Hard-case coverage. For excluded frames with a missing movable class:
    how many have a hand in the frame and how many a hand overlapping the
    missing class (``hand_overlaps_class``). For the final (ok) frames: how many
    have a hand overlapping a red or yellow box. Frames without hand data are
    counted separately and never guessed."""
    s = Counter()
    for run, run_labels in labels.items():
        for fid, lab in run_labels.items():
            if (run, fid) not in hands:
                s["frames_without_hand_data"] += 1
                continue
            hb = hands[(run, fid)]
            if lab["status"] == "excluded" and any(c in MOVABLE_CLASSES for c in lab["missing"]):
                s["excluded_movable_missing"] += 1
                if hb:
                    s["excluded_with_hand"] += 1
                    if any(
                        hand_overlaps_class(
                            hb, lab["candidates"].get(c, []), _nearest_box(run_labels, fid, c)
                        )
                        for c in lab["missing"]
                        if c in MOVABLE_CLASSES
                    ):
                        s["excluded_with_hand_overlapping_missing_class"] += 1
            elif lab["status"] == "ok":
                s["final_frames"] += 1
                if any(_intersects(h, lab["boxes"][c]) for h in hb for c in MOVABLE_CLASSES):
                    s["final_with_hand_over_container"] += 1
    return dict(s)


# --- COCO (pure) ------------------------------------------------------------------------


def build_coco(
    split: str,
    labels: dict[str, dict[int, dict]],
    dims: dict[str, tuple[int, int]],
    classes: list[str],
) -> dict:
    """COCO-style dict from the ``ok`` frames only. Category ids are the index
    in ``classes`` (0-based, contiguous); boxes are xywh in original pixels,
    clipped to the image; ids are assigned in (run, frame) order."""
    images, annotations = [], []
    for run in sorted(labels):
        w, h = dims[run]
        for fid in sorted(labels[run]):
            lab = labels[run][fid]
            if lab["status"] != "ok":
                continue
            image_id = len(images) + 1
            images.append(
                {"id": image_id, "file_name": f"{run}_{fid}.jpg", "width": w, "height": h}
            )
            for cid, cls in enumerate(classes):
                x1, y1, x2, y2 = lab["boxes"][cls]
                x1, y1 = max(0.0, x1), max(0.0, y1)
                x2, y2 = min(float(w), x2), min(float(h), y2)
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": cid,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0,
                    }
                )
    return {
        "info": {"description": "auto-labels (Grounding DINO tiny + frozen rules)", "split": split},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": i, "name": c} for i, c in enumerate(classes)],
    }


# --- I/O layer ---------------------------------------------------------------------------


def load_classes(path: Path = EXPERIMENT_PATH) -> list[str]:
    from contracts import ExperimentDefinition

    return list(ExperimentDefinition.from_json(path).classes)


def load_prompts(path: Path = PROMPTS_PATH, classes: list[str] | None = None) -> dict[str, str]:
    """class -> its ONE phrase. Lowercase, trailing period, exactly one phrasing,
    and the class set equals ``classes`` (experiment.classes)."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    prompts = data["classes"]
    classes = classes if classes is not None else load_classes()
    if set(prompts) != set(classes):
        raise ValueError(f"prompts classes {sorted(prompts)} != experiment classes {classes}")
    out = {}
    for cls in classes:
        phrases = prompts[cls]
        if len(phrases) != 1:
            raise ValueError(f"{cls}: exactly one phrasing is expected, got {phrases}")
        phrase = phrases[0]
        if phrase != phrase.lower() or not phrase.endswith("."):
            raise ValueError(f"{cls}: phrase {phrase!r} must be lowercase and end with a period")
        out[cls] = phrase
    return out


def load_index() -> dict[str, dict]:
    return json.loads((FRAMES_DIR / "index.json").read_text(encoding="utf-8"))


def load_priority_cells() -> list[tuple[str, int]]:
    if not SAMPLE_PATH.exists():
        return []
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    return [(r, int(f)) for r, f in sample["cells"]]


def frame_path(index: dict, run: str, fid: int) -> Path:
    return FRAMES_DIR / index[run]["split"] / f"{run}_{fid}.jpg"


def _all_done() -> set[tuple[str, int, str]]:
    keys: set[tuple[str, int, str]] = set()
    if RAW_DIR.exists():
        for p in sorted(RAW_DIR.glob("*.jsonl")):
            keys |= done_keys(read_raw(p))
    return keys


def _recent_seconds(n: int = 60) -> list[float]:
    """Model seconds of the most recently written records (by file mtime)."""
    if not RAW_DIR.exists():
        return []
    files = sorted(RAW_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
    secs: list[float] = []
    for p in files:
        secs += [r["seconds"] for r in reversed(read_raw(p))]
        if len(secs) >= n:
            break
    return secs[:n]


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            return False
        code = ctypes.c_ulong()
        ok = kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
        kernel32.CloseHandle(handle)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def worker_pid() -> int | None:
    try:
        pid = int(PID_PATH.read_text().strip())
    except (OSError, ValueError):
        return None
    return pid if pid_alive(pid) else None


def cmd_status() -> None:
    index = load_index()
    jobs = plan_jobs(index, load_priority_cells())
    done = _all_done()
    n_done = sum((j.run_id, j.frame_id, j.cls) in done for j in jobs)
    s = status_summary(n_done, len(jobs), _recent_seconds())
    groups = {
        "static": [j for j in jobs if j.cls in STATIC_CLASSES],
        "review_sample_movable": [],
        "other_movable": [],
    }
    pri = set(load_priority_cells())
    for j in jobs:
        if j.cls in MOVABLE_CLASSES:
            groups[
                "review_sample_movable" if (j.run_id, j.frame_id) in pri else "other_movable"
            ].append(j)
    print(f"done {s['done']}/{s['total']} calls ({s['remaining']} remaining)")
    for name, js in groups.items():
        print(f"  {name}: {sum((j.run_id, j.frame_id, j.cls) in done for j in js)}/{len(js)}")
    if s["mean_seconds_per_call"] is None:
        print("seconds per call: n/a (no calls yet)")
    else:
        eta = s["eta_seconds"]
        finish = time.strftime("%Y-%m-%d %H:%M", time.localtime(time.time() + eta))
        print(
            f"seconds per call (model, last {len(_recent_seconds())}): "
            f"{s['mean_seconds_per_call']:.2f}; remaining ~{eta / 3600:.2f} h; "
            f"estimated finish {finish} (local time)"
        )
    pid = worker_pid()
    print(f"worker: {'alive, pid ' + str(pid) if pid else 'not running'}; log {LOG_PATH}")


def cmd_resume() -> None:
    from training.spikes.detector import GroundingDinoSpike

    index = load_index()
    classes = load_classes()
    prompts = load_prompts(classes=classes)
    jobs = plan_jobs(index, load_priority_cells())
    done = _all_done()
    pending = [j for j in jobs if (j.run_id, j.frame_id, j.cls) not in done]
    print(f"{len(pending)} of {len(jobs)} calls pending", flush=True)
    if not pending:
        return
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PID_PATH.write_text(str(os.getpid()))
    detector = GroundingDinoSpike(box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD)
    cur_key, rgb, hw = None, None, (0, 0)
    n = 0
    for job in pending:
        if (job.run_id, job.frame_id) != cur_key:
            img = cv2.imread(str(frame_path(index, job.run_id, job.frame_id)))
            if img is None:
                print(f"cannot read {job.run_id}#{job.frame_id}; skipped", flush=True)
                rgb = None
                cur_key = (job.run_id, job.frame_id)
                continue
            hw = img.shape[:2]
            rgb = np.ascontiguousarray(img[..., ::-1])  # BGR -> RGB at the model boundary
            cur_key = (job.run_id, job.frame_id)
        if rgb is None:
            continue
        t0 = time.perf_counter()
        res = detector.detect(rgb, [prompts[job.cls]])
        seconds = time.perf_counter() - t0
        cands = [
            (d.box, d.score)
            for d in sorted(res.detections, key=lambda d: -d.score)
            if d.score >= SCORE_FLOOR
        ][:MAX_CANDIDATES]
        rec = raw_record(
            job.run_id, job.frame_id, job.cls, prompts[job.cls], hw[1], hw[0], cands, seconds
        )
        with (RAW_DIR / f"{job.run_id}.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        n += 1
        if n % 25 == 0:
            print(
                f"{n} calls this session; last {job.run_id}#{job.frame_id} "
                f"{job.cls} {seconds:.1f}s",
                flush=True,
            )
    print(f"finished: {n} calls this session", flush=True)


def cmd_detach() -> None:
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if worker_pid():
        print(f"already running, pid {worker_pid()}")
        return
    log = LOG_PATH.open("a", encoding="utf-8")
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = (
            0x00000008 | 0x00000200
        )  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    env = dict(os.environ, HF_HUB_OFFLINE="1", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(  # noqa: S603
        [sys.executable, "-m", "training.autolabel", "--resume"],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        env=env,
        close_fds=True,
        **kwargs,
    )
    PID_PATH.write_text(str(proc.pid))
    print(f"started pid {proc.pid}; log {LOG_PATH}")


def _hands_cache() -> dict[tuple[str, int], list[Box]]:
    out: dict[tuple[str, int], list[Box]] = {}
    if HANDS_DIR.exists():
        for p in HANDS_DIR.glob("*.jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    out[(r["run_id"], r["frame_id"])] = [tuple(b) for b in r["hands"]]
    return out


def cmd_hands() -> None:
    from perception.hands import HandTracker

    index = load_index()
    HANDS_DIR.mkdir(parents=True, exist_ok=True)
    tracker = HandTracker()
    for run in sorted(index):
        path = HANDS_DIR / f"{run}.jsonl"
        ids = sorted(index[run]["frame_ids"])
        if path.exists() and len(read_raw(path)) == len(ids):
            continue
        tracker.reset()
        fps = index[run]["fps"]
        lines = []
        for fid in ids:
            img = cv2.imread(str(frame_path(index, run, fid)))
            hands = tracker.process(img, fid / fps)
            boxes = []
            for h in hands:
                xs = [p[0] for p in h.landmarks_px]
                ys = [p[1] for p in h.landmarks_px]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])
            lines.append(json.dumps({"run_id": run, "frame_id": fid, "hands": boxes}))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{run}: {len(ids)} frames", flush=True)
    tracker.close()


def cmd_build() -> None:
    if not RULES_PATH.exists():
        raise SystemExit(f"{RULES_PATH} is missing: freeze the rules first")
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))
    index = load_index()
    classes = load_classes()
    all_labels: dict[str, dict[int, dict]] = {}
    dims: dict[str, tuple[int, int]] = {}
    for run in sorted(index):
        records = read_raw(RAW_DIR / f"{run}.jsonl")
        fids = sorted(index[run]["frame_ids"])
        cache: dict[int, np.ndarray | None] = {}

        def img_of(fid: int, run=run, cache=cache):
            if fid not in cache:
                cache.clear()
                cache[fid] = cv2.imread(str(frame_path(index, run, fid)))
            return cache[fid]

        def sv(fid, box, img_of=img_of):
            img = img_of(fid)
            return None if img is None else mean_sat_val_in_box(img, box)

        def hs(fid, box, img_of=img_of):
            img = img_of(fid)
            return None if img is None else patch_hue_sat(img, box)

        all_labels[run] = build_run_labels(run, fids, records, rules, sv, hs)
        if records:
            dims[run] = (records[0]["width"], records[0]["height"])
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict = {"rules_version": rules["rules_version"], "splits": {}}
    missing_rows = []
    for split in SPLIT_ORDER:
        runs = [r for r in all_labels if index[r]["split"] == split]
        coco = build_coco(split, {r: all_labels[r] for r in runs if r in dims}, dims, classes)
        (COCO_DIR / f"{split}.json").write_text(json.dumps(coco), encoding="utf-8")
        counts = Counter()
        reasons = Counter()
        for r in runs:
            for fid, lab in all_labels[r].items():
                counts[lab["status"]] += 1
                if lab["status"] == "excluded":
                    reasons[lab["reason"]] += 1
                    missing_rows.append([split, r, fid, ",".join(lab["missing"]), lab["reason"]])
        summary["splits"][split] = {
            "frames": sum(counts.values()),
            "ok": counts["ok"],
            "excluded": counts["excluded"],
            "pending": counts["pending"],
            "excluded_by_reason": dict(sorted(reasons.items())),
            "images_in_coco": len(coco["images"]),
            "annotations_in_coco": len(coco["annotations"]),
        }
    with (LABELS_DIR / "missing_frames.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "run_id", "frame_id", "missing_classes", "reason"])
        w.writerows(missing_rows)
    hands = _hands_cache()
    if hands:
        summary["hard_cases"] = hard_case_stats(all_labels, hands)
    serial = {
        r: {
            str(f): {
                **lab,
                "boxes": {c: list(b) for c, b in lab["boxes"].items()},
                "candidates": {c: [list(b) for b in bs] for c, bs in lab["candidates"].items()},
            }
            for f, lab in labs.items()
        }
        for r, labs in all_labels.items()
    }
    (LABELS_DIR / "frame_labels.json").write_text(json.dumps(serial), encoding="utf-8")
    (LABELS_DIR / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--resume", action="store_true", help="run the labeling queue here")
    g.add_argument("--detach", action="store_true", help="run the queue as a detached process")
    g.add_argument("--status", action="store_true")
    g.add_argument("--build", action="store_true", help="raw cache -> COCO per split (offline)")
    g.add_argument("--hands", action="store_true", help="hand boxes for hard-case counts")
    args = ap.parse_args(argv)
    if args.resume:
        cmd_resume()
    elif args.detach:
        cmd_detach()
    elif args.status:
        cmd_status()
    elif args.build:
        cmd_build()
    else:
        cmd_hands()


if __name__ == "__main__":
    main()
