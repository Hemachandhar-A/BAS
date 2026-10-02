"""End to end, file based, no browser: a simulated editor session (run by node on the real
editor_core.js) writes an overlay; build_dataset validates and applies it. Skipped when node
is not installed."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from training.autolabel_rules import StaticResult
from training.build_dataset import build_dataset
from training.label_editor.build import build_frames, render_page

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]
W, H = 160, 96
CORE = Path(__file__).parents[3] / "training" / "label_editor" / "editor_core.js"
RULES = {
    "rules_version": "v3",
    "static": {
        "sat_max": 61.0,
        "val_min": 128.0,
        "static_bands": {
            "outer_box": [0.1, 0.9],
            "tray": [0.01, 0.5],
            "start_button": [0.001, 0.5],
        },
    },
    "container": {
        "container_band": [0.03, 0.12],
        "area_ceiling": 0.12,
        "small_floor": 0.03,
        "width_max": 1.0e9,
        "height_max": 1.0e9,
        "colour": {},
    },
}


def _boxes():
    return {
        "outer_box": [5.0, 5.0, 90.0, 90.0],
        "tray": [100.0, 10.0, 150.0, 50.0],
        "red_box": [20.0, 20.0, 50.0, 45.0],
        "yellow_box": [20.0, 55.0, 50.0, 80.0],
        "start_button": [110.0, 70.0, 140.0, 90.0],
    }


RUNS = {"a1": ("train", [0, 30, 60, 90]), "v1": ("val", [0, 30])}
ACCEPTED = [22.0, 22.0, 52.0, 47.0]  # the red candidate the Lead will accept on a1#60


def _world(tmp_path):
    frames_dir = tmp_path / "frames"
    manifest, index, labels, raw = [], {}, {}, {}
    for run, (split, fids) in RUNS.items():
        manifest.append({"run_id": run, "split": split, "width": str(W), "height": str(H)})
        index[run] = {"split": split, "fps": 30.0, "frame_ids": fids}
        (frames_dir / split).mkdir(parents=True, exist_ok=True)
        labels[run], raw[run] = {}, []
        for fid in fids:
            cv2.imwrite(
                str(frames_dir / split / f"{run}_{fid}.jpg"),
                np.full((H, W, 3), 40 + fid, dtype=np.uint8),
            )
            boxes = _boxes()
            lab = {"status": "ok", "boxes": boxes, "missing": [], "reason": ""}
            if (run, fid) in {("a1", 60), ("a1", 90)}:  # red missing: excluded by the labeler
                del boxes["red_box"]
                lab = {
                    "status": "excluded",
                    "boxes": boxes,
                    "missing": ["red_box"],
                    "reason": "missing:red_box",
                }
            labels[run][str(fid)] = lab
        raw["a1"] = [
            {
                "run_id": "a1",
                "frame_id": f,
                "class": "red_box",
                "width": W,
                "height": H,
                "candidates": [
                    {"box": [60.0, 30.0, 70.0, 40.0], "score": 0.9},
                    {"box": ACCEPTED, "score": 0.4},
                ],
            }
            for f in (60, 90)
        ]
    return manifest, index, labels, raw, frames_dir


def _run_session(html: str, tmp_path: Path, script: str) -> dict:
    """Run ``script`` (JS) with D = the page's EDITOR_DATA and LE = editor_core.js; it must
    ``console.log(JSON.stringify(overlay))``."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    data = html.split("window.EDITOR_DATA = ", 1)[1].split(";</script>", 1)[0]
    driver = tmp_path / "session.js"
    core = json.dumps(str(CORE))
    header = (
        f"const LE = require({core});\n"
        f"const D = {data};\n"  # the inlined JSON is valid JS ('<\\/' is a legal escape)
        "const byFile = (f, k) => D.frames.find((x) => x.file === f && (!k || x.kind === k));\n"
    )
    driver.write_text(header + script, encoding="utf-8")
    r = subprocess.run([node, str(driver)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def _page(tmp_path, **kw):
    manifest, index, labels, raw, frames_dir = _world(tmp_path)
    statics = StaticResult(
        boxes={c: tuple(_boxes()[c]) for c in ("outer_box", "tray", "start_button")},
        start_button_white=True,
        start_button_measure=(10, 200),
    )
    frames, info = build_frames(
        split="train",
        classes=CLASSES,
        index={"a1": index["a1"]},
        labels=labels,
        raw_by_run=raw,
        rules=RULES,
        statics_by_run={"a1": statics},
        measure_factory=lambda run, fid: lambda box: None,
        img_dir="../frames/train",
        **kw,
    )
    return (
        render_page("train", CLASSES, frames, info),
        frames,
        (manifest, index, labels, frames_dir),
    )


def _build(tmp_path, world, overlay):
    manifest, index, labels, frames_dir = world
    corr = tmp_path / "corrections"
    corr.mkdir(exist_ok=True)
    (corr / "train.json").write_text(json.dumps(overlay), encoding="utf-8")
    report = build_dataset(
        classes=CLASSES,
        manifest_rows=manifest,
        index=index,
        frame_labels=labels,
        frames_dir=frames_dir,
        corrections_dir=corr,
        out_dir=tmp_path / "dataset",
        report_path=tmp_path / "reports" / "dataset.json",
        generated_at="2026-10-03T00:00:00",
    )
    coco = json.loads((tmp_path / "dataset" / "train" / "_annotations.coco.json").read_text())
    return report, coco


def _by_image(coco):
    name_of = {c["id"]: c["name"] for c in coco["categories"]}
    files = {i["id"]: i["file_name"] for i in coco["images"]}
    out: dict[str, dict] = {}
    for a in coco["annotations"]:
        out.setdefault(files[a["image_id"]], {})[name_of[a["category_id"]]] = a["bbox"]
    return out


SESSION = """
const C = D.classes, st = {};
// a1#60: accept the red candidate number 2, then verify
const f60 = byFile("a1_60.jpg", "excluded");
let s = LE.freshState(f60);
s = LE.acceptProposal(s, "red_box", f60.proposals.red_box[1], f60.width, f60.height);
st[f60.key] = LE.setVerified(C, s);
// a1#90: excluded by the Lead
const f90 = byFile("a1_90.jpg", "excluded");
st[f90.key] = LE.setExcluded(LE.freshState(f90), true);
console.log(JSON.stringify(LE.assembleOverlay(D, st).overlay));
"""


def test_overlay_from_a_simulated_session_validates_and_brings_the_excluded_frame_back(tmp_path):
    html, frames, world = _page(tmp_path)
    assert [f["file"] for f in frames] == ["a1_60.jpg", "a1_90.jpg"]
    # the proposal list is what the session picks from: both cached candidates, with reasons
    props = frames[0]["proposals"]["red_box"]
    assert [p["score"] for p in props] == [0.9, 0.4]
    overlay = _run_session(html, tmp_path, SESSION)
    assert overlay["split"] == "train" and overlay["version"] == 1
    assert overlay["frames"]["a1_90.jpg"]["excluded"] is True
    entry = overlay["frames"]["a1_60.jpg"]
    assert [b["class"] for b in entry["boxes"]] == CLASSES  # all five, experiment order
    assert entry["verified"] is True

    report, coco = _build(tmp_path, world, overlay)  # build_dataset's own assertions pass
    boxes = _by_image(coco)
    assert "a1_60.jpg" in boxes  # the previously excluded frame is back ...
    x1, y1, x2, y2 = ACCEPTED
    assert boxes["a1_60.jpg"]["red_box"] == [x1, y1, x2 - x1, y2 - y1]  # ... with the accepted box
    assert set(boxes["a1_60.jpg"]) == set(CLASSES)
    assert "a1_90.jpg" not in boxes  # still dropped
    tr = report["splits"]["train"]
    assert tr["frames_corrected"] == 1 and tr["frames_verified"] == 1
    assert tr["frames_excluded_overlay"] == 1


def test_a_frame_entry_replaces_the_frame_wholesale_and_static_override_covers_the_run(tmp_path):
    script = """
    const st = {};
    const sf = byFile("a1_0.jpg", "static");
    st[sf.key] = LE.setBox(LE.freshState(sf), "tray", [101, 11, 151, 51], 160, 96);
    console.log(JSON.stringify(LE.assembleOverlay(D, st).overlay));
    """
    # the page has the static check frames too, so the session has a static view to edit
    html, frames, world = _page(tmp_path, static=True)
    overlay = _run_session(html, tmp_path, script)
    assert overlay["static_overrides"] == {"a1": {"tray": [101.0, 11.0, 151.0, 51.0]}}
    _, coco = _build(tmp_path, world, overlay)
    boxes = _by_image(coco)
    # every ok frame of the run got the new tray; the frame boxes otherwise unchanged
    for name in ("a1_0.jpg", "a1_30.jpg"):
        assert boxes[name]["tray"] == [101.0, 11.0, 50.0, 40.0]
        assert boxes[name]["red_box"] == [20.0, 20.0, 30.0, 25.0]
    assert "a1_60.jpg" not in boxes  # still excluded by the labeler (no frame entry)


def test_a_frame_entry_overrides_the_static_override_only_where_the_user_set_that_box(tmp_path):
    html, frames, world = _page(tmp_path, static=True)
    script = """
    const st = {};
    const sf = byFile("a1_0.jpg", "static");
    st[sf.key] = LE.setBox(LE.freshState(sf), "tray", [101, 11, 151, 51], 160, 96);
    const f = byFile("a1_60.jpg", "excluded");
    let s = LE.acceptProposal(LE.freshState(f), "red_box", f.proposals.red_box[1], 160, 96);
    st[f.key] = LE.setVerified(D.classes, s);
    console.log(JSON.stringify(LE.assembleOverlay(D, st).overlay));
    """
    overlay = _run_session(html, tmp_path, script)
    _, coco = _build(tmp_path, world, overlay)
    boxes = _by_image(coco)
    # the entry replaces a1#60 wholesale, and carries the run's fixed tray (not the old one)
    assert boxes["a1_60.jpg"]["tray"] == [101.0, 11.0, 50.0, 40.0]
    assert boxes["a1_60.jpg"]["red_box"][0] == ACCEPTED[0]
