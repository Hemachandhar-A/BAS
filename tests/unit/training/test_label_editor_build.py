"""Unit tests for training/label_editor/build.py (the page generator, pure and file
based): frame order and kinds, gold pick, proposals with scores, area and rejection
reasons, class order and colours (BGR in Python, RGB in CSS), safe inlining."""

from __future__ import annotations

import json

import pytest

from training.autolabel_rules import StaticResult
from training.label_editor.build import (
    PageError,
    build_frames,
    css_color,
    explain,
    pick_gold,
    proposals_for_class,
    render_page,
)

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]
W, H = 800, 400
RULES = {
    "rules_version": "v3",
    "static": {
        "sat_max": 61.0,
        "val_min": 128.0,
        "static_bands": {
            "outer_box": [0.18, 0.33],
            "tray": [0.019, 0.188],
            "start_button": [0.001, 0.1076],
        },
    },
    "container": {
        "container_band": [0.03, 0.07],
        "area_ceiling": 0.07,
        "small_floor": 0.03,
        "width_max": 1.0e9,
        "height_max": 1.0e9,
        "colour": {},
    },
}
STATICS = {
    "outer_box": (10.0, 10.0, 400.0, 390.0),
    "tray": (500.0, 20.0, 700.0, 200.0),
    "start_button": (520.0, 300.0, 640.0, 380.0),
}


def _cand(box, score):
    return {"box": list(box), "score": score}


def _no_colour(_box):
    return None


def _statics():
    return StaticResult(
        boxes=dict(STATICS), start_button_white=True, start_button_measure=(10, 200)
    )


# --- explain / colour ------------------------------------------------------------------


def test_css_color_converts_bgr_to_rgb_hex():
    assert css_color((0, 0, 255)) == "#ff0000"  # red_box: BGR blue=0 green=0 red=255
    assert css_color((255, 128, 0)) == "#0080ff"


def test_explain_gives_numbers_for_each_rejection_code():
    box = (0.0, 0.0, 40.0, 32.0)  # 1280 px of 320000 = 0.0040
    p = RULES["container"]
    assert "0.0040" in explain("out_of_band", box, W, H, p, "red_box")
    assert "0.0300" in explain("out_of_band", box, W, H, p, "red_box")
    assert "overlaps" in explain("overlaps_tray", box, W, H, p, "red_box")
    assert "tray" in explain("overlaps_tray", box, W, H, p, "red_box")
    assert "higher score" in explain("not_highest_score", box, W, H, p, "red_box")
    wide = {**p, "width_max": 30.0}
    assert "40" in explain("too_wide", box, W, H, wide, "red_box")
    assert "30" in explain("too_wide", box, W, H, wide, "red_box")
    assert explain("something_new", box, W, H, p, "red_box") == "something_new"


# --- proposals ---------------------------------------------------------------------------


def test_proposals_list_every_cached_candidate_with_score_area_and_reason():
    cands = [
        _cand((100, 100, 130, 125), 0.55),  # tiny: 750 px = 0.0023 -> out_of_band
        _cand((20, 20, 380, 380), 0.40),  # whole outer box -> out_of_band (too large)
        _cand((510, 30, 690, 190), 0.30),  # 28800 px = 0.09 > 0.07 -> out_of_band
        _cand((560, 40, 640, 100), 0.25),  # 4800 px = 0.015, below the 0.03 floor
    ]
    props = proposals_for_class("red_box", cands, W, H, RULES, _statics(), _no_colour)
    assert [p["n"] for p in props] == [1, 2, 3, 4]
    assert [p["score"] for p in props] == [0.55, 0.40, 0.30, 0.25]
    assert props[0]["box"] == [100.0, 100.0, 130.0, 125.0]
    assert props[0]["area_frac"] == pytest.approx(750 / 320000, abs=1e-6)
    assert all(p["reason_code"] == "out_of_band" for p in props)
    assert all(p["reason"] for p in props)


def test_proposals_include_rules_rejected_candidates_such_as_overlaps():
    # a candidate of container size lying on the tray: rejected by the exclusion rule
    cands = [_cand((500, 20, 630, 170), 0.5)]  # 19500 px = 0.061, IoU 0.54 with the tray
    props = proposals_for_class("red_box", cands, W, H, RULES, _statics(), _no_colour)
    assert props[0]["reason_code"] == "overlaps_tray"
    assert "tray" in props[0]["reason"]


def test_proposals_keep_cache_order_and_cap_nothing():
    cands = [_cand((100 + i, 100, 160 + i, 150), 0.2 + i / 100) for i in range(10)]
    props = proposals_for_class("yellow_box", cands, W, H, RULES, _statics(), _no_colour)
    assert len(props) == 10
    assert [p["n"] for p in props] == list(range(1, 11))


def test_static_class_proposals_say_band_or_consensus():
    cands = [_cand((0, 0, 20, 20), 0.9), _cand((20, 20, 370, 300), 0.8)]
    props = proposals_for_class("outer_box", cands, W, H, RULES, _statics(), _no_colour)
    assert props[0]["reason_code"] == "out_of_band"
    assert props[1]["reason_code"] == "not_run_consensus"


# --- gold pick ------------------------------------------------------------------------------


def _split_index(frames=24):
    return {
        "v1": {"split": "val", "frame_ids": [i * 15 for i in range(frames)]},
        "v2": {"split": "val", "frame_ids": [i * 15 for i in range(6)]},
        "t1": {"split": "test", "frame_ids": [i * 15 for i in range(frames)]},
    }


def test_pick_gold_is_seeded_stratified_in_time_and_stays_in_the_split():
    idx = _split_index()
    a = pick_gold(idx, "val", 6, seed=1)
    assert a == pick_gold(idx, "val", 6, seed=1)
    assert a != pick_gold(idx, "val", 6, seed=2)
    assert {r for r, _ in a} == {"v1", "v2"}  # no test run
    v1 = sorted(f for r, f in a if r == "v1")
    assert len(v1) == 6
    ids = idx["v1"]["frame_ids"]
    # one frame from each sixth of the run, so a pick never bunches in one stretch
    chunks = [set(ids[i * 4 : (i + 1) * 4]) for i in range(6)]
    assert all(len(c & set(v1)) == 1 for c in chunks)
    assert len([f for r, f in a if r == "v2"]) == 6  # run with exactly 6 frames: all of them


def test_pick_gold_takes_all_frames_of_a_short_run():
    idx = {"s": {"split": "test", "frame_ids": [0, 15, 30]}}
    assert pick_gold(idx, "test", 6, seed=1) == [("s", 0), ("s", 15), ("s", 30)]


# --- frame list --------------------------------------------------------------------------------


def _labels():
    ok = {c: list(b) for c, b in STATICS.items()}
    ok["red_box"] = [100.0, 100.0, 180.0, 160.0]
    ok["yellow_box"] = [100.0, 200.0, 180.0, 260.0]
    no_red = {c: b for c, b in ok.items() if c != "red_box"}
    return {
        "a": {
            "0": {"status": "ok", "boxes": ok, "missing": [], "reason": ""},
            "15": {"status": "excluded", "boxes": no_red, "missing": ["red_box"], "reason": "x"},
            "30": {"status": "pending", "boxes": {}, "missing": [], "reason": ""},
        },
        "b": {
            "0": {"status": "ok", "boxes": ok, "missing": [], "reason": ""},
            "15": {"status": "excluded", "boxes": no_red, "missing": ["red_box"], "reason": "x"},
            "30": {"status": "ok", "boxes": ok, "missing": [], "reason": ""},
        },
    }


def _raw():
    def rec(run, fid, cls, cands):
        return {
            "run_id": run,
            "frame_id": fid,
            "class": cls,
            "width": W,
            "height": H,
            "candidates": cands,
        }

    return {
        "a": [rec("a", 15, "red_box", [_cand((100, 100, 130, 125), 0.5)])],
        "b": [
            rec(
                "b", 15, "red_box", [_cand((20, 20, 380, 380), 0.4), _cand((90, 90, 170, 150), 0.3)]
            )
        ],
    }


def _index(split="train"):
    return {
        "a": {"split": split, "frame_ids": [0, 15, 30]},
        "b": {"split": split, "frame_ids": [0, 15, 30]},
        "z": {"split": "val", "frame_ids": [0, 15]},
    }


def _frames(**kw):
    args = {
        "split": "train",
        "classes": CLASSES,
        "index": _index(),
        "labels": _labels(),
        "raw_by_run": _raw(),
        "rules": RULES,
        "statics_by_run": {"a": _statics(), "b": _statics()},
        "measure_factory": lambda run, fid: _no_colour,
        "img_dir": "../frames/train",
    }
    args.update(kw)
    return build_frames(**args)


def test_excluded_frames_come_first_grouped_by_run_and_pending_is_skipped():
    frames, info = _frames()
    assert [(f["kind"], f["run"], f["frame"]) for f in frames] == [
        ("excluded", "a", 15),
        ("excluded", "b", 15),
    ]
    assert info["pending_skipped"] == 1
    f = frames[0]
    assert f["file"] == "a_15.jpg" and f["img"] == "../frames/train/a_15.jpg"
    assert f["width"] == W and f["height"] == H
    assert f["missing"] == ["red_box"]
    assert set(f["auto_boxes"]) == set(CLASSES) - {"red_box"}
    assert [p["n"] for p in f["proposals"]["red_box"]] == [1]
    assert [p["n"] for p in frames[1]["proposals"]["red_box"]] == [1, 2]


def test_other_splits_never_appear():
    frames, _ = _frames()
    assert all(f["run"] in ("a", "b") for f in frames)


def test_gold_frames_follow_excluded_and_dedupe():
    idx = {
        "a": {"split": "val", "frame_ids": [0, 15, 30]},
        "b": {"split": "val", "frame_ids": [0, 15, 30]},
    }
    frames, info = _frames(split="val", index=idx, gold=3)
    kinds = [(f["kind"], f["run"], f["frame"]) for f in frames]
    # excluded first; gold frames (the non-pending frames of each run) after, without repeats
    assert kinds[:2] == [("excluded", "a", 15), ("excluded", "b", 15)]
    assert ("gold", "a", 0) in kinds and ("gold", "b", 30) in kinds
    assert ("gold", "a", 15) not in kinds
    assert len(kinds) == len(set(kinds))
    assert frames[0]["gold"] is True  # an excluded frame that is also a gold pick is tagged
    assert info["gold_frames"] >= 5


def test_static_check_frames_come_last_one_per_run_with_static_boxes_only():
    frames, _ = _frames(static=True)
    static = [f for f in frames if f["kind"] == "static"]
    assert [(f["run"], f["frame"]) for f in static] == [("a", 0), ("b", 0)]
    assert all(set(f["auto_boxes"]) == {"outer_box", "tray", "start_button"} for f in static)
    assert frames[-1]["kind"] == "static"
    assert [f["kind"] for f in frames[:2]] == ["excluded", "excluded"]


def test_only_restricts_to_the_given_cells():
    frames, _ = _frames(only={("b", 15)})
    assert [(f["run"], f["frame"]) for f in frames] == [("b", 15)]


# --- the page ---


def _data_of(html):
    data_json = html.split("window.EDITOR_DATA = ", 1)[1].split(";</script>", 1)[0]
    return json.loads(data_json.replace("<\\/", "</"))


def test_render_page_inlines_data_in_experiment_order_and_is_offline():
    frames, info = _frames()
    html = render_page("train", CLASSES, frames, info)
    assert "<script" in html
    assert "http://" not in html and "https://" not in html and "fetch(" not in html
    data = _data_of(html)
    assert data["classes"] == CLASSES
    assert data["colors"]["red_box"] == "#ff0000"
    assert data["split"] == "train"
    assert len(data["frames"]) == 2


def test_render_page_escapes_script_end_tags_in_data():
    frames, info = _frames()
    frames[0]["proposals"]["red_box"][0]["reason"] = "</script><b>x"
    html = render_page("train", CLASSES, frames, info)
    assert html.count("</script>") == html.count("<script")  # data cannot close a tag early
    assert _data_of(html)["frames"][0]["proposals"]["red_box"][0]["reason"] == "</script><b>x"


def test_render_page_refuses_a_class_without_a_colour():
    frames, info = _frames()
    with pytest.raises(PageError, match="colour"):
        render_page("train", [*CLASSES, "mystery"], frames, info)


def test_page_js_parses_with_node(tmp_path):
    import shutil
    import subprocess

    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    frames, info = _frames()
    html = render_page("train", CLASSES, frames, info)
    scripts = html.split("<script>")[1:]
    assert len(scripts) == 3
    for i, chunk in enumerate(scripts):
        f = tmp_path / f"s{i}.js"
        f.write_text(chunk.split("</script>", 1)[0], encoding="utf-8")
        r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_static_check_uses_a_frame_whose_movable_calls_are_pending_if_its_static_boxes_exist():
    labels = _labels()
    labels["a"]["0"] = {
        "status": "pending",
        "boxes": {c: list(b) for c, b in STATICS.items()},
        "missing": [],
        "reason": "",
    }
    labels["b"]["0"] = {"status": "pending", "boxes": {}, "missing": [], "reason": ""}
    frames, _ = _frames(static=True, labels=labels)
    assert [(f["run"], f["frame"]) for f in frames if f["kind"] == "static"] == [("a", 0)]
