"""Unit tests for the pure parts of training/autolabel.py: the resumable raw
cache, job planning, per-run label building, exclusion reasons, COCO output,
the hand-overlap hard-case count and the status summary. No model here."""

from __future__ import annotations

import json

import pytest

from training.autolabel import (
    Job,
    build_coco,
    build_run_labels,
    done_keys,
    hand_overlaps_class,
    hard_case_stats,
    plan_jobs,
    raw_record,
    read_raw,
    status_summary,
)

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]
W, H = 848, 480
OUTER = (60.0, 40.0, 340.0, 440.0)  # 0.34 of the frame
TRAY = (400.0, 60.0, 620.0, 300.0)  # 0.13
CARD = (480.0, 340.0, 600.0, 425.0)  # 0.025
RED = (100.0, 120.0, 264.0, 240.0)  # ~0.048
YEL = (110.0, 280.0, 274.0, 400.0)


def _rules():
    return {
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
            "container_band": [0.0324, 0.0694],
            "area_ceiling": 0.0694,
            "small_floor": 0.0324,
            "width_max": 1.0e9,
            "height_max": 1.0e9,
            "colour": {},
        },
    }


def _raw(run, fid, cls, boxes, phrase="p."):
    return raw_record(
        run,
        fid,
        cls,
        phrase,
        W,
        H,
        [(b, 0.6) for b in boxes],
        seconds=4.0,
        thresholds=(0.2, 0.2, 0.2),
    )


# --- raw cache ----------------------------------------------------------------


def test_raw_record_shape_matches_the_spike_cache():
    rec = _raw("x001", 30, "red_box", [RED])
    assert rec["run_id"] == "x001" and rec["frame_id"] == 30 and rec["class"] == "red_box"
    assert rec["candidates"][0] == {"box": list(RED), "score": 0.6}
    assert {"phrase", "width", "height", "box_threshold", "text_threshold", "score_floor"} <= set(
        rec
    )


def test_read_raw_skips_a_truncated_last_line(tmp_path):
    p = tmp_path / "x001.jsonl"
    good = json.dumps(_raw("x001", 0, "red_box", [RED]))
    p.write_text(good + "\n" + '{"run_id": "x001", "fra', encoding="utf-8")
    recs = read_raw(p)
    assert len(recs) == 1
    assert done_keys(recs) == {("x001", 0, "red_box")}


def test_read_raw_missing_file_is_empty(tmp_path):
    assert read_raw(tmp_path / "nope.jsonl") == []


def test_read_raw_corruption_before_the_last_line_is_an_error(tmp_path):
    p = tmp_path / "x001.jsonl"
    good = json.dumps(_raw("x001", 0, "red_box", [RED]))
    p.write_text("not json\n" + good + "\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_raw(p)


# --- plan_jobs ------------------------------------------------------------------


def _index():
    return {
        "x001": {"split": "train", "frame_ids": list(range(0, 300, 30))},  # 10 frames
        "x002": {"split": "val", "frame_ids": [0, 30, 60]},
    }


def test_plan_has_static_jobs_on_spread_frames_and_movable_jobs_on_every_frame():
    jobs = plan_jobs(_index(), priority_cells=[])
    static = [j for j in jobs if j.cls in ("outer_box", "tray", "start_button")]
    movable = [j for j in jobs if j.cls in ("red_box", "yellow_box")]
    assert len(movable) == 2 * 13
    # x001: 5 static frames x 3 classes; x002 has 3 frames <= 5 so all 3
    assert len(static) == 5 * 3 + 3 * 3
    assert {j.frame_id for j in static if j.run_id == "x001"} == {0, 60, 120, 210, 270}


def test_plan_order_statics_first_then_priority_movable_then_the_rest():
    jobs = plan_jobs(_index(), priority_cells=[("x001", 90), ("x001", 120)])
    kinds = ["s" if j.cls in ("outer_box", "tray", "start_button") else "m" for j in jobs]
    first_m = kinds.index("m")
    assert set(kinds[:first_m]) == {"s"} and "s" not in kinds[first_m:]
    pri = [(j.run_id, j.frame_id) for j in jobs[first_m : first_m + 4]]
    assert pri == [("x001", 90), ("x001", 90), ("x001", 120), ("x001", 120)]


def test_plan_has_no_duplicate_jobs_and_is_deterministic():
    a = plan_jobs(_index(), priority_cells=[("x001", 90)])
    assert len(a) == len(set(a))
    assert a == plan_jobs(_index(), priority_cells=[("x001", 90)])


def test_job_is_hashable_and_ordered_by_fields():
    assert Job("x001", 0, "red_box") == Job("x001", 0, "red_box")


# --- build_run_labels --------------------------------------------------------------


def _run_records(with_movable=True, hide_red_at=None):
    recs = []
    for fid in (0, 30, 60):
        recs += [
            _raw("x001", fid, "outer_box", [OUTER]),
            _raw("x001", fid, "tray", [TRAY]),
            _raw("x001", fid, "start_button", [CARD]),
        ]
        if with_movable:
            recs.append(_raw("x001", fid, "red_box", [] if fid == hide_red_at else [RED]))
            recs.append(_raw("x001", fid, "yellow_box", [YEL]))
    return recs


def _label(recs, frame_ids=(0, 30, 60), static_ids=None):
    return build_run_labels(
        "x001",
        list(frame_ids),
        recs,
        _rules(),
        static_ids=static_ids,
        measure_sv=lambda fid, box: (20.0, 220.0),
        measure_hs=lambda fid, box: (2.0, 120.0),
    )


def test_static_boxes_are_the_same_box_on_every_frame_of_the_run():
    labels = _label(_run_records(), frame_ids=(0, 30, 60, 90), static_ids=(0, 30, 60))
    boxes = {fid: labels[fid]["boxes"] for fid in labels}
    assert all(boxes[f]["outer_box"] == OUTER for f in (0, 30, 60, 90))
    assert all(boxes[f]["start_button"] == CARD for f in (0, 30, 60, 90))


def test_frame_with_every_class_is_labeled_ok():
    labels = _label(_run_records())
    for fid in (0, 30, 60):
        assert labels[fid]["status"] == "ok" and labels[fid]["missing"] == []
        assert set(labels[fid]["boxes"]) == set(CLASSES)


def test_frame_with_a_missing_movable_class_is_excluded_with_a_reason():
    labels = _label(_run_records(hide_red_at=30))
    assert labels[30]["status"] == "excluded"
    assert labels[30]["missing"] == ["red_box"]
    assert labels[30]["reason"] == "missing:red_box"
    assert labels[0]["status"] == "ok" and labels[60]["status"] == "ok"


def test_frames_without_movable_calls_are_pending_not_excluded():
    labels = _label(_run_records(with_movable=False))
    assert {labels[f]["status"] for f in labels} == {"pending"}


def test_run_with_a_missing_static_class_excludes_all_its_frames():
    recs = [r for r in _run_records() if r["class"] != "tray"]
    recs += [_raw("x001", f, "tray", []) for f in (0, 30, 60)]  # asked, nothing found
    labels = _label(recs)
    assert all(labels[f]["status"] == "excluded" for f in labels)
    assert labels[0]["reason"].startswith("static_missing:tray")


def test_a_frame_missing_both_movable_classes_names_both():
    recs = [
        r
        for r in _run_records()
        if not (r["frame_id"] == 0 and r["class"] in ("red_box", "yellow_box"))
    ]
    recs += [_raw("x001", 0, "red_box", []), _raw("x001", 0, "yellow_box", [])]
    labels = _label(recs)
    assert labels[0]["missing"] == ["red_box", "yellow_box"]
    assert labels[0]["reason"] == "missing:red_box,yellow_box"


def test_rejected_candidates_are_kept_for_the_hard_case_count():
    recs = _run_records()
    recs = [r for r in recs if not (r["frame_id"] == 30 and r["class"] == "red_box")]
    recs.append(_raw("x001", 30, "red_box", [(100.0, 100.0, 700.0, 460.0)]))  # far too large
    labels = _label(recs)
    assert labels[30]["status"] == "excluded"
    assert labels[30]["candidates"]["red_box"] == [(100.0, 100.0, 700.0, 460.0)]


# --- hand_overlaps_class -------------------------------------------------------------


def test_hand_overlap_uses_candidates_or_the_nearest_chosen_box():
    hand = (90.0, 110.0, 150.0, 170.0)
    assert hand_overlaps_class([hand], candidate_boxes=[RED], nearest_box=None)
    assert hand_overlaps_class([hand], candidate_boxes=[], nearest_box=RED)
    assert not hand_overlaps_class([hand], candidate_boxes=[YEL], nearest_box=YEL)
    assert not hand_overlaps_class([], candidate_boxes=[RED], nearest_box=RED)


# --- build_coco ----------------------------------------------------------------------


def test_coco_ids_follow_the_class_order_and_boxes_are_xywh():
    labels = {
        0: {
            "status": "ok",
            "boxes": {
                "outer_box": OUTER,
                "tray": TRAY,
                "red_box": RED,
                "yellow_box": YEL,
                "start_button": CARD,
            },
        },
        30: {"status": "excluded", "boxes": {}, "missing": ["red_box"]},
        60: {"status": "pending", "boxes": {}},
    }
    coco = build_coco("train", {"x001": labels}, {"x001": (W, H)}, CLASSES)
    assert [c["name"] for c in coco["categories"]] == CLASSES
    assert [c["id"] for c in coco["categories"]] == [0, 1, 2, 3, 4]
    assert [i["file_name"] for i in coco["images"]] == ["x001_0.jpg"]
    assert len(coco["annotations"]) == 5
    red = next(a for a in coco["annotations"] if a["category_id"] == 2)
    assert red["bbox"] == [100.0, 120.0, 164.0, 120.0]
    assert red["area"] == pytest.approx(164.0 * 120.0)
    ids = [a["id"] for a in coco["annotations"]]
    assert ids == sorted(set(ids))


def test_coco_clips_boxes_to_the_image():
    labels = {0: {"status": "ok", "boxes": {c: (-5.0, -5.0, 900.0, 500.0) for c in CLASSES}}}
    coco = build_coco("train", {"x001": labels}, {"x001": (W, H)}, CLASSES)
    assert coco["annotations"][0]["bbox"] == [0.0, 0.0, 848.0, 480.0]


def test_coco_round_trip_looks_up_a_known_box_by_name():
    labels = {
        0: {
            "status": "ok",
            "boxes": {
                "outer_box": OUTER,
                "tray": TRAY,
                "red_box": RED,
                "yellow_box": YEL,
                "start_button": CARD,
            },
        }
    }
    coco = json.loads(json.dumps(build_coco("train", {"x001": labels}, {"x001": (W, H)}, CLASSES)))
    name_of = {c["id"]: c["name"] for c in coco["categories"]}
    by_name = {name_of[a["category_id"]]: a["bbox"] for a in coco["annotations"]}
    assert by_name["start_button"] == [480.0, 340.0, 120.0, 85.0]


# --- status ----------------------------------------------------------------------------


def test_status_summary_counts_and_eta():
    s = status_summary(done=300, total=1000, recent_seconds=[4.0, 4.2, 3.8])
    assert s["done"] == 300 and s["total"] == 1000 and s["remaining"] == 700
    assert s["mean_seconds_per_call"] == pytest.approx(4.0)
    assert s["eta_seconds"] == pytest.approx(2800.0)


def test_status_summary_with_no_calls_yet_has_no_eta():
    s = status_summary(done=0, total=10, recent_seconds=[])
    assert s["mean_seconds_per_call"] is None and s["eta_seconds"] is None


# --- hard_case_stats ---------------------------------------------------------------------


def test_hard_case_stats_counts_excluded_and_final_frames():
    recs = _run_records(hide_red_at=30)
    recs = [r for r in recs if not (r["frame_id"] == 30 and r["class"] == "red_box")]
    recs.append(_raw("x001", 30, "red_box", [(100.0, 100.0, 700.0, 460.0)]))  # rejected: too large
    labels = {"x001": _label(recs)}
    hand_on_red = (90.0, 110.0, 150.0, 170.0)
    far_hand = (700.0, 400.0, 760.0, 470.0)
    hands = {("x001", 0): [far_hand], ("x001", 30): [hand_on_red], ("x001", 60): [hand_on_red]}
    st = hard_case_stats(labels, hands)
    assert st["excluded_movable_missing"] == 1
    assert st["excluded_with_hand"] == 1
    assert st["excluded_with_hand_overlapping_missing_class"] == 1
    assert st["final_frames"] == 2
    assert st["final_with_hand_over_container"] == 1  # frame 60 only; frame 0's hand is far away


def test_hard_case_stats_never_guesses_frames_without_hand_data():
    labels = {"x001": _label(_run_records())}
    st = hard_case_stats(labels, {})
    assert st["frames_without_hand_data"] == 3
    assert "final_frames" not in st


# --- prompts.yaml ------------------------------------------------------------------------------


def _yaml(tmp_path, body):
    p = tmp_path / "prompts.yaml"
    p.write_text(body, encoding="utf-8")
    return p


def test_repo_prompts_match_the_experiment_classes_and_have_one_phrase_each():
    from training.autolabel import load_classes, load_prompts

    classes = load_classes()
    prompts = load_prompts(classes=classes)
    assert list(prompts) == classes
    assert prompts["start_button"] == "a white index card."


def test_prompts_reject_two_phrasings(tmp_path):
    from training.autolabel import load_prompts

    body = "classes:\n  red_box: ['a red box.', 'a red container.']\n"
    with pytest.raises(ValueError, match="exactly one"):
        load_prompts(_yaml(tmp_path, body), classes=["red_box"])


def test_prompts_need_a_trailing_period_and_lowercase(tmp_path):
    from training.autolabel import load_prompts

    with pytest.raises(ValueError, match="period"):
        load_prompts(_yaml(tmp_path, "classes:\n  red_box: ['a red box']\n"), classes=["red_box"])
    with pytest.raises(ValueError, match="lowercase"):
        load_prompts(_yaml(tmp_path, "classes:\n  red_box: ['A red box.']\n"), classes=["red_box"])


def test_prompts_classes_must_equal_the_experiment_classes(tmp_path):
    from training.autolabel import load_prompts

    with pytest.raises(ValueError, match="classes"):
        load_prompts(
            _yaml(tmp_path, "classes:\n  red_box: ['a red box.']\n"), classes=["red_box", "tray"]
        )
