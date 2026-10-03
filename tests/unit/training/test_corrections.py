"""training/corrections.py: the only tool that edits the hand-made overlay. It backs the
file up first (timestamp and sha256 prefix in the name), refuses unknown frames and splits,
refuses to exclude a corrected frame without --force, and re-validates the whole file with
build_dataset's checks before the original is replaced."""

from __future__ import annotations

import json

import pytest

from training import corrections as C
from training.build_dataset import BuildError

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]
BOX = [10.0, 10.0, 50.0, 50.0]
EXCLUDED = {"excluded": True, "verified": False, "boxes": []}


def _tables():
    manifest = [
        {"run_id": "a1", "split": "train", "width": "160", "height": "96"},
        {"run_id": "v1", "split": "val", "width": "160", "height": "96"},
    ]
    index = {
        "a1": {"split": "train", "fps": 30.0, "frame_ids": [0, 30, 60]},
        "v1": {"split": "val", "fps": 30.0, "frame_ids": [0]},
    }
    return manifest, index


def _overlay(frames=None):
    return {"version": 1, "split": "train", "static_overrides": {}, "frames": frames or {}}


def _corrected():
    return {
        "excluded": False,
        "verified": True,
        "boxes": [{"class": c, "xyxy": list(BOX)} for c in CLASSES],
    }


def test_known_frames_are_the_split_files():
    _, index = _tables()
    assert C.known_frames(index, "train") == {"a1_0.jpg", "a1_30.jpg", "a1_60.jpg"}
    assert C.known_frames(index, "val") == {"v1_0.jpg"}


def test_exclude_adds_the_exact_entry_and_leaves_the_input_untouched():
    ov = _overlay({"a1_0.jpg": dict(EXCLUDED)})
    new, status = C.exclude_frame(ov, "a1_30.jpg", {"a1_0.jpg", "a1_30.jpg"})
    assert new["frames"]["a1_30.jpg"] == EXCLUDED
    assert status == "excluded"
    assert "a1_30.jpg" not in ov["frames"]


def test_exclude_an_already_excluded_frame_is_a_noop():
    ov = _overlay({"a1_0.jpg": dict(EXCLUDED)})
    new, status = C.exclude_frame(ov, "a1_0.jpg", {"a1_0.jpg"})
    assert new == ov and status == "already_excluded"


def test_exclude_a_corrected_frame_is_refused_unless_forced():
    ov = _overlay({"a1_0.jpg": _corrected()})
    with pytest.raises(C.CorrectionError, match="corrected"):
        C.exclude_frame(ov, "a1_0.jpg", {"a1_0.jpg"})
    new, status = C.exclude_frame(ov, "a1_0.jpg", {"a1_0.jpg"}, force=True)
    assert new["frames"]["a1_0.jpg"] == EXCLUDED
    assert status == "excluded_forced"


def test_exclude_an_unknown_frame_is_refused():
    with pytest.raises(C.CorrectionError, match="unknown frame"):
        C.exclude_frame(_overlay(), "zz_9.jpg", {"a1_0.jpg"})


def test_backup_name_has_sha_prefix_and_timestamp_and_copies_the_bytes(tmp_path):
    p = tmp_path / "train.json"
    data = b'{"x": 1}'
    p.write_bytes(data)
    b = C.backup(p, now="20261003T101500")
    assert b.name == f"train.json.bak-{C.sha256_prefix(data)}-20261003T101500"
    assert b.read_bytes() == data
    assert len(C.sha256_prefix(b"abc")) == 12


def _write_inputs(tmp_path, overlay):
    d = tmp_path / "corrections"
    d.mkdir()
    (d / "train.json").write_bytes(json.dumps(overlay, indent=1).encode("utf-8"))  # LF
    return d


def test_run_exclude_end_to_end_backs_up_then_rewrites(tmp_path):
    manifest, index = _tables()
    d = _write_inputs(tmp_path, _overlay({"a1_0.jpg": _corrected()}))
    before = (d / "train.json").read_bytes()
    out = C.run_exclude(d, "train", "a1_60.jpg", CLASSES, manifest, index, now="T1")
    assert out["status"] == "excluded"
    assert (d / out["backup"]).read_bytes() == before
    after = json.loads((d / "train.json").read_text(encoding="utf-8"))
    assert after["frames"]["a1_60.jpg"] == EXCLUDED
    assert after["frames"]["a1_0.jpg"] == _corrected()  # the rest is untouched
    assert (d / "train.json").read_text(encoding="utf-8") == json.dumps(after, indent=1)
    assert (out["entries"], out["excluded"], out["corrected"]) == (2, 1, 1)
    raw = (d / "train.json").read_bytes()
    assert b"\r" not in raw  # line endings are not changed (a Windows text write would add CR)
    assert raw == json.dumps(after, indent=1).encode("utf-8")


def test_run_exclude_noop_writes_no_backup_and_keeps_the_file(tmp_path):
    manifest, index = _tables()
    d = _write_inputs(tmp_path, _overlay({"a1_0.jpg": dict(EXCLUDED)}))
    before = (d / "train.json").read_bytes()
    out = C.run_exclude(d, "train", "a1_0.jpg", CLASSES, manifest, index, now="T1")
    assert out["status"] == "already_excluded" and out["backup"] is None
    assert (d / "train.json").read_bytes() == before
    assert [p.name for p in d.iterdir()] == ["train.json"]


def test_run_exclude_refuses_a_corrected_frame_and_changes_nothing(tmp_path):
    manifest, index = _tables()
    d = _write_inputs(tmp_path, _overlay({"a1_0.jpg": _corrected()}))
    before = (d / "train.json").read_bytes()
    with pytest.raises(C.CorrectionError):
        C.run_exclude(d, "train", "a1_0.jpg", CLASSES, manifest, index, now="T1")
    assert (d / "train.json").read_bytes() == before
    assert [p.name for p in d.iterdir()] == ["train.json"]


def test_run_exclude_refuses_unknown_split_missing_file_and_a_frame_of_another_split(tmp_path):
    manifest, index = _tables()
    d = _write_inputs(tmp_path, _overlay())
    with pytest.raises(C.CorrectionError, match="split"):
        C.run_exclude(d, "valid", "v1_0.jpg", CLASSES, manifest, index, now="T1")
    with pytest.raises(C.CorrectionError, match="no overlay"):
        C.run_exclude(d, "val", "v1_0.jpg", CLASSES, manifest, index, now="T1")
    with pytest.raises(C.CorrectionError, match="unknown frame"):
        C.run_exclude(d, "train", "v1_0.jpg", CLASSES, manifest, index, now="T1")


def test_run_exclude_refuses_a_file_with_nul_bytes(tmp_path):
    manifest, index = _tables()
    d = tmp_path / "corrections"
    d.mkdir()
    (d / "train.json").write_bytes(b"\x00" * 50)
    with pytest.raises(C.CorrectionError, match="NUL"):
        C.run_exclude(d, "train", "a1_0.jpg", CLASSES, manifest, index, now="T1")


def test_run_exclude_revalidates_the_whole_file_and_keeps_the_original_on_failure(tmp_path):
    manifest, index = _tables()
    bad = _corrected()
    bad["boxes"][0]["xyxy"] = [10.0, 10.0, 999.0, 50.0]  # outside the 160x96 image
    d = _write_inputs(tmp_path, _overlay({"a1_0.jpg": bad}))
    before = (d / "train.json").read_bytes()
    with pytest.raises(BuildError, match="outside"):
        C.run_exclude(d, "train", "a1_60.jpg", CLASSES, manifest, index, now="T1")
    assert (d / "train.json").read_bytes() == before
    assert [p.name for p in d.iterdir()] == ["train.json"]
