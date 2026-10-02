"""Unit tests for training/sample_frames.py (F14 stage 2): stride, 32x32
thumbnail de-duplication, the 1 s keep rule, the per-run cap, and a tiny
end-to-end pass over a synthetic video."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from training.sample_frames import (
    DIFF_THRESHOLD,
    select_frame_ids,
    stride_for,
    thumb_diff,
    thumbnail,
)

# --- stride_for ---------------------------------------------------------------


@pytest.mark.parametrize("fps,stride", [(30.0, 15), (29.95, 15), (25.0, 12), (60.0, 30), (1.0, 1)])
def test_stride_is_half_the_fps_rounded(fps, stride):
    assert stride_for(fps) == stride


def test_stride_never_below_one():
    assert stride_for(0.4) == 1


# --- thumbnail / thumb_diff ---------------------------------------------------


def test_thumbnail_is_32x32_float_grayscale():
    img = np.zeros((480, 848, 3), dtype=np.uint8)
    t = thumbnail(img)
    assert t.shape == (32, 32)
    assert t.dtype == np.float32


def test_thumbnail_uses_bgr_order_for_gray():
    blue = np.zeros((64, 64, 3), dtype=np.uint8)
    blue[..., 0] = 255  # BGR blue
    red = np.zeros((64, 64, 3), dtype=np.uint8)
    red[..., 2] = 255
    # BT.601 gray: blue 0.114*255 ~ 29, red 0.299*255 ~ 76
    assert thumbnail(blue).mean() == pytest.approx(29.0, abs=1.0)
    assert thumbnail(red).mean() == pytest.approx(76.0, abs=1.0)


def test_thumb_diff_identical_is_zero_and_in_unit_scale():
    a = thumbnail(np.full((100, 100, 3), 100, dtype=np.uint8))
    b = thumbnail(np.full((100, 100, 3), 100 + 51, dtype=np.uint8))
    assert thumb_diff(a, a) == 0.0
    assert thumb_diff(a, b) == pytest.approx(51 / 255, abs=1e-3)


# --- select_frame_ids ---------------------------------------------------------


def _flat(level):
    return np.full((32, 32), level, dtype=np.float32)


def test_first_candidate_is_always_kept():
    assert select_frame_ids([(0, _flat(10))], src_fps=30.0) == [0]


def test_static_scene_is_kept_once_per_second_only():
    # candidates every 15 frames (0.5 s) of an identical scene
    cands = [(i * 15, _flat(100)) for i in range(9)]  # 0 .. 120 = 4 s
    kept = select_frame_ids(cands, src_fps=30.0)
    assert kept == [0, 30, 60, 90, 120]


def test_changed_scene_is_kept_even_inside_one_second():
    big = (DIFF_THRESHOLD * 255 + 3) * 1.0
    cands = [(0, _flat(100)), (15, _flat(100 + big)), (30, _flat(100 + big))]
    kept = select_frame_ids(cands, src_fps=30.0)
    # frame 15 differs from kept frame 0; frame 30 equals the last kept (15) and
    # only 0.5 s has passed -> dropped
    assert kept == [0, 15]


def test_diff_at_threshold_is_not_kept():
    at = DIFF_THRESHOLD * 255
    cands = [(0, _flat(100)), (15, _flat(100 + at))]
    assert select_frame_ids(cands, src_fps=30.0) == [0]


def test_reference_is_last_kept_not_last_candidate():
    # slow drift: every step is small vs the previous candidate, but the sum
    # exceeds the threshold against the last KEPT frame
    step = DIFF_THRESHOLD * 255 * 0.4
    cands = [(i * 5, _flat(100 + i * step)) for i in range(6)]  # 1/6 s apart
    kept = select_frame_ids(cands, src_fps=30.0)
    # i=3 is 1.2x the threshold away from frame 0 -> kept; i=4, 5 are only
    # 0.4x and 0.8x away from the new reference (frame 15) -> dropped
    assert kept == [0, 15]


def test_cap_subsamples_evenly_and_keeps_first_and_last():
    big = 100.0
    cands = [(i * 15, _flat(50 + (i % 2) * big)) for i in range(300)]  # all differ
    kept = select_frame_ids(cands, src_fps=30.0, cap=120)
    assert len(kept) == 120
    assert kept[0] == 0
    assert kept[-1] == 299 * 15
    assert kept == sorted(set(kept))


def test_cap_not_applied_below_the_limit():
    cands = [(i * 15, _flat(50 + (i % 2) * 100.0)) for i in range(50)]
    assert len(select_frame_ids(cands, src_fps=30.0, cap=120)) == 50


def test_deterministic():
    cands = [(i * 15, _flat(50 + (i * 37) % 90)) for i in range(60)]
    assert select_frame_ids(cands, 30.0) == select_frame_ids(cands, 30.0)


# --- end to end on a synthetic clip --------------------------------------------


def _write_clip(path, n_frames=75, fps=30.0, size=(160, 96)):
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened()
    for i in range(n_frames):
        img = np.full((h, w, 3), 60, dtype=np.uint8)
        if i >= 40:  # a bright block appears at frame 40
            img[20:80, 30:120] = 230
        vw.write(img)
    vw.release()


def test_sample_run_writes_named_jpegs_at_native_resolution(tmp_path):
    from training.sample_frames import sample_run

    video = tmp_path / "video.mp4"
    _write_clip(video)
    out = tmp_path / "frames" / "train"
    ids = sample_run("x900", video, out, src_fps=30.0)
    assert ids[0] == 0
    assert ids == sorted(ids)
    assert 45 in ids or 60 in ids  # the scene change is picked up promptly
    for fid in ids:
        p = out / f"x900_{fid}.jpg"
        assert p.exists()
    img = cv2.imread(str(out / f"x900_{ids[0]}.jpg"))
    assert img.shape == (96, 160, 3)
    assert sorted(p.name for p in out.iterdir()) == sorted(f"x900_{i}.jpg" for i in ids)


def test_sample_run_is_deterministic(tmp_path):
    from training.sample_frames import sample_run

    video = tmp_path / "video.mp4"
    _write_clip(video)
    a = sample_run("x900", video, tmp_path / "a", src_fps=30.0)
    b = sample_run("x900", video, tmp_path / "b", src_fps=30.0)
    assert a == b
