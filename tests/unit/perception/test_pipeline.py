"""F2/F3 (essential-features.md sections 2-3): perception/pipeline.py -- PerceptionPipeline
implements contracts.Perception from a Detector and a hand tracker. Fakes only: determinism,
reset, the model_stamp format, frame-level exceptions, timestamp monotonicity."""

from __future__ import annotations

import logging
import re

import numpy as np
import pytest

from contracts import Detection, Frame, Hand, Perception, PerceptionFrame, Pose, compose_model_stamp
from perception import hands as hands_mod
from perception.pipeline import PerceptionPipeline

pytestmark = [pytest.mark.F2, pytest.mark.F3]

H_SHA = "h" * 64
P_SHA = "p" * 64
D_SHA = "d" * 64


class FakeDetector:
    name = "yolo11n"
    sha256 = D_SHA
    stamp_label = "yolo11n"

    def __init__(self):
        self.calls = []
        self.resets = 0
        self.fail_on: set[int] = set()

    def detect(self, frame_bgr):
        k = int(frame_bgr[0, 0, 0])
        self.calls.append(k)
        if k in self.fail_on:
            raise RuntimeError(f"boom {k}")
        return [Detection(label="tray", conf=0.5 + k / 1000, box=(1.0, 2.0, 3.0 + k, 4.0))]

    def reset(self):
        self.resets += 1


def _hand(x: float) -> Hand:
    return Hand(handedness="Left", score=0.9, landmarks_px=[(x, x)] * 21)


class FakeHands:
    weights_sha256 = H_SHA

    def __init__(self):
        self.ts: list[float] = []
        self.resets = 0

    def process(self, image, t):
        self.ts.append(t)
        return [_hand(float(image[0, 0, 0]))]

    def reset(self):
        self.resets += 1
        self.ts = []


class FakePose:
    weights_sha256 = P_SHA

    def __init__(self):
        self.resets = 0

    def process(self, image, t):
        return Pose(score=0.5, landmarks_px=[(1.0, 1.0)] * 33)

    def reset(self):
        self.resets += 1


def frame(i: int, t: float | None = None) -> Frame:
    img = np.full((8, 8, 3), i, np.uint8)
    return Frame(frame_id=i, t=i / 10 if t is None else t, image=img)


def make(**kw):
    return PerceptionPipeline(FakeDetector(), FakeHands(), **kw)


def test_it_implements_the_perception_protocol():
    p = make()
    assert isinstance(p, Perception)
    assert isinstance(p.process(frame(3)), PerceptionFrame)


def test_process_carries_frame_id_t_all_detections_and_hands_and_no_pose_by_default():
    p = make()
    out = p.process(frame(7, t=0.7))
    assert (out.frame_id, out.t) == (7, 0.7)
    assert [d.label for d in out.detections] == ["tray"] and out.detections[0].conf == 0.507
    assert len(out.hands) == 1 and out.hands[0].landmarks_px[0] == (7.0, 7.0)
    assert out.pose is None


def test_pose_is_present_only_when_a_pose_tracker_is_given():
    p = make(pose=FakePose())
    assert p.process(frame(1)).pose is not None


def test_same_input_gives_the_same_output():
    a, b = make(), make()
    fs = [frame(i) for i in range(5)]
    assert [a.process(f) for f in fs] == [b.process(f) for f in fs]
    a.reset()
    assert [a.process(f) for f in fs] == [b.process(f) for f in fs]


def test_the_image_is_not_modified():
    p = make()
    f = frame(4)
    before = f.image.copy()
    p.process(f)
    assert np.array_equal(f.image, before)


def test_model_stamp_is_composed_by_the_contract_with_the_detector_label():
    p = make()
    assert p.model_stamp == compose_model_stamp(D_SHA, H_SHA, "none", detector_name="yolo11n")
    assert re.fullmatch(r"yolo11n:dddddddd\|hand:hhhhhhhh\|pose:none", p.model_stamp)
    q = make(pose=FakePose())
    assert q.model_stamp.endswith("|pose:pppppppp")
    d = FakeDetector()
    d.stamp_label = "rfdetr-nano"
    assert PerceptionPipeline(d, FakeHands()).model_stamp.startswith("rfdetr-nano:dddddddd|")


def test_the_stamp_hashes_are_read_once_not_per_frame():
    class CountingHands(FakeHands):
        reads = 0

        @property
        def weights_sha256(self):
            CountingHands.reads += 1
            return H_SHA

    p = PerceptionPipeline(FakeDetector(), CountingHands())
    for i in range(5):
        p.process(frame(i))
    assert CountingHands.reads == 1 and p.model_stamp


def test_reset_resets_every_stateful_part_and_keeps_the_stamp():
    det, hands, pose = FakeDetector(), FakeHands(), FakePose()
    p = PerceptionPipeline(det, hands, pose=pose)
    stamp = p.model_stamp
    p.process(frame(1))
    p.reset()
    assert (det.resets, hands.resets, pose.resets) == (1, 1, 1)
    assert p.model_stamp == stamp


def test_hand_timestamps_come_from_frame_t():
    p = make()
    for i in range(4):
        p.process(frame(i, t=i * 0.5))
    assert p._hands.ts == [0.0, 0.5, 1.0, 1.5]


def test_a_failing_frame_is_logged_with_its_id_and_raised_for_the_loop_to_skip(caplog):
    det = FakeDetector()
    det.fail_on = {2}
    p = PerceptionPipeline(det, FakeHands())
    p.process(frame(1))
    with caplog.at_level(logging.WARNING, logger="perception.pipeline"):
        with pytest.raises(RuntimeError, match="boom 2"):
            p.process(frame(2))
    assert any("frame 2" in r.getMessage() for r in caplog.records)
    assert p.process(frame(3)).frame_id == 3  # the next frame is processed normally


def test_a_failing_hand_tracker_also_fails_the_frame_loudly_not_silently():
    class BadHands(FakeHands):
        def process(self, image, t):
            raise ValueError("landmarker died")

    p = PerceptionPipeline(FakeDetector(), BadHands())
    with pytest.raises(ValueError):
        p.process(frame(0))


# --- the real HandTracker keeps its timestamps strictly increasing --------------------------


class _Result:
    hand_landmarks: list = []
    handedness: list = []


class _Landmarker:
    def __init__(self):
        self.stamps: list[int] = []

    def detect_for_video(self, image, timestamp_ms):
        if self.stamps and timestamp_ms <= self.stamps[-1]:
            raise ValueError("timestamps must increase")  # MediaPipe's own rule
        self.stamps.append(timestamp_ms)
        return _Result()

    def close(self):
        pass


def test_timestamps_reaching_mediapipe_are_strictly_increasing_even_for_equal_frame_t(
    monkeypatch,
):
    lm = _Landmarker()
    monkeypatch.setattr(hands_mod.HandTracker, "_create_landmarker", lambda self: lm)
    monkeypatch.setattr(hands_mod, "_to_mp_image", lambda image: image)
    ht = hands_mod.HandTracker.__new__(hands_mod.HandTracker)
    ht._last_timestamp_ms = None
    ht._landmarker = lm
    p = PerceptionPipeline(FakeDetector(), ht, hand_sha256=H_SHA)
    for t in (0.0, 0.0001, 0.0002, 0.5, 0.5):
        p.process(frame(1, t=t))
    assert lm.stamps == sorted(set(lm.stamps)) and len(lm.stamps) == 5
