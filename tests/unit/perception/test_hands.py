"""F3 (essential-features.md section 3): perception/hands.py, tested as
plumbing only, against a fake MediaPipe landmarker -- per the 2026-09-29
ISSUES.md DECISION ("defer P1.2's real-footage check"), this session makes
no claim about real hardware behaviour (gloves survival, landmark
accuracy). No `.task` model file is vendored yet, so every test
monkeypatches the landmarker factory rather than touching a real one."""

from __future__ import annotations

import numpy as np
import pytest

from contracts import Hand, Pose
from perception import hands

pytestmark = pytest.mark.F3


class _FakeCategory:
    def __init__(self, category_name: str, score: float) -> None:
        self.category_name = category_name
        self.score = score


class _FakeLandmark:
    def __init__(self, x: float, y: float, visibility: float | None = 1.0) -> None:
        self.x = x
        self.y = y
        self.visibility = visibility


class _FakeHandResult:
    def __init__(self, hand_landmarks: list, handedness: list) -> None:
        self.hand_landmarks = hand_landmarks
        self.handedness = handedness


class _FakePoseResult:
    def __init__(self, pose_landmarks: list) -> None:
        self.pose_landmarks = pose_landmarks


class _FakeLandmarker:
    """Stands in for both HandLandmarker and PoseLandmarker: records every
    detect_for_video timestamp so tests can check monotonicity, and pops
    from a scripted list of canned results."""

    def __init__(self, results: list | None = None, default_factory=None) -> None:
        self._results = list(results) if results is not None else []
        self._default_factory = default_factory or (lambda: _FakeHandResult([], []))
        self.timestamps: list[int] = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms: int):
        self.timestamps.append(timestamp_ms)
        if self._results:
            return self._results.pop(0)
        return self._default_factory()

    def close(self) -> None:
        self.closed = True


def _make_image(height: int = 20, width: int = 30) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def _hand_tracker_with_fake(monkeypatch, created: list[_FakeLandmarker]) -> hands.HandTracker:
    def fake_create_from_options(_options):
        fake = _FakeLandmarker()
        created.append(fake)
        return fake

    monkeypatch.setattr(
        hands.mp_vision.HandLandmarker, "create_from_options", fake_create_from_options
    )
    return hands.HandTracker(model_path="fake_hand.task")


def _pose_tracker_with_fake(monkeypatch, created: list[_FakeLandmarker]) -> hands.PoseTracker:
    def fake_create_from_options(_options):
        fake = _FakeLandmarker(default_factory=lambda: _FakePoseResult([]))
        created.append(fake)
        return fake

    monkeypatch.setattr(
        hands.mp_vision.PoseLandmarker, "create_from_options", fake_create_from_options
    )
    return hands.PoseTracker(model_path="fake_pose.task")


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------


def test_no_hand_returns_empty_list_not_an_error(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _hand_tracker_with_fake(monkeypatch, created)
    result = tracker.process(_make_image(), t=0.0)
    assert result == []


def test_hand_landmarks_are_converted_to_original_pixel_coordinates(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _hand_tracker_with_fake(monkeypatch, created)
    landmarks = [_FakeLandmark(x=0.5, y=0.25) for _ in range(21)]
    created[0]._results.append(
        _FakeHandResult(
            hand_landmarks=[landmarks],
            handedness=[[_FakeCategory("Left", 0.91)]],
        )
    )

    result = tracker.process(_make_image(height=20, width=30), t=0.0)

    assert len(result) == 1
    hand = result[0]
    assert isinstance(hand, Hand)
    assert hand.handedness == "Left"
    assert hand.score == pytest.approx(0.91)
    assert len(hand.landmarks_px) == 21
    assert hand.landmarks_px[0] == pytest.approx((15.0, 5.0))  # 0.5*30, 0.25*20


def test_two_hands_each_keep_their_own_top_handedness_category(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _hand_tracker_with_fake(monkeypatch, created)
    left_landmarks = [_FakeLandmark(x=0.1, y=0.1) for _ in range(21)]
    right_landmarks = [_FakeLandmark(x=0.9, y=0.9) for _ in range(21)]
    created[0]._results.append(
        _FakeHandResult(
            hand_landmarks=[left_landmarks, right_landmarks],
            handedness=[
                [_FakeCategory("Left", 0.8), _FakeCategory("Right", 0.2)],
                [_FakeCategory("Right", 0.95)],
            ],
        )
    )

    result = tracker.process(_make_image(), t=0.0)

    assert [h.handedness for h in result] == ["Left", "Right"]
    assert result[0].score == pytest.approx(0.8)


def test_timestamps_are_milliseconds_and_strictly_increasing(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _hand_tracker_with_fake(monkeypatch, created)
    for t in (0.0, 1 / 30, 2 / 30):
        tracker.process(_make_image(), t=t)
    assert created[0].timestamps == [0, 33, 67]


def test_timestamp_collision_bumps_by_one_millisecond(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _hand_tracker_with_fake(monkeypatch, created)
    # Two frames close enough to round to the same millisecond.
    tracker.process(_make_image(), t=0.0001)
    tracker.process(_make_image(), t=0.0002)
    assert created[0].timestamps == [0, 1]
    assert created[0].timestamps[1] > created[0].timestamps[0]


def test_reset_recreates_the_landmarker_and_restarts_timestamps(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _hand_tracker_with_fake(monkeypatch, created)
    tracker.process(_make_image(), t=5.0)
    assert created[0].timestamps == [5000]

    tracker.reset()

    assert created[0].closed is True
    assert len(created) == 2  # a fresh landmarker was created
    tracker.process(_make_image(), t=0.0)
    assert created[1].timestamps == [0]  # timestamps restarted


def test_close_closes_the_underlying_landmarker(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _hand_tracker_with_fake(monkeypatch, created)
    tracker.close()
    assert created[0].closed is True


# ---------------------------------------------------------------------------
# PoseTracker
# ---------------------------------------------------------------------------


def test_pose_is_none_when_disabled_or_no_pose_detected(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _pose_tracker_with_fake(monkeypatch, created)
    result = tracker.process(_make_image(), t=0.0)
    assert result is None


def test_pose_landmarks_convert_to_pixels_with_visibility_mean_score(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _pose_tracker_with_fake(monkeypatch, created)
    landmarks = [_FakeLandmark(x=0.5, y=0.5, visibility=0.8) for _ in range(33)]
    created[0]._results.append(_FakePoseResult(pose_landmarks=[landmarks]))

    result = tracker.process(_make_image(height=10, width=10), t=0.0)

    assert isinstance(result, Pose)
    assert len(result.landmarks_px) == 33
    assert result.landmarks_px[0] == pytest.approx((5.0, 5.0))
    assert result.score == pytest.approx(0.8)


def test_pose_score_defaults_missing_visibility_to_one(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _pose_tracker_with_fake(monkeypatch, created)
    landmarks = [_FakeLandmark(x=0.1, y=0.1, visibility=None) for _ in range(33)]
    created[0]._results.append(_FakePoseResult(pose_landmarks=[landmarks]))

    result = tracker.process(_make_image(), t=0.0)

    assert result.score == pytest.approx(1.0)


def test_pose_reset_recreates_the_landmarker(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _pose_tracker_with_fake(monkeypatch, created)
    tracker.process(_make_image(), t=3.0)
    tracker.reset()
    assert created[0].closed is True
    assert len(created) == 2


# ---------------------------------------------------------------------------
# weights_sha256
# ---------------------------------------------------------------------------


def test_weights_sha256_hashes_the_model_file(tmp_path, monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    model_path = tmp_path / "hand_landmarker.task"
    model_path.write_bytes(b"fake model bytes")

    def fake_create_from_options(_options):
        fake = _FakeLandmarker()
        created.append(fake)
        return fake

    monkeypatch.setattr(
        hands.mp_vision.HandLandmarker, "create_from_options", fake_create_from_options
    )
    tracker = hands.HandTracker(model_path=model_path)

    import hashlib

    expected = hashlib.sha256(b"fake model bytes").hexdigest()
    assert tracker.weights_sha256 == expected
