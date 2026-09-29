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
        self.received_images: list = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms: int):
        self.timestamps.append(timestamp_ms)
        self.received_images.append(image)
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
# _to_mp_image: the BGR->RGB model boundary (rule 16). Builds a real
# mp.Image -- no model file needed, per ISSUES.md 2026-09-29 P2 review.
# ---------------------------------------------------------------------------


def test_to_mp_image_produces_contiguous_rgb_with_correct_channel_order() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)  # BGR
    image[1, 2] = (100, 150, 200)
    image[3, 4] = (5, 6, 7)

    mp_image = hands._to_mp_image(image)
    view = mp_image.numpy_view()

    assert view.flags["C_CONTIGUOUS"]
    assert tuple(view[0, 0]) == (30, 20, 10)
    assert tuple(view[1, 2]) == (200, 150, 100)
    assert tuple(view[3, 4]) == (7, 6, 5)


def test_to_mp_image_handles_a_non_contiguous_input_view() -> None:
    base = np.zeros((8, 10, 3), dtype=np.uint8)
    base[2, 4] = (1, 2, 3)
    sliced = base[::2, ::2]  # e.g. a decimated/cropped frame view
    assert not sliced.flags["C_CONTIGUOUS"]

    view = hands._to_mp_image(sliced).numpy_view()

    assert tuple(view[1, 2]) == (3, 2, 1)  # base[2,4] -> sliced[1,2]


def test_to_mp_image_does_not_mutate_the_caller_array() -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    original = image.copy()

    hands._to_mp_image(image)

    assert np.array_equal(image, original)


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


def test_hand_tracker_passes_a_correctly_converted_image_to_the_landmarker(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _hand_tracker_with_fake(monkeypatch, created)
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[2, 3] = (10, 20, 30)  # BGR

    tracker.process(image, t=0.0)

    view = created[0].received_images[0].numpy_view()
    assert view.flags["C_CONTIGUOUS"]
    assert tuple(view[2, 3]) == (30, 20, 10)


def test_hand_tracker_handles_a_non_contiguous_input_frame(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _hand_tracker_with_fake(monkeypatch, created)
    base = np.zeros((8, 10, 3), dtype=np.uint8)
    base[2, 4] = (9, 8, 7)
    frame = base[::2, ::2]
    assert not frame.flags["C_CONTIGUOUS"]

    tracker.process(frame, t=0.0)

    view = created[0].received_images[0].numpy_view()
    assert tuple(view[1, 2]) == (7, 8, 9)  # base[2,4] -> frame[1,2]


def test_hand_tracker_is_deterministic_across_reset(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _hand_tracker_with_fake(monkeypatch, created)

    def scripted_result() -> _FakeHandResult:
        landmarks = [_FakeLandmark(x=0.3, y=0.4) for _ in range(21)]
        return _FakeHandResult(
            hand_landmarks=[landmarks], handedness=[[_FakeCategory("Left", 0.77)]]
        )

    created[0]._results.append(scripted_result())
    first = tracker.process(_make_image(), t=0.0)

    tracker.reset()
    created[1]._results.append(scripted_result())
    second = tracker.process(_make_image(), t=0.0)

    assert first == second


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


def test_pose_tracker_passes_a_correctly_converted_image_to_the_landmarker(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _pose_tracker_with_fake(monkeypatch, created)
    image = np.zeros((6, 8, 3), dtype=np.uint8)
    image[4, 1] = (11, 22, 33)  # BGR

    tracker.process(image, t=0.0)

    view = created[0].received_images[0].numpy_view()
    assert view.flags["C_CONTIGUOUS"]
    assert tuple(view[4, 1]) == (33, 22, 11)


def test_pose_tracker_handles_a_non_contiguous_input_frame(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _pose_tracker_with_fake(monkeypatch, created)
    base = np.zeros((8, 10, 3), dtype=np.uint8)
    base[6, 8] = (3, 4, 5)
    frame = base[::2, ::2]
    assert not frame.flags["C_CONTIGUOUS"]

    tracker.process(frame, t=0.0)

    view = created[0].received_images[0].numpy_view()
    assert tuple(view[3, 4]) == (5, 4, 3)  # base[6,8] -> frame[3,4]


def test_pose_tracker_is_deterministic_across_reset(monkeypatch) -> None:
    created: list[_FakeLandmarker] = []
    tracker = _pose_tracker_with_fake(monkeypatch, created)

    def scripted_result() -> _FakePoseResult:
        landmarks = [_FakeLandmark(x=0.2, y=0.6, visibility=0.9) for _ in range(33)]
        return _FakePoseResult(pose_landmarks=[landmarks])

    created[0]._results.append(scripted_result())
    first = tracker.process(_make_image(), t=0.0)

    tracker.reset()
    created[1]._results.append(scripted_result())
    second = tracker.process(_make_image(), t=0.0)

    assert first == second


# ---------------------------------------------------------------------------
# weights_sha256
# ---------------------------------------------------------------------------


def test_weights_sha256_raises_file_not_found_when_no_model_file_is_vendored(
    tmp_path, monkeypatch
) -> None:
    # Documents current behaviour (ISSUES.md 2026-09-29 P2 review): no
    # .task file is vendored yet, so weights_sha256 -- which re-reads and
    # hashes the file on every access, by design, see the docstring --
    # raises rather than returning a stale/cached value.
    created: list[_FakeLandmarker] = []

    def fake_create_from_options(_options):
        fake = _FakeLandmarker()
        created.append(fake)
        return fake

    monkeypatch.setattr(
        hands.mp_vision.HandLandmarker, "create_from_options", fake_create_from_options
    )
    tracker = hands.HandTracker(model_path=tmp_path / "missing.task")

    with pytest.raises(FileNotFoundError):
        _ = tracker.weights_sha256


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
