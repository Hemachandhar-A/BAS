"""perception/hands.py -- HandTracker and PoseTracker (F3,
essential-features.md section 3). Owner: P1.

Per the 2026-09-29 ISSUES.md DECISION ("defer P1.2's real-footage check"),
this session builds and tests the MediaPipe Tasks-API wrapper as plumbing
only, against a fake landmarker (no real `.task` model file is vendored
yet -- see the 2026-09-28 "MediaPipe model files must be vendored"
DECISION entry, still open). It makes no claim about real performance
(gloves survival, landmark accuracy) until the real-footage check runs.
"""

from __future__ import annotations

from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode

from contracts import Hand, Pose, sha256_of_file

DEFAULT_HAND_MODEL_PATH = Path("weights/hand_landmarker.task")
DEFAULT_POSE_MODEL_PATH = Path("weights/pose_landmarker_lite.task")

_DEFAULT_CONFIDENCE = 0.5
_RUNNING_MODE = VisionTaskRunningMode.VIDEO


def _next_timestamp_ms(frame_t: float, last_timestamp_ms: int | None) -> int:
    """essential-features.md section 3, step 2: VIDEO mode requires a
    strictly increasing timestamp -- MediaPipe raises ValueError otherwise.
    Bump by +1ms whenever two frames round to the same value."""
    timestamp_ms = int(round(frame_t * 1000))
    if last_timestamp_ms is not None and timestamp_ms <= last_timestamp_ms:
        timestamp_ms = last_timestamp_ms + 1
    return timestamp_ms


def _to_pixels(landmarks, width: int, height: int) -> list[tuple[float, float]]:
    # essential-features.md section 3, step 3: normalized [0,1] -> pixels.
    # Nobody downstream ever sees normalized coordinates (contract rule).
    return [(float(lm.x) * width, float(lm.y) * height) for lm in landmarks]


def _to_mp_image(image: np.ndarray) -> mp.Image:
    # BGR (OpenCV native) everywhere; convert to RGB only at this model
    # boundary (AGENTS.md rule 16). `image[..., ::-1]` alone is a
    # negative-stride, non-contiguous view -- MediaPipe's native Image
    # reads it as raw contiguous bytes, so it silently sees scrambled
    # channels and out-of-bounds memory (ISSUES.md, 2026-09-29 P2 review).
    # np.ascontiguousarray copies it into a real C-contiguous RGB buffer
    # without mutating the caller's (BGR) array.
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(image[..., ::-1]))


class HandTracker:
    """Wraps MediaPipe's HandLandmarker (Tasks API, VIDEO mode). Never
    raises on a frame with no hand -- returns ``[]`` (contract rule)."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_HAND_MODEL_PATH,
        num_hands: int = 2,
        min_detection_confidence: float = _DEFAULT_CONFIDENCE,
        min_presence_confidence: float = _DEFAULT_CONFIDENCE,
        min_tracking_confidence: float = _DEFAULT_CONFIDENCE,
    ) -> None:
        self._model_path = Path(model_path)
        self._num_hands = num_hands
        self._min_detection_confidence = min_detection_confidence
        self._min_presence_confidence = min_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._last_timestamp_ms: int | None = None
        self._landmarker = self._create_landmarker()

    def _create_landmarker(self):
        options = mp_vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=_RUNNING_MODE,
            num_hands=self._num_hands,
            min_hand_detection_confidence=self._min_detection_confidence,
            min_hand_presence_confidence=self._min_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        return mp_vision.HandLandmarker.create_from_options(options)

    @property
    def weights_sha256(self) -> str:
        """For ``compose_model_stamp`` (contracts.py), assembled by
        ``perception/pipeline.py`` at P1.6. Re-reads and hashes the whole
        file on every access -- fine for a one-off stamp, but P1.6 should
        read it once at construction / model_stamp composition time, not
        per frame (ISSUES.md, 2026-09-29 P2 review; not cached here since
        that's a larger change than this property needs today)."""
        return sha256_of_file(self._model_path)

    def process(self, image: np.ndarray, t: float) -> list[Hand]:
        height, width = image.shape[:2]
        timestamp_ms = _next_timestamp_ms(t, self._last_timestamp_ms)
        result = self._landmarker.detect_for_video(_to_mp_image(image), timestamp_ms)
        self._last_timestamp_ms = timestamp_ms

        hands: list[Hand] = []
        for landmarks, categories in zip(
            result.hand_landmarks, result.handedness, strict=True
        ):
            top = categories[0]  # essential-features.md section 3, step 4: as reported
            hands.append(
                Hand(
                    handedness=top.category_name,
                    score=top.score,
                    landmarks_px=_to_pixels(landmarks, width, height),
                )
            )
        return hands

    def reset(self) -> None:
        """essential-features.md section 3, step 2: VIDEO-mode timestamps
        restart at each run, so the landmarker is recreated."""
        self._landmarker.close()
        self._last_timestamp_ms = None
        self._landmarker = self._create_landmarker()

    def close(self) -> None:
        self._landmarker.close()


class PoseTracker:
    """Wraps MediaPipe's PoseLandmarker. Off by default
    (``RuntimeConfig.enable_pose=False``, essential-features.md section 3
    step 5) -- construct one only when pose is enabled."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_POSE_MODEL_PATH,
        min_detection_confidence: float = _DEFAULT_CONFIDENCE,
        min_presence_confidence: float = _DEFAULT_CONFIDENCE,
        min_tracking_confidence: float = _DEFAULT_CONFIDENCE,
    ) -> None:
        self._model_path = Path(model_path)
        self._min_detection_confidence = min_detection_confidence
        self._min_presence_confidence = min_presence_confidence
        self._min_tracking_confidence = min_tracking_confidence
        self._last_timestamp_ms: int | None = None
        self._landmarker = self._create_landmarker()

    def _create_landmarker(self):
        options = mp_vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self._model_path)),
            running_mode=_RUNNING_MODE,
            num_poses=1,
            min_pose_detection_confidence=self._min_detection_confidence,
            min_pose_presence_confidence=self._min_presence_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )
        return mp_vision.PoseLandmarker.create_from_options(options)

    @property
    def weights_sha256(self) -> str:
        """See ``HandTracker.weights_sha256``: re-hashes on every access,
        documented rather than cached (ISSUES.md, 2026-09-29 P2 review)."""
        return sha256_of_file(self._model_path)

    def process(self, image: np.ndarray, t: float) -> Pose | None:
        height, width = image.shape[:2]
        timestamp_ms = _next_timestamp_ms(t, self._last_timestamp_ms)
        result = self._landmarker.detect_for_video(_to_mp_image(image), timestamp_ms)
        self._last_timestamp_ms = timestamp_ms

        if not result.pose_landmarks:
            return None
        landmarks = result.pose_landmarks[0]  # num_poses=1
        # essential-features.md/contracts.py don't say how Pose.score is
        # derived from MediaPipe output (unlike Hand, which gets a
        # handedness Category.score directly) -- disclosed judgment call:
        # mean per-landmark visibility, defaulting an unset visibility to
        # 1.0. Recorded in ISSUES.md, 2026-09-29 (P1.3 DECISION). Not
        # exercised by any Tier-1 rule since pose is off by default.
        visibilities = [
            lm.visibility if lm.visibility is not None else 1.0 for lm in landmarks
        ]
        score = float(np.mean(visibilities))
        return Pose(score=score, landmarks_px=_to_pixels(landmarks, width, height))

    def reset(self) -> None:
        self._landmarker.close()
        self._last_timestamp_ms = None
        self._landmarker = self._create_landmarker()

    def close(self) -> None:
        self._landmarker.close()
