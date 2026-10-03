"""perception/pipeline.py -- PerceptionPipeline, the real ``contracts.Perception`` (F2 + F3,
IMPLEMENTATION_PLAN.md 5.1). Owner: P1.

``process(frame)`` runs the detector and the hand tracker (and the pose tracker only when one
was given) on the BGR frame and returns a ``PerceptionFrame`` with *every* detection at or above
``DETECTOR_MIN_CONF`` -- the tracker's ``detector_conf_floor`` decides, not perception. Hand
timestamps are derived from ``frame.t`` by ``HandTracker`` itself (milliseconds, strictly
increasing). Nothing here reads a clock or draws a random number: the same frames give the same
output.

A frame-level failure is logged with its frame id and **re-raised**: ``runtime/loop.py`` catches
an exception from ``process`` (a warning, then it skips that frame), and so does the cache
builder. ``model_stamp`` is composed once, at construction, by ``contracts.compose_model_stamp``
with the detector's stamp label and ``pose:none`` when pose is off.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from contracts import Frame, PerceptionFrame, compose_model_stamp, sha256_of_file
from perception.detector import DEFAULT_MANIFEST_PATH, Detector, load_detector

logger = logging.getLogger(__name__)


class PerceptionPipeline:
    """Detector + hands (+ optional pose) behind ``contracts.Perception``."""

    def __init__(
        self,
        detector: Detector,
        hands: Any,
        pose: Any = None,
        *,
        hand_sha256: str | None = None,
        pose_sha256: str | None = None,
    ) -> None:
        self._detector = detector
        self._hands = hands
        self._pose = pose
        hand_sha = hand_sha256 or hands.weights_sha256  # hashed once, here, not per frame
        pose_sha = "none" if pose is None else (pose_sha256 or pose.weights_sha256)
        self.model_stamp: str = compose_model_stamp(
            detector.sha256, hand_sha, pose_sha, detector_name=detector.stamp_label
        )

    @property
    def detector_name(self) -> str:
        return self._detector.name

    def process(self, frame: Frame) -> PerceptionFrame:
        try:
            detections = self._detector.detect(frame.image)
            hands = self._hands.process(frame.image, frame.t)
            pose = None if self._pose is None else self._pose.process(frame.image, frame.t)
        except Exception:
            logger.error("perception failed on frame %d", frame.frame_id, exc_info=True)
            raise
        return PerceptionFrame(
            frame_id=frame.frame_id, t=frame.t, detections=detections, hands=hands, pose=pose
        )

    def reset(self) -> None:
        """A fresh run: the hand (and pose) landmarker is recreated so VIDEO-mode timestamps
        restart; the detector clears whatever state it keeps."""
        self._detector.reset()
        self._hands.reset()
        if self._pose is not None:
            self._pose.reset()


def load_pipeline(
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    detector: str | None = None,
    *,
    enable_pose: bool = False,
    device: str = "auto",
) -> PerceptionPipeline:
    """Build the real pipeline at launch: the detector by ``load_detector`` (argument, then
    ``$SIH_DETECTOR``, then the manifest), the hand tracker from the manifest's ``hand`` entry
    with its sha256 checked. Pose needs a ``pose`` manifest entry; without one it is refused."""
    from perception.hands import HandTracker, PoseTracker  # mediapipe: only when really built

    mpath = Path(manifest_path)
    manifest = json.loads(mpath.read_text(encoding="utf-8"))
    det = load_detector(mpath, detector, device=device)

    def checked(entry: dict, what: str) -> Path:
        path = mpath.parent / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"{what} model not found: {path}")
        if sha256_of_file(path) != entry["sha256"]:
            raise ValueError(f"{what} model sha256 differs from the manifest: {path}")
        return path

    hands = HandTracker(model_path=checked(manifest["hand"], "hand"))
    pose = None
    if enable_pose:
        if "pose" not in manifest:
            raise ValueError("enable_pose needs a 'pose' entry in the manifest; there is none")
        pose = PoseTracker(model_path=checked(manifest["pose"], "pose"))
    return PerceptionPipeline(det, hands, pose)
