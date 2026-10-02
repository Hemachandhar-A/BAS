"""contracts.py -- the single source of truth for shapes, vocabularies and
signatures that cross a person boundary (IMPLEMENTATION_PLAN.md Part 1, R1).

This module is frozen after G0 (IMPLEMENTATION_PLAN.md R4). Agents never edit
it after G0: found a gap? Append a CONTRACT entry to ISSUES.md, work against a
TEMP_<n> stub, and keep going (IMPLEMENTATION_PLAN.md Part 6). ISSUES.md is
the log of open contract questions and decisions for this file -- read it
before touching anything here.

Import purity (tested): this module imports only pydantic, numpy, pyyaml and
the standard library. It must import cleanly in an environment with no torch,
mediapipe or rfdetr installed.

Conventions (essential-features.md section 0), restated because they cause
the most bugs:
- Time ``t`` is float seconds since the source started, never wall-clock.
- Pixels are original-frame pixels, origin top-left; boxes are
  ``(x1, y1, x2, y2)`` with ``x1 <= x2`` and ``y1 <= y2``.
- Images are ``np.ndarray (H, W, 3) uint8`` in BGR (OpenCV native). Convert to
  RGB only at a model boundary (RF-DETR, MediaPipe), never earlier.
- IDs (class names, step ids) are ``lower_snake_case``.
- Confidence and score values are in ``[0, 1]``.
- No non-finite (``NaN``/``inf``) numbers cross a contract boundary.
"""

from __future__ import annotations

import hashlib
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

import numpy as np
import yaml
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ---------------------------------------------------------------------------
# Project-wide constants (disclosed judgment calls, defined once; see
# essential-features.md section 0 and IMPLEMENTATION_PLAN.md 5.8)
# ---------------------------------------------------------------------------

DETECTOR_MIN_CONF: float = 0.10
"""Deliberately far below ``PerceptionConfig.detector_conf_floor`` (0.30) so
the tracker's floor is what decides, and tuning has room."""

CAPTURE_WIDTH: int = 1280
CAPTURE_HEIGHT: int = 720
CAPTURE_FPS: int = 30

MAX_SPOKEN_WORDS: int = 8
"""A ``StepDef.say`` is a phrase, not a sentence (checked by the experiment
lint, IMPLEMENTATION_PLAN.md 7.3 -- not enforced here because it is a style
check, not a structural one)."""

RECENT_ALERTS_CAP: int = 20
DASHBOARD_POLL_MS: int = 500

_SNAKE_CASE_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_snake_case(value: str) -> str:
    if not _SNAKE_CASE_RE.match(value):
        raise ValueError(f"{value!r} must be lower_snake_case")
    return value


def _finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{value!r} must be finite (no NaN/inf)")
    return value


FiniteFloat = Annotated[float, AfterValidator(_finite)]
Unit = Annotated[float, AfterValidator(_finite), Field(ge=0.0, le=1.0)]
SnakeCaseStr = Annotated[str, AfterValidator(_is_snake_case)]


def compose_model_stamp(detector_sha256: str, hand_sha256: str, pose_sha256: str) -> str:
    """The one place the ``model_stamp`` string is composed, so P1 (writer)
    and P2 (reader/verifier) can never disagree on its format.

    Format: ``"rfdetr-nano:<sha256[:8]>|hand:<sha256[:8]>|pose:<sha256[:8]>"``
    (IMPLEMENTATION_PLAN.md 5.2).
    """
    return (
        f"rfdetr-nano:{detector_sha256[:8]}"
        f"|hand:{hand_sha256[:8]}"
        f"|pose:{pose_sha256[:8]}"
    )


def sha256_of_file(path: Path) -> str:
    """Shared helper so every sha256 in ``model_stamp`` / manifests is
    computed the same way (streamed, so large weight files are fine)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class ContractViolation(Exception):
    """Raised when calling code breaks an invariant this module defines
    (e.g. an unknown step_id passed to ``Engine.on_state_event``, or a
    ``JsonlLogger`` sequence gap). Never coerced -- callers must not catch
    this to paper over a bug."""


class SourceError(Exception):
    """Raised by ``open_source`` (perception/camera.py) at construction when
    a video/camera source cannot be opened or the first read fails."""


# ---------------------------------------------------------------------------
# Section 1 -- Ingestion (F1): Frame, FrameSource
# ---------------------------------------------------------------------------


class Frame(BaseModel):
    """One raw frame from a ``FrameSource``. ``t`` is seconds since the
    source started; for a file this is ``frame_id / fps``, for a live camera
    it is a monotonic clock read at capture time (the only clock read
    allowed in the perception layer, other than inside camera.py itself)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    frame_id: int = Field(ge=0)
    t: FiniteFloat
    image: np.ndarray

    @field_validator("image")
    @classmethod
    def _check_image(cls, v: np.ndarray) -> np.ndarray:
        if not isinstance(v, np.ndarray):
            raise TypeError("Frame.image must be a numpy.ndarray")
        if v.ndim != 3 or v.shape[2] != 3:
            raise ValueError("Frame.image must have shape (H, W, 3)")
        if v.dtype != np.uint8:
            raise ValueError("Frame.image must be dtype uint8 (BGR)")
        return v


@runtime_checkable
class FrameSource(Protocol):
    """Implemented by ``perception/camera.py``. Constructed via the factory
    ``open_source(source: str) -> FrameSource`` (also in
    ``perception/camera.py``); ``source`` is an all-digit camera index, a
    file path, or a stream URL. ``SourceError`` is raised at construction,
    never from ``read()``. ``perception.camera.DecimatedSource`` wraps any
    ``FrameSource`` and implements this same protocol."""

    fps: float | None
    exhausted: bool

    def read(self) -> Frame | None:
        """A file at end-of-stream sets ``exhausted = True`` and returns
        ``None``. A live camera returning ``None`` with ``exhausted = False``
        is a transient read failure, not end-of-stream."""
        ...

    def close(self) -> None: ...


# ---------------------------------------------------------------------------
# Section 2 -- Detection, hands/pose, PerceptionFrame (F2, F3)
# ---------------------------------------------------------------------------


class Detection(BaseModel):
    """One detected object instance. ``label`` must be a member of the
    active ``ExperimentDefinition.classes`` (checked by the caller, since
    this model has no experiment in scope)."""

    label: SnakeCaseStr
    conf: Unit
    box: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

    @field_validator("box")
    @classmethod
    def _check_box(
        cls, v: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = v
        if x1 > x2 or y1 > y2:
            raise ValueError(f"box {v!r} must have x1<=x2 and y1<=y2")
        return v


class Hand(BaseModel):
    """21 hand landmarks in original-image pixels (never normalized).
    ``handedness`` is informational only -- no Tier-1 rule consumes it,
    because MediaPipe assumes a mirrored (selfie) image and this system's
    camera is not mirrored."""

    handedness: Literal["Left", "Right"]
    score: Unit
    landmarks_px: list[tuple[FiniteFloat, FiniteFloat]] = Field(
        min_length=21, max_length=21
    )


class Pose(BaseModel):
    """33 body-pose landmarks in original-image pixels. ``pose = None`` on
    ``PerceptionFrame`` whenever pose tracking is disabled
    (``PerceptionConfig``/pipeline ``enable_pose=False``, the default)."""

    score: Unit
    landmarks_px: list[tuple[FiniteFloat, FiniteFloat]] = Field(
        min_length=33, max_length=33
    )


class PerceptionFrame(BaseModel):
    """Pure evidence, no experiment logic: everything P1 owns ends here, and
    everything P2 owns starts here (IMPLEMENTATION_PLAN.md 5.1). Detections
    below ``PerceptionConfig.detector_conf_floor`` are still present here --
    the floor is applied by ``StateTracker``, not by perception."""

    frame_id: int = Field(ge=0)
    t: FiniteFloat
    detections: list[Detection] = Field(default_factory=list)
    hands: list[Hand] = Field(default_factory=list)
    pose: Pose | None = None


class PerceptionCacheHeader(BaseModel):
    """Line 1 of ``data/cache/<run_id>/perception.jsonl``; every later line
    is a ``PerceptionFrame``. ``fps`` is the rate the cache was built at
    (``RuntimeConfig.target_fps``), not the source's native fps. A cache is
    valid for replay only while its ``model_stamp`` equals the replaying
    ``Perception.model_stamp``."""

    run_id: str
    experiment_id: str
    fps: FiniteFloat
    model_stamp: str


@runtime_checkable
class Perception(Protocol):
    """Implemented by ``perception/pipeline.py``'s ``PerceptionPipeline``.
    Consumed by P2's runtime/harness (the sanctioned cross-boundary call,
    IMPLEMENTATION_PLAN.md R3). Deterministic for fixed weights and input."""

    model_stamp: str

    def process(self, frame: Frame) -> PerceptionFrame: ...

    def reset(self) -> None:
        """Recreates any stateful sub-model (e.g. the MediaPipe VIDEO-mode
        landmarker, whose timestamps must restart) for a fresh run."""
        ...


# ---------------------------------------------------------------------------
# Section 3 -- Experiment definition and the WhenRule vocabulary (F4)
#
# Rule semantics (evaluated by state/tracker.py against a floor-filtered
# PerceptionFrame -- "best" detection of a label = the highest-confidence
# one, ties broken by first-in-list; see IMPLEMENTATION_PLAN.md 5.3):
#
#   inside(label, container)        true iff both label and container have a
#                                    best detection, and the label box's
#                                    centroid lies within the container box.
#                                    False if either is undetected.
#   outside(label, container)       true iff label has a best detection and
#                                    it is NOT inside container. If container
#                                    has no detection at all, the rule is
#                                    vacuously true (it cannot be inside a
#                                    container that is not there). False if
#                                    label itself is undetected.
#   hand_touching(label)            true iff label has a best detection,
#                                    grown by PerceptionConfig.touch_margin_frac
#                                    on each side, and at least one FINGERTIP
#                                    landmark (MediaPipe indices 4, 8, 12, 16,
#                                    20) of any hand lies inside the grown box.
#                                    Other landmarks (wrist, palm) do not count.
#   absent(label)                   true iff label has no detection at or
#                                    above the confidence floor this frame.
#   present(label)                  true iff label has at least one
#                                    detection at or above the floor.
#
# This is a DECISION taken at G0 (2026-09-28): the five kinds above are the
# closed WhenRule vocabulary (IMPLEMENTATION_PLAN.md 7.4, F4: "all five rule
# kinds"). See ISSUES.md.
# ---------------------------------------------------------------------------


class InsideRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["inside"] = "inside"
    label: SnakeCaseStr
    container: SnakeCaseStr


class OutsideRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["outside"] = "outside"
    label: SnakeCaseStr
    container: SnakeCaseStr


class HandTouchingRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["hand_touching"] = "hand_touching"
    label: SnakeCaseStr


class AbsentRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["absent"] = "absent"
    label: SnakeCaseStr


class PresentRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    type: Literal["present"] = "present"
    label: SnakeCaseStr


WhenRule = Annotated[
    InsideRule | OutsideRule | HandTouchingRule | AbsentRule | PresentRule,
    Field(discriminator="type"),
]

WHEN_RULE_TYPES: frozenset[str] = frozenset(
    {"inside", "outside", "hand_touching", "absent", "present"}
)


_AnyRule = InsideRule | OutsideRule | HandTouchingRule | AbsentRule | PresentRule


def _rule_labels(rule: _AnyRule) -> set[str]:
    labels = {rule.label}
    if isinstance(rule, (InsideRule, OutsideRule)):
        labels.add(rule.container)
    return labels


class StepDef(BaseModel):
    """One step of the canonical, single-valid-order sequence. ``when`` is
    the AND of its rules (all must be true in the same processed frame for
    the step to be considered true that frame; see IMPLEMENTATION_PLAN.md
    5.3.2). ``say`` is a short phrase, not a sentence (word-count is checked
    by the experiment lint, not here)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: SnakeCaseStr
    display_name: str
    say: str
    when: list[WhenRule] = Field(min_length=1)


class ExperimentDefinition(BaseModel):
    """``config/experiment.json``. A DRAFT until G2 (IMPLEMENTATION_PLAN.md
    Part 9); changing it follows the contract process (Part 6) because it
    changes what the video goldens expect."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: SnakeCaseStr
    version: str
    classes: list[SnakeCaseStr] = Field(min_length=1)
    steps: list[StepDef] = Field(min_length=1)

    @model_validator(mode="after")
    def _check_structure(self) -> ExperimentDefinition:
        if len(set(self.classes)) != len(self.classes):
            raise ValueError("ExperimentDefinition.classes must be unique")
        ids = [s.step_id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("ExperimentDefinition.steps must have unique step_id values")
        classes = set(self.classes)
        for step in self.steps:
            for rule in step.when:
                unknown = _rule_labels(rule) - classes
                if unknown:
                    raise ValueError(
                        f"step {step.step_id!r} rule references unknown "
                        f"class(es) {sorted(unknown)!r}"
                    )
        return self

    @property
    def step_ids(self) -> list[str]:
        return [s.step_id for s in self.steps]

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentDefinition:
        import json

        return cls.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Section 4 -- Tunable config (F13): PerceptionConfig, RuntimeConfig
# ---------------------------------------------------------------------------


class PerceptionConfig(BaseModel):
    """Disclosed judgment-call defaults (IMPLEMENTATION_PLAN.md 5.3, 5.8),
    tuned on ``val`` only at P2.6, at the real processing rate.
    ``extra="forbid"`` so a misspelled YAML key fails loudly instead of
    silently reverting to a default (ISSUES.md, 2026-09-28 CONTRACT)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector_conf_floor: Unit = 0.30
    confirm_conf: Unit = 0.60
    hysteresis_frames: int = Field(default=5, ge=1)
    release_frames: int = Field(default=5, ge=1)
    baseline_frames: int = Field(default=10, ge=0)
    touch_margin_frac: float = Field(default=0.10, ge=0.0)

    @model_validator(mode="after")
    def _check_floors(self) -> PerceptionConfig:
        if self.confirm_conf < self.detector_conf_floor:
            raise ValueError("confirm_conf must be >= detector_conf_floor")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> PerceptionConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


class RuntimeConfig(BaseModel):
    """``extra="forbid"`` for the same reason as ``PerceptionConfig``
    (ISSUES.md, 2026-09-28 CONTRACT)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = "0"
    target_fps: float = Field(default=15.0, gt=0.0)
    stream_fps: float = Field(default=10.0, gt=0.0)
    jpeg_quality: int = Field(default=80, ge=1, le=100)
    narrate_next_step: bool = True
    alert_on_repeat: bool = True
    alert_cooldown_s: float = Field(default=5.0, ge=0.0)
    feed_timeout_s: float = Field(default=3.0, gt=0.0)
    enable_pose: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8443, ge=1, le=65535)
    allowed_client_ips: list[str] = Field(default_factory=lambda: ["127.0.0.1"])
    tls_cert: str | None = None
    tls_key: str | None = None
    video_dir: str = "runs_out/video"
    log_dir: str = "runs_out/logs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> RuntimeConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


# ---------------------------------------------------------------------------
# Section 5 -- StateTracker output (F4, F13): StateEvent
# ---------------------------------------------------------------------------


class StateEvent(BaseModel):
    """Emitted by ``state/tracker.py``'s ``StateTracker.update`` when an
    armed step's rules have held true for ``hysteresis_frames`` consecutive
    processed frames (IMPLEMENTATION_PLAN.md 5.3.3). ``confidence`` is the
    minimum, over the hysteresis window, of the confidences of the best
    detections the step's rules reference (plus the touching hand's score
    for ``hand_touching``). ``evidence`` is diagnostic only -- the engine
    never branches on it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    t: FiniteFloat
    step_id: SnakeCaseStr
    confidence: Unit
    uncertain: bool
    evidence: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class StateTracker(Protocol):
    """Implemented by ``state/tracker.py``. Pure logic: no models, no clock
    reads (IMPLEMENTATION_PLAN.md rule 8). Time arrives on the data."""

    def update(self, frame: PerceptionFrame) -> list[StateEvent]: ...

    def reset(self) -> None:
        """Starts a fresh baseline window; latches any step already true."""
        ...


# ---------------------------------------------------------------------------
# Section 6 -- Sequence engine output (F5, F6, F8): EngineEvent, RunSummary
# ---------------------------------------------------------------------------

RunState = Literal["idle", "running", "completed"]
StepStatus = Literal["pending", "confirmed", "skipped", "completed_late"]
DeviationType = Literal["omission", "out_of_order", "repeat"]
ConfidenceTag = Literal["confirmed", "flagged_uncertain"]
EngineEventKind = Literal["step_confirmed", "deviation_detected", "run_completed"]
SpeakPriority = Literal["alert", "info"]


class StepProgress(BaseModel):
    """One row of ``Engine.snapshot()`` / ``StatusResponse.steps``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    step_id: SnakeCaseStr
    status: StepStatus
    t_confirmed: FiniteFloat | None = None


class RunSummary(BaseModel):
    """Attached to the ``run_completed`` ``EngineEvent``. ``pos`` (percent of
    sequence correct) = ``1 - min(d / n, 1)`` where ``d`` is the
    Damerau-Levenshtein distance (RapidFuzz) between the canonical step ids
    and ``observed_sequence``, and ``n = len(canonical ids)``. Repeats are
    included in ``observed_sequence``. ``all_steps_done`` is true iff no
    step is ``skipped`` or still ``pending`` when the run ends."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    pos: Unit
    all_steps_done: bool
    aborted: bool = False
    observed_sequence: list[str] = Field(default_factory=list)
    skipped_step_ids: list[str] = Field(default_factory=list)
    late_step_ids: list[str] = Field(default_factory=list)


class EngineEvent(BaseModel):
    """Emitted by ``SequenceEngine.on_state_event`` / ``.finish`` per the
    decision table in IMPLEMENTATION_PLAN.md 5.4. ``expected_step_id`` is
    the first pending step *before* this event; ``next_step_id`` the first
    pending step *after* it. ``speak`` is the exact TTS text (or ``None``),
    matching the plan's templates so the goldens can pin it verbatim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    t: FiniteFloat
    kind: EngineEventKind
    step_id: SnakeCaseStr | None = None
    deviation_type: DeviationType | None = None
    skipped_step_ids: list[str] = Field(default_factory=list)
    expected_step_id: SnakeCaseStr | None = None
    next_step_id: SnakeCaseStr | None = None
    confidence_tag: ConfidenceTag
    speak: str | None = None
    summary: RunSummary | None = None

    @model_validator(mode="after")
    def _check_kind_fields(self) -> EngineEvent:
        if self.kind == "deviation_detected" and self.deviation_type is None:
            raise ValueError("deviation_detected events must set deviation_type")
        if self.kind == "run_completed" and self.summary is None:
            raise ValueError("run_completed events must carry a RunSummary")
        return self


@runtime_checkable
class Engine(Protocol):
    """Implemented by ``engine/sequence.py``'s ``SequenceEngine``.
    Constructed as ``SequenceEngine(experiment: ExperimentDefinition, config:
    RuntimeConfig)``. Raises ``ContractViolation`` for an unknown
    ``step_id``. Never reads a clock; time arrives on ``StateEvent.t`` /
    the ``t`` passed to ``start``/``finish``."""

    run_state: RunState

    def start(self, t: float) -> None: ...

    def on_state_event(self, event: StateEvent) -> list[EngineEvent]: ...

    def finish(self, t: float) -> list[EngineEvent]:
        """Ends the run early: ``run_completed(aborted=True)`` if it was
        running, ``[]`` if it already completed or never started."""
        ...

    def snapshot(self) -> list[StepProgress]: ...


class ExpectedDeviation(BaseModel):
    """One entry of ``RunScript.expected_deviations``. For ``omission``,
    ``step_ids`` is the skipped ids (canonical order); for ``out_of_order``
    / ``repeat`` it is ``[the observed step]``. Derived, never hand-edited --
    see ``engine/reference.py``'s ``derive_expected_deviations``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deviation_type: DeviationType
    step_ids: list[SnakeCaseStr] = Field(min_length=1)


# ---------------------------------------------------------------------------
# Section 7 -- Outputs (F7, F9): Speaker, LogEntry, LogSink
# ---------------------------------------------------------------------------


@runtime_checkable
class Speaker(Protocol):
    """Implemented by ``outputs/tts.py``'s ``TTSWorker`` (a real worker
    process) and by ``FakeSpeaker`` (test double, records calls in order).
    ``say`` never blocks and never raises; an ``alert`` pre-empts whatever is
    currently speaking, an ``info`` is queued."""

    def say(self, text: str, priority: SpeakPriority) -> None: ...

    def close(self) -> None: ...


LogEventType = Literal[
    "run_started",
    "step_confirmed",
    "deviation_detected",
    "run_completed",
    "feed_lost",
    "feed_restored",
]


class LogEntry(BaseModel):
    """One line of ``runs_out/logs/<run_id>.jsonl``, one per *event*, never
    per frame. ``seq`` is a per-run counter starting at 0, owned by the
    router and asserted by the logger (gap -> ``ContractViolation``).
    ``t_wall`` is optional at construction and is stamped by the logger from
    its injected ``wall_clock`` (IMPLEMENTATION_PLAN.md 5.5; ISSUES.md,
    2026-09-28 CONTRACT) -- it is the only wall-clock read in the system, and
    must be timezone-aware once set. ``t_video`` is the originating event's
    ``t``."""

    model_config = ConfigDict(extra="forbid")

    seq: int = Field(ge=0)
    run_id: str
    t_wall: datetime | None = None
    t_video: FiniteFloat
    event_type: LogEventType
    step_id: SnakeCaseStr | None = None
    expected_step_id: SnakeCaseStr | None = None
    deviation_type: DeviationType | None = None
    skipped_step_ids: list[str] = Field(default_factory=list)
    confidence_tag: ConfidenceTag | None = None
    detail: str

    @field_validator("t_wall")
    @classmethod
    def _check_tz_aware(cls, v: datetime | None) -> datetime | None:
        if v is not None and v.tzinfo is None:
            raise ValueError("LogEntry.t_wall must be timezone-aware when set")
        return v


@runtime_checkable
class LogSink(Protocol):
    """Implemented by ``outputs/logger.py``'s ``JsonlLogger(path,
    wall_clock=...)``."""

    def write(self, entry: LogEntry) -> None: ...


# ---------------------------------------------------------------------------
# Section 8 -- HTTP surface (F10, F12): API_ROUTES, StatusResponse
# ---------------------------------------------------------------------------

API_ROUTES: dict[str, frozenset[str]] = {
    "/": frozenset({"GET"}),
    "/video_feed": frozenset({"GET"}),
    "/api/status": frozenset({"GET"}),
    "/api/experiment": frozenset({"GET"}),
    "/api/log": frozenset({"GET"}),
    "/api/run/start": frozenset({"POST"}),
    "/api/run/reset": frozenset({"POST"}),
}
"""The complete Flask URL map for ``server/app.py``. Every route requires
Basic auth AND a client-IP allowlist check (IMPLEMENTATION_PLAN.md 5.6). A
test asserts the live Flask app's URL map equals this dict exactly, so the
routes can never silently drift from the contract."""


class StatusResponse(BaseModel):
    """Body of ``GET /api/status``, built from ``Engine.snapshot()`` plus
    runtime feed/fps state. ``recent_alerts`` is newest-last, capped at
    ``RECENT_ALERTS_CAP``. Polled by the dashboard every
    ``DASHBOARD_POLL_MS``."""

    model_config = ConfigDict(extra="forbid")

    run_id: str | None
    run_state: RunState
    feed_ok: bool
    fps: float | None = None
    steps: list[StepProgress]
    expected_step_id: SnakeCaseStr | None = None
    next_step_id: SnakeCaseStr | None = None
    next_step_say: str | None = None
    recent_alerts: list[EngineEvent] = Field(default_factory=list, max_length=RECENT_ALERTS_CAP)
    generated_at: datetime


class RunControlResponse(BaseModel):
    """Body of ``POST /api/run/start`` and ``POST /api/run/reset``."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    run_id: str
    run_state: RunState


# ---------------------------------------------------------------------------
# Section 9 -- Recorded-run shape (F14): RunScript
# ---------------------------------------------------------------------------

ScriptType = Literal["correct", "skip", "swap", "repeat", "idle", "robustness"]
Split = Literal["train", "val", "test"]


class RunScript(BaseModel):
    """``runs/<run_id>/script.json``, written by ``perception/record.py``.
    ``performed_steps`` is the performer's declaration -- it IS ground truth
    (no per-frame labeling). ``expected_deviations`` is derived, never
    hand-edited (``engine/reference.py``); a contract test fails if any
    file is stale relative to the reference engine."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    experiment_id: str
    script_type: ScriptType
    split: Split
    fps: FiniteFloat
    camera_setup_id: str
    operator: str
    performed_steps: list[SnakeCaseStr] = Field(default_factory=list)
    expected_deviations: list[ExpectedDeviation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Section 10 -- The small file seams (config/acceptance.yaml, reports/*.json)
#
# Added at G0 per ISSUES.md 2026-09-28 CONTRACT ("the small file seams are
# not typed"): AcceptanceConfig and ReportHeader. Report bodies otherwise
# stay dicts (IMPLEMENTATION_PLAN.md 5.8) -- only the header is pinned.
# ---------------------------------------------------------------------------


class DetectorAcceptance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    min_recall_per_class: Unit
    min_map50: Unit


class PipelineAcceptance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    min_pipeline_fps: float = Field(gt=0.0)


class ReplayAcceptance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    exact_deviation_match: bool
    max_mismatched_runs: int = Field(ge=0)


class LabelReviewAcceptance(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    max_bad_fraction_per_class: Unit


class AcceptanceConfig(BaseModel):
    """``config/acceptance.yaml``. Written by P1 at P1.5, amended by P2 at
    P2.6, each noted in ISSUES.md (IMPLEMENTATION_PLAN.md Part 8)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector: DetectorAcceptance
    pipeline: PipelineAcceptance
    replay: ReplayAcceptance
    label_review: LabelReviewAcceptance

    @classmethod
    def from_yaml(cls, path: str | Path) -> AcceptanceConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


class ReportHeader(BaseModel):
    """The common header every ``reports/*.json`` file carries
    (IMPLEMENTATION_PLAN.md 5.8): ``generated_at``, the ``model_stamp``
    where relevant, and the stamps of its inputs. Report bodies are
    feature-specific dicts alongside this header, not modeled here."""

    model_config = ConfigDict(extra="forbid")

    generated_at: datetime
    model_stamp: str | None = None
    input_stamps: dict[str, str] = Field(default_factory=dict)

    @field_validator("generated_at")
    @classmethod
    def _check_tz_aware(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("ReportHeader.generated_at must be timezone-aware")
        return v
