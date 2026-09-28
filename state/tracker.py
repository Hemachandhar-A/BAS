"""state/tracker.py -- StateTracker (F4, F13; IMPLEMENTATION_PLAN.md 5.3).

Pure logic: no models, no clock reads (AGENTS.md rule 8). Time arrives on
``PerceptionFrame.t``; everything here is a function of the sequence of
frames passed to ``update()``.

Per-frame algorithm (IMPLEMENTATION_PLAN.md 5.3):
1. Floor-filter detections; "best" detection per label is the
   highest-confidence one, ties broken by first-in-list.
2. A step's truth is the AND of its ``when`` rules (contracts.py section 3
   semantics).
3. Lifecycle per step, counted in received frames: the first
   ``baseline_frames`` frames are baseline (no events; a step already true
   is latched); a step is armed unless latched or already fired; a latched
   or fired step re-arms after ``release_frames`` consecutive false frames;
   an armed step whose truth holds for ``hysteresis_frames`` consecutive
   frames emits one ``StateEvent`` and disarms.
4. ``confidence`` is the minimum, over the hysteresis window, of the
   confidences of the best detections the step's rules reference (plus the
   touching hand's score for ``hand_touching``); ``uncertain = confidence <
   confirm_conf``.
5. Same-frame ties emit in canonical experiment order (the order steps
   appear in ``ExperimentDefinition.steps``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from contracts import (
    AbsentRule,
    Detection,
    ExperimentDefinition,
    Hand,
    HandTouchingRule,
    InsideRule,
    OutsideRule,
    PerceptionConfig,
    PerceptionFrame,
    PresentRule,
    StateEvent,
    StepDef,
    WhenRule,
)

Box = tuple[float, float, float, float]


def _floor_filter(detections: list[Detection], floor: float) -> list[Detection]:
    return [d for d in detections if d.conf >= floor]


def _best_detection(label: str, detections: list[Detection]) -> Detection | None:
    best: Detection | None = None
    for d in detections:
        if d.label != label:
            continue
        if best is None or d.conf > best.conf:
            best = d
    return best


def _centroid(box: Box) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _point_in_box(point: tuple[float, float], box: Box) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2


def _grow_box(box: Box, frac: float) -> Box:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - frac * w, y1 - frac * h, x2 + frac * w, y2 + frac * h)


def _evaluate_rule(
    rule: WhenRule,
    detections: list[Detection],
    hands: list[Hand],
    config: PerceptionConfig,
) -> tuple[bool, list[float]]:
    """Returns (truth, confidences_referenced_this_frame). ``detections`` is
    already floor-filtered. Semantics per contracts.py section 3."""

    if isinstance(rule, InsideRule):
        label_d = _best_detection(rule.label, detections)
        container_d = _best_detection(rule.container, detections)
        if label_d is None or container_d is None:
            return False, []
        truth = _point_in_box(_centroid(label_d.box), container_d.box)
        return truth, ([label_d.conf, container_d.conf] if truth else [])

    if isinstance(rule, OutsideRule):
        label_d = _best_detection(rule.label, detections)
        if label_d is None:
            return False, []
        container_d = _best_detection(rule.container, detections)
        if container_d is None:
            return True, [label_d.conf]  # vacuously true: container not there
        truth = not _point_in_box(_centroid(label_d.box), container_d.box)
        return truth, ([label_d.conf, container_d.conf] if truth else [])

    if isinstance(rule, HandTouchingRule):
        label_d = _best_detection(rule.label, detections)
        if label_d is None:
            return False, []
        grown = _grow_box(label_d.box, config.touch_margin_frac)
        for hand in hands:
            for lm in hand.landmarks_px:
                if _point_in_box(lm, grown):
                    return True, [label_d.conf, hand.score]
        return False, []

    if isinstance(rule, AbsentRule):
        label_d = _best_detection(rule.label, detections)
        return label_d is None, []

    if isinstance(rule, PresentRule):
        label_d = _best_detection(rule.label, detections)
        if label_d is None:
            return False, []
        return True, [label_d.conf]

    raise AssertionError(f"unhandled WhenRule kind: {rule!r}")  # pragma: no cover


def _evaluate_step(
    step: StepDef,
    detections: list[Detection],
    hands: list[Hand],
    config: PerceptionConfig,
) -> tuple[bool, list[float]]:
    refs: list[float] = []
    for rule in step.when:
        truth, rule_refs = _evaluate_rule(rule, detections, hands, config)
        if not truth:
            return False, []
        refs.extend(rule_refs)
    return True, refs


@dataclass
class _StepState:
    armed: bool = True
    hysteresis_count: int = 0
    false_streak: int = 0
    window_refs: list[float] = field(default_factory=list)


class StateTracker:
    """Implements ``contracts.StateTracker`` (duck-typed; ``runtime_checkable``).

    ``confidence`` for a step whose rules never reference a detection across
    the whole hysteresis window (a bare ``absent``-only step) defaults to
    ``1.0`` -- an underspecified case in essential-features.md/Plan 5.3,
    recorded as a DECISION in ISSUES.md (P2.1).
    """

    def __init__(self, experiment: ExperimentDefinition, config: PerceptionConfig) -> None:
        self._experiment = experiment
        self._config = config
        self._frames_seen = 0
        self._states: dict[str, _StepState] = {}
        self.reset()

    def reset(self) -> None:
        """Starts a fresh baseline window; latches any step already true."""
        self._frames_seen = 0
        self._states = {step.step_id: _StepState() for step in self._experiment.steps}

    def update(self, frame: PerceptionFrame) -> list[StateEvent]:
        self._frames_seen += 1
        in_baseline = self._frames_seen <= self._config.baseline_frames
        detections = _floor_filter(frame.detections, self._config.detector_conf_floor)

        events: list[StateEvent] = []
        for step in self._experiment.steps:  # canonical order -> same-frame ties order
            state = self._states[step.step_id]
            truth, refs = _evaluate_step(step, detections, frame.hands, self._config)

            if state.armed:
                if not truth:
                    state.hysteresis_count = 0
                    state.window_refs = []
                    continue
                if in_baseline:
                    # Already true during baseline: latch, no events emitted.
                    state.armed = False
                    state.hysteresis_count = 0
                    state.window_refs = []
                    state.false_streak = 0
                    continue
                state.hysteresis_count += 1
                state.window_refs.extend(refs)
                if state.hysteresis_count >= self._config.hysteresis_frames:
                    confidence = min(state.window_refs) if state.window_refs else 1.0
                    events.append(
                        StateEvent(
                            t=frame.t,
                            step_id=step.step_id,
                            confidence=confidence,
                            uncertain=confidence < self._config.confirm_conf,
                            evidence={"window_refs": list(state.window_refs)},
                        )
                    )
                    state.armed = False
                    state.hysteresis_count = 0
                    state.window_refs = []
                    state.false_streak = 0
            else:
                if truth:
                    state.false_streak = 0
                else:
                    state.false_streak += 1
                    if state.false_streak >= self._config.release_frames:
                        state.armed = True
                        state.false_streak = 0
                        state.hysteresis_count = 0
                        state.window_refs = []

        return events
