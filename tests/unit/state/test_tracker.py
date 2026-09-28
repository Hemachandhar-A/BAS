"""state/tracker.py tests (P2.1; IMPLEMENTATION_PLAN.md Part 10, 5.3; F4, F13).

All hand-built PerceptionFrames -- no models, no cache, no clock. Each test
builds a tiny throwaway ExperimentDefinition (never config/experiment.json,
which is P2.2's fixture territory) so rule-kind and lifecycle behavior is
isolated from the real experiment's shape.
"""

from __future__ import annotations

import pytest

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
    StepDef,
)
from state.tracker import StateTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _experiment(rule, classes: list[str]) -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_id="fixture",
        version="0",
        classes=classes,
        steps=[StepDef(step_id="s1", display_name="S1", say="Do it", when=[rule])],
    )


def _two_step_experiment(rule_a, rule_b, classes: list[str]) -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_id="fixture",
        version="0",
        classes=classes,
        steps=[
            StepDef(step_id="s1", display_name="S1", say="Do it", when=[rule_a]),
            StepDef(step_id="s2", display_name="S2", say="Do it too", when=[rule_b]),
        ],
    )


def _frame(t: float, detections=(), hands=()) -> PerceptionFrame:
    return PerceptionFrame(
        frame_id=int(round(t * 1000)), t=t, detections=list(detections), hands=list(hands)
    )


def _det(label: str, conf: float, box) -> Detection:
    return Detection(label=label, conf=conf, box=box)


def _hand(landmarks_px, score: float = 0.9, handedness: str = "Right") -> Hand:
    pts = list(landmarks_px) + [landmarks_px[-1]] * (21 - len(landmarks_px))
    return Hand(handedness=handedness, score=score, landmarks_px=pts)


IMMEDIATE = PerceptionConfig(
    baseline_frames=0,
    hysteresis_frames=1,
    release_frames=1,
    detector_conf_floor=0.30,
    confirm_conf=0.60,
    touch_margin_frac=0.10,
)

INSIDE_OUTER = (0.0, 0.0, 100.0, 100.0)
INSIDE_BOX = (10.0, 10.0, 20.0, 20.0)  # centroid (15, 15) -- inside INSIDE_OUTER
OUTSIDE_BOX = (200.0, 200.0, 210.0, 210.0)  # centroid (205, 205) -- outside INSIDE_OUTER


def _fires(
    experiment: ExperimentDefinition, config: PerceptionConfig, frame: PerceptionFrame
) -> bool:
    tracker = StateTracker(experiment, config)
    return len(tracker.update(frame)) == 1


# ---------------------------------------------------------------------------
# F4 -- the five WhenRule kinds, table-driven (best detection, grown box,
# missing container, absent/present)
# ---------------------------------------------------------------------------


@pytest.mark.F4
def test_inside_true_when_centroid_in_container() -> None:
    exp = _experiment(InsideRule(label="box", container="tray"), ["box", "tray"])
    frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])
    assert _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_inside_false_when_centroid_outside_container() -> None:
    exp = _experiment(InsideRule(label="box", container="tray"), ["box", "tray"])
    frame = _frame(0.0, [_det("box", 0.9, OUTSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])
    assert not _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
@pytest.mark.parametrize(
    "present_labels",
    [["box"], ["tray"], []],
    ids=["container_missing", "label_missing", "both_missing"],
)
def test_inside_false_when_either_detection_missing(present_labels: list[str]) -> None:
    exp = _experiment(InsideRule(label="box", container="tray"), ["box", "tray"])
    dets = []
    if "box" in present_labels:
        dets.append(_det("box", 0.9, INSIDE_BOX))
    if "tray" in present_labels:
        dets.append(_det("tray", 0.9, INSIDE_OUTER))
    assert not _fires(exp, IMMEDIATE, _frame(0.0, dets))


@pytest.mark.F4
def test_best_detection_is_highest_confidence_not_first_in_list() -> None:
    # First-listed detection says "inside" but is lower-confidence; the
    # higher-confidence (second-listed) detection says "outside" and must
    # be the one that decides the rule (IMPLEMENTATION_PLAN.md 5.3.1).
    exp = _experiment(InsideRule(label="box", container="tray"), ["box", "tray"])
    frame = _frame(
        0.0,
        [
            _det("box", 0.40, INSIDE_BOX),
            _det("box", 0.95, OUTSIDE_BOX),
            _det("tray", 0.9, INSIDE_OUTER),
        ],
    )
    assert not _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_best_detection_tie_breaks_first_in_list() -> None:
    # Equal confidence: the first-listed detection wins (IMPLEMENTATION_PLAN.md 5.3.1).
    exp = _experiment(InsideRule(label="box", container="tray"), ["box", "tray"])
    frame = _frame(
        0.0,
        [
            _det("box", 0.9, INSIDE_BOX),
            _det("box", 0.9, OUTSIDE_BOX),
            _det("tray", 0.9, INSIDE_OUTER),
        ],
    )
    assert _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_outside_true_when_not_inside_container() -> None:
    exp = _experiment(OutsideRule(label="box", container="tray"), ["box", "tray"])
    frame = _frame(0.0, [_det("box", 0.9, OUTSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])
    assert _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_outside_false_when_inside_container() -> None:
    exp = _experiment(OutsideRule(label="box", container="tray"), ["box", "tray"])
    frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])
    assert not _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_outside_vacuously_true_when_container_missing() -> None:
    exp = _experiment(OutsideRule(label="box", container="tray"), ["box", "tray"])
    frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX)])  # no tray detection at all
    assert _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_outside_false_when_label_missing() -> None:
    exp = _experiment(OutsideRule(label="box", container="tray"), ["box", "tray"])
    frame = _frame(0.0, [_det("tray", 0.9, INSIDE_OUTER)])
    assert not _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_hand_touching_true_within_grown_margin() -> None:
    # box (0,0,10,10), touch_margin_frac=0.10 -> grown to (-1,-1,11,11).
    exp = _experiment(HandTouchingRule(label="button"), ["button"])
    frame = _frame(
        0.0,
        [_det("button", 0.9, (0.0, 0.0, 10.0, 10.0))],
        [_hand([(10.5, 5.0)])],
    )
    assert _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_hand_touching_false_outside_grown_margin() -> None:
    exp = _experiment(HandTouchingRule(label="button"), ["button"])
    frame = _frame(
        0.0,
        [_det("button", 0.9, (0.0, 0.0, 10.0, 10.0))],
        [_hand([(11.5, 5.0)])],
    )
    assert not _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_hand_touching_false_when_no_hands() -> None:
    exp = _experiment(HandTouchingRule(label="button"), ["button"])
    frame = _frame(0.0, [_det("button", 0.9, (0.0, 0.0, 10.0, 10.0))], [])
    assert not _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_hand_touching_false_when_label_missing() -> None:
    exp = _experiment(HandTouchingRule(label="button"), ["button"])
    frame = _frame(0.0, [], [_hand([(0.5, 0.5)])])
    assert not _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_absent_true_when_never_detected() -> None:
    exp = _experiment(AbsentRule(label="box"), ["box"])
    assert _fires(exp, IMMEDIATE, _frame(0.0, []))


@pytest.mark.F4
def test_absent_false_when_detected_above_floor() -> None:
    exp = _experiment(AbsentRule(label="box"), ["box"])
    frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX)])
    assert not _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_present_true_when_detected_above_floor() -> None:
    exp = _experiment(PresentRule(label="box"), ["box"])
    frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX)])
    assert _fires(exp, IMMEDIATE, frame)


@pytest.mark.F4
def test_present_false_when_never_detected() -> None:
    exp = _experiment(PresentRule(label="box"), ["box"])
    assert not _fires(exp, IMMEDIATE, _frame(0.0, []))


# ---------------------------------------------------------------------------
# F13 -- floor, confirm, hysteresis, release, baseline
# ---------------------------------------------------------------------------


@pytest.mark.F13
def test_below_floor_detection_is_treated_as_absent() -> None:
    # detector_conf_floor=0.30: a 0.20 detection must be ignored entirely,
    # so `present` reads false and `absent` reads true.
    config = PerceptionConfig(baseline_frames=0, hysteresis_frames=1, release_frames=1)
    present_exp = _experiment(PresentRule(label="box"), ["box"])
    absent_exp = _experiment(AbsentRule(label="box"), ["box"])
    frame = _frame(0.0, [_det("box", 0.20, INSIDE_BOX)])
    assert not _fires(present_exp, config, frame)
    assert _fires(absent_exp, config, frame)


@pytest.mark.F13
def test_below_floor_detection_cannot_satisfy_inside() -> None:
    config = PerceptionConfig(baseline_frames=0, hysteresis_frames=1, release_frames=1)
    exp = _experiment(InsideRule(label="box", container="tray"), ["box", "tray"])
    frame = _frame(0.0, [_det("box", 0.10, INSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])
    assert not _fires(exp, config, frame)


@pytest.mark.F13
def test_uncertain_flag_true_between_floor_and_confirm_conf() -> None:
    config = PerceptionConfig(
        baseline_frames=0, hysteresis_frames=2, release_frames=2, confirm_conf=0.60
    )
    exp = _experiment(InsideRule(label="box", container="tray"), ["box", "tray"])
    tracker = StateTracker(exp, config)
    frame = _frame(0.0, [_det("box", 0.45, INSIDE_BOX), _det("tray", 0.45, INSIDE_OUTER)])
    assert tracker.update(frame) == []
    events = tracker.update(frame)
    assert len(events) == 1
    assert events[0].confidence == pytest.approx(0.45)
    assert events[0].uncertain is True


@pytest.mark.F13
def test_uncertain_flag_false_above_confirm_conf() -> None:
    config = PerceptionConfig(
        baseline_frames=0, hysteresis_frames=2, release_frames=2, confirm_conf=0.60
    )
    exp = _experiment(InsideRule(label="box", container="tray"), ["box", "tray"])
    tracker = StateTracker(exp, config)
    frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])
    tracker.update(frame)
    events = tracker.update(frame)
    assert len(events) == 1
    assert events[0].confidence == pytest.approx(0.9)
    assert events[0].uncertain is False


@pytest.mark.F13
def test_hysteresis_requires_consecutive_frames() -> None:
    config = PerceptionConfig(baseline_frames=0, hysteresis_frames=3, release_frames=1)
    exp = _experiment(PresentRule(label="box"), ["box"])
    tracker = StateTracker(exp, config)
    true_frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX)])
    assert tracker.update(true_frame) == []
    assert tracker.update(true_frame) == []
    events = tracker.update(true_frame)
    assert len(events) == 1
    assert events[0].step_id == "s1"


@pytest.mark.F13
def test_one_frame_glitch_never_fires() -> None:
    config = PerceptionConfig(baseline_frames=0, hysteresis_frames=3, release_frames=1)
    exp = _experiment(PresentRule(label="box"), ["box"])
    tracker = StateTracker(exp, config)
    true_frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX)])
    false_frame = _frame(0.0, [])
    # true, true, FALSE, true, true, FALSE, true, true -- never 3 in a row.
    pattern = [
        true_frame,
        true_frame,
        false_frame,
        true_frame,
        true_frame,
        false_frame,
        true_frame,
        true_frame,
    ]
    all_events = [ev for frame in pattern for ev in tracker.update(frame)]
    assert all_events == []


@pytest.mark.F13
def test_release_requires_consecutive_false_frames_before_rearm() -> None:
    config = PerceptionConfig(baseline_frames=0, hysteresis_frames=1, release_frames=2)
    exp = _experiment(PresentRule(label="box"), ["box"])
    tracker = StateTracker(exp, config)
    true_frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX)])
    false_frame = _frame(0.0, [])

    assert len(tracker.update(true_frame)) == 1  # fires once, disarms

    assert tracker.update(true_frame) == []  # still true: not armed, no refire
    assert tracker.update(false_frame) == []  # false streak = 1 (< release_frames)
    assert tracker.update(true_frame) == []  # streak reset to 0, still not armed
    assert tracker.update(false_frame) == []  # false streak = 1
    assert tracker.update(false_frame) == []  # false streak = 2 -> re-arms (this frame is false)
    assert len(tracker.update(true_frame)) == 1  # armed again, fires immediately (hysteresis=1)


@pytest.mark.F13
def test_baseline_latches_a_step_already_true_and_suppresses_events() -> None:
    config = PerceptionConfig(baseline_frames=3, hysteresis_frames=2, release_frames=2)
    exp = _experiment(InsideRule(label="box", container="tray"), ["box", "tray"])
    tracker = StateTracker(exp, config)
    inside_frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])

    # Baseline (3 frames): rule is true throughout -> latched at frame 1, no events.
    for _ in range(3):
        assert tracker.update(inside_frame) == []

    # Still true after baseline, but latched -- must not fire just because
    # hysteresis_frames worth of true frames have now elapsed.
    for _ in range(5):
        assert tracker.update(inside_frame) == []


@pytest.mark.F13
def test_baseline_true_step_fires_only_after_leave_and_return() -> None:
    # ISSUES.md, 2026-09-28 CONTRACT: a true-at-baseline step must fire only
    # after it has gone false for release_frames and then true for
    # hysteresis_frames (the mechanism the experiment lint depends on).
    config = PerceptionConfig(baseline_frames=2, hysteresis_frames=2, release_frames=2)
    exp = _experiment(InsideRule(label="box", container="tray"), ["box", "tray"])
    tracker = StateTracker(exp, config)
    inside_frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])
    outside_frame = _frame(0.0, [_det("box", 0.9, OUTSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])

    # Baseline: true both frames -> latched.
    assert tracker.update(inside_frame) == []
    assert tracker.update(inside_frame) == []

    # Still latched post-baseline while true.
    assert tracker.update(inside_frame) == []

    # Leave: release_frames=2 consecutive false frames re-arms it.
    assert tracker.update(outside_frame) == []
    assert tracker.update(outside_frame) == []

    # Return: hysteresis_frames=2 consecutive true frames now fires.
    assert tracker.update(inside_frame) == []
    events = tracker.update(inside_frame)
    assert len(events) == 1
    assert events[0].step_id == "s1"


@pytest.mark.F13
def test_baseline_false_step_arms_normally_after_baseline() -> None:
    config = PerceptionConfig(baseline_frames=2, hysteresis_frames=2, release_frames=2)
    exp = _experiment(PresentRule(label="box"), ["box"])
    tracker = StateTracker(exp, config)
    false_frame = _frame(0.0, [])
    true_frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX)])

    assert tracker.update(false_frame) == []  # baseline frame 1
    assert tracker.update(false_frame) == []  # baseline frame 2
    assert tracker.update(true_frame) == []  # hysteresis count 1
    events = tracker.update(true_frame)  # hysteresis count 2 -> fires
    assert len(events) == 1


@pytest.mark.F13
def test_absent_rule_latches_at_baseline_like_any_other_rule() -> None:
    config = PerceptionConfig(baseline_frames=2, hysteresis_frames=1, release_frames=1)
    exp = _experiment(AbsentRule(label="box"), ["box"])
    tracker = StateTracker(exp, config)
    no_box = _frame(0.0, [])
    with_box = _frame(0.0, [_det("box", 0.9, INSIDE_BOX)])

    # box never shows up during baseline -> absent(box) is true throughout,
    # latched, and must not fire post-baseline while it stays true.
    assert tracker.update(no_box) == []
    assert tracker.update(no_box) == []
    assert tracker.update(no_box) == []

    # box appears (rule goes false) for release_frames, then leaves again.
    assert tracker.update(with_box) == []
    events = tracker.update(no_box)
    assert len(events) == 1


# ---------------------------------------------------------------------------
# Same-frame ties, confidence edge case, determinism
# ---------------------------------------------------------------------------


@pytest.mark.F13
def test_same_frame_ties_emit_in_canonical_step_order() -> None:
    config = PerceptionConfig(baseline_frames=0, hysteresis_frames=1, release_frames=1)
    exp = _two_step_experiment(
        PresentRule(label="a"), PresentRule(label="b"), ["a", "b"]
    )
    tracker = StateTracker(exp, config)
    frame = _frame(0.0, [_det("a", 0.9, INSIDE_BOX), _det("b", 0.9, INSIDE_BOX)])
    events = tracker.update(frame)
    assert [ev.step_id for ev in events] == ["s1", "s2"]


@pytest.mark.F13
def test_confidence_defaults_to_one_when_no_detection_referenced() -> None:
    # DECISION (ISSUES.md, P2.1): an absent-only step's window references no
    # detection at all, so confidence defaults to 1.0.
    config = PerceptionConfig(baseline_frames=0, hysteresis_frames=1, release_frames=1)
    exp = _experiment(AbsentRule(label="box"), ["box"])
    tracker = StateTracker(exp, config)
    events = tracker.update(_frame(0.0, []))
    assert len(events) == 1
    assert events[0].confidence == 1.0
    assert events[0].uncertain is False


def test_tracker_is_deterministic() -> None:
    config = PerceptionConfig(baseline_frames=2, hysteresis_frames=2, release_frames=2)
    exp = _two_step_experiment(
        InsideRule(label="box", container="tray"),
        HandTouchingRule(label="button"),
        ["box", "tray", "button"],
    )

    def build_frames() -> list[PerceptionFrame]:
        inside = _frame(0.0, [_det("box", 0.9, INSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])
        outside = _frame(0.0, [_det("box", 0.9, OUTSIDE_BOX), _det("tray", 0.9, INSIDE_OUTER)])
        touch = _frame(
            0.0,
            [_det("button", 0.9, (0.0, 0.0, 10.0, 10.0))],
            [_hand([(5.0, 5.0)])],
        )
        return [inside, inside, outside, outside, touch, touch, inside, inside]

    def run() -> list[list[dict]]:
        tracker = StateTracker(exp, config)
        return [[ev.model_dump() for ev in tracker.update(f)] for f in build_frames()]

    assert run() == run()


def test_reset_starts_a_fresh_baseline_window() -> None:
    config = PerceptionConfig(baseline_frames=1, hysteresis_frames=1, release_frames=1)
    exp = _experiment(PresentRule(label="box"), ["box"])
    tracker = StateTracker(exp, config)
    true_frame = _frame(0.0, [_det("box", 0.9, INSIDE_BOX)])

    assert tracker.update(true_frame) == []  # baseline frame -> latched
    tracker.reset()
    assert tracker.update(true_frame) == []  # baseline again after reset -> latched, not firing
