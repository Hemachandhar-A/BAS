"""hand_touching is fingertip-only (D74; ISSUES.md 2026-10-02 P2 DECISION).

True iff at least one fingertip landmark (MediaPipe 4, 8, 12, 16, 20) of any
hand lies inside the label's best box grown by ``touch_margin_frac``. Wrist,
palm and the other landmarks never count (the arm-crossing case). Hand-built
frames only; no models, no clock.
"""

from __future__ import annotations

import pytest

from contracts import (
    Detection,
    ExperimentDefinition,
    Hand,
    HandTouchingRule,
    PerceptionConfig,
    PerceptionFrame,
    StepDef,
)
from state.tracker import StateTracker

FINGERTIP_LANDMARKS = (4, 8, 12, 16, 20)  # MediaPipe fingertip indices

BOX = (0.0, 0.0, 10.0, 10.0)  # grown by 0.10 -> (-1, -1, 11, 11)
FAR = (-1000.0, -1000.0)
IN = (5.0, 5.0)
WRIST, PALM = 0, 9

IMMEDIATE = PerceptionConfig(
    baseline_frames=0,
    hysteresis_frames=1,
    release_frames=1,
    detector_conf_floor=0.30,
    confirm_conf=0.60,
    touch_margin_frac=0.10,
)


def _experiment() -> ExperimentDefinition:
    return ExperimentDefinition(
        experiment_id="fixture",
        version="0",
        classes=["button"],
        steps=[
            StepDef(
                step_id="s1",
                display_name="S1",
                say="Do it",
                when=[HandTouchingRule(label="button")],
            )
        ],
    )


def _hand(at: dict[int, tuple[float, float]], score: float = 0.9) -> Hand:
    """All 21 landmarks far away except those named in ``at``."""
    pts = [at.get(i, FAR) for i in range(21)]
    return Hand(handedness="Right", score=score, landmarks_px=pts)


def _frame(t: float, hands=(), conf: float = 0.9, with_button: bool = True) -> PerceptionFrame:
    dets = [Detection(label="button", conf=conf, box=BOX)] if with_button else []
    return PerceptionFrame(frame_id=int(round(t * 1000)), t=t, detections=dets, hands=list(hands))


def _events(frame: PerceptionFrame, config: PerceptionConfig = IMMEDIATE):
    return StateTracker(_experiment(), config).update(frame)


def test_fingertip_constant_is_the_mediapipe_tips() -> None:
    from state.tracker import FINGERTIP_LANDMARKS as impl

    assert impl == FINGERTIP_LANDMARKS


@pytest.mark.F4
def test_fingertip_inside_grown_box_is_true() -> None:
    assert len(_events(_frame(0.0, [_hand({8: IN})]))) == 1


@pytest.mark.F4
@pytest.mark.parametrize("idx", [WRIST, PALM])
def test_non_fingertip_inside_with_fingertips_outside_is_false(idx: int) -> None:
    # The arm-crossing case: wrist or palm over the card, every fingertip away.
    assert _events(_frame(0.0, [_hand({idx: IN})])) == []


@pytest.mark.F4
def test_all_non_fingertip_landmarks_never_count() -> None:
    others = {i: IN for i in range(21) if i not in FINGERTIP_LANDMARKS}
    assert _events(_frame(0.0, [_hand(others)])) == []


@pytest.mark.F4
@pytest.mark.parametrize("idx", [4, 8, 12, 16, 20])
def test_each_fingertip_index_alone_is_true(idx: int) -> None:
    assert len(_events(_frame(0.0, [_hand({idx: IN})]))) == 1


@pytest.mark.F4
def test_grown_edge_is_inclusive_and_just_past_is_false() -> None:
    # Grown right edge is exactly x = 11.0; the box test is inclusive.
    assert len(_events(_frame(0.0, [_hand({8: (11.0, 5.0)})]))) == 1
    assert _events(_frame(0.0, [_hand({8: (11.01, 5.0)})])) == []
    assert len(_events(_frame(0.0, [_hand({8: (-1.0, -1.0)})]))) == 1
    assert _events(_frame(0.0, [_hand({8: (-1.01, 5.0)})])) == []


@pytest.mark.F4
def test_two_hands_only_one_with_fingertip_inside_is_true() -> None:
    palm_only = _hand({PALM: IN})
    tip = _hand({12: IN})
    assert len(_events(_frame(0.0, [palm_only, tip]))) == 1
    assert len(_events(_frame(0.0, [tip, palm_only]))) == 1


@pytest.mark.F4
def test_no_hands_is_false() -> None:
    assert _events(_frame(0.0, [])) == []


@pytest.mark.F4
def test_label_undetected_or_below_floor_is_false() -> None:
    hand = _hand({8: IN})
    assert _events(_frame(0.0, [hand], with_button=False)) == []
    assert _events(_frame(0.0, [hand], conf=0.29)) == []


@pytest.mark.F4
def test_touching_hand_score_is_in_event_confidence() -> None:
    events = _events(_frame(0.0, [_hand({8: IN}, score=0.5)], conf=0.9))
    assert len(events) == 1
    assert events[0].confidence == pytest.approx(0.5)
    assert events[0].uncertain is True


@pytest.mark.F4
def test_touching_hand_is_the_one_with_the_fingertip() -> None:
    # The score comes from the hand whose fingertip is inside, not the palm-only hand.
    palm_only = _hand({PALM: IN}, score=0.31)
    tip = _hand({8: IN}, score=0.95)
    events = _events(_frame(0.0, [palm_only, tip]))
    assert events[0].confidence == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# End to end through the lifecycle (hysteresis_frames = 5, the default)
# ---------------------------------------------------------------------------

HOLD = PerceptionConfig(baseline_frames=0, hysteresis_frames=5, release_frames=5)


def _run(frames: list[PerceptionFrame]):
    tracker = StateTracker(_experiment(), HOLD)
    out = []
    for f in frames:
        out.extend(tracker.update(f))
    return out


@pytest.mark.F4
def test_palm_crossing_for_many_frames_fires_nothing() -> None:
    frames = [_frame(i / 8, [_hand({PALM: IN, WRIST: IN})]) for i in range(60)]
    assert _run(frames) == []


@pytest.mark.F4
def test_fingertip_held_hysteresis_frames_fires_once() -> None:
    frames = [_frame(i / 8, [_hand({8: IN})]) for i in range(8)]
    assert len(_run(frames)) == 1


@pytest.mark.F4
def test_fingertip_one_frame_short_of_hysteresis_fires_nothing() -> None:
    frames = [_frame(i / 8, [_hand({8: IN})]) for i in range(4)]
    frames.append(_frame(0.5, [_hand({PALM: IN})]))  # palm only: the run is broken
    assert _run(frames) == []
