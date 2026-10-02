"""Unit tests for the pure P1.2 addendum helpers (training/spikes/touch_variants.py):
START touch variants, the tracker-lifecycle press counter, hold-in-seconds
conversion and exact/missed/extra classification."""

from __future__ import annotations

import pytest

from contracts import Hand
from state.tracker import StateTracker
from training.spikes.touch_variants import (
    VARIANTS,
    classify_count,
    hold_frames,
    press_events,
    tally_labels,
    touch_flags,
    touch_flags_from_points,
)

BOX = (100.0, 100.0, 200.0, 180.0)  # w=100, h=80


def hand_with(point: tuple[float, float], idx: int = 8) -> Hand:
    pts = [(0.0, 0.0)] * 21
    pts[idx] = point
    return Hand(handedness="Left", score=0.9, landmarks_px=pts)


def test_variants_registry() -> None:
    assert set(VARIANTS) == {"V0", "V1a", "V1b", "V1c", "V2"}
    assert VARIANTS["V0"].margin == 0.10 and VARIANTS["V0"].landmarks is None
    assert [VARIANTS[k].margin for k in ("V1a", "V1b", "V1c")] == [0.10, 0.25, 0.50]
    assert VARIANTS["V1a"].landmarks == (8,)
    assert VARIANTS["V2"].landmarks == (4, 8, 12, 16, 20) and VARIANTS["V2"].margin == 0.10


def test_index_tip_inside_and_outside_margin() -> None:
    # 10 px right of the box edge: 0.10 * 100 = 10 -> on the boundary (inclusive)
    h = hand_with((210.0, 140.0))
    assert touch_flags_from_points([[h]], BOX, (8,), 0.10) == [True]
    assert touch_flags_from_points([[hand_with((210.5, 140.0))]], BOX, (8,), 0.10) == [False]
    assert touch_flags_from_points([[hand_with((210.5, 140.0))]], BOX, (8,), 0.25) == [True]
    # vertical margin is a fraction of the height (80): 0.25 -> 20
    assert touch_flags_from_points([[hand_with((150.0, 200.0))]], BOX, (8,), 0.25) == [True]
    assert touch_flags_from_points([[hand_with((150.0, 200.5))]], BOX, (8,), 0.25) == [False]


def test_other_landmark_does_not_count_for_index_only() -> None:
    palm = hand_with((150.0, 140.0), idx=0)  # wrist inside, tip at (0,0)
    assert touch_flags_from_points([[palm]], BOX, None, 0.10) == [True]  # V0 any landmark
    assert touch_flags_from_points([[palm]], BOX, (8,), 0.10) == [False]
    assert touch_flags_from_points([[palm]], BOX, (4, 8, 12, 16, 20), 0.10) == [False]
    thumb = hand_with((150.0, 140.0), idx=4)
    assert touch_flags_from_points([[thumb]], BOX, (4, 8, 12, 16, 20), 0.10) == [True]


def test_no_hand_and_no_box() -> None:
    assert touch_flags_from_points([[]], BOX, (8,), 0.10) == [False]
    assert touch_flags_from_points([[hand_with((150, 140))]], None, (8,), 0.10) == [False]


def test_second_hand_counts() -> None:
    far, near = hand_with((0.0, 0.0)), hand_with((150.0, 140.0))
    assert touch_flags_from_points([[far, near]], BOX, (8,), 0.10) == [True]


def test_touch_flags_uses_variant() -> None:
    hands = [[hand_with((205.0, 140.0))], [hand_with((260.0, 140.0))]]
    assert touch_flags(hands, BOX, "V1a") == [True, False]
    assert touch_flags(hands, BOX, "V1c") == [True, False]  # margin 50 -> x<=250
    assert touch_flags([[hand_with((240.0, 140.0))]], BOX, "V1c") == [True]


def test_hold_frames_ceil_of_seconds_times_fps() -> None:
    assert hold_frames(0.25, 4) == 1
    assert hold_frames(0.5, 4) == 2
    assert hold_frames(0.75, 4) == 3
    assert hold_frames(1.0, 4) == 4
    assert hold_frames(0.25, 8) == 2
    assert hold_frames(0.75, 8) == 6
    assert hold_frames(0.3, 4) == 2  # 1.2 -> 2
    assert hold_frames(0.01, 4) == 1  # never below one frame


def test_press_requires_hold() -> None:
    flags = [False] * 3 + [True] * 4 + [False] * 5
    assert press_events(flags, baseline=2, hysteresis=4, release=5) == [6]
    assert press_events(flags, baseline=2, hysteresis=5, release=5) == []


def test_baseline_latches_a_true_step() -> None:
    flags = [True] * 6 + [False] * 6 + [True] * 6
    # true during baseline (first 3 frames): latched; never fires while the
    # false run (6) >= release (5) re-arms it, then the later run fires
    assert press_events(flags, baseline=3, hysteresis=2, release=5) == [13]
    # release too long to re-arm: latched forever
    assert press_events(flags, baseline=3, hysteresis=2, release=7) == []


def test_fired_step_rearms_after_release_false_frames() -> None:
    flags = [False, False] + [True] * 3 + [False] * 5 + [True] * 3
    assert press_events(flags, baseline=2, hysteresis=3, release=5) == [4, 12]
    assert press_events(flags, baseline=2, hysteresis=3, release=6) == [4]


def test_true_frames_interrupt_release_count() -> None:
    flags = [False, False, True, True] + [False, False, False, True] + [False] * 3 + [True] * 2
    # fires at 3 (hyst 2); false_streak 3 then True resets; never re-arms
    assert press_events(flags, baseline=2, hysteresis=2, release=4) == [3]


def test_hysteresis_counts_consecutive_only() -> None:
    flags = [False, False] + [True, True, False, True, True, False, True, True, True]
    assert press_events(flags, baseline=2, hysteresis=3, release=5) == [10]


def test_matches_the_real_state_tracker() -> None:
    """The helper must agree with state/tracker.py on a START-only experiment
    for several flag sequences (the tracker is the reference)."""
    from contracts import (
        Detection,
        ExperimentDefinition,
        HandTouchingRule,
        PerceptionConfig,
        PerceptionFrame,
        StepDef,
    )

    exp = ExperimentDefinition.from_json("config/experiment.json")
    step = next(s for s in exp.steps if s.step_id == "start_pressed")
    assert isinstance(step.when[0], HandTouchingRule)
    only = exp.model_copy(update={"steps": [step]})
    cfg = PerceptionConfig(hysteresis_frames=3, release_frames=4, baseline_frames=5)
    seqs = [
        [True] * 9 + [False] * 6 + [True] * 5,
        [False] * 5 + [True] * 2 + [False] + [True] * 4 + [False] * 6 + [True] * 3,
        [False] * 4 + [True] * 6 + [False] * 4 + [True] * 6,
        [False] * 20,
    ]
    card = Detection(label="start_button", conf=0.9, box=BOX)
    inside = hand_with((150.0, 140.0))
    for flags in seqs:
        tr = StateTracker(only, cfg)
        got: list[int] = []
        for i, f in enumerate(flags):
            pf = PerceptionFrame(
                frame_id=i, t=i / 4.0, detections=[card], hands=[inside] if f else []
            )
            if tr.update(pf):
                got.append(i)
        assert press_events(flags, baseline=5, hysteresis=3, release=4) == got, flags
    assert StepDef  # imported for the isinstance check above


@pytest.mark.parametrize(
    ("n", "expected", "label"),
    [
        (1, 1, "exact"),
        (0, 1, "missed"),
        (2, 1, "extra"),
        (0, 0, "exact"),
        (1, 0, "extra"),
        (1, 2, "missed"),
        (2, 2, "exact"),
        (3, 2, "extra"),
    ],
)
def test_classify_count(n: int, expected: int, label: str) -> None:
    assert classify_count(n, expected) == label


def test_tally_labels_overall_and_by_script_type() -> None:
    rows = [
        ("correct", "exact"),
        ("correct", "exact"),
        ("correct", "missed"),
        ("skip", "extra"),
        ("swap", "exact"),
    ]
    t = tally_labels(rows)
    assert t["all"] == {"exact": 3, "missed": 1, "extra": 1, "n": 5}
    assert t["correct"] == {"exact": 2, "missed": 1, "extra": 0, "n": 3}
    assert t["skip"] == {"exact": 0, "missed": 0, "extra": 1, "n": 1}
    assert tally_labels([])["all"] == {"exact": 0, "missed": 0, "extra": 0, "n": 0}
