"""Validates the real Sample Transfer experiment (config/experiment.json)
against state/tracker.py end to end -- exercising the actual StateTracker
against the actual shipped experiment definition, not just the synthetic
single-step fixtures in test_tracker.py.

Physical layout (arbitrary but internally consistent pixel positions):
outer_box, tray and start_button are stationary containers; red_box and
yellow_box start inside outer_box (matching the real rig: ISSUES.md,
2026-09-28 "experiment lint" entry) and are moved through "out" -> "in
tray" -> back "in outer_box" (stowed), one at a time, in canonical step
order, each held for hysteresis_frames frames.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from contracts import Detection, ExperimentDefinition, Hand, PerceptionConfig, PerceptionFrame
from state.tracker import StateTracker

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_PATH = ROOT / "config" / "experiment.json"

OUTER_BOX = (0.0, 0.0, 100.0, 100.0)
TRAY = (200.0, 0.0, 300.0, 100.0)
START_BUTTON = (400.0, 0.0, 410.0, 10.0)

STOWED_RED = (10.0, 10.0, 20.0, 20.0)  # inside OUTER_BOX
STOWED_YELLOW = (30.0, 30.0, 40.0, 40.0)  # inside OUTER_BOX
OUT_RED = (150.0, 150.0, 160.0, 160.0)  # outside both OUTER_BOX and TRAY
OUT_YELLOW = (170.0, 170.0, 180.0, 180.0)  # outside both
TRAY_RED = (210.0, 10.0, 220.0, 20.0)  # inside TRAY
TRAY_YELLOW = (240.0, 10.0, 250.0, 20.0)  # inside TRAY

CONF = 0.95


def _det(label: str, box: tuple[float, float, float, float]) -> Detection:
    return Detection(label=label, conf=CONF, box=box)


def _hand_on_button() -> Hand:
    return Hand(handedness="Right", score=CONF, landmarks_px=[(405.0, 5.0)] * 21)


def _frame(
    t: float,
    red_box: tuple[float, float, float, float],
    yellow_box: tuple[float, float, float, float],
    hand_on_button: bool,
) -> PerceptionFrame:
    detections = [
        _det("outer_box", OUTER_BOX),
        _det("tray", TRAY),
        _det("start_button", START_BUTTON),
        _det("red_box", red_box),
        _det("yellow_box", yellow_box),
    ]
    hands = [_hand_on_button()] if hand_on_button else []
    return PerceptionFrame(frame_id=int(round(t * 1000)), t=t, detections=detections, hands=hands)


@pytest.fixture(scope="module")
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition.from_json(EXPERIMENT_PATH)


@pytest.mark.F4
@pytest.mark.F13
def test_correct_run_fires_all_seven_steps_in_canonical_order(
    experiment: ExperimentDefinition,
) -> None:
    config = PerceptionConfig(baseline_frames=2, hysteresis_frames=2, release_frames=2)
    tracker = StateTracker(experiment, config)

    # (red_box position, yellow_box position, hand on button) per phase, in
    # the same order as config/experiment.json's canonical steps.
    phases = [
        (OUT_RED, STOWED_YELLOW, False),  # red_out
        (TRAY_RED, STOWED_YELLOW, False),  # red_in_tray
        (TRAY_RED, OUT_YELLOW, False),  # yellow_out
        (TRAY_RED, TRAY_YELLOW, False),  # yellow_in_tray
        (TRAY_RED, TRAY_YELLOW, True),  # start_pressed
        (STOWED_RED, TRAY_YELLOW, False),  # red_stowed
        (STOWED_RED, STOWED_YELLOW, False),  # yellow_stowed
    ]
    assert [step.step_id for step in experiment.steps] == [
        "red_out",
        "red_in_tray",
        "yellow_out",
        "yellow_in_tray",
        "start_pressed",
        "red_stowed",
        "yellow_stowed",
    ]

    t = 0.0
    # Baseline: both boxes already stowed, matching the real rig at t=0 --
    # this must latch red_stowed and yellow_stowed without emitting events.
    for _ in range(config.baseline_frames):
        t += 1.0
        assert tracker.update(_frame(t, STOWED_RED, STOWED_YELLOW, False)) == []

    all_events = []
    for red_pos, yellow_pos, hand in phases:
        for _ in range(config.hysteresis_frames):
            t += 1.0
            all_events.extend(tracker.update(_frame(t, red_pos, yellow_pos, hand)))

    assert [ev.step_id for ev in all_events] == experiment.step_ids
    assert len(all_events) == len(experiment.steps)
    for ev in all_events:
        assert ev.uncertain is False
        assert ev.confidence == pytest.approx(CONF)


@pytest.mark.F4
@pytest.mark.F13
def test_omitted_step_never_fires_and_nothing_else_leaks_through(
    experiment: ExperimentDefinition,
) -> None:
    # A sanity check on the flip side of the correct-run test: if red_box
    # never leaves outer_box, red_out/red_in_tray/red_stowed must never
    # fire, and nothing about the (independent) yellow steps is disturbed.
    config = PerceptionConfig(baseline_frames=2, hysteresis_frames=2, release_frames=2)
    tracker = StateTracker(experiment, config)

    t = 0.0
    for _ in range(config.baseline_frames):
        t += 1.0
        tracker.update(_frame(t, STOWED_RED, STOWED_YELLOW, False))

    all_events = []
    # red_box stays stowed throughout; only yellow's steps are performed.
    for yellow_pos, hand in [(OUT_YELLOW, False), (TRAY_YELLOW, False), (TRAY_YELLOW, True)]:
        for _ in range(config.hysteresis_frames):
            t += 1.0
            all_events.extend(tracker.update(_frame(t, STOWED_RED, yellow_pos, hand)))

    fired = {ev.step_id for ev in all_events}
    assert fired == {"yellow_out", "yellow_in_tray", "start_pressed"}
