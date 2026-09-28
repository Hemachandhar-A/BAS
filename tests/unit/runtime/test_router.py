"""runtime/loop.py -- LatestFrameStore and Router. The headline test here
is GOLD-1 "through the loop" (IMPLEMENTATION_PLAN.md Part 10, P2.4): the
same skipped-step scenario as engine/test_sequence.py's GOLD-1, but driven
through a real StateTracker via PerceptionFrames and routed through the
full Router (JsonlLogger + FakeSpeaker), not just the bare Engine."""

from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path

import pytest

from contracts import (
    ContractViolation,
    Detection,
    ExperimentDefinition,
    PerceptionConfig,
    PerceptionFrame,
    RuntimeConfig,
    StateEvent,
)
from engine.sequence import SequenceEngine
from outputs.tts import FakeSpeaker
from runtime.loop import LatestFrameStore, Router
from state.tracker import StateTracker

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"
CONF = 0.95


@pytest.fixture()
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition.from_json(FIXTURE_PATH)


def _pframe(frame_id: int, t: float, present_label: str | None) -> PerceptionFrame:
    detections = (
        [Detection(label=present_label, conf=CONF, box=(0.0, 0.0, 10.0, 10.0))]
        if present_label
        else []
    )
    return PerceptionFrame(frame_id=frame_id, t=t, detections=detections)


# ---------------------------------------------------------------------------
# LatestFrameStore
# ---------------------------------------------------------------------------


def test_latest_frame_store_starts_empty_and_always_holds_the_newest_put() -> None:
    store = LatestFrameStore()
    assert store.get() is None

    from harness.fakes import make_blank_frame

    f0 = make_blank_frame(0, 0.0)
    f1 = make_blank_frame(1, 1.0)
    store.put(f0)
    assert store.get() is f0
    store.put(f1)
    assert store.get() is f1  # overwritten, not queued


# ---------------------------------------------------------------------------
# GOLD-1 through the loop
# ---------------------------------------------------------------------------


@pytest.mark.F5
@pytest.mark.F6
@pytest.mark.F8
@pytest.mark.F9
@pytest.mark.F13
@pytest.mark.gold_symbolic
def test_gold_1_single_omission_through_state_tracker_and_router(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    perception_config = PerceptionConfig(baseline_frames=2, hysteresis_frames=2, release_frames=2)
    tracker = StateTracker(experiment, perception_config)
    runtime_config = RuntimeConfig()  # narrate_next_step=True, alert_on_repeat=True (defaults)
    engine = SequenceEngine(experiment, runtime_config)
    speaker = FakeSpeaker()
    run_id = "gold1-test"
    router = Router(
        experiment,
        runtime_config,
        tracker,
        engine,
        speaker,
        log_dir=tmp_path,
        run_id_factory=lambda: run_id,
    )

    router.start(t=0.0)
    assert router.run_state == "running"

    # 2 baseline frames (nothing present) then 2-frame windows of item_a,
    # item_c, item_d -- item_b (step s2) is never performed, so s3 firing
    # while s2 is still pending must yield exactly one omission.
    frames = [
        _pframe(0, 0.0, None),
        _pframe(1, 1.0, None),
        _pframe(2, 2.0, "item_a"),
        _pframe(3, 3.0, "item_a"),
        _pframe(4, 4.0, "item_c"),
        _pframe(5, 5.0, "item_c"),
        _pframe(6, 6.0, "item_d"),
        _pframe(7, 7.0, "item_d"),
    ]
    for frame in frames:
        router.process_perception_frame(frame)

    assert router.run_state == "completed"

    # Exactly one alert, matching GOLD-1's "must never regress" requirement.
    alert_calls = [call for call in speaker.calls if call[1] == "alert"]
    assert alert_calls == [("Step skipped: Step two", "alert")]

    assert speaker.calls == [
        ("Do step one", "info"),  # spoken by start() at run start
        ("Do step two", "info"),  # step_confirmed(s1) narrates the next step
        ("Step skipped: Step two", "alert"),  # the one alert
        ("Experiment ended with skipped steps", "info"),  # run_completed
        # step_confirmed(s4) narrates nothing further (no next step), so it
        # speaks nothing -- but it must still be logged (checked below).
    ]

    log_path = tmp_path / f"{run_id}.jsonl"
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]

    assert [entry["seq"] for entry in lines] == list(range(len(lines)))
    assert all(entry["run_id"] == run_id for entry in lines)

    by_type = [entry["event_type"] for entry in lines]
    assert by_type == [
        "run_started",
        "step_confirmed",  # s1
        "deviation_detected",  # omission at s3
        "step_confirmed",  # s4
        "run_completed",
    ]

    confirmed_s4 = lines[3]
    assert confirmed_s4["step_id"] == "s4"
    assert confirmed_s4["detail"] == "Confirmed: Step four"  # logged even though nothing was said

    omission = lines[2]
    assert omission["deviation_type"] == "omission"
    assert omission["skipped_step_ids"] == ["s2"]
    assert omission["detail"] == "Step skipped: Step two"

    completed = lines[4]
    assert completed["detail"] == "Experiment ended with skipped steps"


# ---------------------------------------------------------------------------
# Router's own "idle" bookkeeping (ISSUES.md 2026-09-28 P2.2 DECISION: Engine
# never reports "idle" again after finish()/completion -- Router must).
# ---------------------------------------------------------------------------


def _router(experiment: ExperimentDefinition, tmp_path: Path, run_id: str = "r") -> Router:
    runtime_config = RuntimeConfig()
    return Router(
        experiment,
        runtime_config,
        StateTracker(experiment, PerceptionConfig()),
        SequenceEngine(experiment, runtime_config),
        FakeSpeaker(),
        log_dir=tmp_path,
        run_id_factory=lambda: run_id,
    )


def test_router_reports_idle_before_start_and_after_reset(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    router = _router(experiment, tmp_path)
    assert router.run_state == "idle"

    router.start(t=0.0)
    assert router.run_state == "running"

    router.reset(t=1.0)
    # The underlying Engine only ever reaches "completed", never "idle"
    # again on its own -- Router must still report "idle" here.
    assert router.run_state == "idle"


def test_router_reset_while_running_writes_run_completed_aborted(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    events = []
    runtime_config = RuntimeConfig()
    router = Router(
        experiment,
        runtime_config,
        StateTracker(experiment, PerceptionConfig()),
        SequenceEngine(experiment, runtime_config),
        FakeSpeaker(),
        log_dir=tmp_path,
        run_id_factory=lambda: "aborted-run",
        on_engine_event=events.append,
    )
    router.start(t=0.0)
    router.reset(t=5.0)

    assert len(events) == 1
    assert events[0].kind == "run_completed"
    assert events[0].summary is not None
    assert events[0].summary.aborted is True


def test_router_reset_while_idle_is_a_noop_for_the_engine(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    events = []
    runtime_config = RuntimeConfig()
    router = Router(
        experiment,
        runtime_config,
        StateTracker(experiment, PerceptionConfig()),
        SequenceEngine(experiment, runtime_config),
        FakeSpeaker(),
        log_dir=tmp_path,
        run_id_factory=lambda: "never-started",
        on_engine_event=events.append,
    )
    router.reset(t=0.0)  # never started -- nothing to finish
    assert events == []
    assert router.run_state == "idle"


def test_router_start_while_already_running_raises_without_corrupting_state(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    """A caller bug (calling start() twice without reset()) must not
    silently discard the first run's tracker/log state -- it must raise
    before mutating anything, matching SequenceEngine's own re-entrancy
    guard (ISSUES.md, 2026-09-28 P2.2 DECISION)."""
    router = _router(experiment, tmp_path, run_id="first-run")
    router.start(t=0.0)
    assert router.run_id == "first-run"

    with pytest.raises(ContractViolation):
        router.start(t=1.0)

    # State from the first, legitimately-started run must be untouched.
    assert router.run_state == "running"
    assert router.run_id == "first-run"
    log_path = tmp_path / "first-run.jsonl"
    assert log_path.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # only the original run_started, nothing extra


# ---------------------------------------------------------------------------
# Router is safe to call from a different thread than the inference loop
# (essential-features.md section 0: run control and the inference loop are
# on different threads; P2.5's Flask routes will call start()/reset() while
# the inference thread concurrently calls process_perception_frame()).
# ---------------------------------------------------------------------------


def test_router_start_and_process_frame_are_serialized_across_threads(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    counter = itertools.count()
    runtime_config = RuntimeConfig()
    router = Router(
        experiment,
        runtime_config,
        StateTracker(experiment, PerceptionConfig()),
        SequenceEngine(experiment, runtime_config),
        FakeSpeaker(),
        log_dir=tmp_path,
        run_id_factory=lambda: f"concurrent-run-{next(counter)}",
    )
    router.start(t=0.0)

    errors: list[Exception] = []
    stop = threading.Event()

    def _hammer_frames() -> None:
        frame = PerceptionFrame(frame_id=0, t=0.0)
        while not stop.is_set():
            try:
                router.process_perception_frame(frame)
            except Exception as exc:  # pragma: no cover - failure path only
                errors.append(exc)
                return

    worker = threading.Thread(target=_hammer_frames, daemon=True)
    worker.start()
    for _ in range(20):
        router.reset(t=0.0)
        router.start(t=0.0)
    stop.set()
    worker.join(timeout=5.0)

    assert errors == []
    assert router.run_state == "running"


# ---------------------------------------------------------------------------
# process_state_event / process_perception_frame and the tracker=None mode
# ---------------------------------------------------------------------------


def test_process_perception_frame_without_a_tracker_raises(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    runtime_config = RuntimeConfig()
    router = Router(
        experiment,
        runtime_config,
        tracker=None,
        engine=SequenceEngine(experiment, runtime_config),
        speaker=FakeSpeaker(),
        log_dir=tmp_path,
        run_id_factory=lambda: "scripted",
    )
    router.start(t=0.0)
    with pytest.raises(ContractViolation):
        router.process_perception_frame(PerceptionFrame(frame_id=0, t=0.0))


def test_process_state_event_works_with_no_tracker_and_replays_gold_1(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    runtime_config = RuntimeConfig()
    speaker = FakeSpeaker()
    router = Router(
        experiment,
        runtime_config,
        tracker=None,
        engine=SequenceEngine(experiment, runtime_config),
        speaker=speaker,
        log_dir=tmp_path,
        run_id_factory=lambda: "scripted",
    )
    router.start(t=0.0)
    for i, step_id in enumerate(["s1", "s3", "s4"], start=1):
        router.process_state_event(
            StateEvent(t=float(i), step_id=step_id, confidence=1.0, uncertain=False)
        )
    alert_calls = [call for call in speaker.calls if call[1] == "alert"]
    assert alert_calls == [("Step skipped: Step two", "alert")]


# ---------------------------------------------------------------------------
# feed_lost / feed_restored are only logged while a run is active
# ---------------------------------------------------------------------------


def test_feed_events_are_dropped_silently_while_idle(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    router = _router(experiment, tmp_path)
    router.write_feed_lost(0.0, "no frames")  # idle -- must not raise, must not write
    router.write_feed_restored(0.0, "back")
    assert not (tmp_path / "r.jsonl").exists()


def test_feed_lost_and_restored_are_logged_while_running(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    router = _router(experiment, tmp_path, run_id="feed-run")
    router.start(t=0.0)
    router.write_feed_lost(1.0, "No frame received for 3.0s")
    router.write_feed_restored(2.0, "Feed restored")

    lines = [
        json.loads(line)
        for line in (tmp_path / "feed-run.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    types = [entry["event_type"] for entry in lines]
    assert types == ["run_started", "feed_lost", "feed_restored"]
    assert lines[1]["detail"] == "No frame received for 3.0s"
    assert lines[2]["detail"] == "Feed restored"
