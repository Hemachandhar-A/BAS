"""engine/sequence.py -- the 5.4 decision table, GOLD-1..5 (Plan 7.2),
determinism and POS. Runs against the fixture four-step experiment
(fixtures/experiment_4step.json), never config/experiment.json, so
revising the real experiment never changes what these goldens expect."""

from __future__ import annotations

from pathlib import Path

import pytest

from contracts import ContractViolation, ExperimentDefinition, RuntimeConfig, StateEvent
from engine.sequence import SequenceEngine

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"


@pytest.fixture()
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition.from_json(FIXTURE_PATH)


def _run(
    experiment: ExperimentDefinition,
    performed_steps: list[str],
    config: RuntimeConfig | None = None,
    run_id: str = "test-run",
):
    """Starts a fresh engine and feeds performed_steps as StateEvents with
    increasing t, confidence=1.0, uncertain=False. Returns (engine, all
    EngineEvents in emission order)."""
    engine = SequenceEngine(experiment, config or RuntimeConfig())
    engine.start(0.0, run_id=run_id)
    all_events = []
    for i, step_id in enumerate(performed_steps, start=1):
        all_events.extend(
            engine.on_state_event(
                StateEvent(t=float(i), step_id=step_id, confidence=1.0, uncertain=False)
            )
        )
    return engine, all_events


def _display_name(experiment: ExperimentDefinition, step_id: str) -> str:
    return next(s.display_name for s in experiment.steps if s.step_id == step_id)


# ---------------------------------------------------------------------------
# GOLD-1 -- the golden test: s1, s3, s4 (s2 skipped). Must never regress.
# ---------------------------------------------------------------------------


@pytest.mark.F5
@pytest.mark.F6
@pytest.mark.gold_symbolic
def test_gold_1_single_omission_never_regresses(experiment: ExperimentDefinition) -> None:
    engine, events = _run(experiment, ["s1", "s3", "s4"])

    deviations = [e for e in events if e.kind == "deviation_detected"]
    assert len(deviations) == 1
    dev = deviations[0]
    assert dev.deviation_type == "omission"
    assert dev.skipped_step_ids == ["s2"]
    assert dev.speak == f"Step skipped: {_display_name(experiment, 's2')}"
    assert dev.confidence_tag == "confirmed"

    completed = [e for e in events if e.kind == "run_completed"]
    assert len(completed) == 1
    summary = completed[0].summary
    assert summary is not None
    assert summary.all_steps_done is False
    assert summary.aborted is False
    assert summary.pos == pytest.approx(0.75)
    assert summary.skipped_step_ids == ["s2"]
    assert engine.run_state == "completed"


# ---------------------------------------------------------------------------
# GOLD-2 -- mid swap: s1, s3, s2, s4
# ---------------------------------------------------------------------------


@pytest.mark.F5
@pytest.mark.F6
@pytest.mark.gold_symbolic
def test_gold_2_mid_swap_omission_then_out_of_order(experiment: ExperimentDefinition) -> None:
    engine, events = _run(experiment, ["s1", "s3", "s2", "s4"])

    deviations = [e for e in events if e.kind == "deviation_detected"]
    assert len(deviations) == 2
    assert deviations[0].deviation_type == "omission"
    assert deviations[0].skipped_step_ids == ["s2"]
    assert deviations[1].deviation_type == "out_of_order"
    assert deviations[1].step_id == "s2"

    completed = [e for e in events if e.kind == "run_completed"]
    assert len(completed) == 1
    summary = completed[0].summary
    assert summary is not None
    assert summary.late_step_ids == ["s2"]
    assert summary.pos == pytest.approx(0.75)
    assert engine.run_state == "completed"


# ---------------------------------------------------------------------------
# GOLD-3 -- clean run: s1, s2, s3, s4
# ---------------------------------------------------------------------------


@pytest.mark.F5
@pytest.mark.F8
@pytest.mark.gold_symbolic
def test_gold_3_clean_run_zero_deviations(experiment: ExperimentDefinition) -> None:
    config = RuntimeConfig(narrate_next_step=True)
    engine, events = _run(experiment, ["s1", "s2", "s3", "s4"], config=config)

    deviations = [e for e in events if e.kind == "deviation_detected"]
    assert deviations == []

    confirmed = [e for e in events if e.kind == "step_confirmed"]
    assert [e.step_id for e in confirmed] == ["s1", "s2", "s3", "s4"]
    # spoken sequence: next-step phrases after s1, s2, s3, then nothing
    # (no next step) before "Experiment complete" on run_completed.
    assert confirmed[0].speak == _say(experiment, "s2")
    assert confirmed[1].speak == _say(experiment, "s3")
    assert confirmed[2].speak == _say(experiment, "s4")
    assert confirmed[3].speak is None

    completed = [e for e in events if e.kind == "run_completed"]
    assert len(completed) == 1
    assert completed[0].speak == "Experiment complete"
    summary = completed[0].summary
    assert summary is not None
    assert summary.all_steps_done is True
    assert summary.pos == pytest.approx(1.0)
    assert summary.skipped_step_ids == []
    assert summary.late_step_ids == []


def _say(experiment: ExperimentDefinition, step_id: str) -> str:
    return next(s.say for s in experiment.steps if s.step_id == step_id)


@pytest.mark.F8
@pytest.mark.gold_symbolic
def test_narrate_next_step_false_silences_next_step_speech(
    experiment: ExperimentDefinition,
) -> None:
    config = RuntimeConfig(narrate_next_step=False)
    _, events = _run(experiment, ["s1", "s2"], config=config)
    confirmed = [e for e in events if e.kind == "step_confirmed"]
    assert all(e.speak is None for e in confirmed)


# ---------------------------------------------------------------------------
# GOLD-4 -- repeat: s1, s2, s2, s3, s4
# ---------------------------------------------------------------------------


@pytest.mark.F5
@pytest.mark.F6
@pytest.mark.gold_symbolic
def test_gold_4_repeat_alerts_when_enabled(experiment: ExperimentDefinition) -> None:
    engine, events = _run(experiment, ["s1", "s2", "s2", "s3", "s4"])

    deviations = [e for e in events if e.kind == "deviation_detected"]
    assert len(deviations) == 1
    assert deviations[0].deviation_type == "repeat"
    assert deviations[0].step_id == "s2"
    assert deviations[0].speak == f"Repeated: {_display_name(experiment, 's2')}"

    completed = [e for e in events if e.kind == "run_completed"]
    assert completed[0].summary is not None
    assert completed[0].summary.all_steps_done is True
    assert engine.run_state == "completed"


@pytest.mark.F6
@pytest.mark.gold_symbolic
def test_gold_4_repeat_silent_when_alert_on_repeat_false(
    experiment: ExperimentDefinition,
) -> None:
    config = RuntimeConfig(alert_on_repeat=False)
    _, events = _run(experiment, ["s1", "s2", "s2", "s3", "s4"], config=config)
    deviations = [e for e in events if e.kind == "deviation_detected"]
    assert len(deviations) == 1
    assert deviations[0].deviation_type == "repeat"
    assert deviations[0].speak is None


# ---------------------------------------------------------------------------
# GOLD-5 -- idle: nothing performed
# ---------------------------------------------------------------------------


@pytest.mark.F5
@pytest.mark.gold_symbolic
def test_gold_5_idle_run_emits_nothing_until_finished(experiment: ExperimentDefinition) -> None:
    config = RuntimeConfig()
    engine = SequenceEngine(experiment, config)
    engine.start(0.0, run_id="idle-run")

    snapshot = engine.snapshot()
    assert all(s.status == "pending" for s in snapshot)

    events = engine.finish(5.0)
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "run_completed"
    assert ev.speak == "Run ended early"
    assert ev.summary is not None
    assert ev.summary.aborted is True
    assert ev.summary.all_steps_done is False
    assert ev.summary.observed_sequence == []
    assert ev.summary.pos == pytest.approx(0.0)
    assert engine.run_state == "completed"

    # a second finish() is a no-op once completed
    assert engine.finish(6.0) == []


# ---------------------------------------------------------------------------
# Other decision-table rows / contract edges not covered above
# ---------------------------------------------------------------------------


@pytest.mark.F5
def test_events_ignored_before_start(experiment: ExperimentDefinition) -> None:
    engine = SequenceEngine(experiment, RuntimeConfig())
    assert engine.run_state == "idle"
    events = engine.on_state_event(
        StateEvent(t=0.0, step_id="s1", confidence=1.0, uncertain=False)
    )
    assert events == []


@pytest.mark.F5
def test_events_ignored_after_completion(experiment: ExperimentDefinition) -> None:
    engine, _ = _run(experiment, ["s1", "s2", "s3", "s4"])
    assert engine.run_state == "completed"
    events = engine.on_state_event(
        StateEvent(t=99.0, step_id="s1", confidence=1.0, uncertain=False)
    )
    assert events == []


@pytest.mark.F5
def test_unknown_step_id_raises_contract_violation(experiment: ExperimentDefinition) -> None:
    engine = SequenceEngine(experiment, RuntimeConfig())
    engine.start(0.0)
    with pytest.raises(ContractViolation):
        engine.on_state_event(
            StateEvent(t=0.0, step_id="s99", confidence=1.0, uncertain=False)
        )


@pytest.mark.F5
def test_swapping_the_last_pair_reports_skipped_not_out_of_order(
    experiment: ExperimentDefinition,
) -> None:
    # Disclosed consequence, Plan 5.4: the run completes the moment the
    # last canonical step is first seen; events after completion are
    # ignored, so the late s3 never gets a chance to be "out_of_order".
    engine, events = _run(experiment, ["s1", "s2", "s4", "s3"])

    completed = [e for e in events if e.kind == "run_completed"]
    assert len(completed) == 1
    assert completed[0].summary is not None
    assert completed[0].summary.skipped_step_ids == ["s3"]
    assert completed[0].summary.late_step_ids == []

    # the s3 event after completion produced nothing
    deviations = [e for e in events if e.kind == "deviation_detected"]
    assert len(deviations) == 1
    assert deviations[0].deviation_type == "omission"
    assert deviations[0].skipped_step_ids == ["s3"]


@pytest.mark.F5
def test_confidence_tag_reflects_uncertain_flag(experiment: ExperimentDefinition) -> None:
    engine = SequenceEngine(experiment, RuntimeConfig())
    engine.start(0.0)
    events = engine.on_state_event(
        StateEvent(t=0.0, step_id="s1", confidence=0.45, uncertain=True)
    )
    assert events[0].confidence_tag == "flagged_uncertain"


@pytest.mark.F5
def test_expected_and_next_step_ids_track_first_pending(
    experiment: ExperimentDefinition,
) -> None:
    engine = SequenceEngine(experiment, RuntimeConfig())
    engine.start(0.0)
    events = engine.on_state_event(
        StateEvent(t=0.0, step_id="s1", confidence=1.0, uncertain=False)
    )
    assert events[0].expected_step_id == "s1"
    assert events[0].next_step_id == "s2"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


@pytest.mark.F5
def test_determinism_same_input_same_output(experiment: ExperimentDefinition) -> None:
    performed = ["s1", "s3", "s2", "s4"]
    _, events_a = _run(experiment, performed, run_id="run-a")
    _, events_b = _run(experiment, performed, run_id="run-a")
    assert events_a == events_b
