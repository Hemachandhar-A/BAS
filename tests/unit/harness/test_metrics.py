"""harness/metrics.py -- per-run verdict, replay observation, cause diagnosis and the dynamic
experiment lint (IMPLEMENTATION_PLAN.md 7.3), on hand-built frames."""

from __future__ import annotations

from pathlib import Path

import pytest

from contracts import (
    Detection,
    ExpectedDeviation,
    ExperimentDefinition,
    PerceptionConfig,
    PerceptionFrame,
    RuntimeConfig,
)
from harness.metrics import (
    diagnose,
    lint_run,
    observe,
    step_truths,
    verdict,
)

FIXTURE = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"
CFG = PerceptionConfig(hysteresis_frames=2, release_frames=2, baseline_frames=1)


@pytest.fixture()
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition.from_json(FIXTURE)


def _frame(i: int, *labels: str, conf: float = 0.9) -> PerceptionFrame:
    return PerceptionFrame(
        frame_id=i,
        t=i / 10,
        detections=[Detection(label=lab, conf=conf, box=(0.0, 0.0, 10.0, 10.0)) for lab in labels],
    )


def _run(*per_frame: tuple[str, ...], conf: float = 0.9) -> list[PerceptionFrame]:
    return [_frame(i, *labels, conf=conf) for i, labels in enumerate(per_frame)]


def _correct_frames() -> list[PerceptionFrame]:
    # baseline frame, then each item shown for 3 frames in order
    return _run((), *[("item_a",)] * 3, *[("item_b",)] * 3, *[("item_c",)] * 3, *[("item_d",)] * 3)


# --- verdict ---------------------------------------------------------------------------------


def test_a_correct_run_matches(experiment: ExperimentDefinition) -> None:
    obs = observe(experiment, _correct_frames(), CFG, RuntimeConfig())
    assert obs.fired == ["s1", "s2", "s3", "s4"]
    assert obs.deviations == []
    v = verdict([], ["s1", "s2", "s3", "s4"], obs)
    assert v.match and v.extra_events == 0 and v.missing_events == 0 and v.uncertain == 0


def test_a_skip_run_matches_only_with_the_expected_omission(
    experiment: ExperimentDefinition,
) -> None:
    frames = _run((), *[("item_a",)] * 3, *[("item_c",)] * 3, *[("item_d",)] * 3)
    obs = observe(experiment, frames, CFG, RuntimeConfig())
    assert obs.deviations == [("omission", ("s2",))]
    expected = [ExpectedDeviation(deviation_type="omission", step_ids=["s2"])]
    assert verdict(expected, ["s1", "s3", "s4"], obs).match
    # the same observation against "no deviation expected" is a mismatch with an extra deviation
    v = verdict([], ["s1", "s3", "s4"], obs)
    assert not v.match and v.extra_deviations == 1


def test_a_missed_step_is_missing_and_a_mismatch(experiment: ExperimentDefinition) -> None:
    # item_b is shown for a single frame (< hysteresis 2): s2 never fires
    frames = _run((), *[("item_a",)] * 3, ("item_b",), *[("item_c",)] * 3, *[("item_d",)] * 3)
    obs = observe(experiment, frames, CFG, RuntimeConfig())
    v = verdict([], ["s1", "s2", "s3", "s4"], obs)
    assert not v.match
    assert v.missing_events == 1 and v.extra_events == 0
    assert v.performed_in_order is False
    assert v.missing_deviations == 0 and v.extra_deviations == 1  # an unexpected omission of s2


def test_extra_events_and_uncertain_are_counted(experiment: ExperimentDefinition) -> None:
    frames = _run(  # s1 again after s2, before the run completes: a repeat
        (), *[("item_a",)] * 3, *[("item_b",)] * 3, *[("item_a",)] * 3,
        *[("item_c",)] * 3, *[("item_d",)] * 3,
    )  # fmt: skip
    obs = observe(experiment, frames, CFG, RuntimeConfig())
    v = verdict([], ["s1", "s2", "s3", "s4"], obs)
    assert not v.match and v.extra_events >= 1 and v.extra_deviations >= 1
    unc = observe(
        experiment,
        _run((), *[("item_a",)] * 3, conf=0.5),
        PerceptionConfig(
            detector_conf_floor=0.3,
            confirm_conf=0.6,
            hysteresis_frames=2,
            release_frames=2,
            baseline_frames=1,
        ),
        RuntimeConfig(),
    )
    assert unc.fired == ["s1"] and unc.uncertain == 1
    assert verdict([], ["s1"], unc).uncertain == 1


def test_deviation_order_matters(experiment: ExperimentDefinition) -> None:
    obs = observe(
        experiment,
        _run((), *[("item_c",)] * 3, *[("item_a",)] * 3, *[("item_b",)] * 3),
        CFG,
        RuntimeConfig(),
    )
    # s3 first: omission of s1, s2; then s1 late (out_of_order), s2 late (out_of_order)
    assert obs.deviations == [
        ("omission", ("s1", "s2")),
        ("out_of_order", ("s1",)),
        ("out_of_order", ("s2",)),
    ]
    exp = [
        ExpectedDeviation(deviation_type="omission", step_ids=["s1", "s2"]),
        ExpectedDeviation(deviation_type="out_of_order", step_ids=["s2"]),
        ExpectedDeviation(deviation_type="out_of_order", step_ids=["s1"]),
    ]
    assert not verdict(exp, ["s3", "s2", "s1"], obs).match  # same set, wrong order


# --- diagnose --------------------------------------------------------------------------------


def test_diagnose_names_the_cause_of_a_missed_step(experiment: ExperimentDefinition) -> None:
    frames = _run((), *[("item_a",)] * 3, ("item_b",), *[("item_c",)] * 3, *[("item_d",)] * 3)
    obs = observe(experiment, frames, CFG, RuntimeConfig())
    causes = diagnose(experiment, frames, CFG, ["s1", "s2", "s3", "s4"], obs)
    assert len(causes) == 1 and "s2" in causes[0]
    assert "hysteresis" in causes[0]  # true for 1 frame, shorter than hysteresis_frames=2


def test_diagnose_never_true_reports_the_missing_box(experiment: ExperimentDefinition) -> None:
    frames = _run((), *[("item_a",)] * 3, *[("item_c",)] * 3, *[("item_d",)] * 3)
    obs = observe(experiment, frames, CFG, RuntimeConfig())
    causes = diagnose(experiment, frames, CFG, ["s1", "s2", "s3", "s4"], obs)
    assert "s2" in causes[0] and "item_b" in causes[0] and "no box" in causes[0]


# --- dynamic experiment lint -----------------------------------------------------------------


def test_step_truths_apply_the_floor(experiment: ExperimentDefinition) -> None:
    frames = [_frame(0, "item_a", conf=0.2), _frame(1, "item_a", conf=0.9)]
    t = step_truths(experiment, frames, PerceptionConfig())
    assert t["s1"] == [False, True]


def test_lint_passes_when_a_step_true_at_baseline_goes_false_before_its_turn(
    experiment: ExperimentDefinition,
) -> None:
    # item_d is present at baseline (s4 latched), disappears for 3 frames, then returns at its turn
    frames = _run(
        ("item_d",), ("item_a",), ("item_a",), ("item_a",),
        ("item_b",), ("item_b",), ("item_c",), ("item_c",), ("item_d",), ("item_d",),
    )  # fmt: skip
    cfg = PerceptionConfig(hysteresis_frames=2, release_frames=2, baseline_frames=1)
    assert lint_run(experiment, frames, cfg, RuntimeConfig()) == []


def test_lint_flags_a_step_that_never_goes_false(experiment: ExperimentDefinition) -> None:
    frames = _run(
        ("item_d",), ("item_d", "item_a"), ("item_d", "item_a"), ("item_d", "item_b"),
        ("item_d", "item_b"), ("item_d", "item_c"), ("item_d", "item_c"),
    )  # fmt: skip
    cfg = PerceptionConfig(hysteresis_frames=2, release_frames=2, baseline_frames=1)
    viol = lint_run(experiment, frames, cfg, RuntimeConfig())
    assert [v["step_id"] for v in viol] == ["s4"]
    assert "false" in viol[0]["reason"]


def test_lint_ignores_steps_not_true_at_baseline(experiment: ExperimentDefinition) -> None:
    assert lint_run(experiment, _correct_frames(), CFG, RuntimeConfig()) == []
