"""Regression tests for the four CONTRACT fixes applied to contracts.py at
G0 (ISSUES.md, 2026-09-28 entries)."""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

import contracts


# 1. non-finite numbers must be rejected on the four fields the ISSUES.md
#    entry named as gaps.
@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_state_event_t_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValidationError):
        contracts.StateEvent(t=bad, step_id="red_out", confidence=0.9, uncertain=False)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_engine_event_t_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValidationError):
        contracts.EngineEvent(
            t=bad, kind="step_confirmed", step_id="red_out", confidence_tag="confirmed"
        )


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_log_entry_t_video_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValidationError):
        contracts.LogEntry(
            seq=0, run_id="001", t_video=bad, event_type="run_started", detail="Run started"
        )


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_run_script_fps_rejects_non_finite(bad: float) -> None:
    with pytest.raises(ValidationError):
        contracts.RunScript(
            run_id="001-correct",
            experiment_id="sample_transfer",
            script_type="correct",
            split="train",
            fps=bad,
            camera_setup_id="S1",
            operator="O1",
        )


# 2. extra="forbid" on PerceptionConfig / RuntimeConfig catches a typo'd key.
def test_perception_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        contracts.PerceptionConfig(hysteresis_frmes=7)  # deliberate typo


def test_runtime_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        contracts.RuntimeConfig(target_fp=15)  # deliberate typo


# 3. LogEntry.t_wall is optional at construction, stamped by the logger
#    later, and must be timezone-aware once set.
def test_log_entry_t_wall_optional_at_construction() -> None:
    entry = contracts.LogEntry(
        seq=0, run_id="001", t_video=1.0, event_type="run_started", detail="Run started"
    )
    assert entry.t_wall is None


def test_log_entry_t_wall_rejects_naive_datetime() -> None:
    with pytest.raises(ValidationError):
        contracts.LogEntry(
            seq=0,
            run_id="001",
            t_wall=datetime(2026, 1, 1),  # no tzinfo
            t_video=1.0,
            event_type="run_started",
            detail="Run started",
        )


def test_log_entry_t_wall_accepts_aware_datetime() -> None:
    entry = contracts.LogEntry(
        seq=0,
        run_id="001",
        t_wall=datetime(2026, 1, 1, tzinfo=UTC),
        t_video=1.0,
        event_type="run_started",
        detail="Run started",
    )
    assert entry.t_wall is not None and entry.t_wall.tzinfo is not None


# 4. Pydantic models now exist for the small file seams.
def test_acceptance_config_round_trips() -> None:
    cfg = contracts.AcceptanceConfig.model_validate(
        {
            "detector": {"min_recall_per_class": 0.85, "min_map50": 0.80},
            "pipeline": {"min_pipeline_fps": 8},
            "replay": {"exact_deviation_match": True, "max_mismatched_runs": 0},
            "label_review": {"max_bad_fraction_per_class": 0.10},
        }
    )
    assert cfg.pipeline.min_pipeline_fps == 8


def test_acceptance_config_rejects_unknown_key() -> None:
    with pytest.raises(ValidationError):
        contracts.AcceptanceConfig.model_validate(
            {
                "detector": {"min_recall_per_class": 0.85, "min_map50": 0.80},
                "pipeline": {"min_pipeline_fps": 8},
                "replay": {"exact_deviation_match": True, "max_mismatched_runs": 0},
                "label_review": {"max_bad_fraction_per_class": 0.10},
                "typo_field": True,
            }
        )


def test_report_header_requires_tz_aware_generated_at() -> None:
    with pytest.raises(ValidationError):
        contracts.ReportHeader(generated_at=datetime(2026, 1, 1))
    header = contracts.ReportHeader(generated_at=datetime(2026, 1, 1, tzinfo=UTC))
    assert header.model_stamp is None
