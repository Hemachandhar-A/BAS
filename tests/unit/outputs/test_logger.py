"""outputs/logger.py tests (P2.3; IMPLEMENTATION_PLAN.md Part 10, 5.5; F9).

No models, no real clock -- ``wall_clock`` is always injected as a fixed
callable so tests are deterministic (AGENTS.md rule 13).
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from contracts import ContractViolation, LogEntry
from outputs.logger import JsonlLogger

PLACEHOLDER = datetime(1970, 1, 1, tzinfo=UTC)


def _entry(seq: int, **overrides) -> LogEntry:
    fields = dict(
        seq=seq,
        run_id="run-1",
        t_wall=PLACEHOLDER,
        t_video=1.0,
        event_type="run_started",
        detail="Run started",
    )
    fields.update(overrides)
    return LogEntry(**fields)


@pytest.mark.F9
def test_seq_must_start_at_zero(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "run.jsonl", wall_clock=lambda: PLACEHOLDER)
    with pytest.raises(ContractViolation):
        logger.write(_entry(1))


@pytest.mark.F9
def test_seq_gap_raises_contract_violation(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "run.jsonl", wall_clock=lambda: PLACEHOLDER)
    logger.write(_entry(0))
    with pytest.raises(ContractViolation):
        logger.write(_entry(2))


@pytest.mark.F9
def test_seq_repeat_raises_contract_violation(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "run.jsonl", wall_clock=lambda: PLACEHOLDER)
    logger.write(_entry(0))
    with pytest.raises(ContractViolation):
        logger.write(_entry(0))


@pytest.mark.F9
def test_write_stamps_t_wall_from_injected_clock(tmp_path: Path) -> None:
    stamped_at = datetime(2026, 9, 28, 12, 0, 0, tzinfo=UTC)
    logger = JsonlLogger(tmp_path / "run.jsonl", wall_clock=lambda: stamped_at)
    logger.write(_entry(0))

    lines = (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert datetime.fromisoformat(row["t_wall"].replace("Z", "+00:00")) == stamped_at
    assert row["t_wall"] != PLACEHOLDER.isoformat()


@pytest.mark.F9
def test_wall_clock_must_be_tz_aware(tmp_path: Path) -> None:
    naive = datetime(2026, 9, 28, 12, 0, 0)
    logger = JsonlLogger(tmp_path / "run.jsonl", wall_clock=lambda: naive)
    with pytest.raises(ContractViolation):
        logger.write(_entry(0))
    assert not (tmp_path / "run.jsonl").exists() or (tmp_path / "run.jsonl").read_text() == ""


@pytest.mark.F9
def test_one_line_per_event_never_per_frame(tmp_path: Path) -> None:
    logger = JsonlLogger(tmp_path / "run.jsonl", wall_clock=lambda: PLACEHOLDER)
    for seq in range(5):
        logger.write(_entry(seq))
    lines = (tmp_path / "run.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    for i, line in enumerate(lines):
        assert json.loads(line)["seq"] == i


@pytest.mark.F9
def test_write_appends_and_never_rewrites(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    logger = JsonlLogger(path, wall_clock=lambda: PLACEHOLDER)
    logger.write(_entry(0))
    first_line = path.read_text(encoding="utf-8")
    logger.write(_entry(1))
    second_read = path.read_text(encoding="utf-8")
    assert second_read.startswith(first_line)


@pytest.mark.F9
def test_write_flushes_and_fsyncs_every_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []
    real_fsync = os.fsync

    def _spy_fsync(fd: int) -> None:
        calls.append(fd)
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", _spy_fsync)
    logger = JsonlLogger(tmp_path / "run.jsonl", wall_clock=lambda: PLACEHOLDER)
    logger.write(_entry(0))
    logger.write(_entry(1))
    assert len(calls) == 2


@pytest.mark.F9
def test_last_complete_line_parses_after_mid_run_kill(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    logger = JsonlLogger(path, wall_clock=lambda: PLACEHOLDER)
    logger.write(_entry(0))
    logger.write(
        _entry(
            1,
            event_type="step_confirmed",
            step_id="stow_red",
            detail="Confirmed: Stow red sample",
        )
    )

    # Simulate a kill mid-write: a third line is appended but truncated
    # before the writer could finish (no trailing newline, invalid JSON).
    with path.open("a", encoding="utf-8") as f:
        f.write('{"seq": 2, "run_id": "run-1", "t_video": 3.0, "event_typ')

    raw_lines = path.read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 3

    parsed: list[LogEntry] = []
    for line in raw_lines:
        try:
            parsed.append(LogEntry.model_validate_json(line))
        except Exception:
            continue

    assert [e.seq for e in parsed] == [0, 1]
