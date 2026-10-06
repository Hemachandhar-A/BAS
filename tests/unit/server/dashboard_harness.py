"""Helpers for the dashboard's node DOM tests (see dashboard_harness.js): payload builders made
from the real contract models (so the stub API bodies have exactly the server's shape), the node
runner, and small readers over the DOM snapshot it returns."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from contracts import (
    EngineEvent,
    ExperimentDefinition,
    LogEntry,
    StatusResponse,
    StepProgress,
)

ROOT = Path(__file__).resolve().parents[3]
STATIC = ROOT / "server" / "static"
APP_JS = STATIC / "app.js"
INDEX_HTML = STATIC / "index.html"
HARNESS_JS = Path(__file__).with_name("dashboard_harness.js")

EXPERIMENT = ExperimentDefinition.from_json(ROOT / "config" / "experiment.json")
STEP_IDS = EXPERIMENT.step_ids
NAME = {s.step_id: s.display_name for s in EXPERIMENT.steps}
SAY = {s.step_id: s.say for s in EXPERIMENT.steps}
RUN = "live-20261006T181616Z-97a07d"

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def experiment_payload() -> dict[str, Any]:
    return EXPERIMENT.model_dump(mode="json")


def alert(
    t: float,
    step_id: str,
    skipped: list[str],
    speak: str | None,
    deviation_type: str = "omission",
) -> EngineEvent:
    return EngineEvent(
        t=t,
        kind="deviation_detected",
        step_id=step_id,
        deviation_type=deviation_type,  # type: ignore[arg-type]
        skipped_step_ids=skipped,
        confidence_tag="confirmed",
        speak=speak,
    )


def status(
    states: str,
    run_state: str = "running",
    *,
    times: dict[int, float] | None = None,
    feed_ok: bool = True,
    fps: float | None = 9.1,
    run_id: str | None = RUN,
    alerts: list[EngineEvent] | None = None,
    expected: str | None | bool = True,
) -> dict[str, Any]:
    """states: one letter per step: p pending, c confirmed, s skipped, l completed_late.
    expected=True derives the first pending step like the server does."""
    names = {"p": "pending", "c": "confirmed", "s": "skipped", "l": "completed_late"}
    times = times or {}
    steps = [
        StepProgress(step_id=sid, status=names[ch], t_confirmed=times.get(i))  # type: ignore[arg-type]
        for i, (sid, ch) in enumerate(zip(STEP_IDS, states, strict=True))
    ]
    pending = [s.step_id for s in steps if s.status == "pending"]
    if expected is True:
        expected = pending[0] if pending else None
    body = StatusResponse(
        run_id=run_id,
        run_state=run_state,  # type: ignore[arg-type]
        feed_ok=feed_ok,
        fps=fps,
        steps=steps,
        expected_step_id=expected,  # type: ignore[arg-type]
        next_step_id=pending[1] if len(pending) > 1 else None,
        next_step_say=SAY[pending[1]] if len(pending) > 1 else None,
        recent_alerts=alerts or [],
        generated_at=datetime(2026, 10, 6, tzinfo=UTC),
    )
    return body.model_dump(mode="json")


def entry(
    seq: int,
    event_type: str,
    t: float,
    detail: str,
    *,
    step_id: str | None = None,
    tag: str | None = None,
    deviation_type: str | None = None,
    skipped: list[str] | None = None,
    run_id: str = RUN,
) -> dict[str, Any]:
    return LogEntry(
        seq=seq,
        run_id=run_id,
        t_wall=datetime(2026, 10, 6, tzinfo=UTC),
        t_video=t,
        event_type=event_type,  # type: ignore[arg-type]
        step_id=step_id,
        deviation_type=deviation_type,  # type: ignore[arg-type]
        skipped_step_ids=skipped or [],
        confidence_tag=tag,  # type: ignore[arg-type]
        detail=detail,
    ).model_dump(mode="json")


def confirmed(
    seq: int, t: float, step_id: str, tag: str = "confirmed", **kw: Any
) -> dict[str, Any]:
    return entry(
        seq, "step_confirmed", t, f"Confirmed: {NAME[step_id]}", step_id=step_id, tag=tag, **kw
    )


def log(*entries: dict[str, Any], run_id: str = RUN) -> dict[str, Any]:
    return {"run_id": run_id, "entries": list(entries)}


def run_page(tmp_path: Path, scenario: dict[str, Any]) -> dict[str, Any]:
    path = tmp_path / "scenario.json"
    path.write_text(json.dumps(scenario), encoding="utf-8")
    out = subprocess.run(
        ["node", str(HARNESS_JS), str(APP_JS), str(INDEX_HTML), str(path)],
        capture_output=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


# ---- readers over one snapshot -----------------------------------------------------------


def text(snap: dict[str, Any], element_id: str) -> str:
    return snap[element_id]["full"]


def rows(snap: dict[str, Any], element_id: str) -> list[dict[str, Any]]:
    return snap[element_id]["children"]


def step_rows(snap: dict[str, Any]) -> list[dict[str, str]]:
    """One dict per step row: number, title, status label, time, plus the row/status classes."""
    out = []
    for row in rows(snap, "step-list"):
        number, title, state, when = row["children"]
        out.append(
            {
                "n": number["full"],
                "title": title["full"],
                "status": state["full"],
                "time": when["full"],
                "row_cls": row["cls"],
                "status_cls": state["cls"],
            }
        )
    return out


def seg_classes(snap: dict[str, Any]) -> list[str]:
    return [seg["cls"] for seg in rows(snap, "progress")]
