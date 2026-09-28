"""harness/replay.py -- the three replay modes (IMPLEMENTATION_PLAN.md 5.9,
Part 10 P2.4) and the scripts/replay.py CLI. ``ReplayResult`` is a
TEMP_1 stub (ISSUES.md, 2026-09-28 P2.4 CONTRACT) since contracts.py does
not define one yet."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from contracts import ExperimentDefinition, PerceptionCacheHeader, PerceptionFrame, RunScript
from harness.replay import main, replay_from_cache, replay_from_video, replay_scripted

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"


@pytest.fixture()
def experiment() -> ExperimentDefinition:
    return ExperimentDefinition.from_json(FIXTURE_PATH)


# ---------------------------------------------------------------------------
# --script (scripted) mode -- reproduces GOLD-1 without any PerceptionFrame
# ---------------------------------------------------------------------------


def test_replay_scripted_reproduces_gold_1(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    result = replay_scripted(
        experiment, ["s1", "s3", "s4"], run_id="scripted-gold1", log_dir=tmp_path
    )

    assert result.run_id == "scripted-gold1"
    assert result.frames_processed == 3
    assert result.summary is not None
    assert result.summary.pos == pytest.approx(0.75)
    assert result.summary.all_steps_done is False
    assert result.summary.skipped_step_ids == ["s2"]

    deviations = [e for e in result.engine_events if e.kind == "deviation_detected"]
    assert len(deviations) == 1
    assert deviations[0].deviation_type == "omission"

    assert (tmp_path / "scripted-gold1.jsonl").exists()


# ---------------------------------------------------------------------------
# --from-cache mode
# ---------------------------------------------------------------------------


def _write_cache(
    cache_dir: Path, run_id: str, experiment_id: str, frames: list[PerceptionFrame]
) -> None:
    run_dir = cache_dir / run_id
    run_dir.mkdir(parents=True)
    header = PerceptionCacheHeader(
        run_id=run_id, experiment_id=experiment_id, fps=15.0, model_stamp="fake:stamp"
    )
    lines = [header.model_dump_json()] + [f.model_dump_json() for f in frames]
    (run_dir / "perception.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_replay_from_cache_reads_header_and_frames(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    log_dir = tmp_path / "logs"
    frames = [PerceptionFrame(frame_id=i, t=float(i)) for i in range(3)]
    _write_cache(cache_dir, "cached-run", experiment.experiment_id, frames)

    result = replay_from_cache(
        experiment, "cached-run", cache_dir=cache_dir, log_dir=log_dir
    )

    assert result.run_id == "cached-run"
    assert result.frames_processed == 3
    assert (log_dir / "cached-run.jsonl").exists()


def test_replay_from_cache_rejects_a_cache_built_for_a_different_experiment(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, "mismatched-run", "some_other_experiment", frames=[])

    with pytest.raises(ValueError, match="some_other_experiment"):
        replay_from_cache(experiment, "mismatched-run", cache_dir=cache_dir, log_dir=tmp_path)


# ---------------------------------------------------------------------------
# --video mode -- perception/ does not exist yet (P1.1/P1.6 land later)
# ---------------------------------------------------------------------------


def test_replay_from_video_raises_import_error_until_perception_lands(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    with pytest.raises(ImportError):
        replay_from_video(experiment, tmp_path / "does_not_matter.mp4", log_dir=tmp_path)


def test_replay_from_video_closes_the_source_even_if_pipeline_construction_fails(
    experiment: ExperimentDefinition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resource-leak regression check: ``source`` is opened before
    ``PerceptionPipeline`` is constructed, so a failure there must not skip
    closing the already-open source (a real camera/file handle)."""
    closed = []

    class _FakeSource:
        exhausted = True
        fps = None

        def read(self):
            return None

        def close(self):
            closed.append(True)

    class _FailingPipeline:
        def __init__(self, config):
            raise RuntimeError("pipeline construction boom")

    fake_perception_pkg = types.ModuleType("perception")
    fake_camera_mod = types.ModuleType("perception.camera")
    fake_camera_mod.open_source = lambda path: _FakeSource()
    fake_pipeline_mod = types.ModuleType("perception.pipeline")
    fake_pipeline_mod.PerceptionPipeline = _FailingPipeline

    monkeypatch.setitem(sys.modules, "perception", fake_perception_pkg)
    monkeypatch.setitem(sys.modules, "perception.camera", fake_camera_mod)
    monkeypatch.setitem(sys.modules, "perception.pipeline", fake_pipeline_mod)

    with pytest.raises(RuntimeError, match="pipeline construction boom"):
        replay_from_video(experiment, "dummy.mp4", log_dir=tmp_path)

    assert closed == [True]


def test_replay_from_video_skips_a_bad_frame_and_keeps_going(
    experiment: ExperimentDefinition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matches runtime/loop.py's live behavior: a single frame that fails
    perception is skipped, not fatal to the whole replay."""
    frame_ids = [0, 1, 2]

    class _FakeSource:
        fps = None

        def __init__(self):
            self._ids = list(frame_ids)
            self.exhausted = False

        def read(self):
            if not self._ids:
                self.exhausted = True
                return None
            frame_id = self._ids.pop(0)
            return types.SimpleNamespace(frame_id=frame_id, t=float(frame_id))

        def close(self):
            pass

    class _FlakyPipeline:
        def __init__(self, config):
            pass

        def process(self, frame):
            if frame.frame_id == 1:
                raise RuntimeError("boom on frame 1")
            return PerceptionFrame(frame_id=frame.frame_id, t=frame.t)

    fake_perception_pkg = types.ModuleType("perception")
    fake_camera_mod = types.ModuleType("perception.camera")
    fake_camera_mod.open_source = lambda path: _FakeSource()
    fake_pipeline_mod = types.ModuleType("perception.pipeline")
    fake_pipeline_mod.PerceptionPipeline = _FlakyPipeline

    monkeypatch.setitem(sys.modules, "perception", fake_perception_pkg)
    monkeypatch.setitem(sys.modules, "perception.camera", fake_camera_mod)
    monkeypatch.setitem(sys.modules, "perception.pipeline", fake_pipeline_mod)

    result = replay_from_video(experiment, "dummy.mp4", log_dir=tmp_path, run_id="flaky-video")

    assert result.frames_processed == 3  # frame 1 was attempted and counted, then skipped


# ---------------------------------------------------------------------------
# scripts/replay.py CLI (harness.replay.main)
# ---------------------------------------------------------------------------


def test_cli_script_mode_prints_replay_result_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    script = RunScript(
        run_id="cli-script-run",
        experiment_id="fixture_4step",
        script_type="skip",
        split="train",
        fps=15.0,
        camera_setup_id="s1",
        operator="o1",
        performed_steps=["s1", "s3", "s4"],
    )
    script_path = tmp_path / "script.json"
    script_path.write_text(script.model_dump_json(), encoding="utf-8")

    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs_out" / "logs").mkdir(parents=True, exist_ok=True)
    rc = main(["--script", str(script_path), "--experiment", str(FIXTURE_PATH)])

    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["run_id"] == "cli-script-run"
    assert out["summary"]["pos"] == pytest.approx(0.75)


def test_cli_video_mode_reports_a_clean_error_when_perception_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["--video", str(tmp_path / "nope.mp4"), "--experiment", str(FIXTURE_PATH)])
    assert rc == 1
    assert "perception/" in capsys.readouterr().err
