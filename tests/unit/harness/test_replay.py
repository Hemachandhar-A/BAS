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

from contracts import (
    ExperimentDefinition,
    PerceptionCacheHeader,
    PerceptionFrame,
    RunScript,
    RuntimeConfig,
)
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
# ---------------------------------------------------------------------------
# --from-cache: refuses a cache with another fps or model_stamp (R0.2)
# ---------------------------------------------------------------------------


def test_replay_from_cache_refuses_a_different_stamp_or_fps(
    experiment: ExperimentDefinition, tmp_path: Path
) -> None:
    from perception.cache import CacheMismatch

    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, "r1", experiment.experiment_id, [PerceptionFrame(frame_id=0, t=0.0)])

    ok = replay_from_cache(  # header is fps 15.0 / "fake:stamp"
        experiment, "r1", cache_dir=cache_dir, log_dir=tmp_path,
        expected_stamp="fake:stamp", expected_fps=15.0,
    )  # fmt: skip
    assert ok.frames_processed == 1
    with pytest.raises(CacheMismatch, match="model_stamp"):
        replay_from_cache(
            experiment, "r1", cache_dir=cache_dir, log_dir=tmp_path, expected_stamp="other:stamp"
        )
    with pytest.raises(CacheMismatch, match="fps"):
        replay_from_cache(
            experiment, "r1", cache_dir=cache_dir, log_dir=tmp_path, expected_fps=10.0
        )


def test_cli_from_cache_refuses_a_mismatched_cache(
    experiment: ExperimentDefinition, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cache_dir = tmp_path / "cache"  # header: fps 15.0, stamp "fake:stamp"; config: fps 10
    _write_cache(cache_dir, "r1", experiment.experiment_id, [PerceptionFrame(frame_id=0, t=0.0)])
    rc = main(
        ["--from-cache", "r1", "--cache-dir", str(cache_dir), "--experiment", str(FIXTURE_PATH)]
    )
    assert rc == 2
    assert "model_stamp" in capsys.readouterr().err


@pytest.mark.parametrize("split", ["test", "train", "all"])
def test_cli_tune_refuses_any_split_but_val(split: str, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--tune", "--split", split]) == 2
    assert "val split only" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# --video mode (perception.pipeline.load_pipeline, injected here)
# ---------------------------------------------------------------------------


class _FakeSource:
    def __init__(self, n: int = 3, fps: float | None = None) -> None:
        self._ids = list(range(n))
        self.fps = fps
        self.exhausted = False
        self.closed = False

    def read(self):
        if not self._ids:
            self.exhausted = True
            return None
        frame_id = self._ids.pop(0)
        return types.SimpleNamespace(frame_id=frame_id, t=frame_id / 30.0)

    def close(self) -> None:
        self.closed = True


def _stub_camera(monkeypatch: pytest.MonkeyPatch, source: _FakeSource) -> None:
    camera = types.ModuleType("perception.camera")
    camera.open_source = lambda path: source
    monkeypatch.setitem(sys.modules, "perception.camera", camera)


class _FakePipeline:
    def __init__(self, fail_on: int | None = None) -> None:
        self.fail_on = fail_on
        self.seen: list[int] = []
        self.resets = 0

    def process(self, frame):
        self.seen.append(frame.frame_id)
        if frame.frame_id == self.fail_on:
            raise RuntimeError("boom")
        return PerceptionFrame(frame_id=frame.frame_id, t=frame.t)

    def reset(self) -> None:
        self.resets += 1


def test_replay_from_video_defaults_to_load_pipeline(
    experiment: ExperimentDefinition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = _FakePipeline()
    pipeline_mod = types.ModuleType("perception.pipeline")
    pipeline_mod.load_pipeline = lambda: fake
    monkeypatch.setitem(sys.modules, "perception.pipeline", pipeline_mod)
    _stub_camera(monkeypatch, _FakeSource(3))

    result = replay_from_video(
        experiment, "dummy.mp4", runtime_config=RuntimeConfig(target_fps=30.0), log_dir=tmp_path
    )

    assert fake.seen == [0, 1, 2] and fake.resets == 1
    assert result.frames_processed == 3


def test_replay_from_video_raises_import_error_without_the_pipeline_module(
    experiment: ExperimentDefinition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "perception.pipeline", None)
    _stub_camera(monkeypatch, _FakeSource(1))
    with pytest.raises(ImportError):
        replay_from_video(experiment, "dummy.mp4", log_dir=tmp_path)


def test_replay_from_video_closes_the_source_even_if_pipeline_construction_fails(
    experiment: ExperimentDefinition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resource-leak regression check: ``source`` is opened before the pipeline is built, so
    a failure there must not skip closing the already-open source."""
    source = _FakeSource(1)
    _stub_camera(monkeypatch, source)

    def boom():
        raise RuntimeError("pipeline construction boom")

    with pytest.raises(RuntimeError, match="pipeline construction boom"):
        replay_from_video(experiment, "dummy.mp4", log_dir=tmp_path, pipeline_factory=boom)

    assert source.closed


def test_replay_from_video_skips_a_bad_frame_and_keeps_going(
    experiment: ExperimentDefinition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matches runtime/loop.py's live behavior: a frame that fails perception is skipped."""
    _stub_camera(monkeypatch, _FakeSource(3))
    result = replay_from_video(
        experiment, "dummy.mp4", runtime_config=RuntimeConfig(target_fps=30.0),
        log_dir=tmp_path, run_id="flaky-video", pipeline_factory=lambda: _FakePipeline(fail_on=1),
    )  # fmt: skip
    assert result.frames_processed == 3  # frame 1 was attempted and counted, then skipped


def test_replay_from_video_decimates_like_the_cache_builder(
    experiment: ExperimentDefinition, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from harness.replay import decimation_every

    assert decimation_every(29.99, 10.0) == 3  # the real recordings: every 3rd frame
    assert decimation_every(None, 10.0) == 3  # unknown source fps counts as the capture rate
    assert decimation_every(5.0, 10.0) == 1
    fake = _FakePipeline()
    _stub_camera(monkeypatch, _FakeSource(10, fps=29.99))
    replay_from_video(
        experiment, "dummy.mp4", runtime_config=RuntimeConfig(target_fps=10.0),
        log_dir=tmp_path, pipeline_factory=lambda: fake,
    )  # fmt: skip
    assert fake.seen == [0, 3, 6, 9]  # original frame ids are kept
    fake2 = _FakePipeline()
    _stub_camera(monkeypatch, _FakeSource(30, fps=29.99))
    replay_from_video(
        experiment, "dummy.mp4", runtime_config=RuntimeConfig(target_fps=10.0),
        log_dir=tmp_path, pipeline_factory=lambda: fake2, max_frames=4,
    )  # fmt: skip
    assert fake2.seen == [0, 3, 6, 9]

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
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "perception.pipeline", None)  # see the test above
    rc = main(["--video", str(tmp_path / "nope.mp4"), "--experiment", str(FIXTURE_PATH)])
    assert rc == 1
    assert "perception/" in capsys.readouterr().err
