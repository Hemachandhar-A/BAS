"""F14 Stage 8 (essential-features.md section 14): training/benchmark_cpu.py,
tested against a fake model only -- no real detector, no network, no
model download (ISSUES.md, 2026-09-29 "PROVISIONAL CPU latency benchmark").
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from training import benchmark_cpu

pytestmark = pytest.mark.F14


def _make_frames(n: int = 4, height: int = 8, width: int = 10) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8) for _ in range(n)]


class _FakeClock:
    """A deterministic stand-in for time.perf_counter: every call advances
    by a fixed step, so two consecutive calls (benchmark_predict's
    start/end pair for one frame) are always exactly `step` apart."""

    def __init__(self, step: float) -> None:
        self._step = step
        self._t = 0.0

    def __call__(self) -> float:
        self._t += self._step
        return self._t


# ---------------------------------------------------------------------------
# _percentile
# ---------------------------------------------------------------------------


def test_percentile_of_a_sorted_list() -> None:
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert benchmark_cpu._percentile(values, 0) == 10.0
    assert benchmark_cpu._percentile(values, 100) == 50.0
    assert benchmark_cpu._percentile(values, 50) == 30.0


def test_percentile_rejects_an_empty_list() -> None:
    with pytest.raises(ValueError):
        benchmark_cpu._percentile([], 95)


# ---------------------------------------------------------------------------
# _bgr_to_contiguous_rgb
# ---------------------------------------------------------------------------


def test_bgr_to_contiguous_rgb_reverses_channels_and_is_contiguous() -> None:
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[0, 0] = (10, 20, 30)
    rgb = benchmark_cpu._bgr_to_contiguous_rgb(image)
    assert rgb.flags["C_CONTIGUOUS"]
    assert tuple(rgb[0, 0]) == (30, 20, 10)


# ---------------------------------------------------------------------------
# benchmark_predict (the harness itself), against a fake model
# ---------------------------------------------------------------------------


def test_benchmark_predict_excludes_warmup_frames_from_the_report() -> None:
    frames = _make_frames(n=1)
    calls: list[np.ndarray] = []

    def fake_predict(rgb: np.ndarray) -> None:
        calls.append(rgb)

    result = benchmark_cpu.benchmark_predict(
        "fake", fake_predict, frames, warmup_frames=3, timed_frames=5
    )

    assert len(calls) == 8  # warmup + timed frames were all actually run
    assert len(result.per_frame_ms) == 5  # but only the timed ones are reported
    assert result.warmup_frames == 3
    assert result.timed_frames == 5


def test_benchmark_predict_computes_mean_p95_and_fps_from_a_scripted_clock() -> None:
    frames = _make_frames(n=1)
    # Two now() calls per frame (start, end); a fixed step means every
    # frame measures exactly 10ms, deterministically.
    clock = _FakeClock(0.010)

    def fake_predict(rgb: np.ndarray) -> None:
        pass

    result = benchmark_cpu.benchmark_predict(
        "fake", fake_predict, frames, warmup_frames=0, timed_frames=4, now=clock
    )

    assert result.per_frame_ms == pytest.approx([10.0, 10.0, 10.0, 10.0])
    assert result.mean_ms == pytest.approx(10.0)
    assert result.p95_ms == pytest.approx(10.0)
    assert result.fps == pytest.approx(100.0)  # 1000ms / 10ms


def test_benchmark_predict_is_deterministic_given_fixed_inputs() -> None:
    frames = _make_frames(n=2)

    def fake_predict(rgb: np.ndarray) -> None:
        pass

    clock_a = _FakeClock(0.005)
    clock_b = _FakeClock(0.005)
    first = benchmark_cpu.benchmark_predict(
        "fake", fake_predict, frames, warmup_frames=2, timed_frames=6, now=clock_a
    )
    second = benchmark_cpu.benchmark_predict(
        "fake", fake_predict, frames, warmup_frames=2, timed_frames=6, now=clock_b
    )
    assert first.per_frame_ms == second.per_frame_ms
    assert first.mean_ms == second.mean_ms
    assert first.fps == second.fps


def test_benchmark_predict_cycles_frames_shorter_than_the_requested_total() -> None:
    frames = _make_frames(n=2)
    seen_shapes = []

    def fake_predict(rgb: np.ndarray) -> None:
        seen_shapes.append(rgb.shape)

    benchmark_cpu.benchmark_predict("fake", fake_predict, frames, warmup_frames=1, timed_frames=5)
    assert len(seen_shapes) == 6  # cycled through the 2 frames 3 times


def test_benchmark_predict_rejects_empty_frames() -> None:
    with pytest.raises(ValueError):
        benchmark_cpu.benchmark_predict(
            "fake", lambda rgb: None, [], warmup_frames=0, timed_frames=1
        )


def test_benchmark_predict_rejects_zero_timed_frames() -> None:
    with pytest.raises(ValueError):
        benchmark_cpu.benchmark_predict(
            "fake", lambda rgb: None, _make_frames(), warmup_frames=0, timed_frames=0
        )


# ---------------------------------------------------------------------------
# _os_info: platform.release() reports "10" on Windows 11 too (ISSUES.md,
# 2026-09-29) -- windows_build must be the authoritative field there.
# ---------------------------------------------------------------------------


def test_os_info_labels_windows_11_from_the_build_number() -> None:
    class _FakeVersion:
        build = 26200

    info = benchmark_cpu._os_info(system="Windows", getwindowsversion=lambda: _FakeVersion())

    assert info["windows_build"] == 26200
    assert info["windows_release_label"] == "Windows 11"


def test_os_info_labels_windows_10_from_the_build_number() -> None:
    class _FakeVersion:
        build = 19045  # the last Windows 10 build

    info = benchmark_cpu._os_info(system="Windows", getwindowsversion=lambda: _FakeVersion())

    assert info["windows_build"] == 19045
    assert info["windows_release_label"] == "Windows 10"


def test_os_info_windows_11_boundary_is_build_22000() -> None:
    class _JustBelow:
        build = 21999

    class _JustAt:
        build = 22000

    below = benchmark_cpu._os_info(system="Windows", getwindowsversion=lambda: _JustBelow())
    at = benchmark_cpu._os_info(system="Windows", getwindowsversion=lambda: _JustAt())

    assert below["windows_release_label"] == "Windows 10"
    assert at["windows_release_label"] == "Windows 11"


def test_os_info_has_no_windows_fields_on_non_windows() -> None:
    info = benchmark_cpu._os_info(system="Linux")

    assert "windows_build" not in info
    assert "windows_release_label" not in info
    assert info["system"] == "Linux"


def test_environment_info_uses_the_authoritative_windows_build_on_this_machine() -> None:
    # Runs on the real machine (this repo's CI/dev boxes are Windows) --
    # documents that environment_info() no longer relies solely on
    # platform.platform(), which is ambiguous between Windows 10 and 11.
    info = benchmark_cpu.environment_info()
    if info["os"]["system"] == "Windows":
        assert isinstance(info["os"]["windows_build"], int)
        assert info["os"]["windows_release_label"] in ("Windows 10", "Windows 11")


# ---------------------------------------------------------------------------
# build_report / JSON shape
# ---------------------------------------------------------------------------


def test_build_report_has_the_stage_8_shape_and_notes_hands_unmeasured() -> None:
    frames = _make_frames(n=1)
    result = benchmark_cpu.benchmark_predict(
        "fake_model", lambda rgb: None, frames, warmup_frames=1, timed_frames=3
    )

    report = benchmark_cpu.build_report(
        [result], warmup_frames=1, timed_frames=3, frame_width=10, frame_height=8
    )

    assert report["hands"] == benchmark_cpu.HANDS_NOTE
    assert report["warmup_frames"] == 1
    assert report["timed_frames"] == 3
    assert report["frame_size"] == [10, 8]
    assert "environment" in report and "cpu_model" in report["environment"]
    assert len(report["results"]) == 1
    entry = report["results"][0]
    assert entry["name"] == "fake_model"
    assert set(entry) == {
        "name",
        "warmup_frames",
        "timed_frames",
        "mean_ms",
        "p95_ms",
        "fps",
    }  # per_frame_ms is dropped from the summary report

    # The report must be JSON-serializable as-is.
    json.dumps(report)


def test_collect_real_candidates_returns_nothing_without_allow_download() -> None:
    # No network, no import of rfdetr/ultralytics/onnxruntime attempted
    # for real candidates unless explicitly opted in.
    assert benchmark_cpu.collect_real_candidates(allow_download=False) == []


def test_main_writes_a_report_with_no_candidates_when_download_not_allowed(
    tmp_path: Path, capsys
) -> None:
    out_path = tmp_path / "bench.json"
    exit_code = benchmark_cpu.main(
        ["--width", "10", "--height", "8", "--warmup", "0", "--frames", "1", "--out", str(out_path)]
    )
    assert exit_code == 0
    assert out_path.exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["results"] == []
    assert report["hands"] == benchmark_cpu.HANDS_NOTE
