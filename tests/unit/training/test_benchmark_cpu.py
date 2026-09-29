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
# _rgb_to_contiguous_bgr (the YOLO colour-order fix, ISSUES.md 2026-09-29
# P2 review finding 2)
# ---------------------------------------------------------------------------


def test_rgb_to_contiguous_bgr_reverses_channels_and_is_contiguous() -> None:
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = (30, 20, 10)  # RGB
    bgr = benchmark_cpu._rgb_to_contiguous_bgr(rgb)
    assert bgr.flags["C_CONTIGUOUS"]
    assert tuple(bgr[0, 0]) == (10, 20, 30)


def test_bgr_to_rgb_to_bgr_round_trips() -> None:
    original = np.zeros((3, 4, 3), dtype=np.uint8)
    original[1, 2] = (7, 8, 9)
    rgb = benchmark_cpu._bgr_to_contiguous_rgb(original)
    back = benchmark_cpu._rgb_to_contiguous_bgr(rgb)
    assert np.array_equal(original, back)


# ---------------------------------------------------------------------------
# check_raw_output_parity / extract_pytorch_raw_outputs (Phase 1: fakes
# only; Phase 2 feeds this real PyTorch/ONNX output)
# ---------------------------------------------------------------------------


def test_parity_passes_when_outputs_match_within_tolerance() -> None:
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    logits = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    result = benchmark_cpu.check_raw_output_parity(boxes, logits, boxes.copy(), logits.copy())
    assert result.within_tolerance is True
    assert result.max_box_abs_diff == pytest.approx(0.0)
    assert result.max_logit_abs_diff == pytest.approx(0.0)


def test_parity_fails_when_boxes_diverge_beyond_tolerance() -> None:
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    onnx_boxes = boxes + 0.5  # far beyond the default 1e-3 box tolerance
    logits = np.array([[1.0, -2.0]], dtype=np.float32)
    result = benchmark_cpu.check_raw_output_parity(boxes, logits, onnx_boxes, logits.copy())
    assert result.within_tolerance is False
    assert result.max_box_abs_diff == pytest.approx(0.5)


def test_parity_fails_when_logits_diverge_beyond_tolerance() -> None:
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    logits = np.array([[1.0, -2.0]], dtype=np.float32)
    onnx_logits = logits + 1.0  # far beyond the default 1e-2 logit tolerance
    result = benchmark_cpu.check_raw_output_parity(boxes, logits, boxes.copy(), onnx_logits)
    assert result.within_tolerance is False
    assert result.max_logit_abs_diff == pytest.approx(1.0)


def test_parity_respects_custom_tolerances() -> None:
    boxes = np.array([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32)
    onnx_boxes = boxes + 0.01
    logits = np.zeros((1, 2), dtype=np.float32)
    tight = benchmark_cpu.check_raw_output_parity(boxes, logits, onnx_boxes, logits, box_atol=1e-3)
    loose = benchmark_cpu.check_raw_output_parity(boxes, logits, onnx_boxes, logits, box_atol=0.1)
    assert tight.within_tolerance is False
    assert loose.within_tolerance is True


class _FakeInnerModel:
    """Stands in for RFDETR's inner model.model (unoptimized) so that
    model.model.model(tensor) and model.model.inference_model(tensor) are
    both real bound methods, matching the real object's shape -- a naive
    fake with `self.model = self` would let an instance attribute named
    "model" shadow a same-named method, so it's a separate class here."""

    def __init__(self, boxes: np.ndarray, logits: np.ndarray) -> None:
        self._boxes = boxes
        self._logits = logits

    def model(self, tensor):
        return (self._as_tensor(self._boxes), self._as_tensor(self._logits))

    def inference_model(self, tensor):
        return (self._as_tensor(self._boxes), self._as_tensor(self._logits))

    @staticmethod
    def _as_tensor(array: np.ndarray):
        import torch

        return torch.from_numpy(array[None, ...])  # add the batch dim extract_* squeezes back off


class _FakeRawModel:
    """Duck-types the two attributes extract_pytorch_raw_outputs reads:
    _is_optimized_for_inference and model.model / model.inference_model."""

    def __init__(self, boxes: np.ndarray, logits: np.ndarray, optimized: bool = False) -> None:
        self._is_optimized_for_inference = optimized
        self.model = _FakeInnerModel(boxes, logits)


def test_extract_pytorch_raw_outputs_unoptimized_path() -> None:
    pytest.importorskip("torch")
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    logits = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    fake_model = _FakeRawModel(boxes, logits, optimized=False)

    out_boxes, out_logits = benchmark_cpu.extract_pytorch_raw_outputs(fake_model, None)

    assert np.allclose(out_boxes, boxes)
    assert np.allclose(out_logits, logits)


def test_extract_pytorch_raw_outputs_optimized_path() -> None:
    pytest.importorskip("torch")
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    logits = np.array([[0.0, 9.0]], dtype=np.float32)
    fake_model = _FakeRawModel(boxes, logits, optimized=True)

    out_boxes, out_logits = benchmark_cpu.extract_pytorch_raw_outputs(fake_model, None)

    assert np.allclose(out_boxes, boxes)
    assert np.allclose(out_logits, logits)


# ---------------------------------------------------------------------------
# run_suite / summarize_by_candidate / summarize_by_position: shuffled
# repeats, per-run thread recording, a cooldown between runs, and
# separating position effect from candidate effect (ISSUES.md 2026-09-29
# P2 review, finding 4).
# ---------------------------------------------------------------------------


def _make_candidate(name: str, delay_s: float) -> benchmark_cpu.Candidate:
    del delay_s  # unused: no test here depends on nonzero latency

    def predict_fn(rgb: np.ndarray) -> None:
        pass

    return benchmark_cpu.Candidate(name, predict_fn)


def test_run_suite_produces_one_record_per_round_per_candidate() -> None:
    candidates = [_make_candidate("a", 0.0), _make_candidate("b", 0.0)]
    frames = _make_frames(n=1)

    records = benchmark_cpu.run_suite(
        candidates,
        frames,
        warmup_frames=0,
        timed_frames=1,
        repeats=3,
        seed=0,
        cooldown_s=0.0,
        num_threads=1,
    )

    assert len(records) == 6  # 3 repeats x 2 candidates
    assert {r["candidate"] for r in records} == {"a", "b"}
    assert sorted(r["round"] for r in records) == [1, 1, 2, 2, 3, 3]
    assert [r["global_position"] for r in records] == list(range(1, 7))


def test_run_suite_shuffles_candidate_order_across_rounds(monkeypatch) -> None:
    candidates = [_make_candidate(n, 0.0) for n in ("a", "b", "c", "d", "e")]
    frames = _make_frames(n=1)

    records = benchmark_cpu.run_suite(
        candidates,
        frames,
        warmup_frames=0,
        timed_frames=1,
        repeats=4,
        seed=0,
        cooldown_s=0.0,
        num_threads=1,
    )

    orders = []
    for round_idx in (1, 2, 3, 4):
        orders.append(tuple(r["candidate"] for r in records if r["round"] == round_idx))
    assert len(set(orders)) > 1  # not every round used the same fixed order


def test_run_suite_is_reproducible_given_the_same_seed() -> None:
    candidates = [_make_candidate(n, 0.0) for n in ("a", "b", "c")]
    frames = _make_frames(n=1)

    first = benchmark_cpu.run_suite(
        candidates, frames, 0, 1, repeats=3, seed=7, cooldown_s=0.0, num_threads=1
    )
    second = benchmark_cpu.run_suite(
        candidates, frames, 0, 1, repeats=3, seed=7, cooldown_s=0.0, num_threads=1
    )

    assert [r["candidate"] for r in first] == [r["candidate"] for r in second]


def test_run_suite_records_torch_threads_on_every_run() -> None:
    candidates = [_make_candidate("a", 0.0)]
    frames = _make_frames(n=1)

    records = benchmark_cpu.run_suite(
        candidates, frames, 0, 1, repeats=2, seed=0, cooldown_s=0.0, num_threads=2
    )

    import torch

    assert all(r["torch_threads"] == torch.get_num_threads() for r in records)
    assert torch.get_num_threads() == 2


def test_run_suite_records_onnx_intra_op_threads_only_for_that_candidate() -> None:
    plain = _make_candidate("plain", 0.0)
    onnx_like = benchmark_cpu.Candidate("onnx", lambda rgb: None, onnx_intra_op_threads=3)
    frames = _make_frames(n=1)

    records = benchmark_cpu.run_suite(
        [plain, onnx_like], frames, 0, 1, repeats=1, seed=0, cooldown_s=0.0, num_threads=1
    )

    by_name = {r["candidate"]: r for r in records}
    assert by_name["plain"]["onnx_intra_op_threads"] is None
    assert by_name["onnx"]["onnx_intra_op_threads"] == 3


def test_run_suite_sleeps_cooldown_between_runs_but_not_after_the_last(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(benchmark_cpu.time, "sleep", lambda s: sleeps.append(s))
    candidates = [_make_candidate("a", 0.0), _make_candidate("b", 0.0)]
    frames = _make_frames(n=1)

    benchmark_cpu.run_suite(
        candidates, frames, 0, 1, repeats=2, seed=0, cooldown_s=0.5, num_threads=1
    )

    assert sleeps == [0.5, 0.5, 0.5]  # 4 runs total, no cooldown after the last one


def test_summarize_by_candidate_groups_per_repeat_values() -> None:
    records = [
        _fake_record("a", 1, 1, 10.0),
        _fake_record("a", 2, 1, 12.0),
        _fake_record("b", 1, 2, 5.0),
        _fake_record("b", 2, 2, 5.0),
    ]
    summary = benchmark_cpu.summarize_by_candidate(records)
    assert summary["a"]["mean_ms_per_repeat"] == [10.0, 12.0]
    assert summary["a"]["spread_ms"] == pytest.approx(2.0)
    assert summary["b"]["spread_ms"] == pytest.approx(0.0)


def test_summarize_by_position_pools_across_candidates_at_the_same_slot() -> None:
    records = [
        _fake_record("a", 1, 1, 10.0),
        _fake_record("b", 1, 2, 20.0),
        _fake_record("b", 2, 1, 14.0),
        _fake_record("a", 2, 2, 22.0),
    ]
    position_effect = benchmark_cpu.summarize_by_position(records)
    assert position_effect["1"]["n"] == 2
    assert position_effect["1"]["mean_ms_avg"] == pytest.approx((10.0 + 14.0) / 2)
    assert position_effect["2"]["mean_ms_avg"] == pytest.approx((20.0 + 22.0) / 2)


# ---------------------------------------------------------------------------
# build_report / JSON shape
# ---------------------------------------------------------------------------


def _fake_record(candidate: str, round_: int, slot: int, mean_ms: float) -> dict:
    return {
        "candidate": candidate,
        "round": round_,
        "slot": slot,
        "global_position": (round_ - 1) * 2 + slot,
        "torch_threads": 4,
        "onnx_intra_op_threads": None,
        "name": candidate,
        "warmup_frames": 1,
        "timed_frames": 3,
        "mean_ms": mean_ms,
        "p95_ms": mean_ms * 1.1,
        "fps": 1000.0 / mean_ms,
    }


def test_build_report_has_the_stage_8_shape_and_notes_hands_unmeasured() -> None:
    records = [_fake_record("fake_model", 1, 1, 10.0), _fake_record("fake_model", 2, 1, 12.0)]

    report = benchmark_cpu.build_report(
        records,
        warmup_frames=1,
        timed_frames=3,
        frame_width=10,
        frame_height=8,
        repeats=2,
        seed=0,
        cooldown_s=0.0,
        num_threads=4,
    )

    assert report["provisional"] is True
    assert benchmark_cpu.HANDS_NOTE in report["not_measured"]
    assert report["warmup_frames"] == 1
    assert report["timed_frames"] == 3
    assert report["frame_size"] == [10, 8]
    assert report["controls"] == {
        "repeats": 2,
        "seed": 0,
        "cooldown_s": 0.0,
        "num_threads_requested": 4,
        "candidate_order": "shuffled independently each repeat round, fixed seed",
    }
    assert "environment" in report and "cpu_model" in report["environment"]
    assert report["per_run"] == records
    assert report["per_candidate_summary"]["fake_model"]["repeats"] == 2
    assert report["per_candidate_summary"]["fake_model"]["mean_ms_per_repeat"] == [10.0, 12.0]
    assert set(report["position_effect"]) == {"1"}

    # The report must be JSON-serializable as-is.
    json.dumps(report)


def test_collect_real_candidates_returns_nothing_without_allow_download() -> None:
    # No network, no import of rfdetr/ultralytics/onnxruntime attempted
    # for real candidates unless explicitly opted in.
    assert benchmark_cpu.collect_real_candidates(allow_download=False, num_threads=4) == []


def test_main_writes_a_report_with_no_candidates_when_download_not_allowed(
    tmp_path: Path, capsys
) -> None:
    out_path = tmp_path / "bench.json"
    exit_code = benchmark_cpu.main(
        [
            "--width",
            "10",
            "--height",
            "8",
            "--warmup",
            "0",
            "--frames",
            "1",
            "--repeats",
            "1",
            "--cooldown",
            "0",
            "--out",
            str(out_path),
        ]
    )
    assert exit_code == 0
    assert out_path.exists()
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert report["per_run"] == []
    assert report["provisional"] is True
    assert benchmark_cpu.HANDS_NOTE in report["not_measured"]
