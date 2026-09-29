"""F14 Stage 8 (essential-features.md section 14): training/benchmark_cpu.py,
tested against a fake model only -- no real detector, no network, no
model download (ISSUES.md, 2026-09-29 "PROVISIONAL CPU latency benchmark").
"""

from __future__ import annotations

import json
import logging
import sys
import types
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


# ---------------------------------------------------------------------------
# run_parity_check: the end-to-end parity runner (A2). Fakes only here;
# Phase 2 feeds it real model/session output.
# ---------------------------------------------------------------------------


def test_run_parity_check_passes_when_backends_agree() -> None:
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    logits = np.array([[1.0, -2.0]], dtype=np.float32)

    result = benchmark_cpu.run_parity_check(
        lambda tensor: (boxes, logits),
        lambda tensor: (boxes.copy(), logits.copy()),
        None,
        None,
    )

    assert result.within_tolerance is True


def test_run_parity_check_raises_parity_error_when_backends_disagree() -> None:
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    logits = np.array([[1.0, -2.0]], dtype=np.float32)

    with pytest.raises(benchmark_cpu.ParityError):
        benchmark_cpu.run_parity_check(
            lambda tensor: (boxes, logits),
            lambda tensor: (boxes + 1.0, logits),
            None,
            None,
        )


class _FakeInnerModel:
    """Stands in for RFDETR's inner model.model (unoptimized) so that
    model.model.model(tensor) and model.model.inference_model(tensor) are
    both real bound methods, matching the real object's shape -- a naive
    fake with `self.model = self` would let an instance attribute named
    "model" shadow a same-named method, so it's a separate class here."""

    def __init__(self, boxes: np.ndarray, logits: np.ndarray) -> None:
        self._boxes = boxes
        self._logits = logits
        self.last_tensor_dtype = None  # records what dtype inference_model actually received
        # Stands in for the innermost nn.Module's real .training flag (three
        # levels down in the real object: model.model.model.training) --
        # simplified to one level here since the fake has no separate inner
        # module. A fresh RFDETRNano() starts in training mode (confirmed
        # empirically this session), which is exactly the B1 bug: an un-eval'd
        # model.model.model(tensor) call returns num_queries * group_detr
        # (3900) predictions instead of num_queries (300) -- ISSUES.md
        # 2026-09-29 P2 review, BLOCKING B1.
        self.training = True

    def model(self, tensor):
        self.last_tensor_dtype = tensor.dtype
        return (self._as_tensor(self._boxes), self._as_tensor(self._logits))

    def inference_model(self, tensor):
        self.last_tensor_dtype = tensor.dtype
        return (self._as_tensor(self._boxes), self._as_tensor(self._logits))

    @staticmethod
    def _as_tensor(array: np.ndarray):
        import torch

        return torch.from_numpy(array[None, ...])  # add the batch dim extract_* squeezes back off


class _FakeRawModel:
    """Duck-types the attributes extract_pytorch_raw_outputs reads:
    _is_optimized_for_inference, model.model / model.inference_model, and
    (optimized path only) _optimized_dtype."""

    def __init__(self, boxes: np.ndarray, logits: np.ndarray, optimized: bool = False) -> None:
        self._is_optimized_for_inference = optimized
        self.model = _FakeInnerModel(boxes, logits)
        if optimized:
            import torch

            self._optimized_dtype = torch.float64  # deliberately distinct from float32

    def _ensure_eval_mode_for_unoptimized_inference(self) -> None:
        """Duck-types RFDETR.predict()'s real gate (detr.py:2465), which
        extract_pytorch_raw_outputs must now call too on the unoptimized
        path (B1 fix)."""
        self.model.training = False


def test_extract_pytorch_raw_outputs_unoptimized_path() -> None:
    torch = pytest.importorskip("torch")
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    logits = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    fake_model = _FakeRawModel(boxes, logits, optimized=False)
    tensor = torch.zeros((1, 3, 4, 4), dtype=torch.float32)

    out_boxes, out_logits = benchmark_cpu.extract_pytorch_raw_outputs(fake_model, tensor)

    assert np.allclose(out_boxes, boxes)
    assert np.allclose(out_logits, logits)
    assert fake_model.model.last_tensor_dtype == torch.float32  # unchanged on the unoptimized path


def test_extract_pytorch_raw_outputs_unoptimized_path_ensures_eval_mode() -> None:
    """B1 fix (ISSUES.md 2026-09-29 P2 review, BLOCKING, confirmed empirically
    this session against the real RFDETRNano() checkpoint): the raw forward
    this function calls directly (model.model.model(tensor)) must never run
    in training mode. rfdetr's lwdetr.py branches its query count on
    self.training -- 3900 (num_queries * group_detr) in training mode, 300
    (num_queries) in eval -- so an un-eval'd call silently returns 13x too
    many query/box/logit slots. Only predict() used to guard this; this
    function is the __network_only candidate's own forward path and was
    never eval'd itself, which is exactly how it could run before the same
    model's first predict() call in a shuffled benchmark round."""
    torch = pytest.importorskip("torch")
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]], dtype=np.float32)
    logits = np.array([[1.0, -2.0, 3.0]], dtype=np.float32)
    fake_model = _FakeRawModel(boxes, logits, optimized=False)
    assert fake_model.model.training is True  # starts in training mode, like a fresh RFDETRNano()
    tensor = torch.zeros((1, 3, 4, 4), dtype=torch.float32)

    benchmark_cpu.extract_pytorch_raw_outputs(fake_model, tensor)

    assert fake_model.model.training is False  # eval mode is ensured before the raw forward call


def test_extract_pytorch_raw_outputs_optimized_path_casts_to_optimized_dtype() -> None:
    torch = pytest.importorskip("torch")
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32)
    logits = np.array([[0.0, 9.0]], dtype=np.float32)
    fake_model = _FakeRawModel(boxes, logits, optimized=True)
    tensor = torch.zeros((1, 3, 4, 4), dtype=torch.float32)

    out_boxes, out_logits = benchmark_cpu.extract_pytorch_raw_outputs(fake_model, tensor)

    assert np.allclose(out_boxes, boxes)
    assert np.allclose(out_logits, logits)
    # predict()'s own optimized path casts to _optimized_dtype before calling
    # inference_model (ISSUES.md 2026-09-29 P2 review, finding 3) -- confirm
    # this wrapper actually does that, not just that it dispatches correctly.
    assert fake_model.model.last_tensor_dtype == torch.float64


# ---------------------------------------------------------------------------
# run_suite / summarize_by_candidate / summarize_by_position: shuffled
# repeats, per-run thread recording, a cooldown between runs, and
# separating position effect from candidate effect (ISSUES.md 2026-09-29
# P2 review, finding 4).
# ---------------------------------------------------------------------------


def _make_candidate(name: str) -> benchmark_cpu.Candidate:
    def predict_fn(rgb: np.ndarray) -> None:
        pass

    return benchmark_cpu.Candidate(name, predict_fn)


def test_run_suite_produces_one_record_per_round_per_candidate() -> None:
    candidates = [_make_candidate("a"), _make_candidate("b")]
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
    names = ("a", "b", "c", "d", "e")
    candidates = [_make_candidate(n) for n in names]
    frames = _make_frames(n=1)

    records = benchmark_cpu.run_suite(
        candidates,
        frames,
        warmup_frames=0,
        timed_frames=1,
        repeats=8,
        seed=0,
        cooldown_s=0.0,
        num_threads=1,
    )

    orders = []
    slot1_occupants = []
    for round_idx in range(1, 9):
        round_records = [r for r in records if r["round"] == round_idx]
        orders.append(tuple(r["candidate"] for r in round_records))
        slot1_occupants.append(next(r["candidate"] for r in round_records if r["slot"] == 1))

    assert len(set(orders)) > 1  # not every round used the same fixed order
    # Stronger than "more than one distinct order": no single candidate is
    # permanently stuck in (or locked out of) slot 1 across every round --
    # a fixed "always first" bias is exactly what shuffling is meant to
    # remove (ISSUES.md 2026-09-29 P2 review context: fixed order confounded
    # position with candidate identity in the very first provisional pass).
    assert len(set(slot1_occupants)) > 1


def test_run_suite_is_reproducible_given_the_same_seed() -> None:
    candidates = [_make_candidate(n) for n in ("a", "b", "c")]
    frames = _make_frames(n=1)

    first = benchmark_cpu.run_suite(
        candidates, frames, 0, 1, repeats=3, seed=7, cooldown_s=0.0, num_threads=1
    )
    second = benchmark_cpu.run_suite(
        candidates, frames, 0, 1, repeats=3, seed=7, cooldown_s=0.0, num_threads=1
    )

    assert [r["candidate"] for r in first] == [r["candidate"] for r in second]


def test_run_suite_records_torch_threads_on_every_run() -> None:
    torch = pytest.importorskip("torch")
    original_threads = torch.get_num_threads()
    try:
        candidates = [_make_candidate("a")]
        frames = _make_frames(n=1)

        records = benchmark_cpu.run_suite(
            candidates, frames, 0, 1, repeats=2, seed=0, cooldown_s=0.0, num_threads=2
        )

        assert all(r["torch_threads"] == torch.get_num_threads() for r in records)
        assert torch.get_num_threads() == 2
    finally:
        # run_suite leaves the process-global torch thread count changed by
        # design (that's the point of --num-threads); restore it so this
        # test doesn't leak state into whichever test runs next (ISSUES.md
        # 2026-09-29 P2 review, finding 6).
        torch.set_num_threads(original_threads)


def test_run_suite_records_onnx_intra_op_threads_only_for_that_candidate() -> None:
    plain = _make_candidate("plain")
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
    candidates = [_make_candidate("a"), _make_candidate("b")]
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


def test_summarize_by_position_normalizes_per_candidate_before_pooling() -> None:
    records = [
        _fake_record("a", 1, 1, 10.0),
        _fake_record("b", 1, 2, 20.0),
        _fake_record("b", 2, 1, 14.0),
        _fake_record("a", 2, 2, 22.0),
    ]
    # candidate means: a=(10+22)/2=16, b=(20+14)/2=17
    # normalized: a@slot1=10/16, b@slot2=20/17, b@slot1=14/17, a@slot2=22/16
    position_effect = benchmark_cpu.summarize_by_position(records)
    assert position_effect["1"]["n"] == 2
    assert position_effect["1"]["relative_mean"] == pytest.approx((10 / 16 + 14 / 17) / 2)
    assert position_effect["2"]["relative_mean"] == pytest.approx((20 / 17 + 22 / 16) / 2)


def test_summarize_by_position_reports_no_drift_for_constant_candidates_p2_repro() -> None:
    # P2's exact reproduction (ISSUES.md 2026-09-29 review, finding 1):
    # two candidates with perfectly constant per-repeat latency (A=10ms,
    # B=20ms), run in orders AB, AB, BA. The raw pooled-by-slot version
    # reported a fake "position effect" here (slot 1 = 13.33ms, slot 2 =
    # 16.67ms) even though neither candidate drifts at all. Normalizing
    # per candidate first must report exactly the same relative_mean at
    # every slot -- i.e. no drift.
    records = [
        _fake_record("A", 1, 1, 10.0),
        _fake_record("B", 1, 2, 20.0),
        _fake_record("A", 2, 1, 10.0),
        _fake_record("B", 2, 2, 20.0),
        _fake_record("B", 3, 1, 20.0),
        _fake_record("A", 3, 2, 10.0),
    ]
    position_effect = benchmark_cpu.summarize_by_position(records)
    assert position_effect["1"]["relative_mean"] == pytest.approx(1.0)
    assert position_effect["2"]["relative_mean"] == pytest.approx(1.0)


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
    skipped = [{"candidate": "rfdetr_nano_onnxruntime", "reason": "onnxruntime not importable"}]

    report = benchmark_cpu.build_report(
        records,
        skipped,
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
    assert report["skipped"] == skipped
    assert report["parity_check"] is None
    assert report["controls"] == {
        "repeats": 2,
        "seed": 0,
        "cooldown_s": 0.0,
        "num_threads_requested": 4,
        "candidate_order": "shuffled independently each repeat round, fixed seed",
    }
    assert "environment" in report and "cpu_model" in report["environment"]
    assert "library_versions" in report["environment"]
    assert report["per_run"] == records
    assert report["per_candidate_summary"]["fake_model"]["repeats"] == 2
    assert report["per_candidate_summary"]["fake_model"]["mean_ms_per_repeat"] == [10.0, 12.0]
    assert set(report["position_effect"]) == {"1"}

    # The report must be JSON-serializable as-is.
    json.dumps(report)


def test_build_report_names_the_forward_path_and_query_count_per_backend() -> None:
    # B1 correction (ISSUES.md 2026-09-29 P2 review, BLOCKING; 2026-09-30
    # correction entry): all three RF-DETR-Nano forward paths run in eval
    # mode (extract_pytorch_raw_outputs now ensures it -- see that
    # function's docstring), so all three are 300 queries. The plain
    # PyTorch path is directly comparable to "optimized" and "onnx", not
    # a mismatched 3900-query outlier.
    records = [
        _fake_record("rfdetr_nano_pytorch__full", 1, 1, 10.0),
        _fake_record("rfdetr_nano_pytorch_optimized__full", 1, 2, 8.0),
        _fake_record("rfdetr_nano_onnxruntime__full", 1, 3, 5.0),
        _fake_record("yolo11n_pytorch__full", 1, 4, 3.0),
    ]

    report = benchmark_cpu.build_report(
        records,
        [],
        warmup_frames=0,
        timed_frames=1,
        frame_width=10,
        frame_height=8,
        repeats=1,
        seed=0,
        cooldown_s=0.0,
        num_threads=1,
    )

    forward_paths = report["forward_paths"]
    assert forward_paths["rfdetr_nano_pytorch__full"]["queries"] == 300
    assert forward_paths["rfdetr_nano_pytorch_optimized__full"]["queries"] == 300
    assert forward_paths["rfdetr_nano_onnxruntime__full"]["queries"] == 300
    assert forward_paths["yolo11n_pytorch__full"]["queries"] is None
    json.dumps(report)  # still JSON-serializable


def test_build_report_includes_the_parity_result_when_given() -> None:
    records = [_fake_record("fake_model", 1, 1, 10.0)]
    parity = benchmark_cpu.ParityResult(
        max_box_abs_diff=0.0001,
        max_logit_abs_diff=0.001,
        within_tolerance=True,
        box_atol=1e-3,
        logit_atol=1e-2,
    )

    report = benchmark_cpu.build_report(
        records,
        [],
        warmup_frames=0,
        timed_frames=1,
        frame_width=10,
        frame_height=8,
        repeats=1,
        seed=0,
        cooldown_s=0.0,
        num_threads=1,
        parity=parity,
    )

    assert report["parity_check"]["within_tolerance"] is True
    assert report["parity_check"]["max_box_abs_diff"] == pytest.approx(0.0001)

    # The report must be JSON-serializable as-is.
    json.dumps(report)


def test_collect_real_candidates_returns_nothing_without_allow_download() -> None:
    # No network, no import of rfdetr/ultralytics/onnxruntime attempted
    # for real candidates unless explicitly opted in.
    real = benchmark_cpu.collect_real_candidates(allow_download=False, num_threads=4)
    assert real.candidates == []
    assert real.skipped == []
    assert real.pytorch_model is None
    assert real.onnx_session is None


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


# ---------------------------------------------------------------------------
# _yolo11n_candidate: the BGR wiring, tested end to end with a fake
# ultralytics module injected via sys.modules -- no model, no network
# (ISSUES.md 2026-09-29 P2 review, finding 2: this was untested before).
# ---------------------------------------------------------------------------


def test_yolo_candidate_feeds_bgr_not_rgb_to_model_predict(monkeypatch) -> None:
    fake_instances: list[object] = []

    class _FakeYOLOModel:
        def __init__(self, path: str) -> None:
            self.path = path
            self.last_image = None
            fake_instances.append(self)

        def predict(self, image, verbose=False):
            self.last_image = image
            return "fake_result"

    fake_ultralytics = types.ModuleType("ultralytics")
    fake_ultralytics.YOLO = _FakeYOLOModel
    monkeypatch.setitem(sys.modules, "ultralytics", fake_ultralytics)

    candidate = benchmark_cpu._yolo11n_candidate()
    assert candidate is not None
    assert candidate.name == "yolo11n_pytorch__full"
    assert fake_instances[0].path == str(benchmark_cpu._YOLO_CACHE_PATH)

    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[0, 0] = (30, 20, 10)  # RGB: R=30, G=20, B=10

    candidate.predict_fn(rgb)

    received = fake_instances[0].last_image
    assert received is not None
    assert tuple(received[0, 0]) == (10, 20, 30)  # converted to BGR before predict()


# ---------------------------------------------------------------------------
# _rfdetr_onnx_candidate: exercised against a fully faked rfdetr + ONNX
# Runtime stack, injected via sys.modules -- no model, no export, no
# network (ISSUES.md 2026-09-29 P2 review, finding 2: no test at all
# before this session).
# ---------------------------------------------------------------------------


def _install_fake_rfdetr_onnx_stack(monkeypatch: pytest.MonkeyPatch) -> dict:
    calls: dict = {"constructed": 0, "run_calls": 0}
    exported_path = "C:/fake/rfdetr_benchmark/model.onnx"

    class _FakeRFDETRNano:
        def __init__(self) -> None:
            calls["constructed"] += 1

        def export(self, output_dir, format, fp16, verbose):  # noqa: A002
            calls["export_output_dir"] = output_dir
            calls["export_format"] = format
            calls["export_fp16"] = fp16
            return exported_path

    rfdetr_mod = types.ModuleType("rfdetr")
    rfdetr_mod.RFDETRNano = _FakeRFDETRNano
    export_mod = types.ModuleType("rfdetr.export")
    runtime_mod = types.ModuleType("rfdetr.export._runtime")
    decode_mod = types.ModuleType("rfdetr.export._runtime.decode")
    preprocess_mod = types.ModuleType("rfdetr.export._runtime.preprocess")

    def fake_decode_detections(boxes_cwh, logits, image_size, threshold, background_class_id):
        calls["decode_threshold"] = threshold
        calls["decode_background_class_id"] = background_class_id
        return "fake_decoded_detections"

    def fake_preprocess_to_nchw(pil_img, height, width, channels):
        calls["preprocess_calls"] = calls.get("preprocess_calls", 0) + 1
        return np.zeros((1, channels, height, width), dtype=np.float32)

    decode_mod.decode_detections = fake_decode_detections
    preprocess_mod.preprocess_to_nchw = fake_preprocess_to_nchw

    monkeypatch.setitem(sys.modules, "rfdetr", rfdetr_mod)
    monkeypatch.setitem(sys.modules, "rfdetr.export", export_mod)
    monkeypatch.setitem(sys.modules, "rfdetr.export._runtime", runtime_mod)
    monkeypatch.setitem(sys.modules, "rfdetr.export._runtime.decode", decode_mod)
    monkeypatch.setitem(sys.modules, "rfdetr.export._runtime.preprocess", preprocess_mod)

    class _FakeSessionOptions:
        def __init__(self) -> None:
            self.intra_op_num_threads = None
            self.entries: dict[str, str] = {}

        def add_session_config_entry(self, key, value):
            self.entries[key] = value

    class _FakeIOMeta:
        def __init__(self, name: str, shape=None) -> None:
            self.name = name
            self.shape = shape

    class _FakeSession:
        def __init__(self, path, sess_options, providers) -> None:
            calls["session_path"] = path
            calls["session_options"] = sess_options
            calls["session_providers"] = providers

        def get_inputs(self):
            return [_FakeIOMeta("input", (1, 3, 8, 8))]

        def get_outputs(self):
            return [_FakeIOMeta("dets"), _FakeIOMeta("labels")]

        def run(self, output_names, feed):
            calls["run_calls"] += 1
            return [
                np.zeros((1, 300, 4), dtype=np.float32),
                np.zeros((1, 300, 91), dtype=np.float32),
            ]

    ort_mod = types.ModuleType("onnxruntime")
    ort_mod.SessionOptions = _FakeSessionOptions
    ort_mod.InferenceSession = _FakeSession
    monkeypatch.setitem(sys.modules, "onnxruntime", ort_mod)

    return calls


def test_rfdetr_onnx_candidate_builds_full_and_network_only_views(monkeypatch) -> None:
    calls = _install_fake_rfdetr_onnx_stack(monkeypatch)

    candidates, session, input_meta, skip_reason = benchmark_cpu._rfdetr_onnx_candidate(
        num_threads=2
    )

    assert skip_reason is None
    assert session is not None
    assert input_meta == (3, 8, 8)
    assert {c.name for c in candidates} == {
        "rfdetr_nano_onnxruntime__full",
        "rfdetr_nano_onnxruntime__network_only",
    }
    assert all(c.onnx_intra_op_threads == 2 for c in candidates)

    # The export dir is the explicit out-of-repo cache, never the
    # rfdetr default -- and fp16 is explicitly off (module docstring).
    assert calls["export_output_dir"] == str(benchmark_cpu._ONNX_EXPORT_CACHE_DIR)
    assert calls["export_fp16"] is False

    # Thread control and both spinning flags (ISSUES.md 2026-09-29 P2
    # review, finding 5: the earlier comment claimed both but only set one).
    assert calls["session_options"].intra_op_num_threads == 2
    assert calls["session_options"].entries["session.intra_op.allow_spinning"] == "0"
    assert calls["session_options"].entries["session.inter_op.allow_spinning"] == "0"

    # background_class_id=None for this sparse COCO-pretrained checkpoint.
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    full = next(c for c in candidates if c.name.endswith("__full"))
    full.predict_fn(rgb)
    assert calls["run_calls"] == 1
    assert calls["decode_background_class_id"] is None

    # network_only reuses one fixed tensor and never calls decode/PIL again
    # per call -- only session.run.
    preprocess_calls_before = calls["preprocess_calls"]
    network_only = next(c for c in candidates if c.name.endswith("__network_only"))
    network_only.predict_fn(rgb)
    network_only.predict_fn(rgb)
    assert calls["run_calls"] == 3  # 1 from __full + 2 from __network_only
    assert calls["preprocess_calls"] == preprocess_calls_before  # unchanged: no new preprocessing


def test_rfdetr_onnx_candidate_reports_skip_reason_and_warns_when_unimportable(
    monkeypatch, caplog: pytest.LogCaptureFixture
) -> None:
    # None in sys.modules forces `import onnxruntime` to raise ImportError,
    # regardless of whether it's actually installed in this environment.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with caplog.at_level(logging.WARNING, logger="training.benchmark_cpu"):
        candidates, session, input_meta, skip_reason = benchmark_cpu._rfdetr_onnx_candidate(
            num_threads=1
        )

    assert candidates == []
    assert session is None
    assert input_meta is None
    assert skip_reason is not None and "onnxruntime" in skip_reason
    assert any(record.levelno == logging.WARNING for record in caplog.records)
