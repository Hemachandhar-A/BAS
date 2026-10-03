"""S-F2a: training/benchmark_valid.py. The pure helpers only (statistics, the pipeline view, the
frame draw, the timing loop with a fake callable); no model is loaded here."""

from __future__ import annotations

import pytest

from training import benchmark_valid as bv

pytestmark = pytest.mark.F14


def test_stats_median_p95_mean_in_ms():
    s = bv.latency_stats([0.010, 0.020, 0.030, 0.040, 0.100])
    assert s["median_ms"] == pytest.approx(30.0)
    assert s["mean_ms"] == pytest.approx(40.0)
    assert s["p95_ms"] == pytest.approx(88.0)  # 40 + 0.8 * (100 - 40)
    assert s["n"] == 5


def test_stats_refuse_an_empty_list():
    with pytest.raises(ValueError):
        bv.latency_stats([])


def test_pipeline_view_every_frame_is_detector_plus_hands_plus_overhead():
    v = bv.pipeline_view(det_median_ms=100.0, hands_median_ms=20.0, overhead_ms=5.0)
    e1 = v["detector_every_1"]
    assert e1["frame_ms_with_detector"] == pytest.approx(125.0)
    assert e1["avg_frame_ms"] == pytest.approx(125.0)
    assert e1["fps"] == pytest.approx(8.0)
    assert e1["meets_budget"] is True  # 125 ms is exactly the 8 fps budget


def test_pipeline_view_skipping_detector_frames_lowers_the_average_not_the_worst_frame():
    v = bv.pipeline_view(100.0, 20.0, 5.0)
    e2, e3 = v["detector_every_2"], v["detector_every_3"]
    assert e2["avg_frame_ms"] == pytest.approx(75.0) and e2["fps"] == pytest.approx(1000 / 75)
    assert e3["avg_frame_ms"] == pytest.approx(100 / 3 + 25)
    assert e2["frame_ms_with_detector"] == pytest.approx(125.0)  # the slow frame is still slow


def test_pipeline_view_over_budget_is_flagged():
    assert bv.pipeline_view(101.0, 20.0, 5.0)["detector_every_1"]["meets_budget"] is False


def test_draw_frames_is_deterministic_and_cycles_when_short():
    names = [f"f{i}.jpg" for i in range(10)]
    a = bv.draw_frames(names, 25, seed=0)
    assert a == bv.draw_frames(names, 25, seed=0)
    assert len(a) == 25 and set(a) == set(names)
    assert a != bv.draw_frames(names, 25, seed=1)


def test_time_calls_discards_warmup_and_times_only_the_measured_calls():
    calls = []

    def fn(frame):
        calls.append(frame)

    ticks = iter(float(i) for i in range(1000))
    out = bv.time_calls(fn, list(range(7)), warmup=3, timed=4, clock=lambda: next(ticks))
    assert calls == list(range(7))  # warm-up frames run too, but are not in the result
    assert out == [1.0, 1.0, 1.0, 1.0]


def test_time_calls_needs_enough_frames():
    with pytest.raises(ValueError):
        bv.time_calls(lambda f: None, [1, 2, 3], warmup=2, timed=4)


def test_machine_facts_never_raise_and_have_the_core_keys():
    f = bv.machine_facts()
    for k in ("cpu", "logical_cores", "power_plan", "on_ac_power", "torch_threads"):
        assert k in f
