"""P1.7.1: the end-to-end pipeline throughput benchmark. Only the pure helpers are tested; the
measurement itself needs the real weights and videos and is run by hand."""

from __future__ import annotations

import pytest

from training import benchmark_pipeline as B

pytestmark = pytest.mark.F2


def test_summary_has_median_p95_and_the_implied_fps_from_the_median():
    s = B.summarize([0.1] * 90 + [0.2] * 10)
    assert s["n"] == 100
    assert s["median_ms"] == pytest.approx(100.0)
    assert s["p95_ms"] == pytest.approx(200.0)
    assert s["implied_fps"] == pytest.approx(10.0)
    assert s["implied_fps_p95"] == pytest.approx(5.0)


def test_summary_of_nothing_is_an_error_not_a_zero():
    with pytest.raises(ValueError):
        B.summarize([])


def test_clips_are_the_first_valid_runs_in_manifest_order(tmp_path):
    m = tmp_path / "manifest.csv"
    m.write_text("run_id,split\nx001,train\nx002,val\nx003,test\nx004,val\nx005,val\nx006,val\n")
    assert B.pick_clips(m, 3) == ["x002", "x004", "x005"]
    with pytest.raises(ValueError, match="only 4"):
        B.pick_clips(m, 5)
