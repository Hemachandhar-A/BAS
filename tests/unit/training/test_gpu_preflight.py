"""F14 Stage 6 (essential-features.md section 14): training/gpu_preflight.py.
Pure helpers are tested directly; the real model forward/backward is behind a
skip when the pretrained weights are not already cached locally."""

from __future__ import annotations

import pytest

from training import gpu_preflight as gp

pytestmark = pytest.mark.F14


def test_iterations_per_epoch_is_ceil_of_images_over_micro_batch():
    assert gp.iterations_per_epoch(n_train=470, batch_size=4) == 118
    assert gp.iterations_per_epoch(n_train=8, batch_size=4) == 2
    assert gp.iterations_per_epoch(n_train=1, batch_size=4) == 1


@pytest.mark.parametrize("n,b", [(0, 4), (-3, 4), (10, 0)])
def test_iterations_per_epoch_rejects_nonpositive(n, b):
    with pytest.raises(ValueError):
        gp.iterations_per_epoch(n_train=n, batch_size=b)


def test_extrapolate_scales_linearly_and_is_labelled_an_estimate():
    est = gp.extrapolate(seconds_per_iteration=0.5, n_train=470, batch_size=4, epochs=100)
    assert est["iterations_per_epoch"] == 118
    assert est["seconds_per_epoch"] == pytest.approx(59.0)
    assert est["total_seconds"] == pytest.approx(5900.0)
    assert est["is_estimate"] is True
    assert "validation" in est["excludes"]


def test_summarise_times_drops_first_iteration_as_warmup_when_enough_samples():
    s = gp.summarise_times([10.0, 1.0, 3.0])
    assert s["seconds_per_iteration"] == pytest.approx(2.0)
    assert s["warmup_dropped"] is True
    assert s["n_timed"] == 2


def test_summarise_times_keeps_everything_for_one_or_two_samples():
    s = gp.summarise_times([4.0, 2.0])
    assert s["seconds_per_iteration"] == pytest.approx(3.0)
    assert s["warmup_dropped"] is False
    assert gp.summarise_times([5.0])["seconds_per_iteration"] == pytest.approx(5.0)


def test_summarise_times_empty_raises():
    with pytest.raises(ValueError):
        gp.summarise_times([])


def test_time_iterations_uses_injected_clock_and_syncs_each_step():
    calls = []
    ticks = iter([0.0, 1.0, 1.0, 3.0])

    def step():
        calls.append("step")

    times = gp.time_iterations(
        step, 2, clock=lambda: next(ticks), sync=lambda: calls.append("sync")
    )
    assert times == [1.0, 2.0]
    assert calls == ["step", "sync", "step", "sync"]


def test_format_report_labels_estimate_and_cpu_build():
    info = {
        "gpu_name": None,
        "vram_gb": None,
        "cuda_available": False,
        "torch_build": "cpu",
        "torch": "2.14.0+cpu",
        "rfdetr": "1.11.0",
        "device": "cpu",
    }
    timing = {"seconds_per_iteration": 7.5, "n_timed": 2, "warmup_dropped": False}
    est = gp.extrapolate(7.5, n_train=470, batch_size=4, epochs=100)
    text = gp.format_report("rfdetr", info, timing, est, batch_size=4)
    assert "CPU build" in text
    assert "no CUDA device" in text
    assert "ESTIMATE" in text
    assert "7.50" in text
    assert "surrogate" in text


def test_format_report_shows_gpu_name_and_vram():
    info = {
        "gpu_name": "NVIDIA GeForce RTX 4090",
        "vram_gb": 23.99,
        "cuda_available": True,
        "torch_build": "cuda 12.4",
        "torch": "2.5.0+cu124",
        "rfdetr": "1.11.0",
        "device": "cuda",
    }
    timing = {"seconds_per_iteration": 0.2, "n_timed": 4, "warmup_dropped": True}
    est = gp.extrapolate(0.2, 470, 4, 100)
    text = gp.format_report("rfdetr", info, timing, est, batch_size=4)
    assert "RTX 4090" in text and "23.99" in text and "cuda 12.4" in text


def test_require_cuda_without_cuda_exits_nonzero_with_message(monkeypatch, capsys):
    monkeypatch.setattr(
        gp,
        "machine_info",
        lambda device="auto": {
            "gpu_name": None,
            "vram_gb": None,
            "cuda_available": False,
            "torch_build": "cpu",
            "torch": "x",
            "rfdetr": "y",
            "device": "cpu",
        },
    )
    code = gp.main(["--model", "rfdetr", "--n-train", "10", "--epochs", "1", "--require-cuda"])
    assert code == 2
    err = capsys.readouterr().err
    assert "CUDA" in err and "CPU build" in err


def test_weights_missing_raises_instead_of_downloading(tmp_path):
    with pytest.raises(gp.WeightsMissing) as e:
        gp.require_file(tmp_path / "nope.pth", "RF-DETR-Nano")
    assert "nope.pth" in str(e.value)
    assert "download" in str(e.value).lower()


def test_require_file_returns_existing_path(tmp_path):
    f = tmp_path / "w.pth"
    f.write_bytes(b"x")
    assert gp.require_file(f, "x") == f


def _cached() -> bool:
    try:
        return gp.default_weights_path("rfdetr").exists()
    except Exception:
        return False


@pytest.mark.slow
@pytest.mark.skipif(not _cached(), reason="RF-DETR-Nano pretrained weights not cached locally")
def test_real_rfdetr_forward_backward_one_iteration_on_cpu():
    step = gp.build_step("rfdetr", batch_size=2, device="cpu", weights=None, seed=0)
    times = gp.time_iterations(step, 1)
    assert len(times) == 1 and times[0] > 0
