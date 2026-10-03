"""Kernel changes of S-F1d, pure parts: the core-only constraints, the pip dry-run plan, the
temp-directory choice, the output size check, the stage runner and the stage table, and the
RF-DETR probe's result line. The real run happens on Kaggle; nothing here needs a GPU or pip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.kaggle import run_training as R

GB = 1024**3

# --- constraints: only the core ----------------------------------------------------------

IMAGE = {
    "torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128", "torchaudio": "2.11.0+cu128",
    "numpy": "2.1.3", "scipy": "1.16.3", "pillow": "12.3.0", "opencv-python": "4.14.0.94",
    "opencv-python-headless": "4.14.0.94", "torchmetrics": "1.9.0", "pytorch-lightning": "2.6.6",
    "peft": "0.20.0", "transformers": "5.16.1", "pycocotools": "2.0.11", "pydantic": "2.13.5",
}  # fmt: skip


def test_only_the_core_is_frozen_and_pure_python_libraries_are_not():
    assert set(R.FROZEN) == {
        "torch", "torchvision", "torchaudio", "numpy", "scipy", "pillow",
        "opencv-python", "opencv-python-headless", "opencv-contrib-python",
    }  # fmt: skip
    lines = R.constraints_text(IMAGE).splitlines()
    assert {line.split("==")[0] for line in lines} == set(R.FROZEN) - {"opencv-contrib-python"}
    for free in (
        "torchmetrics", "pytorch-lightning", "peft", "transformers", "pycocotools", "pydantic"
    ):  # fmt: skip
        assert not any(line.startswith(free) for line in lines)
    assert "torch==2.11.0+cu128" in lines and "numpy==2.1.3" in lines


def test_opencv_contrib_is_frozen_when_the_image_has_it():
    text = R.constraints_text({**IMAGE, "opencv-contrib-python": "4.14.0.94"})
    assert "opencv-contrib-python==4.14.0.94" in text.splitlines()


def test_the_pip_commands_install_with_constraints_and_dry_run_only_when_asked():
    real = R.pip_install_cmd(["a==1"], Path("c.txt"))
    dry = R.pip_install_cmd(["a==1"], Path("c.txt"), dry_run=True)
    assert "--dry-run" not in real and "--dry-run" in dry
    assert set(real) <= set(dry)
    for bad in ("--upgrade", "-U", "--force-reinstall", "--ignore-installed"):
        assert bad not in dry


def test_pip_dry_run_support_is_read_from_its_help():
    assert R.pip_supports_dry_run("  --dry-run   Don't actually install anything")
    assert not R.pip_supports_dry_run("Usage: pip install [options] <requirement>")


# --- dry-run plan -------------------------------------------------------------------------

PLAN = """\
Collecting rfdetr==1.11.0
Requirement already satisfied: numpy in /usr/lib (2.1.3)
Would install av-14.4.0 pyDeprecate-0.9.0 torchmetrics-1.8.2 ultralytics-thop-2.2.0 rfdetr-1.11.0 torch_hungarian-0.1.0rc0
"""  # noqa: E501


def test_the_dry_run_plan_is_parsed_into_normalised_names_and_versions():
    plan = R.parse_dry_run_plan(PLAN)
    assert ("av", "14.4.0") in plan and ("pydeprecate", "0.9.0") in plan
    assert ("ultralytics-thop", "2.2.0") in plan and ("torch-hungarian", "0.1.0rc0") in plan
    assert dict(plan)["torchmetrics"] == "1.8.2"
    assert R.parse_dry_run_plan("Requirement already satisfied: x") == []


def test_a_plan_that_touches_the_core_is_detected_whatever_the_spelling():
    plan = R.parse_dry_run_plan(
        "Would install Pillow-12.4.0 opencv_python_headless-4.15.0 av-14.4.0"
    )
    assert R.plan_core_changes(plan) == ["opencv-python-headless", "pillow"]
    assert R.plan_core_changes(R.parse_dry_run_plan(PLAN)) == []
    assert R.plan_core_changes(R.parse_dry_run_plan("Would install torch-2.12.0+cu128")) == [
        "torch"
    ]


def test_plan_and_after_install_are_described_as_added_or_changed_packages():
    before = {"torchmetrics": "1.9.0", "numpy": "2.1.3"}
    after = {"torchmetrics": "1.8.2", "numpy": "2.1.3", "av": "14.4.0"}
    assert R.describe_changes(before, after) == [
        "added   av 14.4.0",
        "changed torchmetrics 1.9.0 -> 1.8.2",
    ]
    lines = R.describe_changes(before, R.apply_plan(before, R.parse_dry_run_plan(PLAN)))
    assert "changed torchmetrics 1.9.0 -> 1.8.2" in lines and "added   av 14.4.0" in lines
    assert R.describe_changes(before, before) == []


# --- temp directory -----------------------------------------------------------------------


def test_the_temp_root_is_kaggle_temp_when_it_exists_and_has_space():
    free = {"/kaggle/temp": 50 * GB, "/tmp/work": 10 * GB}
    path, why = R.choose_temp_root(free.get, need_bytes=4 * GB)
    assert path == Path("/kaggle/temp") and "/kaggle/temp" in why


def test_the_temp_root_falls_back_when_kaggle_temp_is_missing_or_too_small():
    path, why = R.choose_temp_root({"/tmp/work": 10 * GB}.get, need_bytes=4 * GB)
    assert path == Path("/tmp/work") and "/kaggle/temp" in why and "/tmp/work" in why
    path, why = R.choose_temp_root({"/kaggle/temp": 1 * GB, "/tmp/work": 10 * GB}.get, 4 * GB)
    assert path == Path("/tmp/work") and "too small" in why


def test_no_temp_root_with_enough_space_is_an_error_naming_both():
    with pytest.raises(R.DiscoveryError, match=r"/kaggle/temp.*/tmp/work"):
        R.choose_temp_root({"/kaggle/temp": 1 * GB, "/tmp/work": 2 * GB}.get, 4 * GB)
    with pytest.raises(R.DiscoveryError):
        R.choose_temp_root({}.get, 4 * GB)


def test_the_free_space_probe_uses_the_nearest_existing_folder(tmp_path):
    assert R.free_bytes(tmp_path / "not" / "yet") is not None
    assert R.free_bytes(tmp_path / "not" / "yet", must_exist=True) is None


# --- output size --------------------------------------------------------------------------


def test_output_size_check_totals_the_tree_and_fails_over_the_limit(tmp_path):
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "a.bin").write_bytes(b"x" * 600)
    (tmp_path / "stray").mkdir()
    (tmp_path / "stray" / "b.bin").write_bytes(b"x" * 1500)
    assert R.dir_size(tmp_path) == 2100
    ok, lines = R.output_size_check(tmp_path, limit_bytes=3000)
    assert ok and "2.1 KB" in lines[0] and "OK" in lines[0]
    ok, lines = R.output_size_check(tmp_path, limit_bytes=2000)
    text = "\n".join(lines)
    assert not ok and "EXCEEDS" in lines[0] and "stray" in text  # the biggest entry is named
    assert R.OUTPUT_LIMIT_BYTES == 2 * GB


def test_output_size_check_on_a_missing_folder_is_zero(tmp_path):
    assert R.dir_size(tmp_path / "nope") == 0


# --- stage runner -------------------------------------------------------------------------


def test_tail_lines_keeps_the_last_n():
    text = "\n".join(f"line {i}" for i in range(100))
    out = R.tail_lines(text, 30)
    assert len(out) == 30 and out[0] == "line 70" and out[-1] == "line 99"
    assert R.tail_lines("", 30) == []


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        self.t += 2.5
        return self.t


def test_a_stage_that_succeeds_records_ok_seconds_and_no_tail(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("fine\n")
    rec = R.run_stage("a", lambda: {"ok": True, "log": log, "detail": {"x": 1}}, clock=Clock())
    assert rec["name"] == "a" and rec["ok"] is True and rec["status"] == "ok"
    assert rec["seconds"] == 2.5 and rec["tail"] == [] and rec["detail"] == {"x": 1}


def test_a_failing_stage_keeps_the_last_30_log_lines_and_the_run_goes_on(tmp_path):
    log = tmp_path / "s.log"
    log.write_text("\n".join(f"out {i}" for i in range(80)))
    bad = R.run_stage("b", lambda: {"ok": False, "log": log, "message": "exit 1"}, clock=Clock())
    assert bad["status"] == "failed" and len(bad["tail"]) == 30 and bad["tail"][-1] == "out 79"
    assert bad["message"] == "exit 1" and bad["ok"] is False

    def boom():
        raise RuntimeError("kaput")

    crashed = R.run_stage("c", boom, clock=Clock())
    assert crashed["status"] == "failed" and any("kaput" in x for x in crashed["tail"])
    assert crashed["message"].startswith("RuntimeError")
    # the runner never raises: a good stage after the crash still runs and is recorded
    after = R.run_stage("d", lambda: {"ok": True}, clock=Clock())
    assert after["status"] == "ok"
    assert len(R.stage_table([bad, crashed, after])) == 4  # header + 3 rows


def test_stage_table_and_verdict():
    ok = {"name": "preflight_rfdetr", "status": "ok", "seconds": 12.0}
    bad = {"name": "dry_run_rfdetr", "status": "failed", "seconds": 300.0, "message": "timed out"}
    skipped = {"name": "train_probe_rfdetr", "status": "skipped", "seconds": 0.0,
               "message": "dry_run_rfdetr failed"}  # fmt: skip
    text = "\n".join(R.stage_table([ok, bad, skipped]))
    assert "preflight_rfdetr" in text and "failed" in text and "skipped" in text
    assert "300.0" in text and "timed out" in text
    assert R.smoke_verdict([ok]) == "SMOKE OK"
    assert R.smoke_verdict([ok, bad, skipped]) == (
        "SMOKE PARTIAL: dry_run_rfdetr, train_probe_rfdetr"
    )
    assert R.smoke_verdict([]) == "SMOKE PARTIAL: no stage ran"


def test_subprocess_result_is_ok_only_on_exit_zero_without_timeout(tmp_path):
    def res(rc, timed_out, s):
        return R.subprocess_result(
            {"returncode": rc, "timed_out": timed_out, "seconds": s}, tmp_path / "l"
        )

    assert res(0, False, 1.0)["ok"] is True
    bad = res(3, False, 1.0)
    assert bad["ok"] is False and "exit 3" in bad["message"]
    hung = res(-15, True, 300)
    assert hung["ok"] is False and "timed out" in hung["message"] and "300" in hung["message"]
    assert R.STAGE_TIMEOUT_S == 300


def test_smoke_report_carries_the_stage_table_and_the_verdict():
    stages = [
        {"name": "a", "status": "ok", "ok": True, "seconds": 1.0, "tail": [], "detail": {}},
        {"name": "b", "status": "failed", "ok": False, "seconds": 2.0, "tail": ["x"], "detail": {}},
    ]  # fmt: skip
    rep = R.smoke_report(
        machine={}, n_train=593, epochs=100, preflight={}, dry_runs={}, smoke_seconds=3.0,
        pip_seconds=None, dataset_stamp="s", problems=["b failed"], stages=stages,
    )  # fmt: skip
    assert rep["stages"] == stages and rep["verdict"] == "SMOKE PARTIAL: b" and not rep["ok"]
    json.dumps(rep)
    rep2 = R.smoke_report(
        machine={}, n_train=1, epochs=1, preflight={}, dry_runs={}, smoke_seconds=1.0,
        pip_seconds=None, dataset_stamp="s", problems=[], stages=stages[:1],
    )  # fmt: skip
    assert rep2["verdict"] == "SMOKE OK" and rep2["ok"]


# --- the RF-DETR probe --------------------------------------------------------------------


def test_the_probe_source_compiles_and_its_result_line_is_parsed():
    compile(R.PROBE_SOURCE, "<probe>", "exec")
    out = (
        'noise\nPROBE_RESULT {"iterations": 3, "seconds_per_iteration": 0.9, "gpu_peak_mb": 4100}\n'
    )
    assert R.parse_probe(out)["gpu_peak_mb"] == 4100
    assert R.parse_probe("no result") is None
    assert R.parse_probe("PROBE_RESULT not json") is None
    cmd = R.probe_cmd(Path("d"), Path("w.pth"), Path("o"), 3)
    assert cmd[1] == "-c" and cmd[2] == R.PROBE_SOURCE and cmd[-1] == "3"


# --- run_smoke with the subprocesses faked ------------------------------------------------


def fake_smoke(tmp_path, monkeypatch, fail: set[str]):
    out, work, code = tmp_path / "work" / "outputs", tmp_path / "work", tmp_path / "code"
    (code / "reports").mkdir(parents=True)
    (code / "reports" / "dataset.json").write_text(
        json.dumps({"splits": {"train": {"images": 593}}, "dataset_stamp": "stamp"})
    )
    monkeypatch.setattr(R, "OUT", out)
    monkeypatch.setattr(R, "WORK", work)
    out.mkdir(parents=True)
    ran: list[str] = []

    probe_line = "PROBE_RESULT " + json.dumps(
        {"iterations": 3, "seconds_each": [2, 1, 1], "seconds_per_iteration": 1.0,
         "gpu_peak_mb": 3000, "gpu_reserved_peak_mb": 3500}
    )  # fmt: skip

    def fake_run_logged(cmd, log, *, cwd, timeout_s, env=None):
        if cmd[1] == "-c":
            key = "probe"
        else:
            kind = "preflight_" if "gpu_preflight" in cmd[2] else "dry_run_"
            key = kind + cmd[cmd.index("--model") + 1]
        ran.append(key)
        assert timeout_s == R.STAGE_TIMEOUT_S
        extra = (probe_line + "\n") if key == "probe" else ""
        Path(log).write_text("line a\nline b\n" + extra)
        return {"returncode": 1 if key in fail else 0, "timed_out": False, "seconds": 1.0}

    monkeypatch.setattr(R, "run_logged", fake_run_logged)
    paths = {
        "code_root": code, "data_root": tmp_path / "data", "tmp_root": tmp_path / "tmp",
        "tmp_note": "using test",
        "weights": {"rf-detr-nano.pth": Path("w"), "yolo11n.pt": Path("y")},
    }  # fmt: skip
    return paths, out, ran


def test_one_failing_stage_does_not_stop_the_others_and_the_verdict_names_it(
    tmp_path, monkeypatch, capsys
):
    paths, out, ran = fake_smoke(tmp_path, monkeypatch, {"preflight_rfdetr"})
    rc = R.run_smoke(paths, {"gpus_seen": 2}, {"seconds": 5.0, "changes": ["added   av 1"]})
    text = capsys.readouterr().out
    assert ran == [
        "preflight_rfdetr",
        "preflight_yolo11n",
        "dry_run_rfdetr",
        "dry_run_yolo11n",
        "probe",
    ]
    assert rc == 1 and text.rstrip().splitlines()[-1] == "SMOKE PARTIAL: preflight_rfdetr"
    rep = json.loads((out / "smoke_report.json").read_text())
    assert [s["status"] for s in rep["stages"]] == ["failed", "ok", "ok", "ok", "ok"]
    assert (
        rep["stages"][0]["tail"] == ["line a", "line b"]
        and rep["stages"][4]["detail"]["probe"]["gpu_peak_mb"] == 3000
    )
    assert rep["verdict"] == "SMOKE PARTIAL: preflight_rfdetr" and rep["pip_changes"] == [
        "added   av 1"
    ]
    assert rep["outputs_size_ok"] and "STAGE TABLE" in text and "3000" in text


def test_all_stages_ok_ends_with_smoke_ok_and_a_failed_rfdetr_dry_run_skips_the_probe(
    tmp_path, monkeypatch, capsys
):
    paths, out, ran = fake_smoke(tmp_path, monkeypatch, set())
    assert R.run_smoke(paths, {}, {}) == 0
    assert capsys.readouterr().out.rstrip().splitlines()[-1] == "SMOKE OK"
    paths, out, ran = fake_smoke(tmp_path / "b", monkeypatch, {"dry_run_rfdetr"})
    R.run_smoke(paths, {}, {})
    assert "probe" not in ran
    rep = json.loads((out / "smoke_report.json").read_text())
    assert rep["stages"][-1]["status"] == "skipped"
    assert rep["verdict"] == "SMOKE PARTIAL: dry_run_rfdetr, train_probe_rfdetr"


def test_an_oversized_output_fails_the_run_but_not_the_stages(tmp_path, monkeypatch, capsys):
    paths, out, _ = fake_smoke(tmp_path, monkeypatch, set())
    monkeypatch.setattr(R, "OUTPUT_LIMIT_BYTES", 10)
    (out / "big.bin").write_bytes(b"x" * 100)
    assert R.run_smoke(paths, {}, {}) == 1
    text = capsys.readouterr().out
    assert "OUTPUT TOO LARGE" in text and "outputs/big.bin" in text
    assert text.rstrip().splitlines()[-1] == "SMOKE OK"  # the stages themselves passed


# --- install_pins with pip faked ----------------------------------------------------------


def fake_pip(tmp_path, monkeypatch, plan_text: str, core_after: dict | None = None):
    out, code = tmp_path / "out", tmp_path / "code"
    out.mkdir()
    code.mkdir()
    (code / "uv.lock").write_text((Path(R.__file__).parents[2] / "uv.lock").read_text("utf-8"))
    monkeypatch.setattr(R, "OUT", out)
    core = {"torch": "2.11.0+cu128", "numpy": "2.1.3"}
    state = {"all": {**core, "torchmetrics": "1.9.0"}, "core": core}
    seen = []

    def installed_versions():
        seen.append(1)  # the first call is "before"; later calls see core_after when given
        return dict(core_after if (core_after and len(seen) > 1) else state["core"])

    monkeypatch.setattr(R, "installed_versions", installed_versions)
    monkeypatch.setattr(R, "installed_all", lambda: dict(state["all"]))
    monkeypatch.setattr(R, "capture", lambda cmd, timeout_s=60: "  --dry-run  x")
    calls: list[list[str]] = []

    def run_logged(cmd, log, *, cwd, timeout_s, env=None):
        calls.append(cmd)
        dry = "--dry-run" in cmd
        Path(log).write_text(plan_text if dry else "Successfully installed x\n")
        if not dry:
            state["all"] = {**state["all"], "torchmetrics": "1.8.2", "av": "14.4.0"}
        return {"returncode": 0, "timed_out": False, "seconds": 2.0}

    monkeypatch.setattr(R, "run_logged", run_logged)
    return code, out, calls


def test_install_pins_dry_runs_first_prints_the_plan_and_reports_what_changed(
    tmp_path, monkeypatch, capsys
):
    code, out, calls = fake_pip(tmp_path, monkeypatch, PLAN)
    res = R.install_pins(code)
    text = capsys.readouterr().out
    assert "--dry-run" in calls[0] and "--dry-run" not in calls[1]  # plan first, then the install
    assert (
        "changed torchmetrics 1.9.0 -> 1.8.2" in res["changes"]
        and "added   av 14.4.0" in res["changes"]
    )
    assert "plan leaves the core alone" in text and "core packages unchanged" in text
    assert (out / "constraints.txt").read_text().splitlines() == [
        "torch==2.11.0+cu128",
        "numpy==2.1.3",
    ]


def test_install_pins_refuses_a_plan_that_changes_a_core_package(tmp_path, monkeypatch):
    code, out, calls = fake_pip(tmp_path, monkeypatch, "Would install torch-2.12.0 av-14.4.0")
    with pytest.raises(SystemExit) as e:
        R.install_pins(code)
    assert e.value.code == 3 and len(calls) == 1  # only the dry run ran; nothing was installed


def test_install_pins_fails_when_the_core_moved_after_the_install(tmp_path, monkeypatch):
    code, out, calls = fake_pip(
        tmp_path, monkeypatch, PLAN, core_after={"torch": "9", "numpy": "2.1.3"}
    )
    with pytest.raises(SystemExit) as e:
        R.install_pins(code)
    assert e.value.code == 3 and len(calls) == 2
