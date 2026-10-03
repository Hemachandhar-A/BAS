"""training/kaggle_pack.py: the package is built from local files only, deterministically, and
carries no credential. Nothing here contacts Kaggle or any network."""

from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

import pytest

from training import kaggle_pack as K
from training.kaggle import run_training as R

CLASSES = ["outer_box", "tray", "red_box", "yellow_box", "start_button"]


def _dataset(tmp_path):
    from training.finetune import make_synthetic_dataset

    return make_synthetic_dataset(tmp_path / "dataset", CLASSES)


def test_zip_is_deterministic_whatever_the_file_times_and_order(tmp_path):
    ds = _dataset(tmp_path)
    entries = K.collect_dataset_files(ds)
    a = K.deterministic_zip(tmp_path / "a.zip", entries)
    import os
    import time

    for _, p in entries:
        os.utime(p, (time.time() + 5000, time.time() + 5000))
    b = K.deterministic_zip(tmp_path / "b.zip", list(reversed(entries)))
    assert a["sha256"] == b["sha256"]
    assert (tmp_path / "a.zip").read_bytes() == (tmp_path / "b.zip").read_bytes()
    assert a["files"] == K.zip_file_hashes(tmp_path / "a.zip")
    with zipfile.ZipFile(tmp_path / "a.zip") as z:
        names = z.namelist()
        assert names == sorted(names)
        assert all(
            i.date_time == K.ZIP_DATE and i.compress_type == zipfile.ZIP_STORED
            for i in z.infolist()
        )
    assert len([n for n in names if n.endswith(".jpg")]) == 6
    assert all(n.split("/")[0] in K.SPLIT_FOLDERS for n in names)


def test_zip_changes_when_a_file_changes(tmp_path):
    ds = _dataset(tmp_path)
    a = K.deterministic_zip(tmp_path / "a.zip", K.collect_dataset_files(ds))
    p = ds / "test" / "_annotations.coco.json"
    p.write_bytes(p.read_bytes() + b" ")
    b = K.deterministic_zip(tmp_path / "b.zip", K.collect_dataset_files(ds))
    assert a["sha256"] != b["sha256"]


def test_a_missing_split_folder_is_refused(tmp_path):
    ds = _dataset(tmp_path)
    import shutil

    shutil.rmtree(ds / "valid")
    with pytest.raises(K.PackError, match="valid"):
        K.collect_dataset_files(ds)


def test_dataset_metadata_is_private_by_default_with_a_cli_valid_licence():
    for meta, slug in (
        (K.dataset_metadata("example-owner"), "sih26174-dataset"),
        (K.code_metadata("example-owner"), "sih26174-code"),
    ):
        assert meta["id"] == f"example-owner/{slug}"
        assert meta["licenses"] == [{"name": "other"}]  # one of the CLI's own license names
        assert 6 <= len(meta["title"]) <= 50 and 6 <= len(slug) <= 50  # limits in the CLI
        assert set(meta) <= {"title", "id", "licenses", "description"}
        assert "isPrivate" not in meta  # `datasets create` is private unless -u is passed


def test_kernel_metadata_uses_only_template_keys_and_both_datasets():
    meta = K.kernel_metadata("example-owner")
    template = {
        "id", "title", "code_file", "language", "kernel_type", "is_private", "enable_gpu",
        "enable_tpu", "enable_internet", "machine_shape", "dataset_sources",
        "competition_sources", "kernel_sources", "model_sources",
    }  # fmt: skip
    assert set(meta) <= template
    assert meta["id"] == "example-owner/sih26174-train"
    assert meta["kernel_type"] == "script" and meta["code_file"] == "run_training.py"
    assert meta["is_private"] == "true" and meta["enable_gpu"] == "true"
    assert meta["enable_internet"] == "true"
    assert meta["dataset_sources"] == [
        "example-owner/sih26174-dataset",
        "example-owner/sih26174-code",
    ]
    assert 5 <= len(meta["title"])  # the CLI refuses shorter titles
    assert meta["title"].lower().replace(" ", "-") == meta["id"].split("/")[1]  # slug matches title


def test_weights_manifest_records_sha_size_and_licence(tmp_path):
    rf, yo = tmp_path / "rf-detr-nano.pth", tmp_path / "yolo11n.pt"
    rf.write_bytes(b"rf")
    yo.write_bytes(b"yolo!")
    m = K.weights_manifest({"rf-detr-nano.pth": rf, "yolo11n.pt": yo})
    assert (
        m["rf-detr-nano.pth"]["licence"] == "Apache-2.0"
        and m["yolo11n.pt"]["licence"] == "AGPL-3.0"
    )
    assert m["yolo11n.pt"]["size_bytes"] == 5 and m["rf-detr-nano.pth"]["sha256"] == K.sha256_file(
        rf
    )


def test_secret_scan_finds_credential_names_and_contents(tmp_path):
    (tmp_path / "ok.json").write_text('{"id": "example-owner/x"}')
    assert K.secret_problems(tmp_path) == []
    (tmp_path / "kaggle.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("token KGAT_abcdef123456")
    (tmp_path / "cfg.json").write_text('{"key": "' + "a1" * 16 + '"}')
    problems = K.secret_problems(tmp_path)
    assert len(problems) == 3


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "training").mkdir(parents=True)
    (repo / "training" / "a.py").write_text("x = 1\n")
    (repo / "config").mkdir()
    (repo / "config" / "c.json").write_text("{}\n")
    (repo / "other.py").write_text("y = 2\n")
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    run("config", "commit.gpgsign", "false")
    run("add", ".")
    run("commit", "-q", "-m", "init")
    return repo


def test_git_archive_has_only_the_named_paths_and_the_commit_is_returned(tmp_path):
    repo = _repo(tmp_path)
    commit = K.git_archive(repo, tmp_path / "code.zip", ("training", "config"))
    assert len(commit) == 40
    names = [n for n in zipfile.ZipFile(tmp_path / "code.zip").namelist() if not n.endswith("/")]
    assert sorted(names) == ["config/c.json", "training/a.py"]
    assert set(K.zip_file_hashes(tmp_path / "code.zip")) == set(names)


def test_dirty_paths_sees_uncommitted_changes_only_in_the_archived_paths(tmp_path):
    repo = _repo(tmp_path)
    paths = ("training", "config")
    assert K.dirty_paths(repo, paths) == []
    (repo / "other.py").write_text("y = 3\n")
    assert K.dirty_paths(repo, paths) == []
    (repo / "training" / "a.py").write_text("x = 2\n")
    assert len(K.dirty_paths(repo, paths)) == 1


def test_git_archive_of_a_missing_path_is_a_pack_error(tmp_path):
    repo = _repo(tmp_path)
    with pytest.raises(K.PackError):
        K.git_archive(repo, tmp_path / "x.zip", ("nonexistent",))


def test_commands_are_in_order_all_network_except_the_local_mode_switch():
    cmds = K.commands("example-owner", Path("data/kaggle_upload"))
    text = [c for _, c in cmds]
    assert text[0].startswith("uv tool run kaggle datasets create -p dataset")
    assert text[1].startswith("uv tool run kaggle datasets create -p code")
    assert text[2].startswith("uv tool run kaggle kernels push -p kernel")
    assert "kernels status example-owner/sih26174-train" in text[3]
    assert "kernels output example-owner/sih26174-train" in text[4]
    assert "--set-mode FULL" in text[5] and not text[5].startswith("uv tool run kaggle")
    assert text[6] == text[2] and "status" in text[7] and "output" in text[8]
    md = K.commands_markdown("example-owner", Path("data/kaggle_upload"))
    assert md.count("needs the Lead's explicit yes (network)") == 8
    assert "GPU T4" in md
    assert "--public" not in md and " -u" not in md  # nothing is made public


def test_set_mode_cli_edits_the_packed_script_only(tmp_path):
    out = tmp_path / "up"
    (out / "kernel").mkdir(parents=True)
    script = out / "kernel" / "run_training.py"
    script.write_text(Path(R.__file__).read_text(encoding="utf-8"), encoding="utf-8")
    assert K.main(["--set-mode", "FULL", "--out", str(out)]) == 0
    assert 'MODE = "FULL"' in script.read_text(encoding="utf-8")
    assert K.main(["--set-mode", "SMOKE", "--out", str(out)]) == 0
    assert 'MODE = "SMOKE"' in script.read_text(encoding="utf-8")
    assert K.main(["--set-mode", "FULL", "--out", str(tmp_path / "none")]) == 2


def test_the_pack_module_never_runs_the_kaggle_cli_or_opens_a_socket():
    src = Path(K.__file__).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith(("#", '"', 'f"')))
    assert "socket" not in code and "urllib" not in code and "requests" not in code
    run_calls = [ln for ln in code.splitlines() if "subprocess.run(" in ln]
    assert len(run_calls) == 1 and '["git"' in run_calls[0]  # the only process it starts is git


def test_owner_comes_from_the_argument_then_the_environment_and_has_no_default():
    assert K.resolve_owner("example-owner", {}) == "example-owner"
    assert K.resolve_owner(None, {"KAGGLE_OWNER": "env-owner"}) == "env-owner"
    assert K.resolve_owner("arg-owner", {"KAGGLE_OWNER": "env-owner"}) == "arg-owner"
    assert K.resolve_owner(None, {}) is None
    assert K.resolve_owner("  ", {"KAGGLE_OWNER": " "}) is None


def test_cli_without_an_owner_is_refused(monkeypatch, capsys):
    monkeypatch.delenv("KAGGLE_OWNER", raising=False)
    with pytest.raises(SystemExit) as e:
        K.main(["--print-commands"])
    assert e.value.code == 2
    assert "KAGGLE_OWNER" in capsys.readouterr().err
    monkeypatch.setenv("KAGGLE_OWNER", "example-owner")
    assert K.main(["--print-commands"]) == 0
    assert "example-owner/sih26174-train" in capsys.readouterr().out
