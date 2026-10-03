"""Kaggle package, session S-F1c: re-pack ONLY the kernel (script + kernel-metadata.json) from the
manifests of the datasets that are already uploaded, tell whether code.zip would change, and
print Kaggle CLI commands with slash-free paths. No network, no Kaggle CLI."""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

import pytest

from training import kaggle_pack as K
from training.kaggle import run_training as R


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _package(tmp_path: Path) -> Path:
    """A minimal already-packed folder: both datasets with manifests, an old kernel."""
    out = tmp_path / "up"
    (out / "dataset").mkdir(parents=True)
    (out / "code").mkdir()
    (out / "kernel").mkdir()
    ds_zip = out / "dataset" / "dataset.zip"
    with zipfile.ZipFile(ds_zip, "w") as z:
        z.writestr("train/a.jpg", b"A")
    code_zip = out / "code" / "code.zip"
    with zipfile.ZipFile(code_zip, "w") as z:
        z.writestr("training/a.py", b"x = 1\n")
    (out / "code" / "rf-detr-nano.pth").write_bytes(b"RF")
    (out / "code" / "yolo11n.pt").write_bytes(b"YO")
    ds = {
        "zip": {"name": "dataset.zip", "sha256": R.sha256_file(ds_zip), "size_bytes": 1},
        "dataset_stamp": "stamp123",
        "images": {"train": 593, "valid": 173, "test": 187},
        "files": {"train/a.jpg": _sha(b"A")},
    }
    code = {
        "zip": {"name": "code.zip", "sha256": R.sha256_file(code_zip), "size_bytes": 1},
        "git_commit": "c" * 40,
        "files": {"training/a.py": _sha(b"x = 1\n")},
        "weights": K.weights_manifest(
            {
                "rf-detr-nano.pth": out / "code" / "rf-detr-nano.pth",
                "yolo11n.pt": out / "code" / "yolo11n.pt",
            }
        ),
    }
    (out / "dataset" / "dataset_manifest.json").write_text(json.dumps(ds))
    (out / "code" / "code_manifest.json").write_text(json.dumps(code))
    (out / "dataset" / "dataset-metadata.json").write_text("{}")
    (out / "code" / "dataset-metadata.json").write_text("{}")
    (out / "kernel" / "run_training.py").write_text("# old kernel\n")
    (out / "package_manifest.json").write_text(
        json.dumps(
            {"git_commit": "c" * 40, "dataset_stamp": "stamp123", "pins": ["p==1"], "files": {}}
        )
    )
    return out


def test_repack_kernel_writes_only_the_kernel_and_bakes_in_the_existing_hashes(tmp_path):
    out = _package(tmp_path)
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file() and "kernel" not in p.parts}
    manifest = K.repack_kernel(out, user="example-owner")
    for p, data in before.items():
        if p.name not in ("package_manifest.json", "COMMANDS.md"):
            assert p.read_bytes() == data, p  # dataset/ and code/ are byte-for-byte unchanged
    script = (out / "kernel" / "run_training.py").read_text(encoding="utf-8")
    ns: dict = {}
    exec(compile(script, "run_training.py", "exec"), ns)  # noqa: S102 - the packed script
    ds = json.loads((out / "dataset" / "dataset_manifest.json").read_text())
    code = json.loads((out / "code" / "code_manifest.json").read_text())
    pkg = ns["PACKAGE"]
    assert ns["MODE"] == "SMOKE"
    assert pkg["dataset_zip_sha256"] == ds["zip"]["sha256"]
    assert pkg["code_zip_sha256"] == code["zip"]["sha256"]
    assert pkg["n_train"] == 593 and pkg["n_valid"] == 173 and pkg["n_test"] == 187
    assert pkg["git_commit"] == "c" * 40 and pkg["dataset_stamp"] == "stamp123"
    assert pkg["weights"] == {n: m["sha256"] for n, m in code["weights"].items()}
    meta = json.loads((out / "kernel" / "kernel-metadata.json").read_text())
    assert meta["id"] == "example-owner/sih26174-train" and meta["code_file"] == "run_training.py"
    assert manifest["files"]["kernel/run_training.py"]["sha256"] == R.sha256_file(
        out / "kernel" / "run_training.py"
    )
    saved = json.loads((out / "package_manifest.json").read_text())
    assert saved["files"] == manifest["files"] and saved["pins"] == ["p==1"]
    assert saved["files"]["code/code.zip"]["sha256"] == code["zip"]["sha256"]


def test_repack_kernel_is_deterministic(tmp_path):
    out = _package(tmp_path)
    a = K.repack_kernel(out, user="example-owner")["files"]["kernel/run_training.py"]["sha256"]
    b = K.repack_kernel(out, user="example-owner")["files"]["kernel/run_training.py"]["sha256"]
    assert a == b


def test_repack_kernel_refuses_a_zip_that_does_not_match_its_manifest(tmp_path):
    out = _package(tmp_path)
    (out / "dataset" / "dataset.zip").write_bytes(b"not the packed zip")
    with pytest.raises(K.PackError, match="dataset.zip"):
        K.repack_kernel(out, user="example-owner")
    out2 = _package(tmp_path / "two")
    (out2 / "code" / "yolo11n.pt").write_bytes(b"changed")
    with pytest.raises(K.PackError, match="yolo11n.pt"):
        K.repack_kernel(out2, user="example-owner")


def test_repack_kernel_needs_an_existing_package(tmp_path):
    with pytest.raises(K.PackError, match="pack first"):
        K.repack_kernel(tmp_path / "empty", user="example-owner")


def _repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "training" / "kaggle").mkdir(parents=True)
    (repo / "training" / "a.py").write_text("x = 1\n")
    (repo / "training" / "kaggle" / "run_training.py").write_text("k = 1\n")
    (repo / "training" / "kaggle_pack.py").write_text("p = 1\n")
    (repo / "config").mkdir()
    (repo / "config" / "c.json").write_text("{}\n")
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, check=True, capture_output=True)  # noqa: E731
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    run("config", "commit.gpgsign", "false")
    run("add", ".")
    run("commit", "-q", "-m", "init")
    return repo, run


def test_code_zip_drift_lists_what_a_new_archive_would_change(tmp_path):
    repo, run = _repo(tmp_path)
    K.git_archive(repo, tmp_path / "code.zip", ("training", "config"))
    manifest = {"files": K.zip_file_hashes(tmp_path / "code.zip")}
    assert K.code_zip_drift(repo, manifest, ("training", "config")) == []
    (repo / "training" / "kaggle" / "run_training.py").write_text("k = 2\n")
    (repo / "training" / "kaggle_pack.py").write_text("p = 2\n")
    (repo / "training" / "new.py").write_text("n = 1\n")
    run("add", ".")
    run("commit", "-q", "-m", "change")
    drift = K.code_zip_drift(repo, manifest, ("training", "config"))
    assert drift == [
        "training/kaggle/run_training.py",
        "training/kaggle_pack.py",
        "training/new.py",
    ]
    # the kernel never runs the packer or its own template from the zip: those do not matter
    assert K.functional_drift(drift) == ["training/new.py"]
    assert K.functional_drift(drift[:2]) == []


def test_a_removed_file_is_drift_too(tmp_path):
    repo, run = _repo(tmp_path)
    K.git_archive(repo, tmp_path / "code.zip", ("training", "config"))
    manifest = {"files": K.zip_file_hashes(tmp_path / "code.zip")}
    run("rm", "-q", "training/a.py")
    run("commit", "-q", "-m", "rm")
    assert K.code_zip_drift(repo, manifest, ("training", "config")) == ["training/a.py"]


def test_commands_use_slash_free_paths_run_from_the_upload_folder():
    cmds = K.commands("example-owner", Path("data/kaggle_upload"))
    kaggle = [c for _, c in cmds if c.startswith("uv tool run kaggle")]
    assert kaggle, "no kaggle commands"
    for c in kaggle:
        for tok in c.split():
            if tok.startswith("-"):
                continue
            assert "/" not in tok or tok.startswith("example-owner/"), c  # only the kernel id
    assert "kernels push -p kernel" in " ".join(kaggle)
    md = K.commands_markdown("example-owner", Path("data/kaggle_upload"))
    assert "cd data/kaggle_upload" in md


def test_the_version_command_for_the_code_dataset_is_exact():
    cmd = K.version_command("changed finetune")
    assert cmd == 'uv tool run kaggle datasets version -p code -m "changed finetune"'
    assert cmd.count('"') == 2 == K.version_command('say "hi"').count('"')


def test_kernel_only_cli_runs_without_a_repo_check_of_the_dataset(tmp_path, capsys):
    out = _package(tmp_path)
    assert K.main(["--kernel-only", "--owner", "example-owner", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "kernels push -p kernel" in text and "Nothing was uploaded" in text
    assert 'MODE = "SMOKE"' in (out / "kernel" / "run_training.py").read_text(encoding="utf-8")
    none = str(tmp_path / "none")
    assert K.main(["--kernel-only", "--owner", "example-owner", "--out", none]) == 2
