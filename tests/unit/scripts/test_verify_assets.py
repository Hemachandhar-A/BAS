"""tests/unit/scripts/test_verify_assets.py -- scripts/verify_assets.py (judge replication kit).

Offline, standard-library-only verifier: every required file is OK / MISSING / BAD HASH, the last
line is READY or NOT READY, the exit code is 0 / 1, and nothing is written."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import verify_assets as va

REPO = Path(__file__).resolve().parents[3]
SLUG = "someone/bas-demo-kit"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


FILES = {
    "weights/detector_yolo11n.pt": b"yolo weights",
    "weights/hand_landmarker.task": b"hand model",
    "demo_videos/01_x015_clean.mp4": b"clip one",
    "demo_videos/playlist.txt": b"01_x015_clean.mp4\n",
}


def make_root(tmp_path: Path, files: dict[str, bytes] | None = None, sep: str = "/") -> Path:
    """A fake repo root: assets list, weights manifest, and the given files on disk."""
    files = FILES if files is None else files
    root = tmp_path / "repo"
    (root / "assets").mkdir(parents=True)
    (root / "weights").mkdir()
    lines = [f"# dataset: {SLUG}", "", "# a comment"]
    for rel, data in FILES.items():
        lines.append(f"{sha(data)}  {rel.replace('/', sep)}")
    (root / "assets" / "ASSETS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "detectors": [
            {
                "name": "yolo11n",
                "file": "detector_yolo11n.pt",
                "sha256": sha(FILES["weights/detector_yolo11n.pt"]),
            }
        ],
        "hand": {
            "file": "hand_landmarker.task",
            "sha256": sha(FILES["weights/hand_landmarker.task"]),
        },
    }
    (root / "weights" / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    for rel, data in files.items():
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return root


def run(root: Path, *extra: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = va.main(["--root", str(root), *extra])
    return code, capsys.readouterr().out


def tree(root: Path) -> dict[str, tuple[int, str]]:
    return {
        p.relative_to(root).as_posix(): (p.stat().st_size, sha(p.read_bytes()))
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_all_present_is_ready(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    code, out = run(make_root(tmp_path), capsys=capsys)
    assert code == 0
    assert out.strip().splitlines()[-1] == "READY"
    assert out.count("[OK]") == len(FILES)
    assert "MISSING" not in out and "BAD HASH" not in out


def test_missing_file_names_the_folder_and_the_dataset(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = dict(FILES)
    del files["demo_videos/01_x015_clean.mp4"]
    code, out = run(make_root(tmp_path, files), capsys=capsys)
    assert code == 1
    assert out.strip().splitlines()[-1] == "NOT READY"
    line = next(x for x in out.splitlines() if "[MISSING]" in x)
    assert "demo_videos/01_x015_clean.mp4" in line
    assert "demo_videos/" in line and SLUG in line


def test_wrong_hash_is_reported(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files = dict(FILES)
    files["weights/detector_yolo11n.pt"] = b"yolo weightz"  # one byte different
    code, out = run(make_root(tmp_path, files), capsys=capsys)
    assert code == 1
    line = next(x for x in out.splitlines() if "[BAD HASH]" in x)
    assert "weights/detector_yolo11n.pt" in line
    assert out.strip().splitlines()[-1] == "NOT READY"


def test_empty_file_is_a_bad_hash_and_says_empty(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = dict(FILES)
    files["weights/hand_landmarker.task"] = b""
    code, out = run(make_root(tmp_path, files), capsys=capsys)
    assert code == 1
    line = next(x for x in out.splitlines() if "[BAD HASH]" in x)
    assert "empty" in line.lower()


def test_wrong_folder_is_missing_with_a_move_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = dict(FILES)
    del files["weights/detector_yolo11n.pt"]
    files["demo_videos/detector_yolo11n.pt"] = FILES["weights/detector_yolo11n.pt"]  # wrong folder
    code, out = run(make_root(tmp_path, files), capsys=capsys)
    assert code == 1
    line = next(x for x in out.splitlines() if "[MISSING]" in x)
    assert "found at demo_videos/detector_yolo11n.pt" in line
    assert "move it to weights/" in line


def test_wrong_folder_at_repo_top_level_is_found_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = dict(FILES)
    del files["weights/hand_landmarker.task"]
    files["hand_landmarker.task"] = FILES["weights/hand_landmarker.task"]
    code, out = run(make_root(tmp_path, files), capsys=capsys)
    assert code == 1
    assert "found at hand_landmarker.task" in out


def test_windows_backslash_paths_in_the_list_are_understood(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(make_root(tmp_path, sep="\\"), capsys=capsys)
    assert code == 0, out
    assert "\\" not in out.split("READY")[0].replace("\\n", "")  # shown with forward slashes


def test_parse_assets_handles_both_path_forms_comments_and_the_slug() -> None:
    text = (
        "# dataset: u/demo\n# dataset-full: u/full\n\n"
        f"{'a' * 64}  weights\\x.pt\n{'b' * 64} *demo_videos/y.mp4\n"
    )
    slug, full, entries = va.parse_assets(text)
    assert (slug, full) == ("u/demo", "u/full")
    assert [(e.rel, e.sha256) for e in entries] == [
        ("weights/x.pt", "a" * 64),
        ("demo_videos/y.mp4", "b" * 64),
    ]


def test_malformed_list_line_is_not_ready(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_root(tmp_path)
    with (root / "assets" / "ASSETS.sha256").open("a", encoding="utf-8") as fh:
        fh.write("this is not a hash line\n")
    code, out = run(root, capsys=capsys)
    assert code == 1
    assert out.strip().splitlines()[-1] == "NOT READY"
    assert "ASSETS.sha256" in out


def test_missing_assets_list_is_not_ready(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_root(tmp_path)
    (root / "assets" / "ASSETS.sha256").unlink()
    code, out = run(root, capsys=capsys)
    assert code == 1
    assert out.strip().splitlines()[-1] == "NOT READY"


def test_a_list_that_disagrees_with_the_weights_manifest_is_flagged(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_root(tmp_path)
    manifest = json.loads((root / "weights" / "MANIFEST.json").read_text(encoding="utf-8"))
    manifest["hand"]["sha256"] = "0" * 64
    (root / "weights" / "MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    code, out = run(root, capsys=capsys)
    assert code == 1
    line = next(x for x in out.splitlines() if "MANIFEST.json" in x)
    assert "hand_landmarker.task" in line and "[BAD HASH]" in line


def test_the_verifier_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    files = dict(FILES)
    del files["demo_videos/playlist.txt"]
    root = make_root(tmp_path, files)
    before = tree(root)
    run(root, capsys=capsys)
    assert tree(root) == before


def test_full_kit_checks_every_run_video_against_the_runs_manifest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = make_root(tmp_path)
    (root / "runs").mkdir()
    rows = ["run_id,split,video_sha256"]
    for run_id, data in {"x001": b"v1", "x002": b"v2"}.items():
        rows.append(f"{run_id},train,{sha(data)}")
        (root / "runs" / run_id).mkdir()
        (root / "runs" / run_id / "video.mp4").write_bytes(data if run_id == "x001" else b"bad")
    (root / "runs" / "manifest.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    code, out = run(root, "--kit", "full", capsys=capsys)
    assert code == 1
    assert any("[BAD HASH]" in x and "runs/x002/video.mp4" in x for x in out.splitlines())
    assert any("[OK]" in x and "runs/x001/video.mp4" in x for x in out.splitlines())
    assert out.strip().splitlines()[-1] == "NOT READY"


def test_the_demo_kit_ignores_the_full_kit_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run(make_root(tmp_path), "--kit", "demo", capsys=capsys)
    assert code == 0
    assert "runs/" not in out


def test_sha_matches_the_repo_helper(tmp_path: Path) -> None:
    from contracts import sha256_of_file

    p = tmp_path / "f.bin"
    p.write_bytes(b"x" * 3_000_000)
    assert va.sha256_of_file(p) == sha256_of_file(p)


def test_no_heavy_import_and_no_network_module() -> None:
    code = (
        "import sys, scripts.verify_assets;"
        "bad={'torch','ultralytics','mediapipe','cv2','numpy','pydantic','urllib.request','requests'}"
        "&set(sys.modules);"
        "sys.exit(1 if bad else 0)"
    )
    proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_committed_list_parses_and_agrees_with_the_weights_manifest() -> None:
    text = (REPO / "assets" / "ASSETS.sha256").read_text(encoding="utf-8")
    slug, _full, entries = va.parse_assets(text)
    assert slug and entries
    manifest = json.loads((REPO / "weights" / "MANIFEST.json").read_text(encoding="utf-8"))
    expected = {f"weights/{d['file']}": d["sha256"] for d in manifest["detectors"]}
    expected[f"weights/{manifest['hand']['file']}"] = manifest["hand"]["sha256"]
    listed = {e.rel: e.sha256 for e in entries}
    assert listed["weights/detector_yolo11n.pt"] == expected["weights/detector_yolo11n.pt"]
    assert listed["weights/hand_landmarker.task"] == expected["weights/hand_landmarker.task"]
    assert any(rel == "demo_videos/playlist.txt" for rel in listed)
