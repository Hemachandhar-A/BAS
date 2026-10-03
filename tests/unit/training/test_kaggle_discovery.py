"""Kaggle kernel, session S-F1c: dataset-root discovery for every layout Kaggle may mount, the
start-of-run listing, the one-GPU rule and the Python 3.13 check. Temp folders stand in for
/kaggle/input; nothing here needs a GPU, the network or the Kaggle CLI."""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from training.kaggle import run_training as R

ROOT = Path(__file__).resolve().parents[3]

FILES = {"train/a.jpg": b"A", "train/_annotations.coco.json": b"{}", "valid/b.jpg": b"B"}


def _expected() -> dict[str, str]:
    return {rel: hashlib.sha256(data).hexdigest() for rel, data in FILES.items()}


def _lay(base: Path, files: dict[str, bytes] = FILES) -> Path:
    for rel, data in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return base


def _zip_of(path: Path, files: dict[str, bytes], prefix: str = "") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as z:
        for rel, data in files.items():
            z.writestr(prefix + rel, data)
    return R.sha256_file(path)


# --- discovery: the layouts ----------------------------------------------------------------


@pytest.mark.parametrize(
    "sub",
    ["", "dataset", "data/dataset", "datasets/example-owner/sih26174-dataset"],
    ids=["a_root", "b_one_extra_folder", "b_two_extra_folders", "c_datasets_owner_slug"],
)
def test_discovery_finds_the_single_root_in_every_layout(tmp_path, sub):
    inp = tmp_path / "input"
    base = _lay(inp / sub if sub else inp)
    (inp / "noise").mkdir(exist_ok=True)
    (inp / "noise" / "readme.txt").write_text("x")
    res = R.find_roots(inp, _expected())
    assert res["matches"] == [base]
    assert R.single_root(res, "dataset", inp) == base


def test_discovery_stops_on_two_candidate_roots_and_lists_both(tmp_path):
    inp = tmp_path / "input"
    a = _lay(inp / "one")
    b = _lay(inp / "datasets" / "o" / "two")
    res = R.find_roots(inp, _expected())
    assert sorted(res["matches"]) == sorted([a, b])
    with pytest.raises(R.DiscoveryError) as e:
        R.single_root(res, "dataset", inp)
    msg = str(e.value)
    assert "ambiguous" in msg and str(a) in msg and str(b) in msg


def test_discovery_with_nothing_found_says_what_was_examined(tmp_path):
    inp = tmp_path / "input"
    (inp / "x" / "y").mkdir(parents=True)
    (inp / "x" / "y" / "other.txt").write_text("no")
    res = R.find_roots(inp, _expected())
    assert res["matches"] == [] and res["dirs_walked"] == 3
    with pytest.raises(R.DiscoveryError) as e:
        R.single_root(res, "dataset", inp)
    msg = str(e.value)
    assert "no directory" in msg and str(inp) in msg and "all 3 manifest files" in msg
    assert "3 directories" in msg


def test_a_root_with_only_some_files_is_not_a_match_but_is_reported(tmp_path):
    inp = tmp_path / "input"
    part = _lay(inp / "half", {"train/a.jpg": b"A"})
    res = R.find_roots(inp, _expected())
    assert res["matches"] == []
    (ex,) = [x for x in res["examined"] if x["root"] == str(part)]
    assert ex["found"] == 1 and ex["total"] == 3
    assert ex["missing"] == ["train/_annotations.coco.json", "valid/b.jpg"]
    with pytest.raises(R.DiscoveryError, match="1 of 3"):
        R.single_root(res, "dataset", inp)


def test_the_missing_list_in_a_report_is_cut_at_ten(tmp_path):
    inp = tmp_path / "input"
    _lay(inp / "p", {"train/f00.jpg": b"x"})
    expected = {f"train/f{i:02d}.jpg": "0" * 64 for i in range(30)}
    (ex,) = R.find_roots(inp, expected)["examined"]
    assert ex["found"] == 1 and len(ex["missing"]) == 10


def test_discovery_does_not_go_deeper_than_five_levels(tmp_path):
    inp = tmp_path / "input"
    _lay(inp / "a" / "b" / "c" / "d" / "e")  # depth 5: found
    assert len(R.find_roots(inp, _expected())["matches"]) == 1
    deep = tmp_path / "input2"
    _lay(deep / "a" / "b" / "c" / "d" / "e" / "f")  # depth 6: not looked at
    assert R.find_roots(deep, _expected())["matches"] == []


def test_discovery_on_a_missing_input_root_finds_nothing(tmp_path):
    res = R.find_roots(tmp_path / "nope", _expected())
    assert res["matches"] == [] and res["dirs_walked"] == 0
    with pytest.raises(R.DiscoveryError, match="does not exist"):
        R.single_root(res, "dataset", tmp_path / "nope")


def test_the_probe_is_the_first_manifest_path_and_every_file_is_then_checked(tmp_path):
    inp = tmp_path / "input"
    _lay(inp / "full")
    _lay(inp / "probe_only", {"train/_annotations.coco.json": b"{}"})  # sorted first
    res = R.find_roots(inp, _expected())
    assert res["matches"] == [inp / "full"]
    assert {x["root"] for x in res["examined"]} == {str(inp / "full"), str(inp / "probe_only")}


# --- staging: tree or zip --------------------------------------------------------------------


def test_stage_copies_a_found_tree_into_the_work_folder(tmp_path):
    inp = tmp_path / "input"
    base = _lay(inp / "datasets" / "o" / "ds")
    (base / "extra.txt").write_text("not in the manifest")
    dest = tmp_path / "work" / "dataset"
    got = R.stage_tree(inp, dest, _expected(), "dataset.zip", [], "dataset")
    assert got == dest and R.check_files(dest, _expected()) == []


def test_stage_unzips_a_zip_at_the_root_layout_d(tmp_path):
    inp = tmp_path / "input"
    sha = _zip_of(inp / "ds" / "dataset.zip", FILES)
    dest = tmp_path / "work" / "dataset"
    got = R.stage_tree(inp, dest, _expected(), "dataset.zip", [sha], "dataset")
    assert got == dest and R.check_files(dest, _expected()) == []


def test_stage_finds_the_data_when_the_zip_holds_an_extra_top_folder(tmp_path):
    inp = tmp_path / "input"
    sha = _zip_of(inp / "dataset.zip", FILES, prefix="dataset/")
    dest = tmp_path / "work" / "dataset"
    got = R.stage_tree(inp, dest, _expected(), "dataset.zip", [sha], "dataset")
    assert got == dest / "dataset" and R.check_files(got, _expected()) == []


def test_stage_refuses_a_zip_with_the_wrong_sha256(tmp_path):
    inp = tmp_path / "input"
    _zip_of(inp / "dataset.zip", FILES)
    with pytest.raises(R.DiscoveryError, match="sha256"):
        R.stage_tree(inp, tmp_path / "w", _expected(), "dataset.zip", ["0" * 64], "dataset")


def test_stage_with_neither_tree_nor_zip_raises_with_the_examination(tmp_path):
    inp = tmp_path / "input"
    (inp / "somewhere").mkdir(parents=True)
    with pytest.raises(R.DiscoveryError, match="no directory"):
        R.stage_tree(inp, tmp_path / "w", _expected(), "dataset.zip", [], "dataset")


def test_stage_stops_on_an_ambiguous_tree(tmp_path):
    inp = tmp_path / "input"
    _lay(inp / "one")
    _lay(inp / "two")
    with pytest.raises(R.DiscoveryError, match="ambiguous"):
        R.stage_tree(inp, tmp_path / "w", _expected(), "dataset.zip", [], "dataset")


def test_a_changed_file_in_the_found_tree_is_caught_by_the_manifest_check(tmp_path):
    inp = tmp_path / "input"
    base = _lay(inp / "ds")
    (base / "valid" / "b.jpg").write_bytes(b"CHANGED")
    dest = tmp_path / "w"
    R.stage_tree(inp, dest, _expected(), "dataset.zip", [], "dataset")
    assert R.check_files(dest, _expected()) == ["sha256 differs: valid/b.jpg"]


def test_find_sentinel_looks_at_most_five_levels_deep_and_prefers_the_shallowest(tmp_path):
    inp = tmp_path / "input"
    (inp / "a" / "b").mkdir(parents=True)
    (inp / "a" / "b" / "m.json").write_text("{}")
    assert R.find_sentinel(inp, "m.json") == inp / "a" / "b"
    (inp / "m.json").write_text("{}")
    assert R.find_sentinel(inp, "m.json") == inp
    far = tmp_path / "far" / "1" / "2" / "3" / "4" / "5" / "6"
    far.mkdir(parents=True)
    (far / "m.json").write_text("{}")
    assert R.find_sentinel(tmp_path / "far", "m.json") is None


def test_weights_are_found_anywhere_and_checked_by_sha256(tmp_path):
    inp = tmp_path / "input"
    d = inp / "datasets" / "o" / "code"
    d.mkdir(parents=True)
    (d / "rf.pth").write_bytes(b"RF")
    (d / "y.pt").write_bytes(b"YO")
    meta = {
        "rf.pth": {"sha256": R.sha256_file(d / "rf.pth")},
        "y.pt": {"sha256": R.sha256_file(d / "y.pt")},
    }
    assert R.locate_weights(inp, meta) == {"rf.pth": d / "rf.pth", "y.pt": d / "y.pt"}
    (d / "y.pt").write_bytes(b"corrupt")
    with pytest.raises(R.DiscoveryError, match="y.pt"):
        R.locate_weights(inp, meta)
    (d / "y.pt").unlink()
    with pytest.raises(R.DiscoveryError, match="not found"):
        R.locate_weights(inp, meta)


# --- the start-of-run listing ----------------------------------------------------------------


def test_listing_is_depth_limited_directories_first_with_sizes_and_counts(tmp_path):
    root = tmp_path / "input"
    (root / "ds" / "train").mkdir(parents=True)
    for i in range(30):
        (root / "ds" / "train" / f"{i}.jpg").write_bytes(b"x" * 10)
    (root / "ds" / "manifest.json").write_bytes(b"y" * 2048)
    deep = root / "l1" / "l2" / "l3" / "l4" / "l5"
    deep.mkdir(parents=True)
    (deep / "hidden.txt").write_text("h")
    lines = R.list_tree(root)
    text = "\n".join(lines)
    assert lines[0].startswith(str(root))
    assert "30 files" in text and "300 B" in text  # a big folder: count and total size
    assert "manifest.json" in text and "2.0 KB" in text  # a file: its size
    assert "hidden.txt" not in text  # beyond four levels
    ds = [i for i, ln in enumerate(lines) if ln.strip().startswith("ds/")][0]
    man = [i for i, ln in enumerate(lines) if "manifest.json" in ln][0]
    assert ds < man
    # directories come before the files beside them
    (root / "zz.txt").write_text("z")
    again = R.list_tree(root)
    assert [i for i, ln in enumerate(again) if "zz.txt" in ln][0] > max(
        i for i, ln in enumerate(again) if ln.startswith("  ") and ln.strip().endswith("/")
    )


def test_listing_is_cut_at_the_line_limit_and_says_so(tmp_path):
    root = tmp_path / "r"
    for i in range(50):
        (root / f"d{i:02d}").mkdir(parents=True)
    lines = R.list_tree(root, max_lines=20)
    assert len(lines) == 21 and "truncated" in lines[-1]
    assert R.list_tree(tmp_path / "missing")[0].endswith("does not exist")


def test_size_text_and_free_disk_line():
    assert R.size_text(0) == "0 B" and R.size_text(1536) == "1.5 KB"
    assert R.size_text(5 * 1024**3) == "5.00 GB"
    line = R.free_disk_line(Path("."))
    assert "free" in line and "GB" in line
    assert "unavailable" in R.free_disk_line(Path("/definitely/not/here/xyz"))


# --- one GPU -----------------------------------------------------------------------------------


def test_single_gpu_env_pins_device_zero_and_keeps_the_rest():
    env = R.single_gpu_env({"PATH": "p", "CUDA_VISIBLE_DEVICES": "0,1"})
    assert env["CUDA_VISIBLE_DEVICES"] == "0" and env["PATH"] == "p"
    assert R.single_gpu_env({})["CUDA_VISIBLE_DEVICES"] == "0"
    src = {"CUDA_VISIBLE_DEVICES": "1"}
    R.single_gpu_env(src)
    assert src == {"CUDA_VISIBLE_DEVICES": "1"}  # the input is not changed


def test_child_env_is_single_gpu_for_every_subprocess(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    assert R.child_env()["CUDA_VISIBLE_DEVICES"] == "0"


def test_a_child_process_really_sees_one_gpu_id(tmp_path):
    code = "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"
    log = tmp_path / "l.log"
    R.run_logged([sys.executable, "-c", code], log, cwd=tmp_path, timeout_s=30, env=R.child_env())
    assert log.read_text().strip() == "0"


def test_gpu_note_names_the_unused_gpu_and_the_effective_batch():
    two = R.gpu_note(2)
    assert "2 GPUs" in two and "deliberately unused" in two and "4 x 4 = 16" in two
    one = R.gpu_note(1)
    assert "1 GPU" in one and "unused" not in one
    assert "no GPU" in R.gpu_note(0)


def test_gpu_fields_are_in_the_smoke_report():
    machine = {"gpus_seen": 2, "gpu_names": ["Tesla T4", "Tesla T4"], "cuda_available": True}
    rep = R.smoke_report(
        machine=machine, n_train=1, epochs=1, preflight={}, dry_runs={}, smoke_seconds=1.0,
        pip_seconds=None, dataset_stamp=None, problems=[],
    )  # fmt: skip
    assert rep["gpus_seen"] == 2 and rep["gpu_used"] == 1
    assert rep["gpu_note"] == R.gpu_note(2)
    json.dumps(rep)


def test_annotate_summary_adds_the_gpu_fields_and_keeps_the_rest(tmp_path):
    p = tmp_path / "train_summary.json"
    p.write_text(json.dumps({"best_val_map": 0.4, "seed": 0}), encoding="utf-8")
    assert R.annotate_summary(p, gpus_seen=2) is True
    d = json.loads(p.read_text(encoding="utf-8"))
    assert d["best_val_map"] == 0.4 and d["gpus_seen"] == 2 and d["gpu_used"] == 1
    assert R.annotate_summary(tmp_path / "none.json", gpus_seen=2) is False
    (tmp_path / "bad.json").write_text("{not json")
    assert R.annotate_summary(tmp_path / "bad.json", gpus_seen=2) is False


# --- Python 3.13: no removed standard-library module in the kernel script or the trainers ------

REMOVED_IN_313 = (
    "aifc audioop cgi cgitb chunk crypt imghdr mailcap msilib nis nntplib ossaudiodev pipes "
    "sndhdr spwd sunau telnetlib uu xdrlib lib2to3 asynchat asyncore smtpd distutils imp"
).split()


def test_no_module_removed_in_python_313_is_imported_by_the_kernel_or_the_trainers():
    import ast

    files = [Path(R.__file__)] + [
        p
        for p in (ROOT / "training").glob("*.py")
        if p.name in {"finetune.py", "gpu_preflight.py", "verify_dataset.py", "coco_to_yolo.py"}
    ]
    assert len(files) == 5
    bad = []
    for f in files:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module]
            bad += [f"{f.name}: {n}" for n in names if n.split(".")[0] in REMOVED_IN_313]
    assert bad == []


# --- manifests, frozen packages, pip conflicts --------------------------------------------------


def test_read_manifest_finds_it_anywhere_and_refuses_different_copies(tmp_path):
    inp = tmp_path / "input"
    (inp / "a").mkdir(parents=True)
    (inp / "datasets" / "o" / "b").mkdir(parents=True)
    with pytest.raises(R.DiscoveryError, match="not found"):
        R.read_manifest(inp, "m.json", "dataset")
    (inp / "datasets" / "o" / "b" / "m.json").write_text('{"x": 1}')
    assert R.read_manifest(inp, "m.json", "dataset") == {"x": 1}
    (inp / "a" / "m.json").write_text('{"x": 1}')  # an identical second mount is fine
    assert R.read_manifest(inp, "m.json", "dataset") == {"x": 1}
    (inp / "a" / "m.json").write_text('{"x": 2}')
    with pytest.raises(R.DiscoveryError, match="ambiguous"):
        R.read_manifest(inp, "m.json", "dataset")


def test_the_constraints_freeze_the_whole_core_stack_when_installed():
    installed = {n: "1.0" for n in R.FROZEN}
    lines = R.constraints_text(installed).splitlines()
    assert lines == [f"{n}==1.0" for n in R.FROZEN]
    for n in "torch torchvision torchaudio numpy opencv-python pillow scipy".split():
        assert n in R.FROZEN
    for n in "pydantic transformers peft pycocotools pytorch-lightning torchmetrics".split():
        assert n not in R.FROZEN  # pure-Python libraries are left to pip (S-F1d)
    assert R.constraints_text({}) == "\n"  # nothing installed: an empty file, no crash


def test_an_optional_pin_is_dropped_when_the_image_has_that_package():
    pins = ["rfdetr[train]==1.11.0", "supervision==0.30.5", "pycocotools==2.0.8",
            "pytorch-lightning==2.5.0", "torchmetrics==1.8.0"]  # fmt: skip
    installed = {"pycocotools": "2.0.11", "pytorch-lightning": "2.6.6"}
    keep, notes = R.drop_installed_optional(pins, installed)
    assert keep == ["rfdetr[train]==1.11.0", "supervision==0.30.5", "torchmetrics==1.8.0"]
    assert len(notes) == 2 and "pycocotools 2.0.11" in notes[0] and "2.0.8" in notes[0]
    assert R.drop_installed_optional(pins, {})[0] == pins


def test_changed_packages_names_what_moved():
    before = {"torch": "2.11", "numpy": "2.0"}
    assert R.changed_packages(before, dict(before)) == []
    assert R.changed_packages(before, {"torch": "2.11", "numpy": "2.1"}) == ["numpy"]
    assert R.changed_packages(before, {"torch": "2.11"}) == ["numpy"]


def test_a_pip_conflict_names_the_package():
    text = (
        "ERROR: Cannot install rfdetr[train]==1.11.0 and transformers==5.16.1 because these "
        "package versions have conflicting dependencies.\n"
        "The conflict is caused by:\n    The user requested transformers==5.16.1 (constraint)\n"
        "ERROR: ResolutionImpossible"
    )
    names = R.conflicting_packages(text)
    assert names[:2] == ["rfdetr", "transformers"] and R.looks_like_conflict(text)
    assert R.conflicting_packages("ERROR: No matching distribution found for foo_bar==9.9") == [
        "foo-bar"
    ]
    inc = "    rfdetr 1.11.0 requires numpy<2, but you have numpy 2.1.0 which is incompatible."
    assert R.conflicting_packages(inc) == ["rfdetr", "numpy"]
    assert R.conflicting_packages("Successfully installed rfdetr-1.11.0") == []
