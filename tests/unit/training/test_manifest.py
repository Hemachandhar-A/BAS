"""weights/MANIFEST.json: hashes match the files on disk, one active detector, and the active
detector is the one evaluated on test (S-F2b). The weights are git-ignored, so the on-disk check
skips on a checkout that does not hold them; the structural checks always run."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "weights"
SHA = re.compile(r"^[0-9a-f]{64}$")


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((WEIGHTS / "MANIFEST.json").read_text(encoding="utf-8"))


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def test_all_sha256_fields_are_well_formed(manifest):
    for d in manifest["detectors"]:
        assert SHA.match(d["sha256"]) and SHA.match(d["training_summary_sha256"])
    assert SHA.match(manifest["hand"]["sha256"])


def test_sha256_and_size_equal_the_files_on_disk(manifest):
    entries = [(d["file"], d["sha256"], d["size_bytes"]) for d in manifest["detectors"]]
    h = manifest["hand"]
    entries.append((h["file"], h["sha256"], h["size_bytes"]))
    present = [(f, s, n) for f, s, n in entries if (WEIGHTS / f).is_file()]
    if not present:
        pytest.skip("weights are git-ignored and not present in this checkout")
    for f, sha, size in present:
        assert (WEIGHTS / f).stat().st_size == size, f
        assert _sha(WEIGHTS / f) == sha, f


def test_exactly_one_detector_is_active_and_it_is_listed(manifest):
    names = [d["name"] for d in manifest["detectors"]]
    assert len(names) == len(set(names)) == 2
    assert manifest["active_detector"] in names
    assert names.count(manifest["active_detector"]) == 1


def test_active_detector_is_the_one_evaluated_on_test(manifest):
    evaluated = [d["name"] for d in manifest["detectors"] if d["test_evaluated"]]
    assert evaluated == [manifest["active_detector"]]
    active = next(d for d in manifest["detectors"] if d["name"] == manifest["active_detector"])
    report = json.loads((ROOT / active["test_report"]).read_text(encoding="utf-8"))
    assert report["split"] == "test"
    assert report["weights_sha256"] == active["sha256"]


def test_inactive_detector_is_marked_not_validated(manifest):
    other = [d for d in manifest["detectors"] if d["name"] != manifest["active_detector"]]
    assert all(d["validated_for_pipeline"] is False and "test_report" not in d for d in other)


def test_valid_reports_exist_and_match_the_weights(manifest):
    for d in manifest["detectors"]:
        rep = json.loads((ROOT / d["valid_report"]).read_text(encoding="utf-8"))
        assert rep["split"] == "valid" and rep["weights_sha256"] == d["sha256"]


def test_classes_equal_the_experiment_in_order(manifest):
    exp = json.loads((ROOT / "config" / "experiment.json").read_text(encoding="utf-8"))
    for d in manifest["detectors"]:
        assert d["classes"] == exp["classes"]
