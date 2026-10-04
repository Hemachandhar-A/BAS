"""harness/settings.py -- config loaders (fail loudly) and the model-free expected stamp."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts import PerceptionCacheHeader
from harness.settings import (
    expected_model_stamp,
    load_perception_config,
    load_runtime_config,
)

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / "weights" / "MANIFEST.json"


def test_the_shipped_runtime_yaml_sets_target_fps_10() -> None:
    cfg = load_runtime_config(ROOT / "config" / "runtime.yaml")
    assert cfg.target_fps == 10


def test_unknown_runtime_key_fails_loudly(tmp_path: Path) -> None:
    p = tmp_path / "runtime.yaml"
    p.write_text("target_fps: 10\ntarget_fpz: 12\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_runtime_config(p)


def test_missing_config_files_fail_loudly(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_runtime_config(tmp_path / "nope.yaml")
    with pytest.raises(FileNotFoundError):
        load_perception_config(tmp_path / "nope.yaml")


def test_unknown_perception_key_fails_loudly(tmp_path: Path) -> None:
    p = tmp_path / "perception.yaml"
    p.write_text("hysteresis_frame: 5\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_perception_config(p)


def test_shipped_perception_yaml_loads() -> None:
    path = ROOT / "config" / "perception.yaml"
    if not path.is_file():
        pytest.skip("config/perception.yaml not written yet")
    load_perception_config(path)


def test_expected_stamp_format_and_detector_choice() -> None:
    yolo = expected_model_stamp(MANIFEST, environ={})
    assert yolo.startswith("yolo11n:") and yolo.endswith("|pose:none")
    rf = expected_model_stamp(MANIFEST, "rfdetr_nano", environ={})
    assert rf.startswith("rfdetr-nano:82d9f126|")
    # precedence: argument, then the environment, then the manifest
    assert expected_model_stamp(MANIFEST, environ={"SIH_DETECTOR": "rfdetr_nano"}) == rf
    env = {"SIH_DETECTOR": "rfdetr_nano"}
    assert expected_model_stamp(MANIFEST, "yolo11n", environ=env) == yolo


def test_unknown_detector_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown detector"):
        expected_model_stamp(MANIFEST, "nope", environ={})


def test_expected_stamp_reproduces_the_real_cache_header() -> None:
    path = ROOT / "data" / "cache" / "x015" / "perception.jsonl"  # a val run
    if not path.is_file():
        pytest.skip("val caches are absent")
    first = path.open(encoding="utf-8").readline()
    header = PerceptionCacheHeader.model_validate_json(first)
    assert header.model_stamp == expected_model_stamp(MANIFEST, "yolo11n", environ={})
