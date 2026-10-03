"""``compose_model_stamp`` (contracts.py): the default reproduces the
original rfdetr-nano format; ``detector_name`` is a validated label
(ISSUES.md, 2026-10-03 CONTRACT entry on the model_stamp detector name)."""

from __future__ import annotations

import hashlib

import pytest

import contracts

D = hashlib.sha256(b"detector").hexdigest()
H = hashlib.sha256(b"hand").hexdigest()
P = hashlib.sha256(b"pose").hexdigest()


def test_default_matches_the_original_output() -> None:
    expected = f"rfdetr-nano:{D[:8]}|hand:{H[:8]}|pose:{P[:8]}"
    assert contracts.compose_model_stamp(D, H, P) == expected


def test_default_equals_explicit_rfdetr_nano() -> None:
    assert contracts.compose_model_stamp(D, H, P) == contracts.compose_model_stamp(
        D, H, P, detector_name="rfdetr-nano"
    )


def test_yolo11n_with_pose_disabled() -> None:
    stamp = contracts.compose_model_stamp(D, H, "none", detector_name="yolo11n")
    assert stamp == f"yolo11n:{D[:8]}|hand:{H[:8]}|pose:none"


@pytest.mark.parametrize(
    "bad",
    ["", "yolo:11", "yolo|11", "yolo 11", " yolo11n", "yolo11n ", "YOLO11n", "Yolo11n",
     "rfdetr_nano", "-yolo", "yolo-", "yolo--11", "yolo\n", "yolo11n\n"],
)
def test_detector_name_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        contracts.compose_model_stamp(D, H, "none", detector_name=bad)


@pytest.mark.parametrize("good", ["yolo11n", "rfdetr-nano", "a", "a-b-c1", "9"])
def test_detector_name_accepts_valid(good: str) -> None:
    assert contracts.compose_model_stamp(D, H, P, detector_name=good).startswith(f"{good}:")


@pytest.mark.parametrize("name", ["rfdetr-nano", "yolo11n"])
@pytest.mark.parametrize("pose", [P, "none"])
def test_stamp_has_three_parts_each_with_one_colon(name: str, pose: str) -> None:
    parts = contracts.compose_model_stamp(D, H, pose, detector_name=name).split("|")
    assert len(parts) == 3
    assert all(part.count(":") == 1 for part in parts)
    assert [part.split(":")[0] for part in parts] == [name, "hand", "pose"]
