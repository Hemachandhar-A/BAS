"""Offline requirement-compatibility check for the Kaggle kernel (S-F1d).

Reads the Requires-Dist metadata of the three pinned packages from THIS venv (importlib.metadata;
nothing is downloaded) and checks every requirement that names a package of the Kaggle image
against the version Kaggle reported in smoke run 2 (Python 3.13.15, linux x86_64). The fixture
below documents those facts. Smoke run 2 stopped on exactly the violation this test pins:
rfdetr[train] 1.11.0 needs ``torchmetrics<1.9.0`` and the image has 1.9.0.
"""

from __future__ import annotations

import importlib.metadata as md
import re

import pytest

packaging = pytest.importorskip("packaging.requirements")
from packaging.requirements import Requirement  # noqa: E402
from packaging.utils import canonicalize_name  # noqa: E402
from packaging.version import Version  # noqa: E402

# Facts from the Kaggle image log of smoke run 2 (kernel version 2).
KAGGLE_PYTHON = "3.13.15"
KAGGLE_VERSIONS = {
    "torch": "2.11.0+cu128", "torchvision": "0.26.0+cu128", "torchaudio": "2.11.0+cu128",
    "numpy": "2.1.3", "opencv-python": "4.14.0.94", "opencv-python-headless": "4.14.0.94",
    "pillow": "12.3.0", "scipy": "1.16.3", "pydantic": "2.13.5", "transformers": "5.16.1",
    "peft": "0.20.0", "pycocotools": "2.0.11", "pytorch-lightning": "2.6.6",
    "torchmetrics": "1.9.0", "matplotlib": "3.10.0", "requests": "2.34.2", "tqdm": "4.67.3",
    "psutil": "5.9.5", "polars": "1.35.2", "cloudpickle": "3.1.2", "filelock": "3.32.5",
    "defusedxml": "0.7.1", "pyyaml": "6.0.3", "nvidia-ml-py": "13.610.43",
}  # fmt: skip
KAGGLE_ENV = {
    "python_version": "3.13", "python_full_version": KAGGLE_PYTHON, "sys_platform": "linux",
    "platform_system": "Linux", "platform_machine": "x86_64", "os_name": "posix",
    "implementation_name": "cpython", "platform_python_implementation": "CPython",
}  # fmt: skip
# The kernel pins (the same three the packer writes) and the extra each is installed with.
PINNED = {
    "rfdetr": ("1.11.0", ("train",)),
    "supervision": ("0.30.5", ()),
    "ultralytics": ("8.4.164", ()),
}


def requirements_of(dist: str, extras: tuple[str, ...], env: dict) -> list[Requirement]:
    """The requirements of ``dist`` that apply in ``env`` with ``extras`` selected."""
    out = []
    for line in md.requires(dist) or []:
        req = Requirement(line)
        if req.marker is None or any(
            req.marker.evaluate({**env, "extra": e}) for e in (*extras, "")
        ):
            out.append(req)
    return out


def check_against(
    reqs: list[tuple[str, Requirement]], kaggle: dict[str, str]
) -> tuple[list[str], list[str]]:
    """(violations, not_on_image): a requirement naming a package of ``kaggle`` whose specifier
    the Kaggle version fails, and the names pip would have to add (not on the image)."""
    versions = {canonicalize_name(k): v for k, v in kaggle.items()}
    bad, absent = [], []
    for owner, req in reqs:
        name = canonicalize_name(req.name)
        if name not in versions:
            if name not in absent:
                absent.append(name)
            continue
        if not req.specifier.contains(Version(versions[name]), prereleases=True):
            bad.append(f"{owner} needs {req.name}{req.specifier}, Kaggle has {versions[name]}")
    return sorted(set(bad)), sorted(absent)


def collect() -> list[tuple[str, Requirement]]:
    reqs: list[tuple[str, Requirement]] = []
    for dist, (version, extras) in PINNED.items():
        assert md.version(dist) == version, (
            f"this venv has {dist} {md.version(dist)}, not {version}"
        )
        reqs += [(f"{dist} {version}", r) for r in requirements_of(dist, extras, KAGGLE_ENV)]
    return reqs


# --- the checker itself, on synthetic input ----------------------------------------------


def test_check_against_reports_a_version_outside_the_specifier_and_ignores_unknown_names():
    reqs = [
        ("a 1", Requirement("torchmetrics[detection]<1.9.0,>=1.8.2")),
        ("a 1", Requirement("numpy")),
        ("a 1", Requirement("torch>=2.2.0")),
        ("a 1", Requirement("av>=14.2")),
        ("a 1", Requirement("torch==2.11.0")),  # a local version tag does not break == without it
    ]
    bad, absent = check_against(
        reqs, {"torchmetrics": "1.9.0", "numpy": "2.1.3", "torch": "2.11.0+cu128"}
    )
    assert bad == ["a 1 needs torchmetrics<1.9.0,>=1.8.2, Kaggle has 1.9.0"]
    assert absent == ["av"]


def test_requirements_follow_markers_extras_and_the_kaggle_environment():
    names = {canonicalize_name(r.name) for r in requirements_of("rfdetr", ("train",), KAGGLE_ENV)}
    assert {"torchmetrics", "pytorch-lightning", "peft", "pycocotools"} <= names
    assert "tensorflow" not in names and "onnx" not in names  # other extras and 3.12-only markers
    plain = {canonicalize_name(r.name) for r in requirements_of("rfdetr", (), KAGGLE_ENV)}
    assert "torchmetrics" not in plain  # the train extra is what brings it


# --- the real packages against the Kaggle facts -------------------------------------------


def test_the_kaggle_image_violates_exactly_one_requirement_torchmetrics():
    bad, absent = check_against(collect(), KAGGLE_VERSIONS)
    print("violations:", *bad, sep="\n  ")
    print("pip would add (not on the image):", ", ".join(absent))
    assert bad == ["rfdetr 1.11.0 needs torchmetrics<1.9.0,>=1.8.2, Kaggle has 1.9.0"]
    for name in ("pydeprecate", "av", "ultralytics-thop", "ultralytics-platform"):
        assert name in absent  # the small packages smoke run 2 saw pip collect
    # everything else the pins name on the image is satisfied by what is already there
    assert not [b for b in bad if "torchmetrics" not in b]


def test_the_core_stack_satisfies_every_pin_so_it_never_needs_to_move():
    core = {"torch", "torchvision", "torchaudio", "numpy", "scipy", "pillow", "opencv-python"}
    bad, _ = check_against(collect(), KAGGLE_VERSIONS)
    assert not [b for b in bad if re.search(rf" needs ({'|'.join(core)})[<>=!~]", b)]
