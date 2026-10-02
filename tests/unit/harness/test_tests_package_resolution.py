"""Regression test for a local-environment hazard, not a product bug: some
installed packages (e.g. ultralytics, the YOLO fallback per AGENTS.md rule
15) ship a top-level ``tests/__init__.py`` into site-packages. Without our
own ``tests/__init__.py``, a *fresh* interpreter doing a plain ``import
tests`` (no pytest import machinery involved -- e.g. the TTSWorker's
spawn-based multiprocessing child in outputs/tts.py, F7, unpickling a
``tests.*``-qualified callable) can resolve ``tests`` to that stray
site-packages package instead of this repo's own ``tests/`` directory.

This must be checked in a subprocess: inside the pytest process itself,
pytest's own importlib import-mode has already pre-populated
``sys.modules['tests']`` pointing at this repo by the time any test body
runs, which masks the hazard -- a plain ``import tests`` here would pass
either way. See ISSUES.md."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]


def test_tests_package_resolves_into_this_repository_in_a_fresh_interpreter() -> None:
    proc = subprocess.run(
        [sys.executable, "-c", "import tests; print(tests.__path__[0])"],
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    resolved_path = Path(proc.stdout.strip()).resolve()
    assert resolved_path == (_REPO_ROOT / "tests").resolve(), (
        f"'import tests' resolved to {resolved_path}, not the repo's own tests/ -- "
        "a stray tests/__init__.py from an installed package (e.g. ultralytics) is "
        "shadowing it"
    )
