"""Runs the pure JS helpers' unit tests (label_editor_core.test.js) with node, when node
is installed (the editor is a dev-time page; node is not a project dependency)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).parent


def test_editor_core_js_helpers():
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed")
    r = subprocess.run(
        [node, str(HERE / "label_editor_core.test.js")], capture_output=True, text=True
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert r.stdout.strip().startswith("ok ")
