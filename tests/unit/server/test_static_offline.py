"""tests/unit/server/test_static_offline.py -- F12's offline requirement
(essential-features.md #12 point 3: "a test scans the served HTML/CSS/JS
for non-local http(s):// resources"). Scans both the source files under
server/static/ and the assembled page server/app.py actually serves."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from server.app import _render_index

STATIC_DIR = Path(__file__).resolve().parents[3] / "server" / "static"
_NON_LOCAL_URL_RE = re.compile(r"https?://", re.IGNORECASE)


@pytest.mark.F12
@pytest.mark.parametrize("filename", ["index.html", "style.css", "app.js"])
def test_static_source_file_has_no_non_local_url(filename: str) -> None:
    text = (STATIC_DIR / filename).read_text(encoding="utf-8")
    assert not _NON_LOCAL_URL_RE.search(text), f"{filename} references a non-local http(s):// URL"


@pytest.mark.F12
def test_rendered_index_page_has_no_non_local_url() -> None:
    rendered = _render_index()
    assert not _NON_LOCAL_URL_RE.search(rendered)


@pytest.mark.F12
def test_app_js_poll_interval_matches_dashboard_poll_ms_constant() -> None:
    from contracts import DASHBOARD_POLL_MS

    text = (STATIC_DIR / "app.js").read_text(encoding="utf-8")
    match = re.search(r"POLL_MS\s*=\s*(\d+)", text)
    assert match is not None, "app.js must define a POLL_MS constant"
    assert int(match.group(1)) == DASHBOARD_POLL_MS
