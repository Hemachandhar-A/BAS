"""Static checks on the dashboard's style and source (no browser needed): the design tokens, WCAG
contrast for every text/background pair the page uses, no text under 14px, the flat look (no
shadows, gradients, pills or uppercase), and that no server string can reach innerHTML."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.unit.server.dashboard_harness import APP_JS, INDEX_HTML, STATIC

pytestmark = pytest.mark.F12

CSS = (STATIC / "style.css").read_text(encoding="utf-8")
HTML = INDEX_HTML.read_text(encoding="utf-8")
JS = APP_JS.read_text(encoding="utf-8")


def strip_css_comments(css: str) -> str:
    return re.sub(r"/\*.*?\*/", "", css, flags=re.S)


CSS_CODE = strip_css_comments(CSS)


def tokens() -> dict[str, str]:
    root = re.search(r":root\s*{(.*?)}", CSS_CODE, re.S)
    assert root is not None
    return {
        name: value.strip().lower()
        for name, value in re.findall(r"--([\w-]+)\s*:\s*([^;]+);", root.group(1))
    }


TOKENS = tokens()

SPEC_TOKENS = {
    "bg": "#14171a",
    "panel": "#1c2024",
    "raised": "#252a30",
    "rule": "#333a42",
    "rule-soft": "#2a3037",
    "text": "#e8ebee",
    "text-2": "#b4bbc3",
    "text-3": "#9aa2ab",
    "blue": "#7bb8f0",
    "on-blue": "#0f1419",
    "red": "#ff9a90",
    "red-bg": "#3a1e1e",
    "red-text": "#ffb4ac",
    "red-border": "#c9544a",
    "amber": "#e8b64c",
    "seg-pending": "#3a4148",
    "num-border": "#3a4148",
    "btn-border": "#5a626c",
    "advisory-border": "#4a515a",
}


def test_the_design_tokens_are_defined_with_the_specified_values() -> None:
    for name, value in SPEC_TOKENS.items():
        assert TOKENS.get(name) == value, name


def test_no_colour_outside_the_tokens() -> None:
    outside = re.sub(r":root\s*{.*?}", "", CSS_CODE, flags=re.S)
    assert not re.findall(r"#[0-9a-fA-F]{3,8}\b|rgba?\(|hsla?\(", outside)
    for name in re.findall(r"var\(--([\w-]+)\)", CSS_CODE):
        assert name in TOKENS, f"var(--{name}) is not defined"
    assert "style=" not in HTML  # no inline colours or sizes either


# ---- WCAG contrast ------------------------------------------------------------------------------


def luminance(hex_colour: str) -> float:
    channels = [int(hex_colour[i : i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg: str, bg: str) -> float:
    a, b = sorted((luminance(TOKENS[fg]), luminance(TOKENS[bg])), reverse=True)
    return (a + 0.05) / (b + 0.05)


# Every text colour on every background it is drawn on: (foreground token, background token, where).
CONTRAST_PAIRS = [
    ("text", "bg", "page text, top bar values, run id"),
    ("text", "panel", "event text, step titles, summary headline"),
    ("text", "raised", "current step row title"),
    ("text-2", "bg", "feed-lost subtitle, status line, brand sub-title"),
    ("text-2", "panel", "log times, spoken line, captions, tile labels"),
    ("text-2", "raised", "step number box text (not current)"),
    ("text-3", "bg", "top bar labels (Run, State, Feed, Processing)"),
    ("text-3", "panel", "'Video time', 'Also spoken aloud', pending status text"),
    ("text-3", "raised", "pending status on the current row; disabled button text"),
    ("blue", "panel", "speaker icon beside the cue"),
    ("on-blue", "blue", "number box of the current step"),
    ("bg", "text", "enabled Start button text"),
    ("red", "bg", "feed-lost state, feed 'Lost'"),
    ("red", "panel", "deviation log row, skipped / completed-late status, red tiles"),
    ("red", "raised", "skipped status on a highlighted row"),
    ("red", "red-bg", "'Voice alert' label"),
    ("red-text", "red-bg", "alert message and time, fetch error banner"),
    ("amber", "panel", "'Low confidence' tag"),
]


def test_every_text_background_pair_meets_wcag_aa() -> None:
    table = [(fg, bg, where, contrast(fg, bg)) for fg, bg, where in CONTRAST_PAIRS]
    for fg, bg, where, ratio in table:
        print(f"{fg:9} on {bg:7} {ratio:5.2f}:1  {where}")
        assert ratio >= 4.5, f"{fg} on {bg} ({where}) is {ratio:.2f}:1, below 4.5:1"


def css_rules() -> list[tuple[str, str]]:
    return re.findall(r"([^{}]+){([^{}]*)}", CSS_CODE)


def test_every_text_and_surface_token_the_css_uses_is_in_the_pair_table() -> None:
    """A new text colour or surface cannot be added without adding (and so checking) its pairs.
    The progress-bar segments (.seg*) are graphics, not text: they are checked at 3:1 below."""
    fg_in_table = {fg for fg, _bg, _ in CONTRAST_PAIRS}
    bg_in_table = {bg for _fg, bg, _ in CONTRAST_PAIRS}
    used_fg = set(re.findall(r"(?<![\w-])color:\s*var\(--([\w-]+)\)", CSS_CODE))
    used_bg = set()
    for selector, body in css_rules():
        if selector.strip().startswith(".seg"):
            continue
        used_bg |= set(re.findall(r"background:\s*var\(--([\w-]+)\)", body))
    assert used_fg <= fg_in_table, used_fg - fg_in_table
    assert used_bg <= bg_in_table, used_bg - bg_in_table


def test_the_progress_segments_that_carry_meaning_are_visible_on_the_panel() -> None:
    """WCAG 1.4.11 (3:1 for graphics). Pending is deliberately quiet (a missing step, not a
    state to act on) and is not asserted; every other segment is."""
    for token in ("text-2", "red", "blue"):
        assert contrast(token, "panel") >= 3, token
    segments = {sel.strip(): body for sel, body in css_rules() if sel.strip().startswith(".seg")}
    assert "var(--text-2)" in segments[".seg-confirmed"]
    assert "var(--red)" in segments[".seg-skipped,\n.seg-completed_late"]
    assert "var(--blue)" in segments[".seg-current"]


# ---- sizes, weights and the flat look -----------------------------------------------------------


def font_sizes() -> list[float]:
    sizes = [float(v) for v in re.findall(r"font-size:\s*([\d.]+)px", CSS_CODE)]
    sizes += [float(v) for v in re.findall(r"font:\s*\d+\s+([\d.]+)px", CSS_CODE)]
    return sizes


def test_no_text_is_smaller_than_14px() -> None:
    sizes = font_sizes()
    assert sizes, "no font sizes found"
    assert min(sizes) >= 14, sorted(set(sizes))
    # every size is a plain px value, so the check above sees all of them
    declared = re.findall(r"font-size:\s*([^;]+);", CSS_CODE)
    assert all(re.fullmatch(r"[\d.]+px", d.strip()) for d in declared), declared
    assert not re.search(r"font:\s*[^;]*\b\d+(\.\d+)?(rem|em|%)", CSS_CODE)
    assert "font-size" not in HTML and "<font" not in HTML


def test_only_the_two_font_weights_400_and_600() -> None:
    weights = set(re.findall(r"font-weight:\s*(\w+)", CSS_CODE))
    weights |= set(re.findall(r"font:\s*(\d{3})\b", CSS_CODE))
    assert weights <= {"400", "600"}, weights
    assert not re.search(r"\b(bold|bolder|lighter)\b", CSS_CODE)


def test_the_look_is_flat() -> None:
    for banned in ("box-shadow", "text-shadow", "filter", "backdrop", "gradient", "blur", "glow"):
        assert banned not in CSS_CODE, banned
    assert "text-transform" not in CSS_CODE and "uppercase" not in CSS_CODE
    radii = [float(v) for v in re.findall(r"border-radius:\s*([\d.]+)px", CSS_CODE)]
    assert radii and max(radii) <= 2
    assert not re.search(r"border-radius:\s*[^;]*(%|rem|em)", CSS_CODE)  # no pills or circles


def test_fonts_are_the_system_stacks() -> None:
    assert TOKENS["font-sans"] == '"segoe ui", system-ui, "helvetica neue", sans-serif'
    assert TOKENS["font-mono"] == 'ui-monospace, "cascadia mono", consolas, monospace'
    assert "tabular-nums" in CSS_CODE and "@font-face" not in CSS_CODE and "@import" not in CSS_CODE


def test_the_layout_numbers_of_the_spec() -> None:
    for fragment in (
        "max-width: 1440px",
        "min-height: 68px",
        "min-height: 44px",
        "min-height: 52px",
        "max-height: 312px",
        "flex: 3 1 560px",
        "flex: 2 1 420px",
        "aspect-ratio: 16 / 9",
        "flex: 0 0 116px",
        "flex: 0 0 28px",
        "font-size: 40px",
        "line-height: 1.15",
        "object-fit: cover",
    ):
        assert fragment in CSS_CODE, fragment
    focus = re.search(r"\.btn:focus-visible\s*{(.*?)}", CSS_CODE, re.S)
    assert (
        focus
        and "2px solid var(--blue)" in focus.group(1)
        and "outline-offset: 2px" in focus.group(1)
    )
    seg = re.search(r"\.seg\s*{(.*?)}", CSS_CODE, re.S)
    assert seg and "height: 6px" in seg.group(1)
    assert re.search(r"\.segs\s*{[^}]*gap:\s*6px", CSS_CODE)


# ---- the markup ---------------------------------------------------------------------------------


def comments_stripped(source: str) -> str:
    source = re.sub(r"/\*.*?\*/", "", source, flags=re.S)
    return re.sub(r"(?m)//.*$", "", source)


def test_no_server_string_can_reach_innerhtml() -> None:
    """The sinks that parse markup are not used at all; the stub DOM in test_dashboard_dom.py also
    throws if any of them is touched while the page renders hostile payloads."""
    code = comments_stripped(JS)
    for sink in (
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "document.write",
        "eval(",
        "new Function",
    ):
        assert sink not in code, sink
    assert "textContent" in code
    # <script> and <style> in the page are the two inlined placeholders only
    assert HTML.count("<script") == 1 and HTML.count("<style") == 1


def test_every_id_the_script_reads_exists_in_the_page() -> None:
    ids = set(re.findall(r'\sid="([^"]+)"', HTML))
    wanted = set(re.findall(r'byId\("([^"]+)"\)', JS))
    assert wanted and wanted <= ids, wanted - ids
    assert len(re.findall(r'\sid="', HTML)) == len(ids), "duplicate id in index.html"


def test_the_page_keeps_the_ids_and_strings_other_tests_rely_on() -> None:
    assert 'id="start-banner"' in HTML and 'id="video"' in HTML and 'src="/video_feed"' in HTML
    assert "/*__STYLE_CSS__*/" in HTML and "/*__APP_JS__*/" in HTML
    for text in ('"waiting to start"', '"running"', '"run finished"'):
        assert text in JS
    assert "POLL_MS = 500" in JS


def test_controls_are_real_buttons_and_the_wordmark_is_the_only_brand() -> None:
    assert len(re.findall(r"<button\b", HTML)) == 2
    assert re.search(r'<button type="button" class="btn btn-start" id="btn-start">', HTML)
    assert "<h1>BAS Experiment Monitor</h1>" in HTML
    assert "<img" in HTML and HTML.count("<img") == 1  # the live feed; no logo or brand mark
    assert "Advisory only" in HTML
    for forbidden in ("detector", "overlay toggle", "Settings", "<nav"):
        assert forbidden not in HTML, forbidden


def test_status_labels_and_the_mapping_are_the_closed_vocabulary() -> None:
    labels = dict(
        re.findall(r'(\w+):\s*"([A-Z][\w ]+)",', JS.split("STATUS_LABEL = {")[1].split("};")[0])
    )
    assert labels == {
        "pending": "Pending",
        "confirmed": "Confirmed",
        "skipped": "Skipped",
        "completed_late": "Completed late",
    }


def test_the_script_only_calls_the_existing_routes() -> None:
    from contracts import API_ROUTES

    called = set(re.findall(r'"(/(?:api/[\w/]+|video_feed))', comments_stripped(JS)))
    assert called <= set(API_ROUTES), called - set(API_ROUTES)
    assert Path(STATIC / "app.js").exists()


# Non-text pairs (graphics): WCAG 1.4.11, 3:1.
NON_TEXT_PAIRS = [("scroll-thumb", "panel", "the event log's scrollbar thumb on its track")]


def test_the_scrolling_log_has_a_dark_scrollbar_with_a_visible_thumb() -> None:
    assert TOKENS["scroll-thumb"] == "#6b737d"
    log_rule = next(body for sel, body in css_rules() if sel.strip() == ".log")
    assert "scrollbar-color: var(--scroll-thumb) var(--panel);" in log_rule
    for fg, bg, where in NON_TEXT_PAIRS:
        ratio = contrast(fg, bg)
        print(f"{fg:12} on {bg:6} {ratio:5.2f}:1  {where} (non-text, 3:1)")
        assert ratio >= 3, f"{fg} on {bg} ({where}) is {ratio:.2f}:1, below 3:1"
