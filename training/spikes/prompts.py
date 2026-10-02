"""Candidate text prompts tried by the P1.2 spike, one entry per phrasing
tried (checkpoint 1 item 2 requires every prompt tried to be written down,
not just the winner). Grounding DINO's documented convention: lowercase,
each phrase ends with a period.

Physical rig (2026-09-30 ISSUES.md "crew footage intake" DECISION): the
START object is a handwritten white index card, not a green coaster; the
tray is a grey hardcover book; the yellow container's hue leans lime/olive.
Prompt wording below is chosen against the real rig, not DATA_COLLECTION.md's
original (now-stale) prop table.
"""

from __future__ import annotations

# class -> ordered list of candidate phrasings tried against real frames.
PROMPT_CANDIDATES: dict[str, list[str]] = {
    "outer_box": [
        "a cardboard box.",
        "a brown cardboard box.",
    ],
    "tray": [
        "a tray.",
        "a grey book.",
    ],
    "red_box": [
        "a red box.",
        "a red container.",
    ],
    "yellow_box": [
        "a yellow box.",
        "a lime green container.",
    ],
    "start_button": [
        "a start button.",
        "a white index card.",
    ],
}


def all_phrases() -> list[str]:
    """Every candidate phrase, in a fixed order, deduplicated."""
    seen: list[str] = []
    for phrases in PROMPT_CANDIDATES.values():
        for phrase in phrases:
            if phrase not in seen:
                seen.append(phrase)
    return seen
