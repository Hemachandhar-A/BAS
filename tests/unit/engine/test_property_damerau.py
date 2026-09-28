"""essential-features.md #6 property test: for random permutations /
omissions / duplications of the canonical list that end *before* the last
step completes, the engine flags >= 1 deviation iff the observed sequence
differs from the canonical prefix of equal length (Damerau-Levenshtein > 0).

Exhaustive, not sampled (AGENTS.md rule 13, determinism): only the
non-last step ids {s1, s2, s3} are drawn, so the run never auto-completes
mid-sequence and no event is ever silently ignored by the "after
completion" rule (that rule is exercised separately in test_sequence.py).
With a 3-symbol alphabet and sequences up to length 6, there are only
sum(3**L for L in range(7)) == 1093 possible sequences -- cheap enough to
check every single one instead of sampling, which gives a complete proof
rather than a probabilistic one.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

import pytest
from rapidfuzz.distance import DamerauLevenshtein

from contracts import ExperimentDefinition, RuntimeConfig, StateEvent
from engine.sequence import SequenceEngine

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"

_NON_LAST_STEP_IDS = ("s1", "s2", "s3")  # excludes the last step, s4
_MAX_LENGTH = 6


def _all_sequences() -> list[tuple[str, ...]]:
    return [
        seq
        for length in range(_MAX_LENGTH + 1)
        for seq in product(_NON_LAST_STEP_IDS, repeat=length)
    ]


@pytest.mark.F5
@pytest.mark.F6
def test_deviation_flagged_iff_observed_differs_from_canonical_prefix() -> None:
    experiment = ExperimentDefinition.from_json(FIXTURE_PATH)
    canonical_ids = experiment.step_ids

    all_sequences = _all_sequences()
    assert len(all_sequences) == 1093  # sanity: 3**0 + 3**1 + ... + 3**6

    saw_a_deviating_case = False
    saw_a_clean_case = False

    for performed in all_sequences:
        engine = SequenceEngine(experiment, RuntimeConfig())
        engine.start(0.0)
        flagged = False
        for i, step_id in enumerate(performed, start=1):
            events = engine.on_state_event(
                StateEvent(t=float(i), step_id=step_id, confidence=1.0, uncertain=False)
            )
            if any(e.kind == "deviation_detected" for e in events):
                flagged = True

        prefix = canonical_ids[: len(performed)]
        differs = DamerauLevenshtein.distance(list(performed), prefix) > 0

        assert flagged == differs, (
            f"performed={performed!r} prefix={prefix!r} "
            f"flagged={flagged} differs={differs}"
        )
        saw_a_deviating_case = saw_a_deviating_case or differs
        saw_a_clean_case = saw_a_clean_case or not differs

    # sanity: the exhaustive search actually exercised both branches of the iff
    assert saw_a_deviating_case
    assert saw_a_clean_case
