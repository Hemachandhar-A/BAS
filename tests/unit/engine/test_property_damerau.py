"""essential-features.md #6 property test: for random permutations /
omissions / duplications of the canonical list that end *before* the last
step completes, the engine flags >= 1 deviation iff the observed sequence
differs from the canonical prefix of equal length (Damerau-Levenshtein > 0).

Fixed seed (AGENTS.md rule 13, determinism). Only the non-last step ids
{s1, s2, s3} are drawn, so the run never auto-completes mid-sequence and no
event is ever silently ignored by the "after completion" rule -- that rule
is exercised separately in test_sequence.py.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
from rapidfuzz.distance import DamerauLevenshtein

from contracts import ExperimentDefinition, RuntimeConfig, StateEvent
from engine.sequence import SequenceEngine

FIXTURE_PATH = Path(__file__).resolve().parents[3] / "fixtures" / "experiment_4step.json"

_NON_LAST_STEP_IDS = ["s1", "s2", "s3"]  # excludes the last step, s4
_TRIALS = 300
_MAX_LENGTH = 6


def _random_sequence(rng: random.Random) -> list[str]:
    length = rng.randint(0, _MAX_LENGTH)
    return [rng.choice(_NON_LAST_STEP_IDS) for _ in range(length)]


@pytest.mark.F5
@pytest.mark.F6
def test_deviation_flagged_iff_observed_differs_from_canonical_prefix() -> None:
    experiment = ExperimentDefinition.from_json(FIXTURE_PATH)
    canonical_ids = experiment.step_ids
    rng = random.Random(20260928)

    saw_a_deviating_case = False
    saw_a_clean_case = False

    for _ in range(_TRIALS):
        performed = _random_sequence(rng)

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
        differs = DamerauLevenshtein.distance(performed, prefix) > 0

        assert flagged == differs, (
            f"performed={performed!r} prefix={prefix!r} "
            f"flagged={flagged} differs={differs}"
        )
        saw_a_deviating_case = saw_a_deviating_case or differs
        saw_a_clean_case = saw_a_clean_case or not differs

    # sanity: the random search actually exercised both branches of the iff
    assert saw_a_deviating_case
    assert saw_a_clean_case
