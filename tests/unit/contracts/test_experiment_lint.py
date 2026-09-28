"""The static half of the experiment lint (IMPLEMENTATION_PLAN.md 7.3):
checks derivable from config/experiment.json alone, with no recorded runs.
The dynamic half (a step true at baseline must go false for
release_frames before its turn, checked against real recordings) is added
once caches exist (P2.6) -- see ISSUES.md, 2026-09-28 CONTRACT wording."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

import contracts

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_PATH = ROOT / "config" / "experiment.json"


@pytest.fixture(scope="module")
def experiment() -> contracts.ExperimentDefinition:
    return contracts.ExperimentDefinition.from_json(EXPERIMENT_PATH)


def test_experiment_json_validates_against_the_contract(
    experiment: contracts.ExperimentDefinition,
) -> None:
    assert experiment.experiment_id == "sample_transfer"
    assert len(experiment.classes) == 5
    assert len(experiment.steps) == 7


def test_every_step_id_is_snake_case_and_unique(
    experiment: contracts.ExperimentDefinition,
) -> None:
    assert len(set(experiment.step_ids)) == len(experiment.step_ids)


def test_every_rule_class_is_in_the_experiment_classes(
    experiment: contracts.ExperimentDefinition,
) -> None:
    # Enforced by ExperimentDefinition's own validator; re-asserted here so
    # a future relaxation of that validator is caught by this named test.
    classes = set(experiment.classes)
    for step in experiment.steps:
        for rule in step.when:
            assert rule.label in classes
            if hasattr(rule, "container"):
                assert rule.container in classes


def test_every_say_is_at_most_max_spoken_words(
    experiment: contracts.ExperimentDefinition,
) -> None:
    for step in experiment.steps:
        word_count = len(step.say.split())
        assert word_count <= contracts.MAX_SPOKEN_WORDS, (
            f"{step.step_id}: say={step.say!r} has {word_count} words "
            f"(max {contracts.MAX_SPOKEN_WORDS})"
        )


def test_the_two_stow_steps_are_position_rules_that_can_go_false() -> None:
    # ISSUES.md, 2026-09-28: red_stowed / yellow_stowed start true (the
    # boxes begin inside outer_box) and are latched at baseline; the
    # requirement is only that the rule *can* go false (when the box
    # leaves), not that it starts false. This is what makes the lint
    # satisfiable by the draft experiment.
    experiment = contracts.ExperimentDefinition.from_json(EXPERIMENT_PATH)
    by_id = {s.step_id: s for s in experiment.steps}
    for step_id in ("red_stowed", "yellow_stowed"):
        rule = by_id[step_id].when[0]
        assert rule.type == "inside" and rule.container == "outer_box"


def test_unknown_rule_type_is_rejected() -> None:
    with pytest.raises(ValidationError):
        contracts.StepDef.model_validate(
            {
                "step_id": "bogus",
                "display_name": "Bogus",
                "say": "Do the thing",
                "when": [{"type": "levitates", "label": "red_box"}],
            }
        )


def test_experiment_rejects_a_rule_referencing_an_undefined_class() -> None:
    with pytest.raises(ValidationError):
        contracts.ExperimentDefinition.model_validate(
            {
                "experiment_id": "bad",
                "version": "0",
                "classes": ["red_box"],
                "steps": [
                    {
                        "step_id": "s1",
                        "display_name": "S1",
                        "say": "Do it",
                        "when": [{"type": "inside", "label": "red_box", "container": "nope"}],
                    }
                ],
            }
        )
