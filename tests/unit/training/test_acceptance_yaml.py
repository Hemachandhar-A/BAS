"""config/acceptance.yaml holds exactly the Plan 5.8 block and validates against
contracts.AcceptanceConfig. eval_detector reads no threshold keys itself: it records the
file's sha256 (and refuses --split test without the file); the thresholds are compared by
whoever reads reports/detector_eval.json."""

from __future__ import annotations

from pathlib import Path

import yaml

import contracts

PATH = Path(__file__).resolve().parents[3] / "config" / "acceptance.yaml"


def test_acceptance_yaml_is_the_plan_5_8_block():
    data = yaml.safe_load(PATH.read_text(encoding="utf-8"))
    assert data == {
        "detector": {"min_recall_per_class": 0.85, "min_map50": 0.80},
        "pipeline": {"min_pipeline_fps": 8},
        "replay": {"exact_deviation_match": True, "max_mismatched_runs": 0},
        "label_review": {"max_bad_fraction_per_class": 0.10},
    }
    contracts.AcceptanceConfig.model_validate(data)


def test_acceptance_yaml_header_records_it_predates_test_evaluation():
    head = PATH.read_text(encoding="utf-8").splitlines()[:5]
    text = " ".join(head)
    assert "2026-10-03" in text and "BEFORE any evaluation on the test split" in text
