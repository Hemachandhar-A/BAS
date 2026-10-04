"""harness/settings.py -- where the replay and tuning entry points get their configuration.

``load_runtime_config`` / ``load_perception_config`` read ``config/runtime.yaml`` and
``config/perception.yaml`` through the contract's ``from_yaml`` (both models forbid unknown keys,
so a misspelled key fails loudly). A missing file is an error too: silently falling back to
``RuntimeConfig``'s default ``target_fps`` of 15 would replay at a different rate than the caches.

``expected_model_stamp`` computes the ``model_stamp`` a cache must carry for the active detector
**without loading any model**: ``contracts.compose_model_stamp`` over the sha256 values recorded
in ``weights/MANIFEST.json`` (the pipeline verifies those hashes when it really loads).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from contracts import PerceptionConfig, RuntimeConfig, compose_model_stamp
from perception.detector import ENV_VAR, STAMP_LABELS

DEFAULT_RUNTIME_PATH = Path("config/runtime.yaml")
DEFAULT_PERCEPTION_PATH = Path("config/perception.yaml")
DEFAULT_MANIFEST_PATH = Path("weights/MANIFEST.json")


def load_runtime_config(path: Path | str = DEFAULT_RUNTIME_PATH) -> RuntimeConfig:
    if not Path(path).is_file():
        raise FileNotFoundError(f"runtime config not found: {path}")
    return RuntimeConfig.from_yaml(path)


def load_perception_config(path: Path | str = DEFAULT_PERCEPTION_PATH) -> PerceptionConfig:
    if not Path(path).is_file():
        raise FileNotFoundError(f"perception config not found: {path}")
    return PerceptionConfig.from_yaml(path)


def _canonical(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def expected_model_stamp(
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
    detector: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    """The stamp of the pipeline ``load_pipeline`` would build (pose off). The detector is
    chosen like ``load_detector`` does: argument, then ``$SIH_DETECTOR``, then the manifest."""
    env = os.environ if environ is None else environ
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    chosen = (detector or "").strip() or (env.get(ENV_VAR) or "").strip()
    chosen = chosen or manifest["active_detector"]
    entries = {_canonical(d["name"]): d for d in manifest["detectors"]}
    entry = entries.get(_canonical(chosen))
    if entry is None:
        raise ValueError(f"unknown detector {chosen!r}; the manifest lists {sorted(entries)}")
    return compose_model_stamp(
        entry["sha256"],
        manifest["hand"]["sha256"],
        "none",
        detector_name=STAMP_LABELS[entry["name"]],
    )
