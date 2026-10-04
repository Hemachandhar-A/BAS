"""harness/metrics.py -- what a replay of one run means (P2.6, IMPLEMENTATION_PLAN.md 5.8/7.3).

* ``observe`` replays frames through ``StateTracker`` + ``SequenceEngine`` in memory (no router,
  no log files, no speech) and returns what happened: the step events that fired, the engine's
  deviations in order, how many events were flagged uncertain.
* ``verdict`` compares an observation with a ``RunScript``. A run MATCHES iff the observed
  deviations equal ``expected_deviations`` (same types, same step ids, same order) AND every step
  of ``performed_steps`` was observed firing in order AND there is no extra deviation (the
  equality already implies that last clause). Besides the boolean it counts extra and missing
  step events, extra and missing deviations, and uncertain flags.
* ``diagnose`` gives a one-line cause per missing or extra event, read from the frames.
* ``lint_run`` is the dynamic half of the experiment lint (Plan 7.3): a step whose rules are true
  inside the baseline window (so the tracker latches it) must go false for at least
  ``release_frames`` consecutive frames before its turn, or it can never fire.

Rule evaluation is ``state.tracker``'s own (imported, never copied, never edited).
Pure functions of their arguments: no clock, no randomness, no file access.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from contracts import (
    ExpectedDeviation,
    ExperimentDefinition,
    HandTouchingRule,
    PerceptionConfig,
    PerceptionFrame,
    RuntimeConfig,
    StateEvent,
    StepDef,
)
from engine.sequence import SequenceEngine
from state.tracker import StateTracker, _best_detection, _evaluate_step, _floor_filter

Deviation = tuple[str, tuple[str, ...]]


@dataclass(frozen=True)
class Observation:
    fired: list[str] = field(default_factory=list)
    fired_frame_index: list[int] = field(default_factory=list)
    fired_t: list[float] = field(default_factory=list)
    deviations: list[Deviation] = field(default_factory=list)
    uncertain: int = 0


@dataclass(frozen=True)
class RunVerdict:
    match: bool
    deviations_equal: bool
    performed_in_order: bool
    extra_events: int
    missing_events: int
    extra_deviations: int
    missing_deviations: int
    uncertain: int


def observe(
    experiment: ExperimentDefinition,
    frames: list[PerceptionFrame],
    perception_config: PerceptionConfig,
    runtime_config: RuntimeConfig,
) -> Observation:
    tracker = StateTracker(experiment, perception_config)
    engine = SequenceEngine(experiment, runtime_config)
    engine.start(frames[0].t if frames else 0.0)
    fired: list[str] = []
    idx: list[int] = []
    ts: list[float] = []
    deviations: list[Deviation] = []
    uncertain = 0
    for i, frame in enumerate(frames):
        for ev in tracker.update(frame):
            _record(ev, i, fired, idx, ts)
            uncertain += ev.uncertain
            for out in engine.on_state_event(ev):
                if out.kind != "deviation_detected":
                    continue
                if out.deviation_type == "omission":
                    deviations.append(("omission", tuple(out.skipped_step_ids)))
                else:
                    assert out.deviation_type is not None and out.step_id is not None
                    deviations.append((out.deviation_type, (out.step_id,)))
    return Observation(fired, idx, ts, deviations, uncertain)


def _record(ev: StateEvent, i: int, fired: list[str], idx: list[int], ts: list[float]) -> None:
    fired.append(ev.step_id)
    idx.append(i)
    ts.append(ev.t)


def _lcs(a: list[str], b: list[str]) -> int:
    row = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, start=1):
            cur = row[j]
            row[j] = prev + 1 if x == y else max(row[j], row[j - 1])
            prev = cur
    return row[len(b)]


def _is_subsequence(needle: list[str], hay: list[str]) -> bool:
    it = iter(hay)
    return all(x in it for x in needle)


def verdict(
    expected: list[ExpectedDeviation], performed: list[str], obs: Observation
) -> RunVerdict:
    want = [(d.deviation_type, tuple(d.step_ids)) for d in expected]
    equal = obs.deviations == want
    in_order = _is_subsequence(performed, obs.fired)
    matched = _lcs(performed, obs.fired)
    have_c, want_c = Counter(obs.deviations), Counter(want)
    return RunVerdict(
        match=equal and in_order,
        deviations_equal=equal,
        performed_in_order=in_order,
        extra_events=len(obs.fired) - matched,
        missing_events=len(performed) - matched,
        extra_deviations=sum((have_c - want_c).values()),
        missing_deviations=sum((want_c - have_c).values()),
        uncertain=obs.uncertain,
    )


# --- per-frame step truth (the tracker's own evaluator) ---------------------------------------


def step_truths(
    experiment: ExperimentDefinition,
    frames: list[PerceptionFrame],
    config: PerceptionConfig,
) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {s.step_id: [] for s in experiment.steps}
    for frame in frames:
        dets = _floor_filter(frame.detections, config.detector_conf_floor)
        for step in experiment.steps:
            out[step.step_id].append(_evaluate_step(step, dets, frame.hands, config)[0])
    return out


def _max_streak(flags: list[bool]) -> int:
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def _rearm_index(truth: list[bool], latch: int, release_frames: int) -> int | None:
    """Frame index at which a step latched at ``latch`` re-arms: the ``release_frames``-th
    consecutive false frame after the latch frame (the tracker's ``false_streak``)."""
    streak = 0
    for i in range(latch + 1, len(truth)):
        streak = 0 if truth[i] else streak + 1
        if streak >= release_frames:
            return i
    return None


def _labels(step: StepDef) -> list[str]:
    out: list[str] = []
    for rule in step.when:
        out.append(rule.label)
        container = getattr(rule, "container", None)
        if container:
            out.append(container)
    return list(dict.fromkeys(out))


def _why_not_fired(
    step: StepDef,
    truth: list[bool],
    frames: list[PerceptionFrame],
    config: PerceptionConfig,
    fired_before: int = 0,
) -> str:
    sid = step.step_id
    n = len(frames)
    if fired_before:
        # performed more often than it fired: the later performances never re-armed the step
        return (
            f"{sid}: fired {fired_before}x but performed more often: the rule never went false "
            f"for release_frames={config.release_frames} consecutive frames after the firing "
            f"(true for up to {_max_streak(truth)} consecutive frames, so the repeat merged "
            "into one long hold)"
        )
    if not any(truth):
        missing = []
        for label in _labels(step):
            k = sum(
                1
                for f in frames
                if _best_detection(label, _floor_filter(f.detections, config.detector_conf_floor))
                is None
            )
            if k:
                missing.append((label, k))
        if any(isinstance(r, HandTouchingRule) for r in step.when):
            label = next(r.label for r in step.when if isinstance(r, HandTouchingRule))
            absent = dict(missing).get(label, 0)
            note = f"; no box for {label} in {absent}/{n} frames" if absent else ""
            return (
                f"{sid}: missed press: no fingertip inside the grown {label} box in any "
                f"frame{note}"
            )
        if missing:
            worst = max(missing, key=lambda m: m[1])
            return (
                f"{sid}: rule never true; no box for {worst[0]} >= "
                f"{config.detector_conf_floor} in {worst[1]}/{n} frames (missing box)"
            )
        return f"{sid}: rule never true although every box was detected (geometry)"
    base = truth[: config.baseline_frames]
    if any(base):
        latch = base.index(True)
        if _rearm_index(truth, latch, config.release_frames) is None:
            return (
                f"{sid}: baseline latch: true in the baseline window and never false for "
                f"{config.release_frames} consecutive frames"
            )
    streak = _max_streak(truth)
    if streak < config.hysteresis_frames:
        return (
            f"{sid}: true for at most {streak} consecutive frame(s), shorter than "
            f"hysteresis_frames={config.hysteresis_frames} (action too short)"
        )
    return f"{sid}: true for up to {streak} consecutive frames but never fired (latched or re-arm)"


def diagnose(
    experiment: ExperimentDefinition,
    frames: list[PerceptionFrame],
    config: PerceptionConfig,
    performed: list[str],
    obs: Observation,
) -> list[str]:
    """One line per performed step that did not fire and per unperformed or repeated firing."""
    truths = step_truths(experiment, frames, config)
    by_id = {s.step_id: s for s in experiment.steps}
    performed_c, fired_c = Counter(performed), Counter(obs.fired)
    causes: list[str] = []
    for sid in dict.fromkeys(performed):
        if performed_c[sid] > fired_c[sid]:
            causes.append(_why_not_fired(by_id[sid], truths[sid], frames, config, fired_c[sid]))
    for sid, n in fired_c.items():
        extra = n - performed_c.get(sid, 0)
        if extra > 0:
            ts = [t for s, t in zip(obs.fired, obs.fired_t, strict=True) if s == sid]
            kind = (
                "extra press"
                if any(isinstance(r, HandTouchingRule) for r in by_id[sid].when)
                else "unperformed or repeated step"
            )
            causes.append(f"{sid}: {kind}: fired {n}x, performed {performed_c.get(sid, 0)}x, "
                          f"at t={', '.join(f'{t:.1f}' for t in ts)}")
    if not causes and obs.fired != performed:
        causes.append(f"fired order {obs.fired} differs from performed order {performed}")
    return causes


# --- dynamic experiment lint ------------------------------------------------------------------


def lint_run(
    experiment: ExperimentDefinition,
    frames: list[PerceptionFrame],
    config: PerceptionConfig,
    runtime_config: RuntimeConfig,
) -> list[dict[str, str]]:
    """Violations of the Plan 7.3 rule for one ``correct`` run (empty list = pass). A step is
    *latched* when its rules are true on a frame of the baseline window. It must go false for
    ``release_frames`` consecutive frames (counted from the frame after the latch, exactly as the
    tracker counts) at or before its turn. The turn starts when the previous canonical step
    fired in the replay of this run (frame 0 for the first step); a step that never got a
    predecessor event falls back to its own firing, then to the end of the run."""
    truths = step_truths(experiment, frames, config)
    obs = observe(experiment, frames, config, runtime_config)
    first_fire: dict[str, int] = {}
    for sid, i in zip(obs.fired, obs.fired_frame_index, strict=True):
        first_fire.setdefault(sid, i)
    violations: list[dict[str, str]] = []
    ids = experiment.step_ids
    for pos, sid in enumerate(ids):
        base = truths[sid][: config.baseline_frames]
        if not any(base):
            continue
        latch = base.index(True)
        rearm = _rearm_index(truths[sid], latch, config.release_frames)
        prev = first_fire.get(ids[pos - 1]) if pos > 0 else 0
        turn = prev if prev is not None else first_fire.get(sid, len(frames))
        if rearm is None:
            reason = (
                f"true in the baseline window and never false for {config.release_frames} "
                "consecutive frames"
            )
        elif rearm > turn:
            reason = (
                f"first false for {config.release_frames} consecutive frames at frame {rearm}, "
                f"after its turn began (frame {turn})"
            )
        else:
            continue
        violations.append({"step_id": sid, "reason": reason})
    return violations
