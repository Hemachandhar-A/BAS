"""engine/sequence.py -- SequenceEngine (F5, F6, F8), IMPLEMENTATION_PLAN.md
5.4. Pure logic: no models, no clock reads (AGENTS.md rule 8) -- time
arrives on ``StateEvent.t`` and the ``t`` passed to ``start``/``finish``.

``RunSummary.run_id`` has no carrier in the ``Engine`` Protocol
(``ISSUES.md``, 2026-09-28 CONTRACT); ``start`` takes an additive optional
``run_id`` keyword as the documented workaround.
"""

from __future__ import annotations

from rapidfuzz.distance import DamerauLevenshtein

from contracts import (
    ConfidenceTag,
    ContractViolation,
    EngineEvent,
    ExperimentDefinition,
    RunState,
    RunSummary,
    RuntimeConfig,
    StateEvent,
    StepProgress,
    StepStatus,
)


class SequenceEngine:
    """Implements ``contracts.Engine``. Constructed as
    ``SequenceEngine(experiment, config)``; reusable across runs by calling
    ``finish()`` (or letting the run complete on its own) and then
    ``start()`` again (used by the runtime's ``/api/run/reset``, Plan
    5.6). ``start()`` while a run is already in progress is a caller bug
    -- it would silently discard that run's completion -- and raises
    ``ContractViolation`` instead."""

    def __init__(self, experiment: ExperimentDefinition, config: RuntimeConfig) -> None:
        self._experiment = experiment
        self._config = config
        self._canonical_ids: list[str] = experiment.step_ids
        self._step_index: dict[str, int] = {sid: i for i, sid in enumerate(self._canonical_ids)}
        self._last_id: str = self._canonical_ids[-1]
        self._display_names: dict[str, str] = {s.step_id: s.display_name for s in experiment.steps}
        self._say: dict[str, str] = {s.step_id: s.say for s in experiment.steps}

        self.run_state: RunState = "idle"
        self._status: dict[str, StepStatus] = dict.fromkeys(self._canonical_ids, "pending")
        self._t_confirmed: dict[str, float | None] = dict.fromkeys(self._canonical_ids, None)
        self._observed: list[str] = []
        self._run_id: str = ""

    def start(self, t: float, run_id: str = "") -> None:
        # t is part of the Engine protocol signature but unused: start()
        # emits no event (essential-features.md #8 -- the runtime speaks
        # the first step and writes run_started, not the engine).
        if self.run_state == "running":
            raise ContractViolation(
                "start() called while a run is already in progress; "
                "call finish() first (IMPLEMENTATION_PLAN.md 5.6)"
            )
        self._status = dict.fromkeys(self._canonical_ids, "pending")
        self._t_confirmed = dict.fromkeys(self._canonical_ids, None)
        self._observed = []
        self._run_id = run_id
        self.run_state = "running"

    def on_state_event(self, event: StateEvent) -> list[EngineEvent]:
        if self.run_state != "running":
            return []
        o = event.step_id
        if o not in self._status:
            raise ContractViolation(f"unknown step_id {o!r}")

        self._observed.append(o)
        expected_before = self._first_pending()
        tag: ConfidenceTag = "flagged_uncertain" if event.uncertain else "confirmed"
        status_o = self._status[o]
        events: list[EngineEvent] = []
        newly_confirmed_last = False

        if status_o == "pending" and o == expected_before:
            self._status[o] = "confirmed"
            self._t_confirmed[o] = event.t
            next_id = self._first_pending()
            speak = (
                self._say[next_id]
                if next_id is not None and self._config.narrate_next_step
                else None
            )
            events.append(
                EngineEvent(
                    t=event.t,
                    kind="step_confirmed",
                    step_id=o,
                    expected_step_id=expected_before,
                    next_step_id=next_id,
                    confidence_tag=tag,
                    speak=speak,
                )
            )
            newly_confirmed_last = o == self._last_id

        elif status_o == "pending":
            o_index = self._step_index[o]
            skipped_ids = [
                sid
                for sid in self._canonical_ids
                if self._status[sid] == "pending" and self._step_index[sid] < o_index
            ]
            for sid in skipped_ids:
                self._status[sid] = "skipped"
            self._status[o] = "confirmed"
            self._t_confirmed[o] = event.t
            next_id = self._first_pending()
            names = ", ".join(self._display_names[sid] for sid in skipped_ids)
            events.append(
                EngineEvent(
                    t=event.t,
                    kind="deviation_detected",
                    step_id=o,
                    deviation_type="omission",
                    skipped_step_ids=skipped_ids,
                    expected_step_id=expected_before,
                    next_step_id=next_id,
                    confidence_tag=tag,
                    speak=f"Step skipped: {names}",
                )
            )
            newly_confirmed_last = o == self._last_id

        elif status_o == "skipped":
            self._status[o] = "completed_late"
            self._t_confirmed[o] = event.t
            next_id = self._first_pending()
            events.append(
                EngineEvent(
                    t=event.t,
                    kind="deviation_detected",
                    step_id=o,
                    deviation_type="out_of_order",
                    expected_step_id=expected_before,
                    next_step_id=next_id,
                    confidence_tag=tag,
                    speak=f"Out of order: {self._display_names[o]}",
                )
            )

        else:  # status_o in {"confirmed", "completed_late"} -> repeat
            next_id = self._first_pending()
            speak = f"Repeated: {self._display_names[o]}" if self._config.alert_on_repeat else None
            events.append(
                EngineEvent(
                    t=event.t,
                    kind="deviation_detected",
                    step_id=o,
                    deviation_type="repeat",
                    expected_step_id=expected_before,
                    next_step_id=next_id,
                    confidence_tag=tag,
                    speak=speak,
                )
            )

        if newly_confirmed_last:
            summary = self._build_summary(aborted=False)
            speak = (
                "Experiment complete"
                if summary.all_steps_done
                else "Experiment ended with skipped steps"
            )
            events.append(
                EngineEvent(
                    t=event.t,
                    kind="run_completed",
                    expected_step_id=self._first_pending(),
                    next_step_id=self._first_pending(),
                    confidence_tag=tag,
                    speak=speak,
                    summary=summary,
                )
            )
            self.run_state = "completed"

        return events

    def finish(self, t: float) -> list[EngineEvent]:
        if self.run_state != "running":
            return []
        summary = self._build_summary(aborted=True)
        self.run_state = "completed"
        pending = self._first_pending()
        return [
            EngineEvent(
                t=t,
                kind="run_completed",
                expected_step_id=pending,
                next_step_id=pending,
                confidence_tag="confirmed",
                speak="Run ended early",
                summary=summary,
            )
        ]

    def snapshot(self) -> list[StepProgress]:
        return [
            StepProgress(
                step_id=sid,
                status=self._status[sid],
                t_confirmed=self._t_confirmed[sid],
            )
            for sid in self._canonical_ids
        ]

    def _first_pending(self) -> str | None:
        for sid in self._canonical_ids:
            if self._status[sid] == "pending":
                return sid
        return None

    def _build_summary(self, aborted: bool) -> RunSummary:
        skipped_ids = [sid for sid in self._canonical_ids if self._status[sid] == "skipped"]
        late_ids = [sid for sid in self._canonical_ids if self._status[sid] == "completed_late"]
        pending_ids = [sid for sid in self._canonical_ids if self._status[sid] == "pending"]
        all_steps_done = not skipped_ids and not pending_ids
        # n >= 1 always: ExperimentDefinition.steps has min_length=1.
        n = len(self._canonical_ids)
        distance = DamerauLevenshtein.distance(self._canonical_ids, self._observed)
        pos = 1.0 - min(distance / n, 1.0)
        return RunSummary(
            run_id=self._run_id,
            pos=pos,
            all_steps_done=all_steps_done,
            aborted=aborted,
            observed_sequence=list(self._observed),
            skipped_step_ids=skipped_ids,
            late_step_ids=late_ids,
        )
