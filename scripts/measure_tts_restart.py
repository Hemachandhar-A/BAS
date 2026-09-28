#!/usr/bin/env python3
"""python scripts/measure_tts_restart.py

Owner: P2. A one-off (re-runnable) measurement tool for
IMPLEMENTATION_PLAN.md Part 10 P2.3: "measure on this machine how long a
worker restart takes and whether interruption is reliable; record it."
Uses the real pyttsx3 backend (outputs.tts.default_engine_factory), so it
needs a working audio device -- run it on each dev/demo machine and paste
the numbers into ISSUES.md.

Measures:
1. Cold start: TTSWorker() construction -> the first worker process signals
   ready (spawn + pyttsx3.init() on this machine).
2. Warm restart: time from an `alert` say() call to the *new* worker
   process signaling ready, while the *old* one was mid-utterance.
3. Interruption reliability: whether the pre-empted process is reliably no
   longer alive shortly after the restart, repeated over several trials.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from outputs.tts import TTSWorker  # noqa: E402


def measure_cold_start() -> float:
    t0 = time.perf_counter()
    worker = TTSWorker()
    ready = worker.wait_ready(timeout=15.0)
    elapsed = time.perf_counter() - t0
    worker.close()
    if not ready:
        raise RuntimeError("worker never became ready -- no audio device?")
    return elapsed


def measure_restart(trials: int = 5) -> list[dict[str, float | bool]]:
    results: list[dict[str, float | bool]] = []
    worker = TTSWorker(cooldown_s=0.0)
    worker.wait_ready(timeout=15.0)
    try:
        for i in range(trials):
            old_process = worker._process
            worker.say(f"This is a long test alert utterance, trial {i}", "alert")
            t0 = time.perf_counter()
            new_ready = worker.wait_ready(timeout=15.0)
            restart_elapsed = time.perf_counter() - t0

            # Give the OS a brief moment to finish tearing the old process
            # down, then check it is reliably dead.
            time.sleep(0.1)
            old_dead = not old_process.is_alive()

            results.append(
                {
                    "trial": i,
                    "new_worker_ready": new_ready,
                    "restart_ready_s": restart_elapsed,
                    "old_process_dead_after_100ms": old_dead,
                }
            )
            time.sleep(0.5)  # let this trial's utterance/no-op settle
    finally:
        worker.close()
    return results


def main() -> None:
    print("Cold start (construction -> first ready):")
    for i in range(3):
        elapsed = measure_cold_start()
        print(f"  trial {i}: {elapsed:.3f}s")

    print("\nWarm restart (alert say() -> new worker ready) + interruption reliability:")
    for row in measure_restart():
        print(
            f"  trial {row['trial']}: restart_ready={row['restart_ready_s']:.3f}s"
            f"  new_ready={row['new_worker_ready']}"
            f"  old_process_dead_after_100ms={row['old_process_dead_after_100ms']}"
        )


if __name__ == "__main__":
    main()
