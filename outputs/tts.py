"""outputs/tts.py -- TTSWorker and FakeSpeaker (F7; IMPLEMENTATION_PLAN.md
5.5; essential-features.md section 7). Both implement ``contracts.Speaker``.

``TTSWorker`` drives ``pyttsx3`` in a separate ``multiprocessing`` process
(spawn start method) because ``runAndWait()`` has long-standing hang reports
on some setups; the documented workaround is to kill the process to
interrupt speech. ``say()`` never blocks on audio playback and never
raises: an ``alert`` clears whatever is queued and pre-empts the current
utterance by terminating the worker and starting a fresh one (a new
``Queue`` is created too, so nothing stale is spoken); an identical
``alert`` text repeated inside ``cooldown_s`` is dropped instead of
restarting. ``info`` is only ever queued. If the worker process cannot
initialize a real engine (no audio device, driver missing) it degrades to
a silent no-op after logging one warning -- ``say()`` still never raises.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import time
from collections.abc import Callable
from typing import Any

from contracts import SpeakPriority

logger = logging.getLogger(__name__)

_TTS_RATE = 170
_TTS_VOLUME = 1.0


def default_engine_factory() -> Any:
    """Real backend: an initialized ``pyttsx3`` engine (SAPI5 / NSSS /
    eSpeak depending on OS). Imported lazily so importing this module never
    requires an audio driver to be present."""
    import pyttsx3

    return pyttsx3.init()


def null_engine_factory() -> Any:
    """Stub backend with no audio device -- used by the one integration
    test (essential-features.md section 7.6: CI has no audio device)."""
    return _NullEngine()


class _NullEngine:
    def setProperty(self, name: str, value: Any) -> None:
        pass

    def getProperty(self, name: str) -> Any:
        return [] if name == "voices" else None

    def say(self, text: str) -> None:
        pass

    def runAndWait(self) -> None:
        pass

    def stop(self) -> None:
        pass


def _choose_english_voice(voices: list[Any]) -> Any | None:
    if not voices:
        return None
    for voice in voices:
        languages = getattr(voice, "languages", None) or []
        for lang in languages:
            text = lang.decode("utf-8", "ignore") if isinstance(lang, bytes) else str(lang)
            if text.lower().startswith("en"):
                return voice
        name = getattr(voice, "name", "") or ""
        if "english" in name.lower():
            return voice
    return voices[0]


def _worker_main(
    queue: mp.Queue,
    ready: Any,
    engine_factory: Callable[[], Any],
) -> None:
    """Entry point of the child process. Reads ``(text, priority)`` pairs
    from ``queue`` until it receives ``None`` (the close sentinel)."""
    try:
        engine = engine_factory()
        engine.setProperty("rate", _TTS_RATE)
        engine.setProperty("volume", _TTS_VOLUME)
        voice = _choose_english_voice(engine.getProperty("voices") or [])
        if voice is not None:
            engine.setProperty("voice", getattr(voice, "id", voice))
        degraded = False
    except Exception:
        logger.warning("TTSWorker: engine init failed, degrading to silent no-op", exc_info=True)
        engine = None
        degraded = True
    ready.set()

    while True:
        item = queue.get()
        if item is None:
            return
        text, _priority = item
        if degraded or engine is None:
            continue
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception:
            logger.warning("TTSWorker: speaking failed, dropping utterance", exc_info=True)


class TTSWorker:
    """Implements ``contracts.Speaker``. One worker process per instance;
    ``close()`` shuts it down. ``clock`` is injectable for tests, since a
    cooldown is genuinely wall-clock (outputs/ is not covered by AGENTS.md
    rule 8's clock ban -- only perception/, state/ and engine/ are)."""

    def __init__(
        self,
        cooldown_s: float = 5.0,
        engine_factory: Callable[[], Any] = default_engine_factory,
        clock: Callable[[], float] = time.monotonic,
        ready_timeout_s: float = 10.0,
    ) -> None:
        self._cooldown_s = cooldown_s
        self._engine_factory = engine_factory
        self._clock = clock
        self._ready_timeout_s = ready_timeout_s
        self._ctx = mp.get_context("spawn")
        self._last_alert_text: str | None = None
        self._last_alert_at: float | None = None
        self._closed = False
        self._process: Any | None = None
        self._queue: mp.Queue | None = None
        self._ready: Any = None
        self._spawn()

    def wait_ready(self, timeout: float | None = None) -> bool:
        """Blocks until the current worker process has finished
        initializing (or timed out). Never called from ``say()`` -- only by
        callers measuring restart latency."""
        return self._ready.wait(timeout)

    def say(self, text: str, priority: SpeakPriority) -> None:
        if self._closed:
            return
        try:
            if priority == "alert":
                now = self._clock()
                in_cooldown = (
                    text == self._last_alert_text
                    and self._last_alert_at is not None
                    and (now - self._last_alert_at) < self._cooldown_s
                )
                if in_cooldown:
                    return
                self._last_alert_text = text
                self._last_alert_at = now
                self._restart()
            assert self._queue is not None
            self._queue.put_nowait((text, priority))
        except Exception:
            logger.warning("TTSWorker.say failed, dropping utterance", exc_info=True)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            assert self._queue is not None
            self._queue.put_nowait(None)
        except Exception:
            pass
        self._terminate_process()

    def _spawn(self) -> None:
        self._queue = self._ctx.Queue()
        self._ready = self._ctx.Event()
        self._process = self._ctx.Process(
            target=_worker_main,
            args=(self._queue, self._ready, self._engine_factory),
            daemon=True,
        )
        self._process.start()

    def _restart(self) -> None:
        self._terminate_process()
        self._spawn()

    def _terminate_process(self) -> None:
        process, self._process = self._process, None
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=1.0)


class FakeSpeaker:
    """Implements ``contracts.Speaker``. Test double: records every call in
    order, no process, no audio. Used by every golden test."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, SpeakPriority]] = []
        self.closed = False

    def say(self, text: str, priority: SpeakPriority) -> None:
        self.calls.append((text, priority))

    def close(self) -> None:
        self.closed = True
