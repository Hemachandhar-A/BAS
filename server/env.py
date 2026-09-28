"""server/env.py -- loads STREAM_USER/STREAM_PASSWORD from a git-ignored
``.env`` file (AGENTS.md rule 14: "No credentials in code, URLs, logs or
git (.env only)"). A minimal hand-rolled parser: ``python-dotenv`` is not in
the locked stack (IMPLEMENTATION_PLAN.md 3.1), and KEY=VALUE-line parsing is
a few lines, not "an algorithm a listed library provides" (AGENTS.md rule 1).
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path = ".env") -> None:
    """Populates ``os.environ`` from ``path`` for any key not already set
    there (a real environment variable always wins). Does nothing if the
    file does not exist -- callers that already export the variables (CI,
    a shell profile) need no ``.env`` file at all."""
    file = Path(path)
    if not file.exists():
        return
    for raw_line in file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def stream_credentials() -> tuple[str, str]:
    """Reads ``STREAM_USER``/``STREAM_PASSWORD`` from the environment
    (call ``load_env_file()`` first to populate it from ``.env``). Raises
    ``RuntimeError`` with a clear message if either is unset -- never falls
    back to a hardcoded default credential."""
    user = os.environ.get("STREAM_USER")
    password = os.environ.get("STREAM_PASSWORD")
    if not user or not password:
        raise RuntimeError(
            "STREAM_USER / STREAM_PASSWORD are not set. Copy .env.example "
            "to .env and fill in real values (AGENTS.md rule 14)."
        )
    return user, password
