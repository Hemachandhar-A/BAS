"""tests/unit/server/test_env.py -- server/env.py (F10; AGENTS.md rule 14:
"No credentials in code, URLs, logs or git (.env only)"). No existing
coverage before this session's review pass -- server/env.py shipped
untested."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from server.env import load_env_file, stream_credentials


@pytest.fixture(autouse=True)
def _clean_stream_env_vars() -> None:
    """STREAM_USER/STREAM_PASSWORD must never leak between tests (or from
    the developer's own real .env / shell) into an assertion about what
    load_env_file() actually set."""
    saved = {k: os.environ.get(k) for k in ("STREAM_USER", "STREAM_PASSWORD", "OTHER_KEY")}
    for k in saved:
        os.environ.pop(k, None)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.mark.F10
def test_load_env_file_sets_simple_key_value_pairs(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("STREAM_USER=alice\nSTREAM_PASSWORD=hunter2\n", encoding="utf-8")
    load_env_file(env_path)
    assert os.environ["STREAM_USER"] == "alice"
    assert os.environ["STREAM_PASSWORD"] == "hunter2"


@pytest.mark.F10
def test_load_env_file_skips_comments_and_blank_lines(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "# a comment\n\nSTREAM_USER=alice\n   # indented comment\nSTREAM_PASSWORD=hunter2\n",
        encoding="utf-8",
    )
    load_env_file(env_path)
    assert os.environ["STREAM_USER"] == "alice"
    assert os.environ["STREAM_PASSWORD"] == "hunter2"


@pytest.mark.F10
def test_load_env_file_strips_matching_quotes(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text('STREAM_USER="alice"\nSTREAM_PASSWORD=\'hunter2\'\n', encoding="utf-8")
    load_env_file(env_path)
    assert os.environ["STREAM_USER"] == "alice"
    assert os.environ["STREAM_PASSWORD"] == "hunter2"


@pytest.mark.F10
def test_load_env_file_value_may_contain_an_equals_sign(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("STREAM_PASSWORD=abc=def123\n", encoding="utf-8")
    load_env_file(env_path)
    assert os.environ["STREAM_PASSWORD"] == "abc=def123"


@pytest.mark.F10
def test_load_env_file_never_overrides_an_existing_env_var(tmp_path: Path) -> None:
    os.environ["STREAM_USER"] = "already-set"
    env_path = tmp_path / ".env"
    env_path.write_text("STREAM_USER=from-file\n", encoding="utf-8")
    load_env_file(env_path)
    assert os.environ["STREAM_USER"] == "already-set"


@pytest.mark.F10
def test_load_env_file_missing_file_is_a_silent_no_op(tmp_path: Path) -> None:
    load_env_file(tmp_path / "does-not-exist.env")  # must not raise
    assert "STREAM_USER" not in os.environ


@pytest.mark.F10
def test_stream_credentials_raises_when_unset() -> None:
    with pytest.raises(RuntimeError, match="not set"):
        stream_credentials()


@pytest.mark.F10
def test_stream_credentials_raises_when_password_is_empty() -> None:
    os.environ["STREAM_USER"] = "alice"
    os.environ["STREAM_PASSWORD"] = ""
    with pytest.raises(RuntimeError, match="not set"):
        stream_credentials()


@pytest.mark.F10
def test_stream_credentials_rejects_the_env_example_placeholder() -> None:
    os.environ["STREAM_USER"] = "changeme"
    os.environ["STREAM_PASSWORD"] = "changeme"
    with pytest.raises(RuntimeError, match="placeholder"):
        stream_credentials()


@pytest.mark.F10
def test_stream_credentials_rejects_placeholder_password_even_with_a_real_username() -> None:
    os.environ["STREAM_USER"] = "alice"
    os.environ["STREAM_PASSWORD"] = "changeme"
    with pytest.raises(RuntimeError, match="placeholder"):
        stream_credentials()


@pytest.mark.F10
def test_stream_credentials_returns_real_values() -> None:
    os.environ["STREAM_USER"] = "alice"
    os.environ["STREAM_PASSWORD"] = "hunter2"
    assert stream_credentials() == ("alice", "hunter2")
