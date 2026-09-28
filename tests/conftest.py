"""Shared test fixtures. The offline guard (IMPLEMENTATION_PLAN.md R10,
7.3) applies to every test in this repo: it is autouse, not opt-in."""

from __future__ import annotations

import socket

import pytest

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost", ""}
_original_connect = socket.socket.connect


def _guarded_connect(self: socket.socket, address):
    host = address[0] if isinstance(address, tuple) else address
    if host not in _LOOPBACK_HOSTS:
        raise RuntimeError(
            f"Outbound network call blocked in tests: {address!r}. "
            "Runtime code must be offline (IMPLEMENTATION_PLAN.md R10)."
        )
    return _original_connect(self, address)


@pytest.fixture(autouse=True)
def _block_outbound_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _guarded_connect)
