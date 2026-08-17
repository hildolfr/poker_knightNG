"""Runtime scaffolding tests for phase-7F entrypoint and bounded admission."""

from __future__ import annotations

import asyncio

import pytest

from types import SimpleNamespace

from poker_knight_ng_service import runtime
from poker_knight_ng_service import identity as identity_mod


def test_runtime_rejects_invalid_bounds() -> None:
    with pytest.raises(ValueError):
        runtime.ServiceRuntime(max_sessions=0)
    with pytest.raises(ValueError):
        runtime.ServiceRuntime(graceful_drain_seconds=-1)


def test_runtime_admission_guard_is_strict() -> None:
    built = runtime.build_runtime(max_sessions=1)
    assert built.active_sessions == 0
    assert built._admit_session()
    assert built.active_sessions == 1
    assert not built._admit_session()
    built._release_session()
    assert built.active_sessions == 0
    assert built._admit_session()
    built._release_session()



def _run_runtime_stop_test(monkeypatch: pytest.MonkeyPatch) -> bool:
    called = {"closed": 0}

    class FakeListener:
        async def close(self) -> None:
            called["closed"] += 1

    async def fake_construct(identity, callback):  # noqa: ARG001
        assert callback is not None
        return FakeListener()

    async def run_case() -> None:
        monkeypatch.setattr(runtime, "construct_listener_with_callback", fake_construct)
        monkeypatch.setattr(
            runtime,
            "resolve_production_identity",
            identity_mod.resolve_production_identity,
        )
        monkeypatch.setattr(
            identity_mod,
            "_getpwnam",
            lambda name: SimpleNamespace(pw_uid=1234, pw_gid=1234),
        )
        monkeypatch.setattr(
            identity_mod,
            "_getgrnam",
            lambda name: SimpleNamespace(gr_gid=1234),
        )

        shutdown = asyncio.Event()
        server = runtime.build_runtime()
        task = asyncio.create_task(server.serve(shutdown))
        await asyncio.sleep(0)
        shutdown.set()
        await task

    asyncio.run(run_case())
    return called["closed"] == 1


def test_runtime_stop_closes_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_runtime_stop_test(monkeypatch)
