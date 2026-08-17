"""Runtime scaffolding tests for phase-7F entrypoint and bounded admission."""

from __future__ import annotations

import asyncio
from typing import cast

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


async def _hold_open_handle(_: object) -> None:
    await asyncio.Event().wait()


def test_runtime_rejects_17th_connection_without_scheduling(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeWriter:
        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    calls = {"count": 0}

    async def delayed_handle(_: object) -> None:
        calls["count"] += 1
        await _hold_open_handle(_)

    monkeypatch.setattr(runtime, "handle_one_session", delayed_handle)

    async def run_case() -> None:
        built = runtime.build_runtime(max_sessions=16)

        writers = [FakeWriter() for _ in range(17)]
        tasks = [
            asyncio.create_task(
                built._on_connection(
                    cast("asyncio.StreamReader", object()),
                    cast("asyncio.StreamWriter", writer),
                )
            )
            for writer in writers
        ]
        await asyncio.sleep(0)

        assert calls["count"] == 16
        assert built.active_sessions == 16
        assert built.session_tasks == 16
        assert all(writer.closed is False for writer in writers[:16])
        assert writers[16].closed

        # no handle should have been scheduled for the rejected edge connection
        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, (type(None), asyncio.CancelledError)) for result in results)
        assert calls["count"] == 16
        assert built.active_sessions == 0
        assert built.session_tasks == 0

    asyncio.run(run_case())



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
