"""Runtime scaffolding tests for phase-7F entrypoint and bounded admission."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from types import SimpleNamespace

from poker_knight_ng_service import identity as identity_mod
from poker_knight_ng_service import runtime


class FakeWriter:
    def __init__(self) -> None:
        self.closed = False
        self._closed = asyncio.Event()
        self.wait_closed_calls = 0

    def close(self) -> None:
        self.closed = True
        if not self._closed.is_set():
            self._closed.set()

    async def wait_closed(self) -> bool | None:
        self.wait_closed_calls += 1
        await self._closed.wait()
        return True


class FakeListener:
    def __init__(self) -> None:
        self.close_calls = 0

    async def close(self) -> None:
        self.close_calls += 1


def test_runtime_rejects_invalid_bounds() -> None:
    with pytest.raises(TypeError):
        runtime.ServiceRuntime(max_sessions="16")
    with pytest.raises(ValueError):
        runtime.ServiceRuntime(max_sessions=15)
    with pytest.raises(ValueError):
        runtime.ServiceRuntime(graceful_drain_seconds=-1)
    with pytest.raises(ValueError):
        runtime.ServiceRuntime(graceful_drain_seconds=6)


def test_runtime_admission_guard_is_strict() -> None:
    built = runtime.build_runtime()
    assert built.active_sessions == 0
    for _ in range(16):
        assert built._admit_session()
    assert not built._admit_session()
    assert built.active_sessions == 16

    for _ in range(16):
        built._release_session()
    assert built.active_sessions == 0


async def _hold_open_handle(_: object, on_admission: object | None = None) -> None:
    del on_admission
    await asyncio.Event().wait()


def test_runtime_rejects_17th_connection_without_scheduling(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"count": 0}
    all_admitted_handlers_entered = asyncio.Event()

    async def delayed_handle(_: object, on_admission: object | None = None) -> None:
        del on_admission
        calls["count"] += 1
        if calls["count"] == runtime.DEFAULT_MAX_SESSIONS:
            all_admitted_handlers_entered.set()
        await _hold_open_handle(_)

    monkeypatch.setattr(runtime, "handle_one_session", delayed_handle)

    async def run_case() -> None:
        built = runtime.build_runtime()

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
        await asyncio.wait_for(all_admitted_handlers_entered.wait(), timeout=1)

        assert calls["count"] == 16
        assert built.active_sessions == 16
        assert built.session_tasks == 16
        assert all(writer.closed is False for writer in writers[:16])
        assert writers[16].closed

        for task in tasks:
            task.cancel()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        assert all(isinstance(result, (type(None), asyncio.CancelledError)) for result in results)
        assert calls["count"] == 16
        assert built.active_sessions == 0
        assert built.session_tasks == 0

    asyncio.run(run_case())


class _NoopListener:
    def __init__(self, called: list[int]) -> None:
        self.called = called

    async def close(self) -> None:
        self.called[0] += 1


def _run_runtime_stop_test(monkeypatch: pytest.MonkeyPatch) -> bool:
    called = [0]
    constructed = asyncio.Event()

    async def fake_construct(identity, callback):  # noqa: ARG001
        assert callback is not None
        constructed.set()
        return _NoopListener(called)

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
        await asyncio.wait_for(constructed.wait(), timeout=1)
        shutdown.set()
        await task

    asyncio.run(run_case())
    return called[0] == 1


def test_runtime_stop_closes_listener(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _run_runtime_stop_test(monkeypatch)


def test_runtime_cancellation_closes_listener_and_clears_readiness(monkeypatch: pytest.MonkeyPatch) -> None:
    class ListenerWithClose:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    async def run_case() -> None:
        listener = ListenerWithClose()
        constructed = asyncio.Event()

        async def fake_construct(identity, callback):  # noqa: ARG001
            constructed.set()
            return listener

        monkeypatch.setattr(runtime, "resolve_production_identity", lambda: object())
        monkeypatch.setattr(runtime, "construct_listener_with_callback", fake_construct)
        service = runtime.ServiceRuntime()
        task = asyncio.create_task(service.serve(asyncio.Event()))
        await asyncio.wait_for(constructed.wait(), timeout=1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert listener.close_calls == 1
        assert service.diagnostics_snapshot()["readiness"] == "not-ready"

    asyncio.run(run_case())


def test_runtime_stop_closes_pre_admission_connection_without_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    entered = asyncio.Event()

    async def delayed_handle(peer, on_admission: object | None = None) -> None:
        del on_admission
        entered.set()
        await peer.wait_closed()

    async def run_case() -> None:
        service = runtime.ServiceRuntime()
        service._listener = FakeListener()
        monkeypatch.setattr(runtime, "handle_one_session", delayed_handle)

        writer = FakeWriter()
        conn_task = asyncio.create_task(
            service._on_connection(cast("asyncio.StreamReader", object()), cast("asyncio.StreamWriter", writer))
        )
        await asyncio.wait_for(entered.wait(), timeout=1)
        await asyncio.wait_for(service.stop(), 1)
        await conn_task

        assert writer.closed

    asyncio.run(run_case())


def test_runtime_waits_for_admitted_connection_after_grace_period(monkeypatch: pytest.MonkeyPatch) -> None:
    async def run_case() -> None:
        proceed = asyncio.Event()
        admitted = asyncio.Event()
        mark = {"admitted": False}

        async def admitted_handle(peer, on_admission=None) -> None:
            assert on_admission is not None
            on_admission()
            mark["admitted"] = True
            admitted.set()
            await proceed.wait()
            peer.close()

        service = runtime.ServiceRuntime(_test_graceful_drain_seconds=0.01)
        monkeypatch.setattr(runtime, "handle_one_session", admitted_handle)

        writer = FakeWriter()
        conn_task = asyncio.create_task(
            service._on_connection(cast("asyncio.StreamReader", object()), cast("asyncio.StreamWriter", writer))
        )
        await asyncio.wait_for(admitted.wait(), timeout=1)

        stop_task = asyncio.create_task(service.stop())
        assert not stop_task.done()
        assert mark["admitted"]
        assert not writer.closed

        proceed.set()
        await conn_task
        await asyncio.wait_for(stop_task, 1)

        assert service.active_sessions == 0
        assert writer.closed

    asyncio.run(run_case())
