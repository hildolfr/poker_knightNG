"""Real Unix-socket coverage for ServiceRuntime's accept boundary."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

import pytest

from poker_knight_ng_service import runtime
from poker_knight_ng_service.responses import serialize_health_response


class _TemporaryListener:
    """Minimal listener facade used only by the runtime's documented test seam."""

    def __init__(self, server: asyncio.AbstractServer, path: Path) -> None:
        self._server = server
        self._path = path

    async def close(self) -> None:
        self._server.close()
        await self._server.wait_closed()
        self._path.unlink(missing_ok=True)


def _factory(path: Path, ready: asyncio.Event):
    async def construct(_: object, callback: Callable[[asyncio.StreamReader, asyncio.StreamWriter], object]):
        server = await asyncio.start_unix_server(callback, path=str(path))
        ready.set()
        return _TemporaryListener(server, path)

    return construct


async def _start(path: Path, monkeypatch: pytest.MonkeyPatch, **runtime_kwargs: object):
    ready = asyncio.Event()
    monkeypatch.setattr(runtime, "resolve_production_identity", lambda: object())
    service = runtime.ServiceRuntime(_test_listener_factory=_factory(path, ready), **runtime_kwargs)
    shutdown = asyncio.Event()
    task = asyncio.create_task(service.serve(shutdown))
    await asyncio.wait_for(ready.wait(), timeout=1.0)
    return service, shutdown, task


async def _stop(shutdown: asyncio.Event, task: asyncio.Task[None]) -> None:
    shutdown.set()
    await asyncio.wait_for(task, timeout=1.0)


def test_real_socket_normal_request_returns_canonical_response_and_closes(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        _, shutdown, task = await _start(tmp_path / "runtime.sock", monkeypatch)
        reader, writer = await asyncio.open_unix_connection(str(tmp_path / "runtime.sock"))
        writer.write(b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n")
        await writer.drain()
        response = await asyncio.wait_for(reader.read(), timeout=1.0)
        assert response == serialize_health_response()
        writer.close()
        await writer.wait_closed()
        await _stop(shutdown, task)

    asyncio.run(scenario())


def test_real_socket_peer_disconnect_during_read_ends_cleanly(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        completed = asyncio.Event()
        original = runtime.handle_one_session

        async def observed(peer, on_admission=None) -> None:
            try:
                await original(peer, on_admission)
            finally:
                completed.set()

        monkeypatch.setattr(runtime, "handle_one_session", observed)
        _, shutdown, task = await _start(tmp_path / "runtime.sock", monkeypatch)
        _, writer = await asyncio.open_unix_connection(str(tmp_path / "runtime.sock"))
        writer.write(b"GET /healthz HTTP/1.1\r\nHost: local\r\n")
        await writer.drain()
        writer.close()
        await writer.wait_closed()
        await asyncio.wait_for(completed.wait(), timeout=1.0)
        await _stop(shutdown, task)

    asyncio.run(scenario())


def test_real_socket_rejects_seventeenth_held_connection(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        count = 0

        async def hold(peer, on_admission=None) -> None:
            nonlocal count
            del on_admission
            count += 1
            if count == runtime.DEFAULT_MAX_SESSIONS:
                entered.set()
            await release.wait()
            peer.close()

        monkeypatch.setattr(runtime, "handle_one_session", hold)
        service, shutdown, task = await _start(tmp_path / "runtime.sock", monkeypatch)
        clients = [await asyncio.open_unix_connection(str(tmp_path / "runtime.sock")) for _ in range(16)]
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        rejected_reader, rejected_writer = await asyncio.open_unix_connection(str(tmp_path / "runtime.sock"))
        assert await asyncio.wait_for(rejected_reader.read(1), timeout=1.0) == b""
        assert service.active_sessions == 16
        assert service.session_tasks == 16
        rejected_writer.close()
        await rejected_writer.wait_closed()
        release.set()
        for _, writer in clients:
            writer.close()
            await writer.wait_closed()
        await _stop(shutdown, task)
        assert service.active_sessions == 0
        assert service.diagnostics_snapshot()["rejected_sessions"] == 1

    asyncio.run(scenario())


def test_real_socket_shutdown_closes_pre_admission_and_drains_admitted(tmp_path, monkeypatch) -> None:
    async def scenario() -> None:
        pre_admission_closed = asyncio.Event()
        pre_admission_started = asyncio.Event()
        admitted = asyncio.Event()
        release_admitted = asyncio.Event()
        invocation = 0

        async def staged(peer, on_admission=None) -> None:
            nonlocal invocation
            invocation += 1
            if invocation == 1:
                pre_admission_started.set()
                try:
                    await peer.read(1)
                finally:
                    pre_admission_closed.set()
                return
            assert on_admission is not None
            await peer.read(1)
            on_admission()
            admitted.set()
            await release_admitted.wait()
            peer.close()

        monkeypatch.setattr(runtime, "handle_one_session", staged)
        service, shutdown, serve_task = await _start(
            tmp_path / "runtime.sock", monkeypatch, _test_graceful_drain_seconds=0.01
        )
        _, pre_writer = await asyncio.open_unix_connection(str(tmp_path / "runtime.sock"))
        await asyncio.wait_for(pre_admission_started.wait(), timeout=1.0)
        _, admitted_writer = await asyncio.open_unix_connection(str(tmp_path / "runtime.sock"))
        admitted_writer.write(b"x")
        await admitted_writer.drain()
        await asyncio.wait_for(admitted.wait(), timeout=1.0)
        stop_task = asyncio.create_task(service.stop())
        release_admitted.set()
        await asyncio.wait_for(stop_task, timeout=1.0)
        assert pre_admission_closed.is_set()
        assert service.active_sessions == 0
        pre_writer.close()
        admitted_writer.close()
        await pre_writer.wait_closed()
        await admitted_writer.wait_closed()
        shutdown.set()
        await asyncio.wait_for(serve_task, timeout=1.0)

    asyncio.run(scenario())
