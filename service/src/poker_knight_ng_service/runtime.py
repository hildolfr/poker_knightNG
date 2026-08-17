"""Runtime orchestration for the service listener.

This module adds Phase-7F runtime scaffolding on top of the existing
listener+session primitives from earlier phases.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .identity import resolve_production_identity
from .listener import L1Listener, construct_listener_with_callback
from .session import handle_one_session
from .stream_adapter import AsyncPeer

DEFAULT_MAX_SESSIONS = 16
_DEFAULT_GRACEFUL_DRAIN_SECONDS = 5.0


def _coerce_float_seconds(value: object) -> float:
    if type(value) is int:
        return float(value)
    if type(value) is float:
        return value
    raise TypeError("graceful_drain_seconds must be numeric")


@dataclass
class _AcceptedSession:
    peer: AsyncPeer
    admitted: bool = False


class ServiceRuntime:
    """Service lifecycle coordinator with bounded session admission."""

    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        graceful_drain_seconds: float = _DEFAULT_GRACEFUL_DRAIN_SECONDS,
        _test_listener_factory: Callable[
            [object, Callable[[asyncio.StreamReader, asyncio.StreamWriter], object]], Awaitable[L1Listener]
        ]
        | None = None,
        _test_graceful_drain_seconds: float | None = None,
    ) -> None:
        if type(max_sessions) is not int:
            raise TypeError("max_sessions must be an int")
        if max_sessions != DEFAULT_MAX_SESSIONS:
            raise ValueError(f"max_sessions is fixed at {DEFAULT_MAX_SESSIONS}")

        graceful_seconds = _coerce_float_seconds(graceful_drain_seconds)
        if not (graceful_seconds >= 0):
            raise ValueError("graceful_drain_seconds must be non-negative")
        if graceful_seconds != _DEFAULT_GRACEFUL_DRAIN_SECONDS:
            raise ValueError(
                f"graceful_drain_seconds is fixed at {_DEFAULT_GRACEFUL_DRAIN_SECONDS}"
            )
        # These underscored seams exist solely to exercise the runtime over a
        # real temporary Unix socket.  Production construction always uses the
        # identity-bound canonical listener and fixed five-second profile.
        if _test_graceful_drain_seconds is not None:
            graceful_seconds = _coerce_float_seconds(_test_graceful_drain_seconds)
            if not (graceful_seconds >= 0):
                raise ValueError("_test_graceful_drain_seconds must be non-negative")

        self._max_sessions = max_sessions
        self._graceful_drain_seconds = graceful_seconds
        self._active_sessions = 0
        # Fixed-cardinality, RAM-only aggregate: no peer, request, path, or
        # exception data is retained for diagnostics.
        self._rejected_sessions = 0
        self._session_tasks: dict[asyncio.Task[None], _AcceptedSession] = {}
        self._listener: L1Listener | None = None
        self._listener_factory = _test_listener_factory or construct_listener_with_callback
        self._stopping = False

    async def _on_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if not self._admit_session():
            writer.close()
            return

        task = asyncio.current_task()
        peer = AsyncPeer(reader, writer)
        state = _AcceptedSession(peer=peer)
        if task is not None:
            self._session_tasks[task] = state

        def mark_admitted() -> None:
            state.admitted = True

        try:
            await handle_one_session(peer, mark_admitted)
        finally:
            self._release_session()
            if task is not None:
                self._session_tasks.pop(task, None)
            # Guarantee the close -> wait_closed lifecycle even when the
            # handler exits early (cancellation or a raised error before it
            # closed the peer): wait_closed() on an unclosed writer blocks
            # forever. AsyncPeer.close() is idempotent, so the normal session
            # path that already closed the peer is unaffected.
            peer.close()
            await peer.wait_closed()

    def _admit_session(self) -> bool:
        if self._stopping:
            self._rejected_sessions += 1
            return False
        if self._active_sessions >= self._max_sessions:
            self._rejected_sessions += 1
            return False
        self._active_sessions += 1
        return True

    def _release_session(self) -> None:
        if self._active_sessions > 0:
            self._active_sessions -= 1

    def _close_non_admitted(self) -> None:
        for state in self._session_tasks.values():
            if not state.admitted:
                state.peer.close()

    def _has_non_admitted_tasks(self) -> bool:
        return any(not state.admitted for state in self._session_tasks.values())

    def _has_admitted_tasks(self) -> bool:
        return any(state.admitted for state in self._session_tasks.values())

    @property
    def active_sessions(self) -> int:
        return self._active_sessions

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    def diagnostics_snapshot(self) -> dict[str, int | str]:
        """Return the fixed, in-process-only operator diagnostics snapshot.

        This is deliberately not an HTTP route: socket filesystem permissions
        remain the only service authorization boundary, and the frozen 204
        ``/healthz`` response remains unchanged.  The snapshot has no
        unbounded labels or request-derived values, so callers can export it
        through a separately authorized local supervisor if one is approved.
        """

        return {
            "schema_version": "poker-knight-ng-runtime-diagnostics-v1",
            "readiness": "ready" if self._listener is not None and not self._stopping else "not-ready",
            "active_sessions": self._active_sessions,
            "max_sessions": self._max_sessions,
            "rejected_sessions": self._rejected_sessions,
        }

    async def serve(self, shutdown: asyncio.Event) -> None:
        """Serve until shutdown is requested."""

        identity = resolve_production_identity()
        self._listener = await self._listener_factory(identity, self._on_connection)
        try:
            await shutdown.wait()
        finally:
            # Cancellation is a process-control path too: retain neither a
            # live listener nor a false-ready diagnostics state.
            await self.stop()

    async def stop(self) -> None:
        """Stop accepting new sessions and wait for admitted work to finish."""

        if self._stopping:
            return
        self._stopping = True

        listener_close: asyncio.Future[None] | None = None
        if self._listener is not None:
            # L1 cleanup waits for accepted transports.  Start it before the
            # drain so it stops accepting immediately, while this runtime can
            # still close or drain the tracked callbacks that release it.
            listener_close = asyncio.ensure_future(self._listener.close())

        self._close_non_admitted()

        # Bounded phase: idle/pre-admission sessions must wind down within
        # the configured graceful-drain window (they were explicitly closed
        # above), so graceful_drain_seconds is an effective upper bound.
        deadline = asyncio.get_running_loop().time() + self._graceful_drain_seconds
        while self._has_non_admitted_tasks() and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.050)

        # A peer can keep its write half open after our close(), leaving its
        # read suspended forever.  The bounded pre-admission phase therefore
        # cancels only the still-idle handler tasks before admitted work drains.
        idle_tasks = [task for task, state in self._session_tasks.items() if not state.admitted]
        for task in idle_tasks:
            task.cancel()
        if idle_tasks:
            await asyncio.gather(*idle_tasks, return_exceptions=True)

        # ADR 0005 section 6: an admitted solve drains without a deadline.
        # Only admitted sessions may hold shutdown open past the bound.
        while self._has_admitted_tasks():
            await asyncio.sleep(0.050)
        if listener_close is not None:
            await listener_close

    @property
    def session_tasks(self) -> int:
        return len(self._session_tasks)


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach fixed runtime profile arguments to an ArgumentParser."""

    parser.epilog = (
        "Production runtime is fixed by ADR 0007/0005: max_sessions=16 and "
        "graceful_drain_seconds=5.0"
    )


def build_runtime(
    *,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    graceful_drain_seconds: float = _DEFAULT_GRACEFUL_DRAIN_SECONDS,
) -> ServiceRuntime:
    """Build a configured runtime object."""

    return ServiceRuntime(max_sessions=max_sessions, graceful_drain_seconds=graceful_drain_seconds)
