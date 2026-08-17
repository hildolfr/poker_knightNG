"""Runtime orchestration for the service listener.

This module adds Phase-7F runtime scaffolding on top of the existing
listener+session primitives from earlier phases.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable

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


class ServiceRuntime:
    """Service lifecycle coordinator with bounded session admission."""

    def __init__(
        self,
        *,
        max_sessions: int = DEFAULT_MAX_SESSIONS,
        graceful_drain_seconds: float = _DEFAULT_GRACEFUL_DRAIN_SECONDS,
    ) -> None:
        if type(max_sessions) is not int or max_sessions <= 0:
            raise ValueError("max_sessions must be a positive integer")
        graceful_seconds = _coerce_float_seconds(graceful_drain_seconds)
        if not (graceful_seconds >= 0):
            raise ValueError("graceful_drain_seconds must be non-negative")

        self._max_sessions = max_sessions
        self._graceful_drain_seconds = graceful_seconds
        self._active_sessions = 0
        # Fixed-cardinality, RAM-only aggregate: no peer, request, path, or
        # exception data is retained for diagnostics.
        self._rejected_sessions = 0
        self._session_tasks: set[asyncio.Task[None]] = set()
        self._listener: L1Listener | None = None
        self._stopping = False

    async def _on_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if not self._admit_session():
            writer.close()
            return

        task = asyncio.current_task()
        if task is not None:
            self._session_tasks.add(task)
        try:
            await handle_one_session(AsyncPeer(reader, writer))
        finally:
            self._release_session()
            if task is not None:
                self._session_tasks.discard(task)

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
        self._listener = await construct_listener_with_callback(identity, self._on_connection)
        await shutdown.wait()
        await self.stop()

    async def stop(self) -> None:
        """Stop accepting new sessions and wait for admitted work to finish."""

        if self._stopping:
            return
        self._stopping = True

        if self._listener is not None:
            await self._listener.close()

        deadline = asyncio.get_running_loop().time() + self._graceful_drain_seconds
        while self._session_tasks and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.050)

        while self._session_tasks:
            await asyncio.sleep(0.050)

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
