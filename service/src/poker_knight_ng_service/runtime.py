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
        if type(graceful_drain_seconds) is not float and type(graceful_drain_seconds) is not int:
            raise TypeError("graceful_drain_seconds must be numeric")
        if graceful_drain_seconds < 0:
            raise ValueError("graceful_drain_seconds must be non-negative")

        self._max_sessions = max_sessions
        self._graceful_drain_seconds = float(graceful_drain_seconds)
        self._active_sessions = 0
        self._listener: L1Listener | None = None
        self._stopping = False

    async def _on_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if not self._admit_session():
            writer.close()
            return
        try:
            await handle_one_session(AsyncPeer(reader, writer))
        finally:
            self._release_session()

    def _admit_session(self) -> bool:
        if self._stopping:
            return False
        if self._active_sessions >= self._max_sessions:
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
        while self._active_sessions > 0 and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.050)

        while self._active_sessions > 0:
            await asyncio.sleep(0.050)


def add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach runtime CLI arguments to an ArgumentParser."""

    parser.add_argument(
        "--max-sessions",
        type=int,
        default=DEFAULT_MAX_SESSIONS,
        help="maximum concurrent accepted sessions",
    )
    parser.add_argument(
        "--graceful-drain-seconds",
        type=float,
        default=_DEFAULT_GRACEFUL_DRAIN_SECONDS,
        help="time to wait for non-admitted work before indefinite admit-hold drain",
    )


def build_runtime(
    *,
    max_sessions: int = DEFAULT_MAX_SESSIONS,
    graceful_drain_seconds: float = _DEFAULT_GRACEFUL_DRAIN_SECONDS,
) -> ServiceRuntime:
    """Build a configured runtime object."""

    return ServiceRuntime(max_sessions=max_sessions, graceful_drain_seconds=graceful_drain_seconds)
