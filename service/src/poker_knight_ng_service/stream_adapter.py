"""Public asyncio stream adapter for one bounded service session."""
from __future__ import annotations

import asyncio
import errno

_PEER_LOSS_ERRNOS = frozenset(
    (errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED, errno.ENOTCONN)
)


def _validate_limit(limit: object) -> int:
    if type(limit) is not int:
        raise TypeError("stream read limit must be an exact integer")
    if limit <= 0:
        raise ValueError("stream read limit must be positive")
    return limit


def _is_peer_loss(failure: OSError) -> bool:
    return failure.errno in _PEER_LOSS_ERRNOS


class AsyncPeer:
    """Weak-referenceable adapter using only public asyncio stream interfaces."""

    __slots__ = (
        "_reader",
        "_writer",
        "_overflow",
        "_close_called",
        "_wait_closed_called",
        "_wait_closed_result",
        "__weakref__",
    )

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self._overflow = bytearray()
        self._close_called = False
        self._wait_closed_called = False
        self._wait_closed_result: bool | None = None

    async def read(self, limit: int) -> bytes:
        """Return at most limit bytes with at most one adapter-owned surplus byte."""

        validated = _validate_limit(limit)
        if self._overflow:
            returned = bytes(self._overflow[:validated])
            del self._overflow[:validated]
            return returned

        chunk = await self._reader.read(validated + 1)
        if type(chunk) is not bytes:
            raise TypeError("StreamReader.read() must return bytes")
        if len(chunk) > validated + 1:
            raise RuntimeError("StreamReader exceeded requested bound")
        if len(chunk) > validated:
            self._overflow.extend(chunk[validated:])
            return chunk[:validated]
        return chunk

    def read_buffered(self, limit: int) -> bytes:
        """Consume only the adapter-owned surplus without touching reader internals."""

        validated = _validate_limit(limit)
        returned = bytes(self._overflow[:validated])
        del self._overflow[:validated]
        return returned

    async def send_response(self, response: bytes) -> bool:
        """Write and drain once; return false only for the closed peer-loss set."""

        if type(response) is not bytes:
            raise TypeError("response must be bytes")
        try:
            self._writer.write(response)
            await self._writer.drain()
        except OSError as failure:
            if _is_peer_loss(failure):
                return False
            raise
        return True

    def close(self) -> None:
        """Invoke StreamWriter.close() at most once."""

        if self._close_called:
            return
        self._close_called = True
        self._writer.close()

    async def wait_closed(self) -> bool | None:
        """Await StreamWriter.wait_closed() at most once."""

        if self._wait_closed_called:
            return self._wait_closed_result
        self._wait_closed_called = True
        try:
            await self._writer.wait_closed()
        except OSError as failure:
            if _is_peer_loss(failure):
                self._wait_closed_result = False
                return False
            raise
        self._wait_closed_result = True
        return True
