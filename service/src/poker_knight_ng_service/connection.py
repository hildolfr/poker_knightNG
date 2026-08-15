"""Bounded incremental assembly for one HTTP/1.1 request."""
from __future__ import annotations

import asyncio
from time import monotonic
from typing import Protocol

from .framing import AdmittedRequest, TransportFailure, _inspect_request_head, admit_request

_HEADER_LIMIT = 8192
_HEADER_TERMINATOR = b"\r\n\r\n"
_READ_TIMEOUT_SECONDS = 5.0
_clock = monotonic
_wait_for = asyncio.wait_for


class AsyncReader(Protocol):
    """Bounded reader interface supplied by a future connection adapter."""

    async def read(self, limit: int) -> bytes:
        """Wait for and return at most ``limit`` bytes."""
        ...

    def read_buffered(self, limit: int) -> bytes:
        """Return at most ``limit`` already-buffered bytes without waiting."""
        ...


async def _read_before(
    reader: AsyncReader,
    limit: int,
    deadline: float,
    timeout_status: int,
) -> bytes:
    remaining = deadline - _clock()
    if remaining <= 0:
        raise TransportFailure(timeout_status)
    try:
        chunk = await _wait_for(reader.read(limit), timeout=remaining)
    except TimeoutError:
        raise TransportFailure(timeout_status) from None
    if _clock() >= deadline:
        raise TransportFailure(timeout_status)
    return chunk


async def read_admitted_request(reader: AsyncReader) -> AdmittedRequest:
    """Read and admit exactly one request without owning a socket or listener."""

    buffered = bytearray()
    header_end: int | None = None
    header_deadline = _clock() + _READ_TIMEOUT_SECONDS
    while header_end is None:
        limit = _HEADER_LIMIT + 1 - len(buffered)
        chunk = await _read_before(reader, limit, header_deadline, 400)
        if not chunk:
            raise TransportFailure(400)
        if len(chunk) > limit:
            raise TransportFailure(400)
        buffered.extend(chunk)
        marker = buffered.find(_HEADER_TERMINATOR)
        if marker >= 0:
            header_end = marker + len(_HEADER_TERMINATOR)
            if header_end > _HEADER_LIMIT:
                raise TransportFailure(431)
        elif len(buffered) > _HEADER_LIMIT:
            raise TransportFailure(431)

    request_head = bytes(buffered[:header_end])
    declared_length = _inspect_request_head(request_head)
    received_body = bytearray(buffered[header_end:])
    if len(received_body) > declared_length:
        raise TransportFailure(400)

    body_deadline = _clock() + _READ_TIMEOUT_SECONDS
    while len(received_body) < declared_length:
        limit = declared_length - len(received_body) + 1
        chunk = await _read_before(reader, limit, body_deadline, 408)
        if not chunk:
            raise TransportFailure(400)
        if len(chunk) > limit:
            raise TransportFailure(400)
        received_body.extend(chunk)
        if len(received_body) > declared_length:
            raise TransportFailure(400)

    surplus = reader.read_buffered(1)
    if surplus:
        raise TransportFailure(400)

    return admit_request(request_head + bytes(received_body))
