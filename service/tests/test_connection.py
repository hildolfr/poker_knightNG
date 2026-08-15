"""Bounded incremental connection-read tests."""
from __future__ import annotations

import asyncio
from collections import deque

import pytest


class ChunkReader:
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = deque(chunks)
        self.limits: list[int] = []

    async def read(self, limit: int) -> bytes:
        self.limits.append(limit)
        if not self._chunks:
            return b""
        chunk = self._chunks.popleft()
        if len(chunk) <= limit:
            return chunk
        self._chunks.appendleft(chunk[limit:])
        return chunk[:limit]

    def read_buffered(self, limit: int) -> bytes:
        if not self._chunks:
            return b""
        chunk = self._chunks.popleft()
        if len(chunk) <= limit:
            return chunk
        self._chunks.appendleft(chunk[limit:])
        return chunk[:limit]


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class AdvancingReader(ChunkReader):
    def __init__(self, clock: FakeClock, *chunks: bytes) -> None:
        super().__init__(*chunks)
        self._clock = clock

    async def read(self, limit: int) -> bytes:
        chunk = await super().read(limit)
        self._clock.now += 1.0
        return chunk


class RecordingWait:
    def __init__(self) -> None:
        self.timeouts: list[float] = []

    async def __call__(self, awaitable: object, *, timeout: float) -> bytes:
        self.timeouts.append(timeout)
        return await awaitable  # type: ignore[misc]


class TimeoutOnCall(RecordingWait):
    def __init__(self, call_number: int) -> None:
        super().__init__()
        self._call_number = call_number

    async def __call__(self, awaitable: object, *, timeout: float) -> bytes:
        self.timeouts.append(timeout)
        if len(self.timeouts) == self._call_number:
            close = getattr(awaitable, "close", None)
            if close is not None:
                close()
            raise TimeoutError
        return await awaitable  # type: ignore[misc]


class HangingBodyReader(ChunkReader):
    def __init__(self, *chunks: bytes) -> None:
        super().__init__(*chunks)
        self.cancelled = False

    async def read(self, limit: int) -> bytes:
        if self._chunks:
            return await super().read(limit)
        self.limits.append(limit)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        raise AssertionError("unreachable")


class CancellationSuppressingReader(ChunkReader):
    def __init__(self, fallback: bytes, *chunks: bytes) -> None:
        super().__init__(*chunks)
        self._fallback = fallback
        self.cancelled = False

    async def read(self, limit: int) -> bytes:
        if self._chunks:
            return await super().read(limit)
        self.limits.append(limit)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            return self._fallback
        raise AssertionError("unreachable")


def test_fragmented_post_is_admitted_with_exact_body() -> None:
    from poker_knight_ng_service.connection import read_admitted_request

    reader = ChunkReader(
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\nContent-T",
        b"ype: application/json\r\nContent-Length: 2\r\n\r",
        b"\n{",
        b"}",
    )

    admitted = asyncio.run(read_admitted_request(reader))

    assert admitted.method == b"POST"
    assert admitted.target == b"/v1/solve"
    assert admitted.body == b"{}"


def test_fragmented_header_consumes_one_monotonic_deadline(monkeypatch: object) -> None:
    import poker_knight_ng_service.connection as connection

    clock = FakeClock()
    wait = RecordingWait()
    reader = AdvancingReader(
        clock,
        b"GET /healthz HTTP/1.1\r\n",
        b"Host: local\r\n",
        b"\r\n",
    )
    monkeypatch.setattr(connection, "_clock", clock)  # type: ignore[attr-defined]
    monkeypatch.setattr(connection, "_wait_for", wait)  # type: ignore[attr-defined]

    admitted = asyncio.run(connection.read_admitted_request(reader))

    assert admitted.target == b"/healthz"
    assert wait.timeouts == [5.0, 4.0, 3.0]


def test_header_timeout_is_empty_400(monkeypatch: object) -> None:
    import pytest
    import poker_knight_ng_service.connection as connection
    from poker_knight_ng_service.framing import TransportFailure

    wait = TimeoutOnCall(1)
    monkeypatch.setattr(connection, "_clock", FakeClock())  # type: ignore[attr-defined]
    monkeypatch.setattr(connection, "_wait_for", wait)  # type: ignore[attr-defined]

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(connection.read_admitted_request(ChunkReader(b"GET ")))

    assert caught.value.status == 400
    assert caught.value.body == b""
    assert wait.timeouts == [5.0]


def test_body_timeout_is_empty_408(monkeypatch: object) -> None:
    import pytest
    import poker_knight_ng_service.connection as connection
    from poker_knight_ng_service.framing import TransportFailure

    wait = TimeoutOnCall(2)
    reader = ChunkReader(
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n"
    )
    monkeypatch.setattr(connection, "_clock", FakeClock())  # type: ignore[attr-defined]
    monkeypatch.setattr(connection, "_wait_for", wait)  # type: ignore[attr-defined]

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(connection.read_admitted_request(reader))

    assert caught.value.status == 408
    assert caught.value.body == b""
    assert wait.timeouts == [5.0, 5.0]


def test_body_gets_fresh_monotonic_deadline_after_header(monkeypatch: object) -> None:
    import poker_knight_ng_service.connection as connection

    clock = FakeClock()
    wait = RecordingWait()
    reader = AdvancingReader(
        clock,
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n",
        b"{",
        b"}",
    )
    monkeypatch.setattr(connection, "_clock", clock)  # type: ignore[attr-defined]
    monkeypatch.setattr(connection, "_wait_for", wait)  # type: ignore[attr-defined]

    admitted = asyncio.run(connection.read_admitted_request(reader))

    assert admitted.body == b"{}"
    assert wait.timeouts == [5.0, 5.0, 4.0]


def _header_with_total_size(total: int) -> bytes:
    head = b"GET /healthz HTTP/1.1\r\nHost: local\r\n"
    head += b"".join(
        f"X-{index}:".encode("ascii") + b"x" * 1000 + b"\r\n"
        for index in range(8)
    )
    remaining = total - len(head) - len(b"\r\n")
    assert 4 <= remaining <= 1028
    head += b"Z:" + b"z" * (remaining - 4) + b"\r\n"
    raw = head + b"\r\n"
    assert len(raw) == total
    return raw


def test_header_at_exact_aggregate_limit_is_admitted() -> None:
    from poker_knight_ng_service.connection import read_admitted_request

    admitted = asyncio.run(read_admitted_request(ChunkReader(_header_with_total_size(8192))))

    assert admitted.target == b"/healthz"


def test_header_one_byte_over_aggregate_limit_is_empty_431() -> None:
    from poker_knight_ng_service.connection import read_admitted_request
    from poker_knight_ng_service.framing import TransportFailure

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(read_admitted_request(ChunkReader(_header_with_total_size(8193))))

    assert caught.value.status == 431
    assert caught.value.body == b""


def test_header_eof_is_empty_400() -> None:
    from poker_knight_ng_service.connection import read_admitted_request
    from poker_knight_ng_service.framing import TransportFailure

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(read_admitted_request(ChunkReader(b"GET /healthz HTTP/1.1\r\n")))

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_body_eof_is_empty_400() -> None:
    from poker_knight_ng_service.connection import read_admitted_request
    from poker_knight_ng_service.framing import TransportFailure

    reader = ChunkReader(
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{"
    )

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(read_admitted_request(reader))

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_one_surplus_body_byte_is_empty_400() -> None:
    from poker_knight_ng_service.connection import read_admitted_request
    from poker_knight_ng_service.framing import TransportFailure

    reader = ChunkReader(
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n{}x"
    )

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(read_admitted_request(reader))

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_separately_buffered_surplus_byte_is_empty_400() -> None:
    from poker_knight_ng_service.connection import read_admitted_request
    from poker_knight_ng_service.framing import TransportFailure

    reader = ChunkReader(
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n",
        b"{}",
        b"x",
    )

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(read_admitted_request(reader))

    assert caught.value.status == 400
    assert caught.value.body == b""


def test_maximum_declared_body_is_admitted_with_one_byte_surplus_probe() -> None:
    from poker_knight_ng_service.connection import read_admitted_request

    reader = ChunkReader(
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: 16384\r\n\r\n",
        b"x" * 16384,
    )

    admitted = asyncio.run(read_admitted_request(reader))

    assert admitted.body == b"x" * 16384
    assert reader.limits == [8193, 16385]


def test_real_body_timeout_cancels_pending_read(monkeypatch: object) -> None:
    import poker_knight_ng_service.connection as connection
    from poker_knight_ng_service.framing import TransportFailure

    reader = HangingBodyReader(
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n"
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        connection, "_READ_TIMEOUT_SECONDS", 0.001
    )

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(connection.read_admitted_request(reader))

    assert caught.value.status == 408
    assert caught.value.body == b""
    assert reader.cancelled is True


def test_declared_payload_overflow_fails_before_body_read() -> None:
    from poker_knight_ng_service.connection import read_admitted_request
    from poker_knight_ng_service.framing import TransportFailure

    reader = ChunkReader(
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: 16385\r\n\r\n",
        b"x" * 16385,
    )

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(read_admitted_request(reader))

    assert caught.value.status == 413
    assert caught.value.body == b""
    assert reader.limits == [8193]


def test_header_deadline_survives_reader_cancellation_suppression(
    monkeypatch: object,
) -> None:
    import poker_knight_ng_service.connection as connection
    from poker_knight_ng_service.framing import TransportFailure

    reader = CancellationSuppressingReader(
        b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        connection, "_READ_TIMEOUT_SECONDS", 0.001
    )

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(connection.read_admitted_request(reader))

    assert caught.value.status == 400
    assert caught.value.body == b""
    assert reader.cancelled is True


def test_body_deadline_survives_reader_cancellation_suppression(
    monkeypatch: object,
) -> None:
    import poker_knight_ng_service.connection as connection
    from poker_knight_ng_service.framing import TransportFailure

    reader = CancellationSuppressingReader(
        b"{}",
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: 2\r\n\r\n",
    )
    monkeypatch.setattr(  # type: ignore[attr-defined]
        connection, "_READ_TIMEOUT_SECONDS", 0.001
    )

    with pytest.raises(TransportFailure) as caught:
        asyncio.run(connection.read_admitted_request(reader))

    assert caught.value.status == 408
    assert caught.value.body == b""
    assert reader.cancelled is True
