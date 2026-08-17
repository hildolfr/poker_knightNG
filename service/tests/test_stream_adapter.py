"""Public asyncio stream adapter contract tests."""
from __future__ import annotations

import asyncio
import errno
from enum import IntEnum
from functools import wraps

import pytest


def _async_test(function):
    @wraps(function)
    def run(*args, **kwargs):
        return asyncio.run(function(*args, **kwargs))

    return run


class _IntSubclass(int):
    pass


class _Number(IntEnum):
    ONE = 1


class _Reader:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = list(chunks)
        self.limits: list[int] = []

    @property
    def _buffer(self):
        pytest.fail("private StreamReader buffer was inspected")

    @property
    def _transport(self):
        pytest.fail("private StreamReader transport was inspected")

    async def read(self, limit: int) -> bytes:
        self.limits.append(limit)
        return self.chunks.pop(0)


class _Writer:
    def __init__(self, operation: str | None = None, failure: BaseException | None = None) -> None:
        self.operation = operation
        self.failure = failure
        self.writes: list[bytes] = []
        self.drain_calls = 0
        self.close_calls = 0
        self.wait_closed_calls = 0

    def _fail(self, operation: str) -> None:
        if self.operation == operation and self.failure is not None:
            raise self.failure

    def write(self, response: bytes) -> None:
        self.writes.append(response)
        self._fail("write")

    async def drain(self) -> None:
        self.drain_calls += 1
        self._fail("drain")

    def close(self) -> None:
        self.close_calls += 1
        self._fail("close")

    async def wait_closed(self) -> None:
        self.wait_closed_calls += 1
        self._fail("wait_closed")


class _ControlSignal(BaseException):
    pass


@_async_test
async def test_reader_owns_one_byte_surplus_and_never_uses_private_state() -> None:
    from poker_knight_ng_service.stream_adapter import AsyncPeer

    reader = _Reader([b"abcd", b"xy", b""])
    peer = AsyncPeer(reader, _Writer())

    assert await peer.read(3) == b"abc"
    assert reader.limits == [4]
    assert peer.read_buffered(1) == b"d"
    assert reader.limits == [4]
    assert peer.read_buffered(1) == b""

    assert await peer.read(1) == b"x"
    assert reader.limits == [4, 2]
    assert await peer.read(5) == b"y"
    assert reader.limits == [4, 2]
    assert await peer.read(2) == b""
    assert reader.limits == [4, 2, 3]


@_async_test
async def test_reader_preserves_short_and_exact_public_reads() -> None:
    from poker_knight_ng_service.stream_adapter import AsyncPeer

    reader = _Reader([b"a", b"bc"])
    peer = AsyncPeer(reader, _Writer())

    assert await peer.read(2) == b"a"
    assert await peer.read(2) == b"bc"
    assert peer.read_buffered(1) == b""
    assert reader.limits == [3, 3]


@pytest.mark.parametrize("invalid", [0, -1, True, _IntSubclass(1), _Number.ONE])
@_async_test
async def test_reader_rejects_nonexact_or_nonpositive_limits(invalid: object) -> None:
    from poker_knight_ng_service.stream_adapter import AsyncPeer

    reader = _Reader([b"forbidden"])
    peer = AsyncPeer(reader, _Writer())

    with pytest.raises((TypeError, ValueError)):
        await peer.read(invalid)  # type: ignore[arg-type]
    with pytest.raises((TypeError, ValueError)):
        peer.read_buffered(invalid)  # type: ignore[arg-type]
    assert reader.limits == []


@_async_test
async def test_reader_fails_if_underlying_public_contract_exceeds_limit_plus_one() -> None:
    from poker_knight_ng_service.stream_adapter import AsyncPeer

    peer = AsyncPeer(_Reader([b"abcd"]), _Writer())
    with pytest.raises(RuntimeError):
        await peer.read(2)


@_async_test
async def test_send_response_writes_and_drains_exactly_once() -> None:
    from poker_knight_ng_service.stream_adapter import AsyncPeer

    writer = _Writer()
    peer = AsyncPeer(_Reader([b""]), writer)

    assert await peer.send_response(b"response") is True
    assert writer.writes == [b"response"]
    assert writer.drain_calls == 1
    with pytest.raises(TypeError):
        await peer.send_response(bytearray(b"response"))  # type: ignore[arg-type]
    assert writer.writes == [b"response"]


_PEER_LOSS = [errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED, errno.ENOTCONN]


@pytest.mark.parametrize("operation", ["write", "drain", "wait_closed"])
@pytest.mark.parametrize("error_number", _PEER_LOSS)
@_async_test
async def test_writer_peer_loss_cartesian_matrix_returns_false(
    operation: str,
    error_number: int,
) -> None:
    from poker_knight_ng_service.stream_adapter import AsyncPeer

    writer = _Writer(operation, OSError(error_number, "peer gone"))
    peer = AsyncPeer(_Reader([b""]), writer)

    if operation == "wait_closed":
        result = await peer.wait_closed()
        assert writer.writes == []
        assert writer.drain_calls == 0
        assert writer.wait_closed_calls == 1
    else:
        result = await peer.send_response(b"response")
        assert writer.writes == [b"response"]
        assert writer.drain_calls == (1 if operation == "drain" else 0)
    assert result is False


@pytest.mark.parametrize("operation", ["write", "drain", "close", "wait_closed"])
@pytest.mark.parametrize("kind", ["ordinary", "base"])
@_async_test
async def test_unlisted_ordinary_and_base_failures_propagate_each_operation(
    operation: str,
    kind: str,
) -> None:
    from poker_knight_ng_service.stream_adapter import AsyncPeer

    failure: BaseException
    if kind == "ordinary":
        failure = RuntimeError(operation)
    else:
        failure = _ControlSignal(operation)
    writer = _Writer(operation, failure)
    peer = AsyncPeer(_Reader([b""]), writer)

    with pytest.raises(type(failure)) as caught:
        if operation == "close":
            peer.close()
        elif operation == "wait_closed":
            await peer.wait_closed()
        else:
            await peer.send_response(b"response")
    assert caught.value is failure
    assert writer.writes == ([b"response"] if operation in {"write", "drain"} else [])
    assert writer.drain_calls == (1 if operation == "drain" else 0)


@pytest.mark.parametrize("operation", ["write", "drain", "wait_closed"])
@_async_test
async def test_unlisted_oserror_is_not_misclassified_as_peer_loss(operation: str) -> None:
    from poker_knight_ng_service.stream_adapter import AsyncPeer

    failure = OSError(errno.EIO, "not peer loss")
    writer = _Writer(operation, failure)
    peer = AsyncPeer(_Reader([b""]), writer)

    with pytest.raises(OSError) as caught:
        if operation == "wait_closed":
            await peer.wait_closed()
        else:
            await peer.send_response(b"response")
    assert caught.value is failure


@_async_test
async def test_close_and_wait_closed_are_sequentially_idempotent() -> None:
    from poker_knight_ng_service.stream_adapter import AsyncPeer

    writer = _Writer()
    peer = AsyncPeer(_Reader([b""]), writer)

    peer.close()
    peer.close()
    assert writer.close_calls == 1
    assert await peer.wait_closed() is True
    assert await peer.wait_closed() is True
    assert writer.wait_closed_calls == 1


@_async_test
async def test_failed_close_and_wait_closed_are_not_retried() -> None:
    from poker_knight_ng_service.stream_adapter import AsyncPeer

    close_failure = RuntimeError("close")
    close_writer = _Writer("close", close_failure)
    close_peer = AsyncPeer(_Reader([b""]), close_writer)
    with pytest.raises(RuntimeError):
        close_peer.close()
    close_peer.close()
    assert close_writer.close_calls == 1

    wait_failure = RuntimeError("wait")
    wait_writer = _Writer("wait_closed", wait_failure)
    wait_peer = AsyncPeer(_Reader([b""]), wait_writer)
    with pytest.raises(RuntimeError):
        await wait_peer.wait_closed()
    assert await wait_peer.wait_closed() is None
    assert wait_writer.wait_closed_calls == 1
