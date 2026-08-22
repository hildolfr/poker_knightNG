"""Listener-free one-request session coordinator tests."""
from __future__ import annotations

import asyncio
from collections import deque
import json
import re
import threading

import pytest

# Bounded test-local handshake timeouts (seconds) mirroring the pattern used in
# test_runtime.py post-fix: explicit awaited, bounded waits instead of
# scheduler-dependent sleep(0)/raw wait() polling. Generous enough never to trip
# spuriously on a loaded CI scheduler, yet hard-bounded so a genuinely stuck
# worker fails loudly instead of hanging.
_WORKER_WAIT_TIMEOUT = 30.0  # cross-thread worker handshake (finish/release)
_START_HANDSHAKE_TIMEOUT = 5.0  # wait for a worker thread to signal it started
_YIELD_TO_LOOP_TIMEOUT = 0.1  # bounded yield so a cancellation can be delivered


async def _await_started(
    started: threading.Event,
    timeout: float = _START_HANDSHAKE_TIMEOUT,
) -> None:
    """Deterministically await a worker-thread start signal (bounded, no busy-wait)."""
    await asyncio.wait_for(
        asyncio.to_thread(started.wait, timeout),
        timeout=timeout + 1.0,
    )
    assert started.is_set()


async def _yield_until_still_running(
    task: asyncio.Task[object],
    timeout: float = _YIELD_TO_LOOP_TIMEOUT,
) -> None:
    """Yield to the loop for a bounded interval and require the task to survive.

    Replaces ``await asyncio.sleep(0)`` used to let a cancellation be delivered
    into an already-cancelled-but-draining task. The task is shielded so the
    bounded ``wait_for`` timeout does not cancel it.
    """
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
    assert not task.done()


class MemorySession:
    def __init__(self, *chunks: bytes) -> None:
        self._chunks = deque(chunks)
        self.sent: list[bytes] = []
        self.close_count = 0

    async def read(self, limit: int) -> bytes:
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

    async def send_response(self, response: bytes) -> bool:
        self.sent.append(response)
        return True

    def close(self) -> None:
        self.close_count += 1


def cpu_request() -> bytes:
    body = (
        b'{"backend":"cpu_reference","board_cards":[],"contract_version":"v1",'
        b'"hero_cards":["As","Kd"],"opponent_count":"1","requested_trials":"1",'
        b'"rng":{"algorithm_id":"poker-knight-ng/philox4x32-10",'
        b'"algorithm_version":"1"},"seed":"0x0000000000000001"}'
    )
    return (
        b"POST /v1/solve HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode("ascii")
        + b"\r\n\r\n"
        + body
    )


def solve_request(body: bytes, *, target: bytes = b"/v1/solve") -> bytes:
    return (
        b"POST " + target + b" HTTP/1.1\r\nHost: local\r\n"
        b"Content-Type: application/json\r\nContent-Length: "
        + str(len(body)).encode("ascii")
        + b"\r\n\r\n"
        + body
    )


def test_health_session_sends_exact_empty_204_and_closes() -> None:
    from poker_knight_ng_service.session import handle_one_session

    session = MemorySession(b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n")

    asyncio.run(handle_one_session(session))

    assert len(session.sent) == 1
    assert session.sent[0].startswith(b"HTTP/1.1 204 \r\n")
    assert session.sent[0].endswith(b"\r\n\r\n")
    assert b"X-Poker-Knight-Request-ID:" not in session.sent[0]
    assert session.close_count == 1


def test_unknown_route_sends_exact_empty_404_and_closes() -> None:
    from poker_knight_ng_service.session import handle_one_session

    session = MemorySession(b"GET /missing HTTP/1.1\r\nHost: local\r\n\r\n")

    asyncio.run(handle_one_session(session))

    assert len(session.sent) == 1
    assert session.sent[0].startswith(b"HTTP/1.1 404 \r\n")
    assert b"Content-Length: 0\r\n" in session.sent[0]
    assert b"Content-Type:" not in session.sent[0]
    assert b"X-Poker-Knight-Request-ID:" not in session.sent[0]
    assert session.sent[0].endswith(b"\r\n\r\n")
    assert session.close_count == 1


def test_cpu_session_executes_once_and_sends_canonical_success() -> None:
    from poker_knight_ng_service.session import handle_one_session

    session = MemorySession(cpu_request())

    asyncio.run(handle_one_session(session))

    assert len(session.sent) == 1
    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 200 \r\n")
    match = re.search(rb"X-Poker-Knight-Request-ID: (pk_[0-9a-f]{32})\r\n", head + b"\r\n")
    assert match is not None
    assert b"Content-Type: application/json\r\n" in head + b"\r\n"
    assert body.endswith(b"\n")
    payload = json.loads(body)
    assert payload["backend"] == "cpu_reference"
    assert payload["completed_trials"] == "1"
    assert "correlation_id" not in payload
    assert session.close_count == 1




def test_auto_route_cuda_session_executes_once_and_marks_cuda_backend(monkeypatch) -> None:
    import poker_knight_ng_service.routing as routing
    import poker_knight_ng_service.session as coordinator

    observed = {}

    async def fake_execute(adapted):
        observed["route"] = object.__getattribute__(adapted, "route")
        observed["backend"] = object.__getattribute__(object.__getattribute__(adapted, "request"), "backend")
        return {
            "backend": observed["backend"],
            "completed_trials": "1",
        }

    monkeypatch.setattr(coordinator, "execute_solve_async", fake_execute)
    body = (
        b'{"backend":"cuda","board_cards":[],"contract_version":"v1",'
        b'"hero_cards":["As","Kd"],"opponent_count":"1","requested_trials":"1",'
        b'"rng":{"algorithm_id":"poker-knight-ng/philox4x32-10",'
        b'"algorithm_version":"1"},"seed":"0x0000000000000001"}'
    )
    session = MemorySession(solve_request(body, target=b"/v1/solve"))

    asyncio.run(coordinator.handle_one_session(session))

    assert observed["route"] == routing.Route.AUTO_SOLVE
    assert observed["backend"] == "cuda"
    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 200 \r\n")
    payload = json.loads(body)
    assert payload["backend"] == "cuda"
    assert payload["completed_trials"] == "1"
    assert session.close_count == 1


def test_explicit_cuda_route_rejects_cpu_backend_before_execution(monkeypatch) -> None:
    import poker_knight_ng_service.session as coordinator

    async def forbidden_execute(adapted):
        raise AssertionError("cpu route entered execution")

    monkeypatch.setattr(coordinator, "execute_solve_async", forbidden_execute)

    body = (
        b'{"backend":"cpu_reference","board_cards":[],"contract_version":"v1",'
        b'"hero_cards":["As","Kd"],"opponent_count":"1","requested_trials":"1",'
        b'"rng":{"algorithm_id":"poker-knight-ng/philox4x32-10",'
        b'"algorithm_version":"1"},"seed":"0x0000000000000001"}'
    )
    session = MemorySession(solve_request(body, target=b"/v1/solve-cuda"))

    asyncio.run(coordinator.handle_one_session(session))

    assert len(session.sent) == 1
    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 400 \r\n")
    payload = json.loads(body)
    assert payload["code"] == "UNSUPPORTED_REQUEST"
    assert session.close_count == 1


def test_accepted_malformed_json_sends_correlated_problem() -> None:
    from poker_knight_ng_service.session import handle_one_session

    session = MemorySession(solve_request(b"{"))

    asyncio.run(handle_one_session(session))

    assert len(session.sent) == 1
    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 400 \r\n")
    match = re.search(rb"X-Poker-Knight-Request-ID: (pk_[0-9a-f]{32})\r\n", head + b"\r\n")
    assert match is not None
    payload = json.loads(body)
    assert payload["code"] == "UNSUPPORTED_REQUEST"
    assert payload["status"] == 400
    assert payload["correlation_id"].encode("ascii") == match.group(1)
    assert body == json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii") + b"\n"
    assert session.close_count == 1


def test_request_id_failure_sends_emergency_internal_without_execution(monkeypatch) -> None:
    import poker_knight_ng_service.session as coordinator
    from poker_knight_ng_service.responses import (
        EMERGENCY_REQUEST_ID,
        RequestIdGenerationFailure,
    )

    executed = False

    def failed_request_id() -> str:
        raise RequestIdGenerationFailure

    async def forbidden_execute(adapted):
        nonlocal executed
        executed = True
        raise AssertionError("request ID failure reached execution")

    monkeypatch.setattr(coordinator, "generate_request_id", failed_request_id)
    monkeypatch.setattr(coordinator, "execute_solve_async", forbidden_execute)
    session = MemorySession(cpu_request())

    asyncio.run(coordinator.handle_one_session(session))

    assert not executed
    assert len(session.sent) == 1
    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 500 \r\n")
    assert f"X-Poker-Knight-Request-ID: {EMERGENCY_REQUEST_ID}\r\n".encode() in head + b"\r\n"
    payload = json.loads(body)
    assert payload["code"] == "INTERNAL_ERROR"
    assert payload["correlation_id"] == EMERGENCY_REQUEST_ID
    assert session.close_count == 1


def test_request_id_failure_render_fault_uses_independent_internal_fallback(monkeypatch) -> None:
    import poker_knight_ng_service.session as coordinator
    from poker_knight_ng_service.responses import (
        EMERGENCY_REQUEST_ID,
        RequestIdGenerationFailure,
    )

    def failed_request_id() -> str:
        raise RequestIdGenerationFailure

    def failed_problem_parts(failure, request_id):
        raise RuntimeError("private-emergency-render-fault")

    async def forbidden_execute(adapted):
        pytest.fail("emergency render fallback reached execution")

    monkeypatch.setattr(coordinator, "generate_request_id", failed_request_id)
    monkeypatch.setattr(coordinator, "_problem_parts", failed_problem_parts)
    monkeypatch.setattr(coordinator, "execute_solve_async", forbidden_execute)
    session = MemorySession(cpu_request())

    asyncio.run(coordinator.handle_one_session(session))

    assert len(session.sent) == 1
    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 500 \r\n")
    assert f"X-Poker-Knight-Request-ID: {EMERGENCY_REQUEST_ID}\r\n".encode() in head + b"\r\n"
    payload = json.loads(body)
    assert payload["code"] == "INTERNAL_ERROR"
    assert payload["correlation_id"] == EMERGENCY_REQUEST_ID
    assert b"private-emergency-render-fault" not in body
    assert session.close_count == 1


def test_unexpected_request_id_fault_uses_emergency_internal_without_execution(monkeypatch) -> None:
    import poker_knight_ng_service.session as coordinator
    from poker_knight_ng_service.responses import EMERGENCY_REQUEST_ID

    def unexpected_request_id() -> str:
        raise RuntimeError("private-rid-fault")

    async def forbidden_execute(adapted):
        pytest.fail("unexpected request-ID fault reached execution")

    monkeypatch.setattr(coordinator, "generate_request_id", unexpected_request_id)
    monkeypatch.setattr(coordinator, "execute_solve_async", forbidden_execute)
    session = MemorySession(cpu_request())

    asyncio.run(coordinator.handle_one_session(session))

    assert len(session.sent) == 1
    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 500 \r\n")
    assert f"X-Poker-Knight-Request-ID: {EMERGENCY_REQUEST_ID}\r\n".encode() in head + b"\r\n"
    payload = json.loads(body)
    assert payload["code"] == "INTERNAL_ERROR"
    assert payload["correlation_id"] == EMERGENCY_REQUEST_ID
    assert b"private-rid-fault" not in body
    assert session.close_count == 1


def test_request_id_process_control_propagates_unchanged_and_closes(monkeypatch) -> None:
    import poker_knight_ng_service.session as coordinator

    class Stop(BaseException):
        pass

    stop = Stop()

    def stopped_request_id() -> str:
        raise stop

    monkeypatch.setattr(coordinator, "generate_request_id", stopped_request_id)
    session = MemorySession(cpu_request())

    with pytest.raises(Stop) as caught:
        asyncio.run(coordinator.handle_one_session(session))

    assert caught.value is stop
    assert session.sent == []
    assert session.close_count == 1


def test_unexpected_result_serialization_fault_sends_internal_without_leak(monkeypatch) -> None:
    import poker_knight_ng_service.session as coordinator

    async def invalid_payload(adapted):
        return {"secret-marker": object()}

    monkeypatch.setattr(coordinator, "execute_solve_async", invalid_payload)
    session = MemorySession(cpu_request())

    asyncio.run(coordinator.handle_one_session(session))

    assert len(session.sent) == 1
    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 500 \r\n")
    payload = json.loads(body)
    assert payload["code"] == "INTERNAL_ERROR"
    assert payload["status"] == 500
    assert "secret-marker" not in body.decode("ascii")
    assert "object" not in body.decode("ascii")
    assert session.close_count == 1


def test_peer_gone_at_send_discards_completed_response_and_closes() -> None:
    from poker_knight_ng_service.session import handle_one_session

    class GoneSession(MemorySession):
        async def send_response(self, response: bytes) -> bool:
            self.sent.append(response)
            return False

    session = GoneSession(cpu_request())

    asyncio.run(handle_one_session(session))

    assert len(session.sent) == 1
    assert session.sent[0].startswith(b"HTTP/1.1 200 \r\n")
    assert session.close_count == 1


def test_parent_cancellation_drains_admitted_worker_before_close(monkeypatch) -> None:
    import poker_knight_ng_service.async_execution as async_execution
    from poker_knight_ng_service.admission import admit_solve
    from poker_knight_ng_service.session import handle_one_session

    started = threading.Event()
    finish = threading.Event()

    def execute(adapted, lease):
        lease._assert_active()
        started.set()
        assert finish.wait(_WORKER_WAIT_TIMEOUT)
        lease._assert_active()
        return {"discarded": True}

    monkeypatch.setattr(async_execution, "_execute_admitted", execute)
    session = MemorySession(cpu_request())

    async def scenario() -> None:
        task = asyncio.create_task(handle_one_session(session))
        await _await_started(started)
        task.cancel()
        await _yield_until_still_running(task)
        assert not task.done()
        assert session.close_count == 0
        held_busy = False
        try:
            admit_solve()
        except Exception as failure:
            held_busy = getattr(failure, "code", None) == "RESOURCE_EXHAUSTED"
        assert held_busy
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert session.sent == []
    assert session.close_count == 1
    lease = admit_solve()
    lease.release()


def test_busy_solve_sends_correlated_503_without_queue() -> None:
    from poker_knight_ng_service.admission import admit_solve
    from poker_knight_ng_service.session import handle_one_session

    held = admit_solve()
    session = MemorySession(cpu_request())
    try:
        asyncio.run(handle_one_session(session))
    finally:
        held.release()

    assert len(session.sent) == 1
    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 503 \r\n")
    payload = json.loads(body)
    assert payload["code"] == "RESOURCE_EXHAUSTED"
    assert payload["retryable"] is True
    assert payload["correlation_id"].encode("ascii") in head
    assert session.close_count == 1


@pytest.mark.parametrize(
    "raw",
    [
        b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n",
        b"GET /missing HTTP/1.1\r\nHost: local\r\n\r\n",
    ],
)
def test_health_and_transport_paths_never_allocate_solve_identity(monkeypatch, raw: bytes) -> None:
    import poker_knight_ng_service.session as coordinator

    def forbidden_request_id() -> str:
        pytest.fail("non-solve path allocated a request ID")

    async def forbidden_execute(adapted):
        pytest.fail("non-solve path reached execution")

    monkeypatch.setattr(coordinator, "generate_request_id", forbidden_request_id)
    monkeypatch.setattr(coordinator, "execute_solve_async", forbidden_execute)
    session = MemorySession(raw)

    asyncio.run(coordinator.handle_one_session(session))

    assert len(session.sent) == 1
    assert b"X-Poker-Knight-Request-ID:" not in session.sent[0]
    assert session.close_count == 1


def test_reader_process_control_propagates_unchanged_and_closes() -> None:
    from poker_knight_ng_service.session import handle_one_session

    class Stop(BaseException):
        pass

    stop = Stop()

    class StoppedSession(MemorySession):
        async def read(self, limit: int) -> bytes:
            raise stop

    session = StoppedSession()

    with pytest.raises(Stop) as caught:
        asyncio.run(handle_one_session(session))

    assert caught.value is stop
    assert session.sent == []
    assert session.close_count == 1


def test_send_infrastructure_fault_propagates_after_close() -> None:
    from poker_knight_ng_service.session import handle_one_session

    failure = RuntimeError("send unavailable")

    class FailedSendSession(MemorySession):
        async def send_response(self, response: bytes) -> bool:
            raise failure

    session = FailedSendSession(b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n")

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(handle_one_session(session))

    assert caught.value is failure
    assert session.sent == []
    assert session.close_count == 1


def test_problem_render_fault_uses_independent_correlated_internal(monkeypatch) -> None:
    import poker_knight_ng_service.session as coordinator

    calls = 0

    def fail_problem_parts(failure, request_id):
        nonlocal calls
        calls += 1
        raise RuntimeError("private-render-fault")

    monkeypatch.setattr(coordinator, "_problem_parts", fail_problem_parts)
    session = MemorySession(solve_request(b"{"))

    asyncio.run(coordinator.handle_one_session(session))

    assert calls == 1
    assert len(session.sent) == 1
    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(b"HTTP/1.1 500 \r\n")
    payload = json.loads(body)
    assert payload["code"] == "INTERNAL_ERROR"
    assert payload["correlation_id"].encode("ascii") in head
    assert b"private-render-fault" not in body
    assert session.close_count == 1


def test_close_fault_cannot_mask_primary_process_control() -> None:
    from poker_knight_ng_service.session import handle_one_session

    class Stop(BaseException):
        pass

    stop = Stop()

    class DoubleFaultSession(MemorySession):
        async def read(self, limit: int) -> bytes:
            raise stop

        def close(self) -> None:
            self.close_count += 1
            raise RuntimeError("close-fault")

    session = DoubleFaultSession()

    with pytest.raises(Stop) as caught:
        asyncio.run(handle_one_session(session))

    assert caught.value is stop
    assert session.sent == []
    assert session.close_count == 1


def test_health_remains_available_while_solve_admission_is_busy() -> None:
    from poker_knight_ng_service.admission import admit_solve
    from poker_knight_ng_service.responses import serialize_health_response
    from poker_knight_ng_service.session import handle_one_session

    held = admit_solve()
    session = MemorySession(b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n")
    try:
        asyncio.run(handle_one_session(session))
    finally:
        held.release()

    assert session.sent == [serialize_health_response()]
    assert session.close_count == 1


@pytest.mark.parametrize(
    ("code", "expected_status"),
    [
        ("UNSUPPORTED_REQUEST", 400),
        ("RNG_REJECTION_EXHAUSTED", 422),
        ("RESOURCE_EXHAUSTED", 503),
        ("INTERNAL_ERROR", 500),
    ],
)
def test_execution_problem_status_mapping_is_closed(monkeypatch, code: str, expected_status: int) -> None:
    import poker_knight_ng_service.session as coordinator

    async def fail_execution(adapted):
        raise coordinator.problem(code)

    monkeypatch.setattr(coordinator, "execute_solve_async", fail_execution)
    session = MemorySession(cpu_request())

    asyncio.run(coordinator.handle_one_session(session))

    head, body = session.sent[0].split(b"\r\n\r\n", 1)
    assert head.startswith(f"HTTP/1.1 {expected_status} \r\n".encode("ascii"))
    payload = json.loads(body)
    assert payload["code"] == code
    assert payload["status"] == expected_status
    assert payload["correlation_id"].encode("ascii") in head
    assert session.close_count == 1


def test_pipelined_second_request_is_transport_failure_before_solve(monkeypatch) -> None:
    import poker_knight_ng_service.session as coordinator

    async def forbidden_execute(adapted):
        pytest.fail("pipelined input reached solve execution")

    monkeypatch.setattr(coordinator, "execute_solve_async", forbidden_execute)
    session = MemorySession(
        cpu_request() + b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"
    )

    asyncio.run(coordinator.handle_one_session(session))

    assert len(session.sent) == 1
    assert session.sent[0].startswith(b"HTTP/1.1 400 \r\n")
    assert b"X-Poker-Knight-Request-ID:" not in session.sent[0]
    assert session.close_count == 1


def test_same_session_has_exactly_one_concurrent_owner(monkeypatch) -> None:
    import poker_knight_ng_service.session as coordinator

    raw = cpu_request()

    class ReplaySession(MemorySession):
        async def read(self, limit: int) -> bytes:
            assert len(raw) <= limit
            return raw

        def read_buffered(self, limit: int) -> bytes:
            return b""

    executions = 0
    entered = asyncio.Event()
    finish = asyncio.Event()

    async def execute_once(adapted):
        nonlocal executions
        executions += 1
        entered.set()
        await finish.wait()
        return {"owner": "first"}

    monkeypatch.setattr(coordinator, "execute_solve_async", execute_once)
    session = ReplaySession()

    async def scenario() -> tuple[object, object]:
        first = asyncio.create_task(coordinator.handle_one_session(session))
        await entered.wait()
        second = asyncio.create_task(coordinator.handle_one_session(session))
        # Deterministically wait for the second session to run its ownership
        # claim (and fail) before releasing the first session's finish gate,
        # instead of a scheduler-dependent await asyncio.sleep(0).
        with pytest.raises(RuntimeError, match="^session is already owned$"):
            await asyncio.wait_for(asyncio.shield(second), timeout=_WORKER_WAIT_TIMEOUT)
        finish.set()
        return await asyncio.gather(first, second, return_exceptions=True)

    results = asyncio.run(scenario())

    assert executions == 1
    assert len(session.sent) == 1
    assert session.sent[0].startswith(b"HTTP/1.1 200 \r\n")
    assert session.close_count == 1
    failures = [result for result in results if isinstance(result, BaseException)]
    assert len(failures) == 1
    assert type(failures[0]) is RuntimeError
    assert str(failures[0]) == "session is already owned"


def test_session_ownership_is_one_shot_and_survives_module_reload() -> None:
    import importlib

    import poker_knight_ng_service.session as coordinator

    session = MemorySession(b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n")
    asyncio.run(coordinator.handle_one_session(session))

    reloaded = importlib.reload(coordinator)
    with pytest.raises(RuntimeError, match="^session is already owned$"):
        asyncio.run(reloaded.handle_one_session(session))

    assert len(session.sent) == 1
    assert session.close_count == 1


def test_non_weak_reference_session_is_rejected_before_io_or_close() -> None:
    from poker_knight_ng_service.session import handle_one_session

    class SlottedSession:
        __slots__ = ("close_count", "read_count", "sent")

        def __init__(self) -> None:
            self.close_count = 0
            self.read_count = 0
            self.sent: list[bytes] = []

        async def read(self, limit: int) -> bytes:
            self.read_count += 1
            return b"GET /healthz HTTP/1.1\r\nHost: local\r\n\r\n"

        def read_buffered(self, limit: int) -> bytes:
            return b""

        async def send_response(self, response: bytes) -> bool:
            self.sent.append(response)
            return True

        def close(self) -> None:
            self.close_count += 1

    session = SlottedSession()

    with pytest.raises(TypeError, match="^sessions must support weak references$"):
        asyncio.run(handle_one_session(session))

    assert session.read_count == 0
    assert session.sent == []
    assert session.close_count == 0
