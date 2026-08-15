"""Cancellation-safe no-listener async execution handoff tests."""
from __future__ import annotations

import asyncio

import pytest


def _adapted():
    from poker_knight_ng_service.adapter import adapt_solve_request
    from poker_knight_ng_service.framing import AdmittedRequest

    body = (
        b'{"backend":"cpu_reference","board_cards":[],"contract_version":"v1",'
        b'"hero_cards":["As","Kd"],"opponent_count":"1","requested_trials":"1",'
        b'"rng":{"algorithm_id":"poker-knight-ng/philox4x32-10",'
        b'"algorithm_version":"1"},"seed":"0x0000000000000001"}'
    )
    admitted = AdmittedRequest(
        method=b"POST",
        target=b"/v1/solve",
        headers=((b"content-type", b"application/json"),),
        body=body,
    )
    return adapt_solve_request(admitted)


def test_busy_rejection_precedes_worker_scheduling(monkeypatch) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    scheduled = False

    class ForbiddenThread:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal scheduled
            scheduled = True
            pytest.fail("busy solve constructed a worker thread")

    monkeypatch.setattr(async_execution, "Thread", ForbiddenThread)
    held = admit_solve()
    try:
        with pytest.raises(ContractProblem) as caught:
            asyncio.run(async_execution.execute_solve_async(_adapted()))
        assert caught.value.code == "RESOURCE_EXHAUSTED"
        assert not scheduled
    finally:
        held.release()


def test_execution_runs_off_loop_with_active_lease(monkeypatch) -> None:
    import threading

    from poker_knight_ng_service import async_execution

    loop_thread = threading.get_ident()
    worker_thread = None

    def execute(adapted, lease):
        nonlocal worker_thread
        lease._assert_active()
        worker_thread = threading.get_ident()
        return {"ok": True}

    monkeypatch.setattr(async_execution, "_execute_admitted", execute)

    assert asyncio.run(async_execution.execute_solve_async(_adapted())) == {"ok": True}
    assert worker_thread is not None
    assert worker_thread != loop_thread


def test_repeated_cancellation_drains_worker_before_release(monkeypatch) -> None:
    import threading

    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    started = threading.Event()
    finish = threading.Event()

    def execute(adapted, lease):
        lease._assert_active()
        started.set()
        assert finish.wait(2)
        lease._assert_active()
        return {"discarded": True}

    monkeypatch.setattr(async_execution, "_execute_admitted", execute)

    async def scenario() -> None:
        task = asyncio.create_task(async_execution.execute_solve_async(_adapted()))
        while not started.is_set():
            await asyncio.sleep(0)

        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()

        with pytest.raises(ContractProblem) as busy:
            await async_execution.execute_solve_async(_adapted())
        assert busy.value.code == "RESOURCE_EXHAUSTED"

        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    lease = admit_solve()
    lease.release()


def test_cancellation_wins_over_ordinary_worker_failure(monkeypatch) -> None:
    import threading

    from poker_knight_ng.contract.errors import problem
    from poker_knight_ng_service import async_execution

    started = threading.Event()
    finish = threading.Event()

    def execute(adapted, lease):
        started.set()
        assert finish.wait(2)
        raise problem("INTERNAL_ERROR")

    monkeypatch.setattr(async_execution, "_execute_admitted", execute)

    async def scenario() -> None:
        task = asyncio.create_task(async_execution.execute_solve_async(_adapted()))
        while not started.is_set():
            await asyncio.sleep(0)
        task.cancel()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def test_immediate_parent_cancel_before_start_creates_no_thread(monkeypatch) -> None:
    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    constructed = False

    class ForbiddenThread:
        def __init__(self, *args, **kwargs) -> None:
            nonlocal constructed
            constructed = True

    monkeypatch.setattr(async_execution, "Thread", ForbiddenThread)

    async def scenario() -> None:
        task = asyncio.create_task(async_execution.execute_solve_async(_adapted()))
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert not constructed
    lease = admit_solve()
    lease.release()


def test_worker_base_exception_propagates_and_releases(monkeypatch) -> None:
    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    class Stop(BaseException):
        pass

    def execute(adapted, lease):
        raise Stop

    monkeypatch.setattr(async_execution, "_execute_admitted", execute)

    async def scenario() -> None:
        with pytest.raises(Stop):
            await async_execution.execute_solve_async(_adapted())

    asyncio.run(scenario())
    lease = admit_solve()
    lease.release()


def test_wait_infrastructure_fault_joins_before_release(monkeypatch) -> None:
    import threading
    import time

    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    completed = threading.Event()

    def execute(adapted, lease):
        time.sleep(0.05)
        lease._assert_active()
        completed.set()
        return {"discarded": True}

    async def failed_sleep(delay):
        raise RuntimeError("loop sleep unavailable")

    monkeypatch.setattr(async_execution, "_execute_admitted", execute)
    monkeypatch.setattr(async_execution.asyncio, "sleep", failed_sleep)

    with pytest.raises(ContractProblem) as caught:
        asyncio.run(async_execution.execute_solve_async(_adapted()))
    assert caught.value.code == "INTERNAL_ERROR"
    assert completed.is_set()

    lease = admit_solve()
    lease.release()


def test_is_alive_fault_joins_before_release(monkeypatch) -> None:
    import threading
    import time

    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    completed = threading.Event()

    def execute(adapted, lease):
        time.sleep(0.05)
        lease._assert_active()
        completed.set()
        return {"discarded": True}

    class FailedAliveThread(threading.Thread):
        def is_alive(self) -> bool:
            raise RuntimeError("is_alive unavailable")

    monkeypatch.setattr(async_execution, "_execute_admitted", execute)
    monkeypatch.setattr(async_execution, "Thread", FailedAliveThread)

    with pytest.raises(ContractProblem) as caught:
        asyncio.run(async_execution.execute_solve_async(_adapted()))
    assert caught.value.code == "INTERNAL_ERROR"
    assert completed.is_set()

    lease = admit_solve()
    lease.release()


def test_parent_join_and_liveness_faults_cannot_release_worker_lease(monkeypatch) -> None:
    import threading
    import time

    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    completed = threading.Event()

    def execute(adapted, lease):
        time.sleep(0.05)
        lease._assert_active()
        completed.set()
        return {"discarded": True}

    class FailedParentSyncThread(threading.Thread):
        def is_alive(self) -> bool:
            raise RuntimeError("is_alive unavailable")

        def join(self, timeout=None) -> None:
            raise RuntimeError("join unavailable")

    monkeypatch.setattr(async_execution, "_execute_admitted", execute)
    monkeypatch.setattr(async_execution, "Thread", FailedParentSyncThread)

    with pytest.raises(ContractProblem) as caught:
        asyncio.run(async_execution.execute_solve_async(_adapted()))
    assert caught.value.code == "INTERNAL_ERROR"
    assert completed.is_set()

    lease = admit_solve()
    lease.release()


def test_wait_process_control_joins_before_propagation(monkeypatch) -> None:
    import threading
    import time

    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    class Stop(BaseException):
        pass

    completed = threading.Event()

    def execute(adapted, lease):
        time.sleep(0.05)
        lease._assert_active()
        completed.set()
        return {"discarded": True}

    async def interrupted_sleep(delay):
        raise Stop

    monkeypatch.setattr(async_execution, "_execute_admitted", execute)
    monkeypatch.setattr(async_execution.asyncio, "sleep", interrupted_sleep)

    with pytest.raises(Stop):
        asyncio.run(async_execution.execute_solve_async(_adapted()))
    assert completed.is_set()

    lease = admit_solve()
    lease.release()


def test_start_failure_after_dispatch_joins_before_release(monkeypatch) -> None:
    import threading
    import time

    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    completed = threading.Event()

    def execute(adapted, lease):
        time.sleep(0.05)
        lease._assert_active()
        completed.set()
        return {"discarded": True}

    class DispatchThenFailThread(threading.Thread):
        def start(self) -> None:
            super().start()
            raise RuntimeError("start failed after dispatch")

    monkeypatch.setattr(async_execution, "_execute_admitted", execute)
    monkeypatch.setattr(async_execution, "Thread", DispatchThenFailThread)

    with pytest.raises(ContractProblem) as caught:
        asyncio.run(async_execution.execute_solve_async(_adapted()))
    assert caught.value.code == "INTERNAL_ERROR"
    assert completed.is_set()

    lease = admit_solve()
    lease.release()


def test_worker_start_fault_maps_internal_and_releases(monkeypatch) -> None:
    import threading

    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    class FailedStartThread(threading.Thread):
        def start(self) -> None:
            raise RuntimeError("thread start unavailable")

    monkeypatch.setattr(async_execution, "Thread", FailedStartThread)

    with pytest.raises(ContractProblem) as caught:
        asyncio.run(async_execution.execute_solve_async(_adapted()))
    assert caught.value.code == "INTERNAL_ERROR"

    lease = admit_solve()
    lease.release()


def test_synchronous_worker_factory_fault_releases_admission(monkeypatch) -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service import async_execution
    from poker_knight_ng_service.admission import admit_solve

    class FailedThread:
        def __init__(self, *args, **kwargs) -> None:
            raise RuntimeError("thread factory unavailable")

    monkeypatch.setattr(async_execution, "Thread", FailedThread)

    with pytest.raises(ContractProblem) as caught:
        asyncio.run(async_execution.execute_solve_async(_adapted()))
    assert caught.value.code == "INTERNAL_ERROR"

    lease = admit_solve()
    lease.release()
