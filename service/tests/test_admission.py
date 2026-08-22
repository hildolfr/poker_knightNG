"""Atomic zero-queue solve-admission tests."""
from __future__ import annotations

from queue import Queue
from threading import Barrier, Event, Thread

import pytest

# Bounded test-local handshake timeout (seconds). Cross-thread event waits use a
# named, generous bound (instead of a magic raw wait(2)) so a slow CI scheduler
# never spuriously trips the winner's release, while a genuinely stuck thread
# still fails loudly.
_THREAD_HANDSHAKE_TIMEOUT = 30.0


def test_second_solve_is_rejected_while_global_lease_is_held() -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.admission import admit_solve

    lease = admit_solve()
    try:
        try:
            admit_solve()
        except ContractProblem as failure:
            assert failure.code == "RESOURCE_EXHAUSTED"
        else:
            raise AssertionError("second solve was admitted")
    finally:
        lease.release()


def test_release_allows_next_solve_to_acquire() -> None:
    from poker_knight_ng_service.admission import admit_solve

    first = admit_solve()
    first.release()
    second = admit_solve()
    second.release()


def test_no_constructible_shadow_gate_or_direct_lease_exists() -> None:
    import poker_knight_ng_service.admission as admission

    lease = admission.admit_solve()
    try:
        assert not hasattr(admission, "_SolveAdmissionGate")
        with pytest.raises(TypeError):
            admission.SolveLease()
    finally:
        lease.release()


def test_module_reload_preserves_held_process_global_token() -> None:
    import importlib
    import poker_knight_ng_service.admission as admission
    from poker_knight_ng.contract.errors import ContractProblem

    lease = admission.admit_solve()
    reloaded = importlib.reload(admission)
    try:
        with pytest.raises(ContractProblem) as caught:
            reloaded.admit_solve()
        assert caught.value.code == "RESOURCE_EXHAUSTED"
    finally:
        lease.release()

    next_lease = reloaded.admit_solve()
    next_lease.release()


def test_double_release_is_internal_error() -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.admission import admit_solve

    lease = admit_solve()
    lease.release()

    with pytest.raises(ContractProblem) as caught:
        lease.release()

    assert caught.value.code == "INTERNAL_ERROR"


def test_forged_lease_release_is_internal_error() -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.admission import SolveLease

    forged = object.__new__(SolveLease)

    with pytest.raises(ContractProblem) as caught:
        forged.release()

    assert caught.value.code == "INTERNAL_ERROR"
    assert caught.value.__cause__ is None


def test_context_releases_without_swallowing_base_exception() -> None:
    from poker_knight_ng_service.admission import admit_solve

    class StopNow(BaseException):
        pass

    with pytest.raises(StopNow):
        with admit_solve():
            raise StopNow

    lease = admit_solve()
    lease.release()


def test_concurrent_double_release_has_one_winner_and_remains_reusable() -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.admission import admit_solve

    lease = admit_solve()
    start = Barrier(2)
    outcomes: Queue[str] = Queue()

    def release() -> None:
        start.wait()
        try:
            lease.release()
        except ContractProblem as failure:
            outcomes.put(failure.code)
        else:
            outcomes.put("RELEASED")

    threads = [Thread(target=release) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(1)
        assert not thread.is_alive()

    released = [outcomes.get_nowait(), outcomes.get_nowait()]
    assert sorted(released) == ["INTERNAL_ERROR", "RELEASED"]

    next_lease = admit_solve()
    next_lease.release()


def test_concurrent_attempts_admit_exactly_one_without_queue() -> None:
    from poker_knight_ng.contract.errors import ContractProblem
    from poker_knight_ng_service.admission import admit_solve

    count = 16
    start = Barrier(count)
    winner_ready = Event()
    release_winner = Event()
    outcomes: Queue[str] = Queue()

    def attempt() -> None:
        start.wait()
        try:
            lease = admit_solve()
        except ContractProblem as failure:
            outcomes.put(failure.code)
            return
        outcomes.put("ADMITTED")
        winner_ready.set()
        if not release_winner.wait(_THREAD_HANDSHAKE_TIMEOUT):
            outcomes.put("WINNER_TIMEOUT")
        lease.release()

    threads = [Thread(target=attempt) for _ in range(count)]
    for thread in threads:
        thread.start()
    assert winner_ready.wait(1)

    # Every first attempt must return while the winner still holds the token.
    initial = [outcomes.get(timeout=1) for _ in range(count)]
    assert initial.count("ADMITTED") == 1
    assert initial.count("RESOURCE_EXHAUSTED") == count - 1
    assert outcomes.empty()

    release_winner.set()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()

    assert outcomes.empty()

    lease = admit_solve()
    lease.release()
