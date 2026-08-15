"""Cancellation-safe async handoff for admitted synchronous execution."""
from __future__ import annotations

import asyncio
from threading import Event, Thread

from poker_knight_ng.contract.errors import ContractProblem, problem

from .adapter import AdaptedSolveRequest
from .admission import SolveLease, admit_solve
from .execution import _execute_admitted, _trusted_snapshot

def _worker(
    adapted: AdaptedSolveRequest,
    lease: SolveLease,
    outcome: list[tuple[bool, object]],
    done: Event,
) -> None:
    try:
        try:
            value = _execute_admitted(adapted, lease)
        except BaseException as exc:
            outcome.append((False, exc))
        else:
            outcome.append((True, value))
    finally:
        try:
            lease.release()
        finally:
            done.set()


def _record_failure(current: BaseException | None, new: BaseException) -> BaseException:
    return new if current is None else current


def _blocking_drain(worker: Thread, done: Event) -> BaseException | None:
    """Wait synchronously until the worker has released admission and exited."""

    failure: BaseException | None = None
    while True:
        try:
            if done.is_set():
                break
        except BaseException as exc:
            failure = _record_failure(failure, exc)
        try:
            worker.join(0.01)
        except BaseException as exc:
            failure = _record_failure(failure, exc)
        try:
            if not worker.is_alive():
                break
        except BaseException as exc:
            failure = _record_failure(failure, exc)
    try:
        worker.join()
    except BaseException as exc:
        failure = _record_failure(failure, exc)
    return failure


async def _wait(
    worker: Thread,
    done: Event,
) -> tuple[asyncio.CancelledError | None, BaseException | None]:
    cancellation: asyncio.CancelledError | None = None
    current = asyncio.current_task()
    while True:
        try:
            if done.is_set() or not worker.is_alive():
                return cancellation, _blocking_drain(worker, done)
        except BaseException as exc:
            drain_failure = _blocking_drain(worker, done)
            return cancellation, drain_failure or exc
        try:
            await asyncio.sleep(0.01)
        except asyncio.CancelledError as exc:
            if cancellation is None:
                cancellation = exc
            if current is not None:
                current.uncancel()
        except BaseException as exc:
            drain_failure = _blocking_drain(worker, done)
            return cancellation, drain_failure or exc


def _process_control(outcome: list[tuple[bool, object]]) -> BaseException | None:
    if len(outcome) == 1:
        success, value = outcome[0]
        if not success and isinstance(value, BaseException) and not isinstance(value, Exception):
            return value
    return None


def _resolve(
    outcome: list[tuple[bool, object]],
    cancellation: asyncio.CancelledError | None,
    wait_failure: BaseException | None,
) -> dict[str, object]:
    process_control = _process_control(outcome)
    if process_control is not None:
        raise process_control
    if wait_failure is not None:
        if isinstance(wait_failure, Exception):
            raise problem("INTERNAL_ERROR") from wait_failure
        raise wait_failure
    if cancellation is not None:
        raise cancellation
    if len(outcome) != 1:
        raise problem("INTERNAL_ERROR")
    success, value = outcome[0]
    if success:
        if type(value) is not dict:
            raise problem("INTERNAL_ERROR")
        return value
    if isinstance(value, ContractProblem):
        raise value  # pyright: ignore[reportGeneralTypeIssues]
    if isinstance(value, Exception):
        raise problem("INTERNAL_ERROR") from value
    raise problem("INTERNAL_ERROR")


async def execute_solve_async(adapted: AdaptedSolveRequest) -> dict[str, object]:
    """Run one admitted solve off-loop and never abandon a started worker."""

    _trusted_snapshot(adapted)
    try:
        done = Event()
        outcome: list[tuple[bool, object]] = []
    except Exception as exc:
        raise problem("INTERNAL_ERROR") from exc
    lease = admit_solve()
    try:
        worker = Thread(
            target=_worker,
            args=(adapted, lease, outcome, done),
            name="poker-knight-ng-solve",
            daemon=False,
        )
    except BaseException as exc:
        lease.release()
        if isinstance(exc, Exception):
            raise problem("INTERNAL_ERROR") from exc
        raise
    try:
        worker.start()
    except BaseException as exc:
        if worker.ident is None:
            lease.release()
        else:
            drain_failure = _blocking_drain(worker, done)
            process_control = _process_control(outcome)
            if process_control is not None:
                raise process_control
            if drain_failure is not None and not isinstance(drain_failure, Exception):
                raise drain_failure
        if isinstance(exc, Exception):
            raise problem("INTERNAL_ERROR") from exc
        raise
    cancellation, wait_failure = await _wait(worker, done)
    return _resolve(outcome, cancellation, wait_failure)
