"""Atomic process-global one-solve admission with no queue."""
from __future__ import annotations

from collections.abc import Callable
from threading import Lock

from poker_knight_ng.contract.errors import problem


class SolveLease:
    """One sealed lease for the process-global solve token."""

    __slots__ = ()

    def __new__(cls) -> SolveLease:
        raise TypeError("solve leases are created only by admit_solve")

    def release(self) -> None:
        """Release this lease exactly once."""

        _release_solve(self)

    def _assert_active(self) -> None:
        """Fail closed unless this exact lease still owns admission."""

        _assert_active_solve(self)

    def __enter__(self) -> SolveLease:
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> bool:
        self.release()
        return False


def _build_admission_api():
    lock = Lock()
    active: SolveLease | None = None

    def acquire() -> SolveLease:
        nonlocal active
        with lock:
            if active is not None:
                raise problem("RESOURCE_EXHAUSTED")
            lease = object.__new__(SolveLease)
            active = lease
            return lease

    def release(lease: SolveLease) -> None:
        nonlocal active
        with lock:
            if active is not lease:
                raise problem("INTERNAL_ERROR")
            active = None

    def assert_active(lease: SolveLease) -> None:
        with lock:
            if active is not lease:
                raise problem("INTERNAL_ERROR")

    return acquire, release, assert_active


admit_solve: Callable[[], SolveLease]
_release_solve: Callable[[SolveLease], None]
_assert_active_solve: Callable[[SolveLease], None]

try:
    admit_solve  # pyright: ignore[reportUnboundVariable]
    _release_solve  # pyright: ignore[reportUnboundVariable]
    _assert_active_solve  # pyright: ignore[reportUnboundVariable]
except NameError:
    admit_solve, _release_solve, _assert_active_solve = _build_admission_api()
del _build_admission_api
