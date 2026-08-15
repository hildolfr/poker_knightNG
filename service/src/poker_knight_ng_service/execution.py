"""Exact admitted synchronous engine execution without a listener."""
from __future__ import annotations

from poker_knight_ng.contract import (
    ContractProblem,
    EquityRequest,
    serialize_equity_result,
)
from poker_knight_ng.contract.errors import problem
from poker_knight_ng.engine import CPUReferenceEngine, CUDAEngine

from .adapter import AdaptedSolveRequest
from .admission import SolveLease, admit_solve
from .routing import Route

_ENGINE_FAILURE_CODES = frozenset(
    {
        "BACKEND_UNAVAILABLE",
        "RESOURCE_EXHAUSTED",
        "RNG_REJECTION_EXHAUSTED",
        "INTERNAL_ERROR",
    }
)


def _trusted_snapshot(adapted: AdaptedSolveRequest) -> tuple[Route, EquityRequest]:
    try:
        if type(adapted) is not AdaptedSolveRequest:
            raise ValueError("unexpected adapted request type")
        route = object.__getattribute__(adapted, "route")
        request = object.__getattribute__(adapted, "request")
        if type(route) is not Route or type(request) is not EquityRequest:
            raise ValueError("unexpected adapted request fields")
        EquityRequest.validate(request)
        backend = object.__getattribute__(request, "backend")
        if route is Route.CPU_SOLVE:
            if backend != "cpu_reference":
                raise ValueError("CPU route/backend mismatch")
        elif route is Route.CUDA_SOLVE:
            if backend != "cuda":
                raise ValueError("CUDA route/backend mismatch")
        else:
            raise ValueError("unexpected solve route")
        return route, request
    except Exception as exc:
        raise problem("INTERNAL_ERROR") from exc


def _execute_admitted(
    adapted: AdaptedSolveRequest,
    lease: SolveLease,
) -> dict[str, object]:
    """Execute only while the exact supplied lease owns global admission."""

    lease._assert_active()
    route, request = _trusted_snapshot(adapted)
    try:
        engine = CPUReferenceEngine() if route is Route.CPU_SOLVE else CUDAEngine()
        result = engine.solve(request)
    except ContractProblem as exc:
        if exc.code in _ENGINE_FAILURE_CODES:
            raise
        raise problem("INTERNAL_ERROR") from exc
    except Exception as exc:
        raise problem("INTERNAL_ERROR") from exc
    try:
        return serialize_equity_result(result, request)
    except Exception as exc:
        raise problem("INTERNAL_ERROR") from exc


def execute_solve(adapted: AdaptedSolveRequest) -> dict[str, object]:
    """Execute one trusted solve while holding the process-global admission lease."""

    _trusted_snapshot(adapted)
    lease = admit_solve()
    try:
        return _execute_admitted(adapted, lease)
    finally:
        lease.release()
