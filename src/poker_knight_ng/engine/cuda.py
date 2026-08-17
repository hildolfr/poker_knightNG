"""Explicit, lazy CUDA engine; it is never the default solve route."""
from time import monotonic_ns
from typing import Any, Callable

from .._cuda_runtime import (
    CupyDeterministicRuntime,
    CudaBackendUnavailable,
    CudaResourceExhausted,
    CudaRngExhausted,
)
from ..contract import EquityRequest, EquityResult, canonical_case_hash
from ..contract.canonical import card_id
from ..contract.errors import ContractProblem, problem
from ..reference.rng import derive_philox_key
from .result import to_equity_result


def _clock(value: object) -> int:
    if type(value) is not int or not 0 <= value <= (1 << 64) - 1:
        raise ValueError("invalid clock")
    return value


_FAILURE_CODE_BY_NAME = {
    "CudaBackendUnavailable": "BACKEND_UNAVAILABLE",
    "CudaResourceExhausted": "RESOURCE_EXHAUSTED",
    "CudaRngExhausted": "RNG_REJECTION_EXHAUSTED",
}


def _runtime_failure_code(exc: BaseException) -> str | None:
    """Map a runtime exception to its public contract code.

    Same-generation instances match by isinstance; exceptions whose class was
    replaced by an importlib.reload or a sys.modules pop+reimport match by
    class name, guarded by the owning module so a foreign exception that
    merely shares a name is never misclassified.
    """
    if isinstance(exc, (CudaBackendUnavailable, CudaResourceExhausted, CudaRngExhausted)):
        return _FAILURE_CODE_BY_NAME[type(exc).__name__]
    if type(exc).__module__ == "poker_knight_ng._cuda_runtime":
        names = {cls.__name__ for cls in type(exc).__mro__}
        for name in names:
            if name in _FAILURE_CODE_BY_NAME:
                return _FAILURE_CODE_BY_NAME[name]
    return None


class CUDAEngine:
    """Run only explicit ``backend='cuda'`` requests against a qualified runtime."""
    def __init__(self, *, runtime: Any = None, clock_ns: Callable[[], object] = monotonic_ns) -> None:
        if runtime is None:
            runtime = CupyDeterministicRuntime()
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self._runtime, self._clock_ns = runtime, clock_ns

    def solve(self, request: EquityRequest) -> EquityResult:
        if type(request) is not EquityRequest:
            raise problem("INTERNAL_ERROR")
        try:
            start = _clock(self._clock_ns())
        except Exception as exc:
            raise problem("INTERNAL_ERROR") from exc
        try:
            EquityRequest.validate(request)
            if object.__getattribute__(request, "backend") != "cuda":
                raise problem("UNSUPPORTED_REQUEST")
            hero = tuple(sorted(card_id(c) for c in object.__getattribute__(request, "hero_cards")))
            board = tuple(sorted(card_id(c) for c in object.__getattribute__(request, "board_cards")))
            opponents = object.__getattribute__(request, "opponent_count")
            count = object.__getattribute__(request, "requested_trials")
            seed = object.__getattribute__(request, "seed")
        except ContractProblem:
            raise
        except Exception as exc:
            raise problem("INTERNAL_ERROR") from exc
        try:
            case_hash = bytes.fromhex(canonical_case_hash(request))
            _digest, key = derive_philox_key(seed, case_hash)
            aggregate = self._runtime.run(hero=hero, board=board, opponents=opponents, key=key, first_simulation_id=0, count=count)
            device, kernel = self._runtime.provenance()
            provenance = ("cuda-deterministic-v1", device, kernel)
            to_equity_result(aggregate, request, 0, provenance=provenance)
            end = _clock(self._clock_ns())
            if end < start:
                raise ValueError("invalid clock")
            return to_equity_result(aggregate, request, end - start, provenance=provenance)
        except Exception as exc:
            code = _runtime_failure_code(exc)
            if code is not None:
                raise problem(code) from exc
            raise problem("INTERNAL_ERROR") from exc
