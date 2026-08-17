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
            # Keep these imported exception classes stable for this engine module's
            # lifetime so a runtime retained across importlib.reload still maps to
            # its public contract error rather than INTERNAL_ERROR.
            if isinstance(exc, CudaBackendUnavailable):
                raise problem("BACKEND_UNAVAILABLE") from exc
            if isinstance(exc, CudaResourceExhausted):
                raise problem("RESOURCE_EXHAUSTED") from exc
            if isinstance(exc, CudaRngExhausted):
                raise problem("RNG_REJECTION_EXHAUSTED") from exc
            raise problem("INTERNAL_ERROR") from exc
