"""Public synchronous CPU equity-engine boundary."""
from ..contract import EquityRequest, EquityResult
from ..contract.errors import ContractProblem, problem
from .interface import Engine
from .local import CPUReferenceEngine
from .cuda import CUDAEngine
from .result import to_equity_result

_DEFAULT_ENGINE = CPUReferenceEngine()


def solve(request: EquityRequest) -> EquityResult:
    """Resolve a request through the CPU-reference route only."""
    return _DEFAULT_ENGINE.solve(request)


def solve_cuda(request: EquityRequest) -> EquityResult:
    """Resolve an explicitly CUDA-bound request without fallback."""
    if type(request) is not EquityRequest:
        raise problem("INTERNAL_ERROR")
    try:
        EquityRequest.validate(request)
        backend = object.__getattribute__(request, "backend")
    except ContractProblem:
        raise
    except Exception as exc:
        raise problem("INTERNAL_ERROR") from exc
    if backend != "cuda":
        raise problem("UNSUPPORTED_REQUEST")
    try:
        return CUDAEngine().solve(request)
    except ContractProblem:
        raise
    except Exception as exc:
        raise problem("INTERNAL_ERROR") from exc


__all__ = [
    "CPUReferenceEngine",
    "CUDAEngine",
    "Engine",
    "solve",
    "solve_cuda",
    "to_equity_result",
]
