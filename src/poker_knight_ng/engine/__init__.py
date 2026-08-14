"""Public synchronous CPU equity-engine boundary."""
from ..contract import EquityRequest, EquityResult
from .interface import Engine
from .local import CPUReferenceEngine
from .cuda import CUDAEngine
from .result import to_equity_result

_DEFAULT_ENGINE = CPUReferenceEngine()


def solve(request: EquityRequest) -> EquityResult:
    """Resolve a request using the only currently available public backend."""
    return _DEFAULT_ENGINE.solve(request)


__all__ = ["CPUReferenceEngine", "CUDAEngine", "Engine", "solve", "to_equity_result"]
