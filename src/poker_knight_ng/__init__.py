"""Poker Knight NG public contract and explicit local engine boundary.

Importing this module deliberately does not import CuPy or compile CUDA sources.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("poker-knight-ng")
except PackageNotFoundError:  # source-tree import before installation
    __version__ = "0+unknown"

from .contract import (
    ContractProblem,
    EquityRequest,
    EquityResult,
    canonical_case_bytes,
    canonical_case_hash,
    serialize_equity_result,
)
from .engine import CPUReferenceEngine, CUDAEngine, Engine, solve, solve_cuda

__all__ = [
    "CPUReferenceEngine",
    "CUDAEngine",
    "ContractProblem",
    "Engine",
    "EquityRequest",
    "EquityResult",
    "canonical_case_bytes",
    "canonical_case_hash",
    "serialize_equity_result",
    "solve",
    "solve_cuda",
    "__version__",
]
