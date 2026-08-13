"""Poker Knight NG contract-only public boundary.

Importing this module deliberately does not import CuPy or compile CUDA sources.
"""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("poker-knight-ng")
except PackageNotFoundError:  # source-tree import before installation
    __version__ = "0+unknown"

from .contract import (ContractProblem, EquityRequest, EquityResult, canonical_case_bytes, canonical_case_hash)

__all__ = ["ContractProblem", "EquityRequest", "EquityResult", "canonical_case_bytes", "canonical_case_hash", "__version__"]
