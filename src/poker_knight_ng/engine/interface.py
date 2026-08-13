"""Minimal public equity-engine protocol."""
from typing import Protocol, runtime_checkable

from ..contract import EquityRequest, EquityResult


@runtime_checkable
class Engine(Protocol):
    """A synchronous engine that resolves one validated equity request."""

    def solve(self, request: EquityRequest) -> EquityResult: ...
