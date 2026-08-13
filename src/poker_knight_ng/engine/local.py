"""Synchronous production CPU-reference engine."""
from time import monotonic_ns
from typing import Callable

from ..contract import EquityRequest, EquityResult, canonical_case_hash
from ..contract.errors import ContractProblem, problem
from ..contract.canonical import card_id
from ..reference.dealer import RngRejectionExhausted
from ..reference.monte_carlo import run_cpu_monte_carlo
from .result import to_equity_result


class CPUReferenceEngine:
    """Execute exactly the contract's CPU reference backend.

    Successful timing includes request/backend validation, canonical preparation,
    reference execution, aggregate validation, and a complete zero-duration
    adapter/contract-validation preflight.  It ends immediately before the final
    immutable result materialization that embeds the measured duration.

    Clock failures that are ``Exception`` subclasses map to INTERNAL_ERROR;
    process-control ``BaseException`` signals deliberately propagate unchanged.
    """

    def __init__(self, *, clock_ns: Callable[[], object] = monotonic_ns) -> None:
        if not callable(clock_ns):
            raise TypeError("clock_ns must be callable")
        self._clock_ns = clock_ns

    def solve(self, request: EquityRequest) -> EquityResult:
        if type(request) is not EquityRequest:
            raise problem("INTERNAL_ERROR")
        try:
            start = self._clock_ns()
            if isinstance(start, bool) or type(start) is not int or not 0 <= start <= (1 << 64) - 1:
                raise ValueError("invalid clock")
        except Exception as exc:
            raise problem("INTERNAL_ERROR") from exc
        try:
            # This is the public input boundary: its ContractProblem codes are preserved.
            # An exact frozen instance can nevertheless carry a forged method in
            # its __dict__, so neither authority may be reached virtually.
            EquityRequest.validate(request)
            backend = object.__getattribute__(request, "backend")
            if backend == "cuda":
                raise problem("BACKEND_UNAVAILABLE")
            # Class validation admits only cpu_reference or cuda.  Keep this
            # defensive branch coupled to that authority rather than trusting
            # an instance-level attribute lookup for execution routing.
            if backend != "cpu_reference":
                raise problem("UNSUPPORTED_REQUEST")
            hero_cards = object.__getattribute__(request, "hero_cards")
            board_cards = object.__getattribute__(request, "board_cards")
            opponent_count = object.__getattribute__(request, "opponent_count")
            requested_trials = object.__getattribute__(request, "requested_trials")
            seed = object.__getattribute__(request, "seed")
        except ContractProblem:
            raise
        except Exception as exc:
            raise problem("INTERNAL_ERROR") from exc
        try:
            aggregate = run_cpu_monte_carlo(
                seed=seed,
                hero_card_ids=tuple(card_id(card) for card in hero_cards),
                board_card_ids=tuple(card_id(card) for card in board_cards),
                opponent_count=opponent_count,
                requested_trials=requested_trials,
                replay_case_hash=bytes.fromhex(canonical_case_hash(request)),
            )
            # Validate the complete closed wire/contract before ending the interval.
            to_equity_result(aggregate, request, 0)
            end = self._clock_ns()
            if isinstance(end, bool) or type(end) is not int or not 0 <= end <= (1 << 64) - 1 or end < start:
                raise ValueError("invalid clock")
            return to_equity_result(aggregate, request, end - start)
        except RngRejectionExhausted as exc:
            raise problem("RNG_REJECTION_EXHAUSTED") from exc
        except Exception as exc:
            raise problem("INTERNAL_ERROR") from exc
