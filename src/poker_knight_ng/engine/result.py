"""Closed conversion from the CPU reference aggregate to the v1 result contract."""
from typing import Callable

from ..contract import EquityRequest, EquityResult, canonical_case_hash
from ..contract.errors import ContractProblem, problem
from ..contract.models import CATEGORIES
from ..reference.monte_carlo import MonteCarloResult

ENGINE_BUILD_ID = "poker-knight-ng-0.1.0"
BACKEND_QUALIFICATION = "cpu-reference-v1"
UINT64_MAX = (1 << 64) - 1


def _duration(value: object) -> int:
    if isinstance(value, bool) or type(value) is not int or not 0 <= value <= UINT64_MAX:
        raise problem("INTERNAL_ERROR")
    return value


def _exact_aggregate_fields(result: MonteCarloResult) -> None:
    """Reject forged authoritative values before the reference invariants run."""
    scalar_names = ("completed_trials", "unique_wins", "losses", "equity_share_units")
    scalars = tuple(object.__getattribute__(result, name) for name in scalar_names)
    rejection_count = object.__getattribute__(result, "rejection_count")
    bins = object.__getattribute__(result, "tie_by_other_winners")
    categories = object.__getattribute__(result, "hero_category_counts")
    if any(type(value) is not int or not 0 <= value <= UINT64_MAX for value in scalars):
        raise ValueError("aggregate scalar is not an exact uint64")
    if type(rejection_count) is not int or rejection_count < 0:
        raise ValueError("aggregate rejection count is not an exact nonnegative integer")
    if type(bins) is not tuple or len(bins) != 6 or type(categories) is not tuple or len(categories) != 9:
        raise ValueError("aggregate bins have invalid exact shape")
    if any(type(value) is not int or not 0 <= value <= UINT64_MAX for value in bins + categories):
        raise ValueError("aggregate bin is not an exact uint64")


def to_equity_result(result: object, request: object, duration_ns: object) -> EquityResult:
    """Internal boundary: expose only INTERNAL_ERROR for malformed internal input."""
    try:
        if type(request) is not EquityRequest:
            raise problem("INTERNAL_ERROR")
        # Fixed authority prevents an exact frozen request instance from
        # replacing validate in its __dict__ at this internal boundary.
        EquityRequest.validate(request)
        opponent_count = object.__getattribute__(request, "opponent_count")
        requested_trials = object.__getattribute__(request, "requested_trials")
        board_cards = object.__getattribute__(request, "board_cards")
        backend = object.__getattribute__(request, "backend")
        seed = object.__getattribute__(request, "seed")
        if type(result) is not MonteCarloResult:
            raise problem("INTERNAL_ERROR")
        duration = _duration(duration_ns)
        _exact_aggregate_fields(result)
        # Invoke the fixed class authority: an exact frozen instance can still
        # have an instance-level ``validate`` attribute forged onto it.  The
        # class implementation reads and validates all authoritative fields
        # through ``_raw_monte_carlo`` without virtual dispatch.
        MonteCarloResult.validate(result, opponent_count, requested_trials, len(board_cards))
        completed = object.__getattribute__(result, "completed_trials")
        wins = object.__getattribute__(result, "unique_wins")
        bins = object.__getattribute__(result, "tie_by_other_winners")
        losses = object.__getattribute__(result, "losses")
        units = object.__getattribute__(result, "equity_share_units")
        categories = object.__getattribute__(result, "hero_category_counts")
        ties = sum(bins)
        raw = {
            "contract_version": "v1", "backend": backend,
            "rng": {"algorithm_id": "poker-knight-ng/philox4x32-10", "algorithm_version": "1"},
            "case_hash": canonical_case_hash(request), "seed": f"0x{seed:016x}",
            "requested_trials": str(requested_trials), "completed_trials": str(completed),
            "unique_wins": str(wins), "ties": str(ties),
            "tie_by_other_winners": {str(i + 1): str(value) for i, value in enumerate(bins)},
            "losses": str(losses), "equity_share_units": str(units),
            "hero_category_counts": {name: str(categories[i]) for i, name in enumerate(CATEGORIES)},
            "probabilities": {
                "unique_win": {"numerator": str(wins), "denominator": str(completed)},
                "tie": {"numerator": str(ties), "denominator": str(completed)},
                "loss": {"numerator": str(losses), "denominator": str(completed)},
                "showdown_equity": {"numerator": str(units), "denominator": str(420 * completed)},
            },
            "timing": {"total_duration_ns": str(duration)},
            "provenance": {"engine_build_id": ENGINE_BUILD_ID, "backend_qualification": BACKEND_QUALIFICATION, "device_id": None, "kernel_id": None},
        }
        return EquityResult.parse(raw, request=request)
    except ContractProblem as exc:
        raise problem("INTERNAL_ERROR") from exc
    except Exception as exc:
        raise problem("INTERNAL_ERROR") from exc
