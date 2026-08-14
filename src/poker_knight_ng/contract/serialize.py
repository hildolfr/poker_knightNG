"""Request-bound conversion from normalized v1 results to the public wire."""

from .errors import ContractProblem, problem
from .models import EquityRequest, EquityResult


def serialize_equity_result(
    result: object,
    request: object,
) -> dict[str, object]:
    """Return the exact v1 result wire after fixed-authority revalidation."""
    try:
        if type(result) is not EquityResult or type(request) is not EquityRequest:
            raise problem("INTERNAL_ERROR")
        EquityRequest.validate(request)

        rng = object.__getattribute__(result, "rng")
        tie_bins = object.__getattribute__(result, "tie_by_other_winners")
        categories = object.__getattribute__(result, "hero_category_counts")
        probabilities = object.__getattribute__(result, "probabilities")
        provenance = object.__getattribute__(result, "provenance")

        raw: dict[str, object] = {
            "contract_version": object.__getattribute__(result, "contract_version"),
            "backend": object.__getattribute__(result, "backend"),
            "rng": {
                "algorithm_id": rng[0],
                "algorithm_version": rng[1],
            },
            "case_hash": object.__getattribute__(result, "case_hash"),
            "seed": f"0x{object.__getattribute__(result, 'seed'):016x}",
            "requested_trials": str(
                object.__getattribute__(result, "requested_trials")
            ),
            "completed_trials": str(
                object.__getattribute__(result, "completed_trials")
            ),
            "unique_wins": str(object.__getattribute__(result, "unique_wins")),
            "ties": str(object.__getattribute__(result, "ties")),
            "tie_by_other_winners": {
                str(index + 1): str(value)
                for index, value in enumerate(tie_bins)
            },
            "losses": str(object.__getattribute__(result, "losses")),
            "equity_share_units": str(
                object.__getattribute__(result, "equity_share_units")
            ),
            "hero_category_counts": {
                name: str(value) for name, value in categories
            },
            "probabilities": {
                name: {
                    "numerator": str(numerator),
                    "denominator": str(denominator),
                }
                for name, numerator, denominator in probabilities
            },
            "timing": {
                "total_duration_ns": str(
                    object.__getattribute__(result, "timing")
                )
            },
            "provenance": {
                "engine_build_id": provenance[0],
                "backend_qualification": provenance[1],
                "device_id": provenance[2],
                "kernel_id": provenance[3],
            },
        }
        EquityResult.parse(raw, request=request)
        return raw
    except ContractProblem as exc:
        raise problem("INTERNAL_ERROR") from exc
    except Exception as exc:
        raise problem("INTERNAL_ERROR") from exc
