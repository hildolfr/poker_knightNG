"""Closed, schema-frozen v1 problem responses."""
from dataclasses import dataclass
import re
from types import MappingProxyType
from typing import Mapping

# These values are intentionally duplicated only from the frozen problem schema.
# Callers select a code, never client-visible wording.
_POLICY_ROWS = (
    ("INVALID_CONTRACT_VERSION", "invalid-contract-version", "Invalid contract version", 400, "The request contract version is not supported.", False),
    ("INVALID_CARD", "invalid-card", "Invalid card", 400, "The request contains an invalid card.", False),
    ("DUPLICATE_CARD", "duplicate-card", "Duplicate card", 400, "The request contains duplicate cards.", False),
    ("INVALID_BOARD_LENGTH", "invalid-board-length", "Invalid board length", 400, "The request has an invalid board length.", False),
    ("INVALID_OPPONENT_COUNT", "invalid-opponent-count", "Invalid opponent count", 400, "The request has an invalid opponent count.", False),
    ("INVALID_TRIAL_COUNT", "invalid-trial-count", "Invalid trial count", 400, "The request has an invalid trial count.", False),
    ("INVALID_SEED", "invalid-seed", "Invalid seed", 400, "The request has an invalid seed.", False),
    ("UNSUPPORTED_FIELD", "unsupported-field", "Unsupported field", 400, "The request contains an unsupported field.", False),
    ("UNSUPPORTED_REQUEST", "unsupported-request", "Unsupported request", 400, "The request is not supported.", False),
    ("UNSUPPORTED_RNG", "unsupported-rng", "Unsupported RNG", 400, "The requested RNG is not supported.", False),
    ("BACKEND_UNAVAILABLE", "backend-unavailable", "Backend unavailable", 503, "The requested backend is currently unavailable.", True),
    ("RNG_REJECTION_EXHAUSTED", "rng-rejection-exhausted", "RNG rejection exhausted", 422, "The deterministic RNG rejection limit was exhausted.", False),
    ("RESOURCE_EXHAUSTED", "resource-exhausted", "Resource exhausted", 503, "Required service resources are currently exhausted.", True),
    ("INTERNAL_ERROR", "internal-error", "Internal error", 500, "The service encountered an internal error.", True),
)
PROBLEM_POLICIES = MappingProxyType(
    {
        code: (slug, title, status, detail, retryable)
        for code, slug, title, status, detail, retryable in _POLICY_ROWS
    }
)
_CORRELATION_ID = re.compile(r"^pk_[0-9a-f]{32}$")
_FIELD = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*|\[[0-9]+\])*$")
_FIELD_CODES = frozenset({"DUPLICATE_CARD", "INVALID_BOARD_LENGTH", "INVALID_CARD", "INVALID_CONTRACT_VERSION", "INVALID_OPPONENT_COUNT", "INVALID_SEED", "INVALID_TRIAL_COUNT", "UNSUPPORTED_FIELD", "UNSUPPORTED_REQUEST", "UNSUPPORTED_RNG"})


@dataclass(frozen=True)
class ContractProblem(Exception):
    code: str

    def __post_init__(self) -> None:
        if self.code not in PROBLEM_POLICIES:
            raise ValueError(f"unknown contract problem code: {self.code}")

    @property
    def detail(self) -> str:
        return PROBLEM_POLICIES[self.code][3]

    def __str__(self) -> str:
        return f"{self.code}: {self.detail}"

    def serialize(self, correlation_id: str, field_errors: tuple[Mapping[str, str], ...] | None = None) -> dict[str, object]:
        if not isinstance(correlation_id, str) or not _CORRELATION_ID.fullmatch(correlation_id):
            raise ValueError("correlation_id must be server-generated pk_ plus 32 lowercase hex digits")
        slug, title, status, detail, retryable = PROBLEM_POLICIES[self.code]
        payload: dict[str, object] = {"type": f"urn:poker-knight-ng:problem:v1:{slug}", "title": title, "status": status, "code": self.code, "detail": detail, "correlation_id": correlation_id, "retryable": retryable}
        if field_errors is not None:
            if self.code not in _FIELD_CODES or len(field_errors) > 32:
                raise ValueError("field_errors are allowed only for validation problems")
            clean = []
            for item in field_errors:
                if set(item) != {"field", "code"} or not isinstance(item["field"], str) or len(item["field"]) > 128 or not _FIELD.fullmatch(item["field"]) or item["code"] not in _FIELD_CODES:
                    raise ValueError("invalid field_errors")
                clean.append({"field": item["field"], "code": item["code"]})
            payload["field_errors"] = clean
        return payload


def problem(code: str) -> ContractProblem:
    return ContractProblem(code)
