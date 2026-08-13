"""Regression tests for v1 problem construction and schema conformance."""
from __future__ import annotations

import json
from importlib.resources import files
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from poker_knight_ng.contract.errors import ContractProblem, PROBLEM_POLICIES, problem

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "contracts" / "v1"
PACKAGED = files("poker_knight_ng.schemas.v1")
CORRELATION_ID = "pk_0123456789abcdef0123456789abcdef"


def _authoritative(name: str) -> dict[str, object]:
    return json.loads((CONTRACTS / name).read_text(encoding="utf-8"))


def _packaged(name: str) -> bytes:
    return (PACKAGED / name).read_bytes()


def test_direct_problem_constructor_rejects_unknown_codes() -> None:
    with pytest.raises(ValueError, match="unknown contract problem code: NOPE"):
        ContractProblem("NOPE")


def test_direct_problem_constructor_is_usable_and_immutable() -> None:
    instance = ContractProblem("INVALID_CARD")

    assert instance.detail == "The request contains an invalid card."
    assert instance.serialize(CORRELATION_ID)["code"] == "INVALID_CARD"
    with pytest.raises(Exception):
        instance.code = "INTERNAL_ERROR"  # type: ignore[misc]


@pytest.mark.parametrize(
    "name",
    ("equity-request.schema.json", "equity-result.schema.json", "problem.schema.json"),
)
def test_packaged_schema_is_byte_identical_to_authoritative_source(name: str) -> None:
    assert _packaged(name) == (CONTRACTS / name).read_bytes()


@pytest.mark.parametrize(
    "name",
    ("equity-request.schema.json", "equity-result.schema.json", "problem.schema.json"),
)
def test_all_packaged_schemas_are_valid_draft_2020_12(name: str) -> None:
    Draft202012Validator.check_schema(json.loads(_packaged(name)))


def test_every_fixed_problem_payload_validates_against_problem_schema() -> None:
    validator = Draft202012Validator(_authoritative("problem.schema.json"))

    assert len(PROBLEM_POLICIES) == 14
    for code in PROBLEM_POLICIES:
        assert not list(validator.iter_errors(problem(code).serialize(CORRELATION_ID))), code


def _problem_branches(value: object) -> list[dict[str, object]]:
    if isinstance(value, dict):
        properties = value.get("properties")
        if (
            isinstance(properties, dict)
            and isinstance(properties.get("code"), dict)
            and "const" in properties["code"]
        ):
            return [value]
        return [branch for child in value.values() for branch in _problem_branches(child)]
    if isinstance(value, list):
        return [branch for child in value for branch in _problem_branches(child)]
    return []


def test_problem_policy_table_exactly_matches_authoritative_schema_branches() -> None:
    expected = {}
    for branch in _problem_branches(_authoritative("problem.schema.json")):
        properties = branch["properties"]
        code = properties["code"]["const"]
        expected[code] = (
            properties["type"]["const"].rsplit(":", 1)[1],
            properties["title"]["const"],
            properties["status"]["const"],
            properties["detail"]["const"],
            properties["retryable"]["const"],
        )

    assert PROBLEM_POLICIES == expected
