"""Draft 2020-12 structural fixtures for v1 request/result schemas."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "contracts" / "v1"


def _validator(name: str) -> Draft202012Validator:
    return Draft202012Validator(
        json.loads((CONTRACTS / name).read_text(encoding="utf-8"))
    )


def _valid_request() -> dict[str, object]:
    return {
        "contract_version": "v1",
        "hero_cards": ["As", "Ah"],
        "board_cards": ["2s", "3h", "Td"],
        "opponent_count": "2",
        "requested_trials": "4",
        "seed": "0x0123456789abcdef",
        "backend": "cpu_reference",
        "rng": {
            "algorithm_id": "poker-knight-ng/philox4x32-10",
            "algorithm_version": "1",
        },
    }


def _valid_result() -> dict[str, object]:
    return {
        "contract_version": "v1",
        "backend": "cpu_reference",
        "rng": {
            "algorithm_id": "poker-knight-ng/philox4x32-10",
            "algorithm_version": "1",
        },
        "case_hash": "fb3c0fa3e41cdd7f89e45b458f17f14174d51f285723c5178c68bd2756fec3eb",
        "seed": "0x0123456789abcdef",
        "requested_trials": "4",
        "completed_trials": "4",
        "unique_wins": "1",
        "ties": "2",
        "tie_by_other_winners": {
            "1": "1",
            "2": "1",
            "3": "0",
            "4": "0",
            "5": "0",
            "6": "0",
        },
        "losses": "1",
        "equity_share_units": "770",
        "hero_category_counts": {
            "high_card": "0",
            "one_pair": "0",
            "two_pair": "0",
            "three_of_a_kind": "0",
            "straight": "0",
            "flush": "0",
            "full_house": "4",
            "four_of_a_kind": "0",
            "straight_flush": "0",
        },
        "probabilities": {
            "unique_win": {"numerator": "1", "denominator": "4"},
            "tie": {"numerator": "2", "denominator": "4"},
            "loss": {"numerator": "1", "denominator": "4"},
            "showdown_equity": {"numerator": "770", "denominator": "1680"},
        },
        "timing": {"total_duration_ns": "1"},
        "provenance": {
            "engine_build_id": "phase1",
            "backend_qualification": "cpu-contract",
            "device_id": None,
            "kernel_id": None,
        },
    }


@pytest.mark.parametrize(
    ("name", "fixture"),
    (
        ("equity-request.schema.json", _valid_request),
        ("equity-result.schema.json", _valid_result),
    ),
)
def test_normative_fixtures_are_structurally_valid(name: str, fixture) -> None:
    assert not list(_validator(name).iter_errors(fixture()))


def test_request_schema_rejects_nested_rng_extra() -> None:
    payload = _valid_request()
    payload["rng"]["__proto__"] = {"polluted": True}

    assert list(_validator("equity-request.schema.json").iter_errors(payload))


def test_result_schema_rejects_nested_fraction_extra() -> None:
    payload = _valid_result()
    payload["probabilities"]["tie"]["__proto__"] = {"polluted": True}

    assert list(_validator("equity-result.schema.json").iter_errors(payload))


def test_result_schema_rejects_provenance_control_characters() -> None:
    payload = _valid_result()
    payload["provenance"]["engine_build_id"] = "build\nFORGED"

    assert list(_validator("equity-result.schema.json").iter_errors(payload))


def test_result_schema_rejects_cuda_with_null_device_and_kernel_ids() -> None:
    payload = _valid_result()
    payload["backend"] = "cuda"
    payload["provenance"]["device_id"] = None
    payload["provenance"]["kernel_id"] = None

    assert list(_validator("equity-result.schema.json").iter_errors(payload))
