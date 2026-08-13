import sys
from importlib.machinery import ModuleSpec
from types import ModuleType

import pytest

from poker_knight_ng.contract import ContractProblem, EquityRequest, canonical_case_bytes, canonical_case_hash
from poker_knight_ng.contract.models import MAX_TRIALS


def request(**changes):
    value = {"contract_version":"v1","hero_cards":["As","Ah"],"board_cards":["2s","3h","Td"],"opponent_count":"2","requested_trials":"4","seed":"0x0123456789abcdef","backend":"cpu_reference","rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1"}}
    value.update(changes)
    return value


def test_valid_request_parses_strict_wire_tokens():
    parsed = EquityRequest.parse(request())
    assert parsed.hero_cards == ("As", "Ah")
    assert parsed.opponent_count == 2


@pytest.mark.parametrize(("change", "code"), [
    ({"hero_cards":["As", "As"]}, "DUPLICATE_CARD"),
    ({"board_cards":["As", "3h", "Td"]}, "DUPLICATE_CARD"),
    ({"hero_cards":["as", "Ah"]}, "INVALID_CARD"),
    ({"board_cards":["2s"]}, "INVALID_BOARD_LENGTH"),
    ({"opponent_count":"07"}, "INVALID_OPPONENT_COUNT"),
    ({"requested_trials":"0"}, "INVALID_TRIAL_COUNT"),
    ({"seed":"0X0123456789abcdef"}, "INVALID_SEED"),
    ({"rng":{"algorithm_id":"other","algorithm_version":"1"}}, "UNSUPPORTED_RNG"),
    ({"legacy_metric":True}, "UNSUPPORTED_FIELD"),
])
def test_invalid_request_fails_closed_with_stable_problem(change, code):
    with pytest.raises(ContractProblem, match=code):
        EquityRequest.parse(request(**change))


@pytest.mark.parametrize(("kwargs", "code"), [
    ({"hero_cards": ("As", "As"), "board_cards": (), "opponent_count": 1, "requested_trials": 1, "seed": 0, "backend": "cpu_reference"}, "DUPLICATE_CARD"),
    ({"hero_cards": ("As", "Ah"), "board_cards": ("2s",), "opponent_count": 1, "requested_trials": 1, "seed": 0, "backend": "cpu_reference"}, "INVALID_BOARD_LENGTH"),
    ({"hero_cards": ("As", "Ah"), "board_cards": (), "opponent_count": True, "requested_trials": 1, "seed": 0, "backend": "cpu_reference"}, "INVALID_OPPONENT_COUNT"),
    ({"hero_cards": ("As", "Ah"), "board_cards": (), "opponent_count": 1, "requested_trials": True, "seed": 0, "backend": "cpu_reference"}, "INVALID_TRIAL_COUNT"),
    ({"hero_cards": ("As", "Ah"), "board_cards": (), "opponent_count": 1, "requested_trials": 1, "seed": True, "backend": "cpu_reference"}, "INVALID_SEED"),
    ({"hero_cards": ("As", "Ah"), "board_cards": (), "opponent_count": 1, "requested_trials": 1, "seed": 0, "backend": True}, "UNSUPPORTED_REQUEST"),
])
def test_forged_direct_request_never_reaches_canonicalization(kwargs, code):
    with pytest.raises(ContractProblem, match=code):
        forged = EquityRequest(**kwargs)
        canonical_case_bytes(forged)
        canonical_case_hash(forged)


def test_valid_direct_normalized_request_has_the_canonical_vector():
    parsed = EquityRequest.parse(request())
    direct = EquityRequest(("As", "Ah"), ("2s", "3h", "Td"), 2, 4, 0x0123456789ABCDEF, "cpu_reference")
    assert canonical_case_bytes(direct) == canonical_case_bytes(parsed)
    assert canonical_case_hash(direct) == canonical_case_hash(parsed)


@pytest.mark.parametrize(("field", "value", "code"), [
    ("hero_cards", ("Xx", "Ah"), "INVALID_CARD"),
    ("hero_cards", ("As",), "INVALID_CARD"),
    ("hero_cards", ("As", "As"), "DUPLICATE_CARD"),
    ("board_cards", ("2s",), "INVALID_BOARD_LENGTH"),
    ("board_cards", ("As", "3h", "Td"), "DUPLICATE_CARD"),
    ("opponent_count", 7, "INVALID_OPPONENT_COUNT"),
    ("opponent_count", 255, "INVALID_OPPONENT_COUNT"),
    ("opponent_count", True, "INVALID_OPPONENT_COUNT"),
    ("requested_trials", 0, "INVALID_TRIAL_COUNT"),
    ("requested_trials", MAX_TRIALS + 1, "INVALID_TRIAL_COUNT"),
    ("requested_trials", True, "INVALID_TRIAL_COUNT"),
    ("seed", -1, "INVALID_SEED"),
    ("seed", 2**64, "INVALID_SEED"),
    ("seed", True, "INVALID_SEED"),
    ("backend", "bogus", "UNSUPPORTED_REQUEST"),
])
def test_post_construction_forgery_is_revalidated_before_canonicalization(field, value, code):
    parsed = EquityRequest.parse(request())
    object.__setattr__(parsed, field, value)
    with pytest.raises(ContractProblem, match=code):
        canonical_case_bytes(parsed)


def test_cuda_is_unavailable_when_cupy_probe_reports_no_package():
    with pytest.raises(ContractProblem, match="BACKEND_UNAVAILABLE"):
        EquityRequest.parse(request(backend="cuda")).require_available_backend()


def test_cuda_is_unavailable_when_cupy_is_importable(monkeypatch):
    cupy = ModuleType("cupy")
    cupy.__spec__ = ModuleSpec("cupy", loader=None)
    monkeypatch.setitem(sys.modules, "cupy", cupy)
    with pytest.raises(ContractProblem, match="BACKEND_UNAVAILABLE"):
        EquityRequest.parse(request(backend="cuda")).require_available_backend()
