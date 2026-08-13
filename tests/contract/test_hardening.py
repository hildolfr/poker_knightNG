import importlib.metadata
import json
from importlib import resources
from pathlib import Path

import pytest

from poker_knight_ng import ContractProblem, EquityRequest, EquityResult, __version__, canonical_case_hash
from poker_knight_ng.contract.errors import PROBLEM_POLICIES, problem


def request(**changes):
    raw = {"contract_version":"v1","hero_cards":["As","Ah"],"board_cards":["2s","3h","Td"],"opponent_count":"2","requested_trials":"4","seed":"0x0123456789abcdef","backend":"cpu_reference","rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1"}}
    raw.update(changes)
    return raw


def result(req, **changes):
    raw = {"contract_version":"v1","backend":req.backend,"rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1"},"case_hash":canonical_case_hash(req),"seed":f"0x{req.seed:016x}","requested_trials":str(req.requested_trials),"completed_trials":"4","unique_wins":"1","ties":"2","tie_by_other_winners":{"1":"1","2":"1","3":"0","4":"0","5":"0","6":"0"},"losses":"1","equity_share_units":"770","hero_category_counts":{"high_card":"0","one_pair":"0","two_pair":"0","three_of_a_kind":"0","straight":"0","flush":"0","full_house":"4","four_of_a_kind":"0","straight_flush":"0"},"probabilities":{"unique_win":{"numerator":"1","denominator":"4"},"tie":{"numerator":"2","denominator":"4"},"loss":{"numerator":"1","denominator":"4"},"showdown_equity":{"numerator":"770","denominator":"1680"}},"timing":{"total_duration_ns":"1"},"provenance":{"engine_build_id":"phase1","backend_qualification":"cpu-contract","device_id":None,"kernel_id":None}}
    raw.update(changes)
    return raw


def test_result_is_request_bound_and_closed():
    req = EquityRequest.parse(request())
    for change in ({"rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1","extra":"x"}}, {"provenance":{"engine_build_id":"bad\nvalue","backend_qualification":"cpu-contract","device_id":None,"kernel_id":None}}, {"case_hash":"0" * 64}, {"seed":"0x0000000000000000"}, {"requested_trials":"5"}):
        with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
            EquityResult.parse(result(req, **change), request=req)


def test_cuda_requires_provenance_ids():
    req = EquityRequest.parse(request(backend="cuda"))
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        EquityResult.parse(result(req), request=req)


@pytest.mark.parametrize("change", [{"opponent_count":2}, {"opponent_count":True}, {"opponent_count":["2"]}, {"board_cards":["xx","3h","Td"]}, {"backend":"unknown"}, {"rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1","x":"y"}}, {"requested_trials":str((2**64-1)//420+1)}, {"rng":{"algorithm_id":1,"algorithm_version":[]}}])
def test_request_rejects_wire_type_and_closure_confusion(change):
    with pytest.raises(ContractProblem):
        EquityRequest.parse(request(**change))


def test_canonical_requires_validated_request():
    with pytest.raises(ContractProblem, match="UNSUPPORTED_REQUEST"):
        canonical_case_hash(object())


def test_problem_policy_is_fixed_and_serializable():
    assert len(PROBLEM_POLICIES) == 14
    with pytest.raises(ValueError):
        problem("NOPE")
    with pytest.raises(TypeError):
        problem("INVALID_CARD", "echo")
    payload = problem("BACKEND_UNAVAILABLE").serialize("pk_" + "a" * 32)
    assert payload["detail"] == "The requested backend is currently unavailable."
    assert "field_errors" not in payload


def test_metadata_and_resources_are_authoritative_and_identical():
    assert __version__ == importlib.metadata.version("poker-knight-ng")
    assert Path(__file__).parents[2].joinpath("README.md").read_bytes() == resources.files("poker_knight_ng").joinpath("README.md").read_bytes()
    for name in ("equity-request.schema.json", "equity-result.schema.json", "problem.schema.json"):
        assert Path(__file__).parents[2].joinpath("contracts/v1", name).read_bytes() == resources.files("poker_knight_ng.schemas.v1").joinpath(name).read_bytes()
        json.loads(resources.files("poker_knight_ng.schemas.v1").joinpath(name).read_text())


def test_problem_policy_table_cannot_be_mutated():
    with pytest.raises(TypeError):
        PROBLEM_POLICIES["INVALID_CARD"] = PROBLEM_POLICIES["INTERNAL_ERROR"]


def test_oversized_numeric_tokens_fail_with_stable_problem():
    raw = {
        "contract_version": "v1",
        "hero_cards": ["As", "Ah"],
        "board_cards": [],
        "opponent_count": "1",
        "requested_trials": "9" * 10_000,
        "seed": "0x0000000000000000",
        "backend": "cpu_reference",
        "rng": {
            "algorithm_id": "poker-knight-ng/philox4x32-10",
            "algorithm_version": "1",
        },
    }
    with pytest.raises(ContractProblem) as caught:
        EquityRequest.parse(raw)
    assert caught.value.code == "INVALID_TRIAL_COUNT"
