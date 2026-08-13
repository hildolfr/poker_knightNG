import pytest

from poker_knight_ng.contract import ContractProblem, EquityRequest, EquityResult


def result(**changes):
    value = {"contract_version":"v1","backend":"cpu_reference","rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1"},"case_hash":"fb3c0fa3e41cdd7f89e45b458f17f14174d51f285723c5178c68bd2756fec3eb","seed":"0x0123456789abcdef","requested_trials":"4","completed_trials":"4","unique_wins":"1","ties":"2","tie_by_other_winners":{"1":"1","2":"1","3":"0","4":"0","5":"0","6":"0"},"losses":"1","equity_share_units":"770","hero_category_counts":{"high_card":"0","one_pair":"0","two_pair":"0","three_of_a_kind":"0","straight":"0","flush":"0","full_house":"4","four_of_a_kind":"0","straight_flush":"0"},"probabilities":{"unique_win":{"numerator":"1","denominator":"4"},"tie":{"numerator":"2","denominator":"4"},"loss":{"numerator":"1","denominator":"4"},"showdown_equity":{"numerator":"770","denominator":"1680"}},"timing":{"total_duration_ns":"1"},"provenance":{"engine_build_id":"phase1","backend_qualification":"cpu-contract","device_id":None,"kernel_id":None}}
    value.update(changes)
    return value


def parsed_request():
    return EquityRequest.parse({"contract_version":"v1","hero_cards":["As","Ah"],"board_cards":["2s","3h","Td"],"opponent_count":"2","requested_trials":"4","seed":"0x0123456789abcdef","backend":"cpu_reference","rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1"}})

def test_result_enforces_phase_zero_invariants_and_exact_fractions():
    parsed = EquityResult.parse(result(), request=parsed_request())
    assert parsed.equity_share_units == 770
    assert parsed.contract_version == "v1"
    assert parsed.backend == "cpu_reference"
    assert parsed.rng == ("poker-knight-ng/philox4x32-10", "1")
    assert parsed.case_hash == result()["case_hash"]
    assert (parsed.seed, parsed.requested_trials, parsed.completed_trials) == (0x0123456789abcdef, 4, 4)
    assert (parsed.unique_wins, parsed.ties, parsed.losses) == (1, 2, 1)
    assert parsed.tie_by_other_winners == (1, 1, 0, 0, 0, 0)
    assert parsed.hero_category_counts[-1] == ("straight_flush", 0)
    assert parsed.probabilities[-1] == ("showdown_equity", 770, 1680)
    assert parsed.timing == 1
    assert parsed.provenance == ("phase1", "cpu-contract", None, None)


@pytest.mark.parametrize("args, kwargs", [
    ((), {}), ((-1,), {}), ((True,), {}), ((2**100,), {}), ((), {"equity_share_units": 770}),
])
def test_result_has_no_public_constructor(args, kwargs):
    with pytest.raises(TypeError):
        EquityResult(*args, **kwargs)


def test_result_parse_detaches_and_freezes_nested_wire_data():
    raw = result()
    parsed = EquityResult.parse(raw, request=parsed_request())
    raw["tie_by_other_winners"]["1"] = "4"
    raw["provenance"]["engine_build_id"] = "forged"
    assert parsed.tie_by_other_winners[0] == 1
    assert parsed.provenance[0] == "phase1"
    with pytest.raises((AttributeError, TypeError)):
        parsed.tie_by_other_winners[0] = 4

@pytest.mark.parametrize("change", [
    {"ties":"1"},
    {"equity_share_units":"771"},
    {"probabilities":{"unique_win":{"numerator":"1","denominator":"4"},"tie":{"numerator":"1","denominator":"4"},"loss":{"numerator":"1","denominator":"4"},"showdown_equity":{"numerator":"770","denominator":"1680"}}},
    {"tie_by_other_winners":{"1":"1","2":"1","3":"1","4":"0","5":"0","6":"0"}},
])
def test_invalid_result_never_parses_as_success(change):
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        EquityResult.parse(result(**change), request=parsed_request())
