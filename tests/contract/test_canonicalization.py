from poker_knight_ng.contract import EquityRequest, canonical_case_bytes, canonical_case_hash


def request(**changes):
    value = {"contract_version":"v1","hero_cards":["Ah","As"],"board_cards":["Td","3h","2s"],"opponent_count":"2","requested_trials":"4","seed":"0x0123456789abcdef","backend":"cpu_reference","rng":{"algorithm_id":"poker-knight-ng/philox4x32-10","algorithm_version":"1"}}
    value.update(changes)
    return value


def test_canonical_case_matches_adr_0003_vector():
    case = EquityRequest.parse(request())
    assert canonical_case_bytes(case).hex() == "17706f6b65722d6b6e696768742d6e672f636173652f7631020c1903000e2202"
    assert canonical_case_hash(case) == "fb3c0fa3e41cdd7f89e45b458f17f14174d51f285723c5178c68bd2756fec3eb"


def test_seed_trials_and_backend_do_not_affect_case_hash():
    assert canonical_case_hash(EquityRequest.parse(request(seed="0x0000000000000000", requested_trials="99", backend="cuda"))) == canonical_case_hash(EquityRequest.parse(request()))
