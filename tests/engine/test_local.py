"""Public CPU engine vertical integration and hostile-boundary tests."""

import json
from pathlib import Path
import subprocess
import sys

import pytest

from poker_knight_ng import ContractProblem, EquityRequest, EquityResult
from poker_knight_ng.contract import canonical_case_hash
from poker_knight_ng.engine import CPUReferenceEngine, Engine, solve, to_equity_result
from poker_knight_ng.reference.dealer import RngRejectionExhausted
from poker_knight_ng.reference.monte_carlo import MonteCarloResult, _run_trial, run_cpu_monte_carlo


def _request(*, hero=("As", "Ah"), board=("2s", "3h", "Td"), opponents=2, trials=4, seed=0x0123_4567_89AB_CDEF, backend="cpu_reference"):
    return EquityRequest(hero, board, opponents, trials, seed, backend)


def _analytical(result):
    return (result.case_hash, result.seed, result.requested_trials, result.completed_trials, result.unique_wins, result.tie_by_other_winners, result.losses, result.equity_share_units, result.hero_category_counts, result.probabilities)


def test_public_solve_returns_contract_result_matching_production_monte_carlo():
    request = _request()
    result = solve(request)
    direct = run_cpu_monte_carlo(seed=request.seed, hero_card_ids=(12, 25), board_card_ids=(0, 14, 34), opponent_count=request.opponent_count, requested_trials=request.requested_trials, replay_case_hash=bytes.fromhex(canonical_case_hash(request)))
    assert type(result) is EquityResult
    assert result.case_hash == canonical_case_hash(request)
    assert result.requested_trials == result.completed_trials == request.requested_trials
    assert (result.unique_wins, result.tie_by_other_winners, result.losses, result.equity_share_units, tuple(value for _, value in result.hero_category_counts)) == (direct.unique_wins, direct.tie_by_other_winners, direct.losses, direct.equity_share_units, direct.hero_category_counts)
    assert result.probabilities == (("unique_win", direct.unique_wins, direct.completed_trials), ("tie", direct.ties, direct.completed_trials), ("loss", direct.losses, direct.completed_trials), ("showdown_equity", direct.equity_share_units, 420 * direct.completed_trials))


def test_public_boundary_is_canonical_order_invariant_and_prefix_reproducible():
    first = solve(_request(trials=12))
    reordered = solve(_request(hero=("Ah", "As"), board=("Td", "2s", "3h"), trials=12))
    prefix = solve(_request(trials=4))
    again = solve(_request(trials=12))
    assert _analytical(first) == _analytical(reordered) == _analytical(again)
    assert prefix.unique_wins + sum(prefix.tie_by_other_winners) + prefix.losses == 4
    request = _request(trials=12)
    digest = bytes.fromhex(canonical_case_hash(request))
    suffix = [_run_trial(seed=request.seed, hero_card_ids=(12, 25), board_card_ids=(0, 14, 34), opponent_count=request.opponent_count, simulation_id=simulation_id, case_hash=digest) for simulation_id in range(4, 12)]
    assert first.unique_wins == prefix.unique_wins + sum(row.unique_wins for row in suffix)
    assert first.tie_by_other_winners == tuple(prefix.tie_by_other_winners[index] + sum(row.tie_by_other_winners[index] for row in suffix) for index in range(6))
    assert first.losses == prefix.losses + sum(row.losses for row in suffix)
    assert first.equity_share_units == prefix.equity_share_units + sum(row.equity_share_units for row in suffix)
    assert tuple(value for _, value in first.hero_category_counts) == tuple(prefix.hero_category_counts[index][1] + sum(row.hero_category_counts[index] for row in suffix) for index in range(9))


def test_public_boundary_matches_frozen_seed_bank_exact_vector():
    vector = json.loads((Path(__file__).parents[2] / "validation/holdem/v1/rng_seed_bank.json").read_text())["exact_vectors"][0]
    request = _request(hero=("As", "Ah"), board=("2s", "3h", "Td"), opponents=vector["opponent_count"], trials=int(vector["requested_trials"]), seed=int(vector["seed"], 16))
    result = solve(request)
    expected = vector["expected"]
    assert str(result.completed_trials) == expected["completed_trials"]
    assert str(result.unique_wins) == expected["unique_wins"]
    assert str(result.losses) == expected["losses"]
    assert str(result.equity_share_units) == expected["equity_share_units"]
    assert {str(i + 1): str(v) for i, v in enumerate(result.tie_by_other_winners)} == expected["tie_by_other_winners"]


def test_cuda_is_explicitly_unavailable_before_reference_execution(monkeypatch):
    import poker_knight_ng.engine.local as local
    monkeypatch.setattr(local, "run_cpu_monte_carlo", lambda **_: pytest.fail("must not execute reference"))
    with pytest.raises(ContractProblem, match="BACKEND_UNAVAILABLE"):
        solve(_request(backend="cuda"))


def _shadow_request_authorities(request):
    object.__setattr__(request, "validate", lambda: None)
    object.__setattr__(request, "require_available_backend", lambda: None)
    return request


def test_public_boundary_rejects_shadowed_cuda_before_cpu_execution(monkeypatch):
    import poker_knight_ng.engine.local as local
    cpu_called = False

    def record_cpu(**_):
        nonlocal cpu_called
        cpu_called = True
        pytest.fail("must not execute CPU")

    monkeypatch.setattr(local, "run_cpu_monte_carlo", record_cpu)
    request = _shadow_request_authorities(_request())
    object.__setattr__(request, "backend", "cuda")
    with pytest.raises(ContractProblem, match="BACKEND_UNAVAILABLE"):
        solve(request)
    assert not cpu_called


@pytest.mark.parametrize(("field", "value", "code"), (
    ("hero_cards", ("bad", "Ah"), "INVALID_CARD"),
    ("hero_cards", ("As", "As"), "DUPLICATE_CARD"),
    ("board_cards", ("2s",), "INVALID_BOARD_LENGTH"),
    ("opponent_count", 0, "INVALID_OPPONENT_COUNT"),
    ("requested_trials", 0, "INVALID_TRIAL_COUNT"),
    ("seed", -1, "INVALID_SEED"),
    ("backend", "not-a-backend", "UNSUPPORTED_REQUEST"),
))
def test_public_boundary_preserves_validation_codes_despite_validate_shadow(monkeypatch, field, value, code):
    import poker_knight_ng.engine.local as local
    monkeypatch.setattr(local, "run_cpu_monte_carlo", lambda **_: pytest.fail("must not execute CPU"))
    request = _shadow_request_authorities(_request())
    object.__setattr__(request, field, value)
    with pytest.raises(ContractProblem) as caught:
        solve(request)
    assert caught.value.code == code


def test_public_boundary_valid_request_with_shadowed_authorities_is_not_blanket_rejected():
    baseline = solve(_request(trials=1))
    shadowed = solve(_shadow_request_authorities(_request(trials=1)))
    assert _analytical(shadowed) == _analytical(baseline)


def test_bad_request_type_or_subclass_is_closed_internal_error():
    class DerivedRequest(EquityRequest):
        pass
    for bad in (object(), DerivedRequest(("As", "Ah"), ("2s", "3h", "Td"), 2, 1, 1, "cpu_reference")):
        with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
            solve(bad)  # type: ignore[arg-type]


def test_forged_request_and_reference_failure_are_closed_internal_errors(monkeypatch):
    request = _request()
    object.__setattr__(request, "hero_cards", ("bad", "Ah"))
    with pytest.raises(ContractProblem, match="INVALID_CARD"):
        solve(request)
    import poker_knight_ng.engine.local as local
    monkeypatch.setattr(local, "run_cpu_monte_carlo", lambda **_: (_ for _ in ()).throw(ValueError("bad internal")))
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        solve(_request())


def test_reference_contract_problem_is_closed_as_internal_error(monkeypatch):
    import poker_knight_ng.engine.local as local
    monkeypatch.setattr(local, "run_cpu_monte_carlo", lambda **_: (_ for _ in ()).throw(ContractProblem("INVALID_CARD")))
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        solve(_request())


def test_reference_rng_rejection_precedes_internal_contract_problem_mapping(monkeypatch):
    import poker_knight_ng.engine.local as local
    monkeypatch.setattr(local, "run_cpu_monte_carlo", lambda **_: (_ for _ in ()).throw(RngRejectionExhausted(0, 0)))
    with pytest.raises(ContractProblem, match="RNG_REJECTION_EXHAUSTED"):
        solve(_request())


def test_result_adapter_closes_forged_or_altered_requests_as_internal_error():
    request = _request(trials=1)
    aggregate = run_cpu_monte_carlo(seed=request.seed, hero_card_ids=(12, 25), board_card_ids=(0, 14, 34), opponent_count=2, requested_trials=1)
    object.__setattr__(request, "hero_cards", ("bad", "Ah"))
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        to_equity_result(aggregate, request, 1)
    object.__setattr__(request, "hero_cards", ("As", "Ah"))
    object.__setattr__(request, "requested_trials", 2)
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        to_equity_result(aggregate, request, 1)


def test_result_adapter_rejects_invalid_request_despite_validate_shadow():
    request = _shadow_request_authorities(_request(trials=1))
    aggregate = _valid_adapter_aggregate()
    object.__setattr__(request, "requested_trials", 0)
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        to_equity_result(aggregate, request, 1)


def test_result_adapter_valid_request_with_validate_shadow_remains_correct():
    request = _shadow_request_authorities(_request(trials=1))
    result = to_equity_result(_valid_adapter_aggregate(), request, 1)
    assert result.backend == "cpu_reference"
    assert result.requested_trials == result.completed_trials == 1


@pytest.mark.parametrize("duration", [True, -1, 1 << 64, 1.0])
def test_result_adapter_rejects_invalid_duration(duration):
    request = _request(trials=1)
    aggregate = run_cpu_monte_carlo(seed=request.seed, hero_card_ids=(12, 25), board_card_ids=(0, 14, 34), opponent_count=2, requested_trials=1)
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        to_equity_result(aggregate, request, duration)


def test_result_adapter_rejects_exact_int_subclass_duration():
    class IntSubclass(int):
        pass
    request = _request(trials=1)
    aggregate = run_cpu_monte_carlo(seed=request.seed, hero_card_ids=(12, 25), board_card_ids=(0, 14, 34), opponent_count=2, requested_trials=1)
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        to_equity_result(aggregate, request, IntSubclass(1))


def test_result_adapter_rejects_subclass_and_forged_impossible_aggregate():
    request = _request(trials=1)
    aggregate = run_cpu_monte_carlo(seed=request.seed, hero_card_ids=(12, 25), board_card_ids=(0, 14, 34), opponent_count=2, requested_trials=1)
    class Derived(MonteCarloResult): pass
    derived = Derived(*aggregate.__dict__.values())
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        to_equity_result(derived, request, 1)
    object.__setattr__(aggregate, "tie_by_other_winners", (0, 0, 1, 0, 0, 0))
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        to_equity_result(aggregate, request, 1)


def _valid_adapter_aggregate():
    return MonteCarloResult(1, 1, (0, 0, 0, 0, 0, 0), 0, 420, (1, 0, 0, 0, 0, 0, 0, 0, 0), 0)


def test_result_adapter_does_not_trust_an_instance_validate_shadow():
    request = _request(trials=1)
    aggregate = _valid_adapter_aggregate()
    object.__setattr__(aggregate, "rejection_count", 6 * ((1 << 32) - 1) + 1)
    object.__setattr__(aggregate, "validate", lambda *_: None)
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        to_equity_result(aggregate, request, 1)


def test_result_adapter_rejects_hostile_exact_aggregate_fields_despite_validate_shadow():
    class IntSubclass(int):
        pass

    class TupleSubclass(tuple):
        pass

    # One representative mutation per authoritative field family; each bypasses
    # the public instance method to prove adapter validation owns this boundary.
    mutations = (
        ("rejection_count", -1), ("rejection_count", True), ("rejection_count", IntSubclass(0)),
        ("rejection_count", 6 * ((1 << 32) - 1) + 1),
        ("completed_trials", True), ("unique_wins", IntSubclass(1)), ("losses", 1 << 64), ("equity_share_units", True),
        ("tie_by_other_winners", TupleSubclass((0, 0, 0, 0, 0, 0))), ("tie_by_other_winners", (0, 0)),
        ("tie_by_other_winners", (True, 0, 0, 0, 0, 0)), ("tie_by_other_winners", (1 << 64, 0, 0, 0, 0, 0)),
        ("hero_category_counts", TupleSubclass((1, 0, 0, 0, 0, 0, 0, 0, 0))), ("hero_category_counts", (1, 0)),
        ("hero_category_counts", (IntSubclass(1), 0, 0, 0, 0, 0, 0, 0, 0)),
        ("hero_category_counts", (1 << 64, 0, 0, 0, 0, 0, 0, 0, 0)),
        ("tie_by_other_winners", (0, 0, 1, 0, 0, 0)), ("losses", 1),
        ("hero_category_counts", (0, 0, 0, 0, 0, 0, 0, 0, 0)), ("equity_share_units", 0),
    )
    request = _request(trials=1)
    for field, value in mutations:
        aggregate = _valid_adapter_aggregate()
        object.__setattr__(aggregate, field, value)
        object.__setattr__(aggregate, "validate", lambda *_: None)
        with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
            to_equity_result(aggregate, request, 1)


@pytest.mark.parametrize("signal", (KeyboardInterrupt, SystemExit, GeneratorExit))
@pytest.mark.parametrize("position", ("first", "second"))
def test_clock_process_control_signals_propagate_unchanged(signal, position):
    request = _request(trials=1)
    if position == "first":
        clock = lambda: (_ for _ in ()).throw(signal())
    else:
        values = iter((20, signal()))

        def clock():
            value = next(values)
            if isinstance(value, BaseException):
                raise value
            return value
    with pytest.raises(signal):
        CPUReferenceEngine(clock_ns=clock).solve(request)


def test_clock_contract_problem_is_closed_as_internal_error_on_both_reads():
    request = _request(trials=1)

    def first_clock_failure():
        raise ContractProblem("INVALID_CARD")

    with pytest.raises(ContractProblem) as first:
        CPUReferenceEngine(clock_ns=first_clock_failure).solve(request)
    assert first.value.code == "INTERNAL_ERROR"
    assert isinstance(first.value.__cause__, ContractProblem)
    assert first.value.__cause__.code == "INVALID_CARD"

    clock_values = iter((20, ContractProblem("INVALID_CARD")))

    def second_clock_failure():
        value = next(clock_values)
        if isinstance(value, Exception):
            raise value
        return value

    with pytest.raises(ContractProblem) as second:
        CPUReferenceEngine(clock_ns=second_clock_failure).solve(request)
    assert second.value.code == "INTERNAL_ERROR"
    assert isinstance(second.value.__cause__, ContractProblem)
    assert second.value.__cause__.code == "INVALID_CARD"


def test_clock_is_exact_and_fails_closed_for_invalid_or_reversed_values():
    request = _request(trials=1)
    assert CPUReferenceEngine(clock_ns=iter((20, 29)).__next__).solve(request).timing == 9
    class IntSubclass(int):
        pass
    for values in ((20, 19), (True, 2), (1, True), (1, 1 << 64), (IntSubclass(1), 2), (1, IntSubclass(2))):
        with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
            CPUReferenceEngine(clock_ns=iter(values).__next__).solve(request)
    def broken(): raise RuntimeError("clock")
    with pytest.raises(ContractProblem, match="INTERNAL_ERROR"):
        CPUReferenceEngine(clock_ns=broken).solve(request)


def test_successful_timing_covers_validation_execution_and_preflight_before_end(monkeypatch):
    import poker_knight_ng.engine.local as local
    events = []
    original_validate = EquityRequest.validate
    original_run = local.run_cpu_monte_carlo
    original_adapter = local.to_equity_result

    def validate(self):
        events.append("validate")
        return original_validate(self)

    def run(**kwargs):
        events.append("reference")
        return original_run(**kwargs)

    def adapter(*args):
        events.append("adapter")
        return original_adapter(*args)

    def clock():
        events.append("clock")
        return (20, 29)[events.count("clock") - 1]

    monkeypatch.setattr(EquityRequest, "validate", validate)
    monkeypatch.setattr(local, "run_cpu_monte_carlo", run)
    monkeypatch.setattr(local, "to_equity_result", adapter)
    request = _request(trials=1)
    events.clear()
    assert CPUReferenceEngine(clock_ns=clock).solve(request).timing == 9
    assert events.index("clock") < events.index("validate") < events.index("reference") < events.index("adapter") < len(events) - 1 - events[::-1].index("clock") < len(events) - 1 - events[::-1].index("adapter")
    assert events.count("adapter") == 2


def test_engine_protocol_and_cpu_only_top_level_import():
    assert isinstance(CPUReferenceEngine(), Engine)
    code = "import poker_knight_ng.engine; import sys; assert 'cupy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
