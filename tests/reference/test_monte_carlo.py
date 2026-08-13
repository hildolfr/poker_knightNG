"""Deterministic CPU Monte Carlo reference accumulator tests."""

import pytest

import poker_knight_ng.reference.monte_carlo as monte_carlo
from poker_knight_ng.reference.cards import CARD_DECK
from poker_knight_ng.reference.dealer import canonical_case_hash, deal_cpu
from poker_knight_ng.reference.enumerate import evaluate_terminal


CASE = dict(seed=0x0123_4567_89AB_CDEF, hero_card_ids=(12, 25), board_card_ids=(0, 14, 34), opponent_count=2)


def _terminal_row(*, simulation_id: int, **case):
    deal = deal_cpu(
        canonical_case_hash=canonical_case_hash(case["hero_card_ids"], case["board_card_ids"], case["opponent_count"]),
        simulation_id=simulation_id,
        **case,
    )
    terminal = evaluate_terminal(
        tuple(CARD_DECK[card] for card in case["hero_card_ids"]),
        tuple(CARD_DECK[card] for card in deal.board_card_ids),
        tuple(tuple(CARD_DECK[card] for card in hole) for hole in deal.opponent_hole_card_ids),
    )
    return terminal, deal.trace.rejection_count


def _zero():
    return monte_carlo.MonteCarloResult(0, 0, (0,) * 6, 0, 0, (0,) * 9, 0)


def _add(left, right):
    return monte_carlo.MonteCarloResult(
        left.completed_trials + right.completed_trials,
        left.unique_wins + right.unique_wins,
        tuple(a + b for a, b in zip(left.tie_by_other_winners, right.tie_by_other_winners)),
        left.losses + right.losses,
        left.equity_share_units + right.equity_share_units,
        tuple(a + b for a, b in zip(left.hero_category_counts, right.hero_category_counts)),
        left.rejection_count + right.rejection_count,
    )


def test_monte_carlo_accumulates_one_deterministic_terminal_trial():
    result = monte_carlo.run_cpu_monte_carlo(requested_trials=1, **CASE)

    terminal, rejections = _terminal_row(simulation_id=0, **CASE)
    assert result.completed_trials == 1
    assert result.unique_wins == terminal.unique_wins
    assert result.tie_by_other_winners == terminal.tie_by_other_winners
    assert result.losses == terminal.losses
    assert result.equity_share_units == terminal.equity_share_units
    assert result.hero_category_counts == terminal.hero_category_counts
    assert result.rejection_count == rejections
    result.validate(CASE["opponent_count"], 1, len(CASE["board_card_ids"]))


@pytest.mark.parametrize("board_card_count,opponent_count", [
    (board_card_count, opponent_count)
    for board_card_count in (0, 3, 4, 5)
    for opponent_count in (1, 6)
])
def test_monte_carlo_result_validation_accepts_exact_topology_rejection_maximum(board_card_count, opponent_count):
    completed_trials = 2
    draw_count = 5 - board_card_count + 2 * opponent_count
    exact_maximum = completed_trials * draw_count * ((1 << 32) - 1)
    result = monte_carlo.MonteCarloResult(completed_trials, completed_trials, (0,) * 6, 0, 420 * completed_trials, (completed_trials,) + (0,) * 8, exact_maximum)

    result.validate(opponent_count, completed_trials, board_card_count)


@pytest.mark.parametrize("board_card_count,opponent_count", [
    (board_card_count, opponent_count)
    for board_card_count in (0, 3, 4, 5)
    for opponent_count in (1, 6)
])
def test_monte_carlo_result_validation_rejects_rejection_count_above_exact_topology_maximum(board_card_count, opponent_count):
    completed_trials = 2
    draw_count = 5 - board_card_count + 2 * opponent_count
    result = monte_carlo.MonteCarloResult(completed_trials, completed_trials, (0,) * 6, 0, 420 * completed_trials, (completed_trials,) + (0,) * 8, completed_trials * draw_count * ((1 << 32) - 1) + 1)

    with pytest.raises(ValueError, match="rejection_count"):
        result.validate(opponent_count, completed_trials, board_card_count)


def test_monte_carlo_result_validation_rejects_uint64_maximum_rejections_for_one_trial():
    result = monte_carlo.MonteCarloResult(1, 1, (0,) * 6, 0, 420, (1,) + (0,) * 8, monte_carlo.UINT64_MAX)

    with pytest.raises(ValueError, match="rejection_count"):
        result.validate(1, 1, 0)


@pytest.mark.parametrize("board_card_count", [True, 0.0, "0", -1, 1, 2, 6])
def test_monte_carlo_result_validation_rejects_malformed_board_topology(board_card_count):
    result = monte_carlo.MonteCarloResult(1, 1, (0,) * 6, 0, 420, (1,) + (0,) * 8, 0)

    with pytest.raises(ValueError, match="board_card_count"):
        result.validate(1, 1, board_card_count)


def test_monte_carlo_is_repeatable_and_semantic_card_order_invariant():
    first = monte_carlo.run_cpu_monte_carlo(requested_trials=12, **CASE)
    again = monte_carlo.run_cpu_monte_carlo(requested_trials=12, **CASE)
    reordered = monte_carlo.run_cpu_monte_carlo(
        requested_trials=12,
        seed=CASE["seed"], hero_card_ids=(25, 12), board_card_ids=(34, 0, 14), opponent_count=2,
    )
    assert first == again == reordered


def test_monte_carlo_seed_changes_a_non_degenerate_trace():
    first = monte_carlo.run_cpu_monte_carlo(requested_trials=12, **CASE)
    changed = monte_carlo.run_cpu_monte_carlo(
        requested_trials=12, seed=0x0123_4567_89AB_CDF0, hero_card_ids=(12, 25), board_card_ids=(0, 14, 34), opponent_count=2,
    )
    assert first != changed


def test_monte_carlo_board_playing_tie_uses_exact_six_way_bin():
    result = monte_carlo.run_cpu_monte_carlo(
        seed=1, hero_card_ids=(26, 39), board_card_ids=(8, 9, 10, 11, 12), opponent_count=6, requested_trials=2,
    )
    assert result == monte_carlo.MonteCarloResult(2, 0, (0, 0, 0, 0, 0, 2), 0, 120, (0,) * 8 + (2,), 0)


@pytest.mark.parametrize("board_card_ids,opponent_count", [
    ((), 1), ((0, 14, 34), 1), ((0, 14, 34, 7), 6), ((0, 14, 34, 7, 8), 6),
])
def test_monte_carlo_matches_independent_deal_and_terminal_evaluation(board_card_ids, opponent_count):
    case = dict(seed=0xABCDEF, hero_card_ids=(12, 25), board_card_ids=board_card_ids, opponent_count=opponent_count)
    result = monte_carlo.run_cpu_monte_carlo(requested_trials=4, **case)
    expected = _zero()
    for simulation_id in range(4):
        terminal, rejections = _terminal_row(simulation_id=simulation_id, **case)
        expected = _add(expected, monte_carlo.MonteCarloResult(
            terminal.completed_trials, terminal.unique_wins, terminal.tie_by_other_winners,
            terminal.losses, terminal.equity_share_units, terminal.hero_category_counts, rejections,
        ))
    assert result == expected
    result.validate(opponent_count, 4, len(board_card_ids))


def test_monte_carlo_prefix_is_exact_sum_of_private_trial_rows():
    case = dict(CASE, requested_trials=9)
    result = monte_carlo.run_cpu_monte_carlo(**case)
    rows = [monte_carlo._run_trial(simulation_id=index, **CASE) for index in range(9)]
    expected = _zero()
    for row in rows:
        expected = _add(expected, row)
    assert result == expected


def test_monte_carlo_accounting_and_frozen_small_aggregate_vector():
    result = monte_carlo.run_cpu_monte_carlo(requested_trials=12, **CASE)
    assert result.unique_wins + sum(result.tie_by_other_winners) + result.losses == 12
    assert sum(result.hero_category_counts) == 12
    assert result.equity_share_units == 420 * result.unique_wins + sum(
        420 // (other_winners + 1) * count
        for other_winners, count in enumerate(result.tie_by_other_winners, start=1)
    )
    assert result.tie_by_other_winners[2:] == (0, 0, 0, 0)
    assert result == monte_carlo.MonteCarloResult(
        12, 9, (0, 0, 0, 0, 0, 0), 3, 3780, (0, 8, 2, 1, 0, 0, 1, 0, 0), 0,
    )  # frozen after independent derivation


@pytest.mark.parametrize("kwargs", [
    {"seed": True}, {"hero_card_ids": (True, 25)}, {"hero_card_ids": (12, 52)},
    {"hero_card_ids": (12, 12)}, {"board_card_ids": (0, 1)}, {"board_card_ids": (0, 14, 34, 52)},
    {"board_card_ids": (12, 14, 34)}, {"opponent_count": True}, {"opponent_count": 0},
    {"opponent_count": 7}, {"requested_trials": True}, {"requested_trials": 0},
])
def test_monte_carlo_rejects_malformed_inputs_before_execution(monkeypatch, kwargs):
    def deal_must_not_run(**_kwargs):
        raise AssertionError("dealer must not run")

    monkeypatch.setattr(monte_carlo, "deal_cpu", deal_must_not_run)
    values = dict(CASE, requested_trials=1)
    values.update(kwargs)
    with pytest.raises(ValueError):
        monte_carlo.run_cpu_monte_carlo(**values)


def test_monte_carlo_rejects_hash_mismatch_and_trial_overflow_before_dealing(monkeypatch):
    def deal_must_not_run(**_kwargs):
        raise AssertionError("dealer must not run")

    monkeypatch.setattr(monte_carlo, "deal_cpu", deal_must_not_run)
    with pytest.raises(ValueError, match="replay hash"):
        monte_carlo.run_cpu_monte_carlo(requested_trials=1, replay_case_hash=bytes(32), **CASE)
    with pytest.raises(ValueError, match="requested_trials"):
        monte_carlo.run_cpu_monte_carlo(requested_trials=monte_carlo.MAX_REQUESTED_TRIALS + 1, **CASE)


def test_validate_case_admits_maximum_requested_trials_for_preflop_six_opponents_without_dealing(monkeypatch):
    def deal_must_not_run(**_kwargs):
        raise AssertionError("dealer must not run")

    monkeypatch.setattr(monte_carlo, "deal_cpu", deal_must_not_run)
    validated = monte_carlo._validate_case(
        CASE["seed"], CASE["hero_card_ids"], (), 6, monte_carlo.MAX_REQUESTED_TRIALS,
    )

    assert validated[4] == monte_carlo.MAX_REQUESTED_TRIALS


def test_validate_case_rejects_requested_trials_above_normative_maximum():
    with pytest.raises(ValueError, match="requested_trials"):
        monte_carlo._validate_case(
            CASE["seed"], CASE["hero_card_ids"], (), 6, monte_carlo.MAX_REQUESTED_TRIALS + 1,
        )


def test_monte_carlo_result_validation_accepts_unbounded_feasible_rejection_maximum_at_requested_trial_limit():
    completed = monte_carlo.MAX_REQUESTED_TRIALS
    draw_count = 5 + 2 * 6
    feasible_maximum = completed * draw_count * monte_carlo.UINT32_MAX
    assert feasible_maximum > monte_carlo.UINT64_MAX
    result = monte_carlo.MonteCarloResult(
        completed, completed, (0,) * 6, 0, 420 * completed,
        (completed,) + (0,) * 8, feasible_maximum,
    )

    result.validate(6, completed, 0)


def test_monte_carlo_result_validation_rejects_rejection_count_above_unbounded_feasible_maximum():
    completed = monte_carlo.MAX_REQUESTED_TRIALS
    feasible_maximum = completed * (5 + 2 * 6) * monte_carlo.UINT32_MAX
    result = monte_carlo.MonteCarloResult(
        completed, completed, (0,) * 6, 0, 420 * completed,
        (completed,) + (0,) * 8, feasible_maximum + 1,
    )

    with pytest.raises(ValueError, match="rejection_count"):
        result.validate(6, completed, 0)


@pytest.mark.parametrize("rejection_count", [-1, True, 0.0])
def test_monte_carlo_result_validation_rejects_non_integer_or_negative_rejection_count(rejection_count):
    result = monte_carlo.MonteCarloResult(1, 1, (0,) * 6, 0, 420, (1,) + (0,) * 8, rejection_count)

    with pytest.raises(ValueError, match="rejection_count"):
        result.validate(1, 1, 0)


@pytest.mark.parametrize("field", [
    "completed_trials", "unique_wins", "losses", "equity_share_units",
])
def test_monte_carlo_result_validation_keeps_authoritative_scalar_counters_uint64_bounded(field):
    values = dict(
        completed_trials=1, unique_wins=1, tie_by_other_winners=(0,) * 6,
        losses=0, equity_share_units=420, hero_category_counts=(1,) + (0,) * 8,
        rejection_count=0,
    )
    values[field] = monte_carlo.UINT64_MAX + 1
    result = monte_carlo.MonteCarloResult(**values)

    with pytest.raises(ValueError, match=field):
        result.validate(1, 1, 0)


@pytest.mark.parametrize("field", ["tie_by_other_winners", "hero_category_counts"])
def test_monte_carlo_result_validation_keeps_authoritative_bins_uint64_bounded(field):
    values = dict(
        completed_trials=1, unique_wins=1, tie_by_other_winners=(0,) * 6,
        losses=0, equity_share_units=420, hero_category_counts=(1,) + (0,) * 8,
        rejection_count=0,
    )
    values[field] = (monte_carlo.UINT64_MAX + 1,) + (0,) * (5 if field == "tie_by_other_winners" else 8)
    result = monte_carlo.MonteCarloResult(**values)

    with pytest.raises(ValueError, match="accounting bin"):
        result.validate(1, 1, 0)


@pytest.mark.parametrize("result", [
    monte_carlo.MonteCarloResult(True, 0, (0,) * 6, 1, 0, (0,) * 9, 0),
    monte_carlo.MonteCarloResult(1, 1.0, (0,) * 6, 0, 420, (1,) + (0,) * 8, 0),
    monte_carlo.MonteCarloResult(1, 1, [0] * 6, 0, 420, (1,) + (0,) * 8, 0),
    monte_carlo.MonteCarloResult(1, 1, (0,) * 5, 0, 420, (1,) + (0,) * 8, 0),
    monte_carlo.MonteCarloResult(1, 1, (0,) * 6, 0, 420, [1] + [0] * 8, 0),
    monte_carlo.MonteCarloResult(1, 1, (0,) * 6, 0, 420, (1,) + (0,) * 7, 0),
])
def test_monte_carlo_result_validation_rejects_invalid_direct_construction(result):
    with pytest.raises(ValueError):
        result.validate(1, 1, 0)


def test_monte_carlo_result_validation_rejects_forged_and_malicious_subclass_instances():
    result = monte_carlo.MonteCarloResult(1, 1, (0,) * 6, 0, 420, (1,) + (0,) * 8, 0)
    object.__setattr__(result, "unique_wins", True)
    with pytest.raises(ValueError):
        result.validate(1, 1, 0)

    class MaliciousResult(monte_carlo.MonteCarloResult):
        def __getattribute__(self, name):
            if name in {"ties", "unique_wins", "losses", "equity_share_units"}:
                return {"ties": 0, "unique_wins": 1, "losses": 0, "equity_share_units": 420}[name]
            return super().__getattribute__(name)

    malicious = MaliciousResult(1, 0, (0,) * 6, 1, 0, (1,) + (0,) * 8, 0)
    with pytest.raises(ValueError):
        monte_carlo.MonteCarloResult.validate(malicious, 1, 1, 0)
    with pytest.raises(ValueError):
        _ = monte_carlo.MonteCarloResult.ties.fget(malicious)
