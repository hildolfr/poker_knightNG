import os
from itertools import combinations as independent_combinations

import pytest

from poker_knight_ng.reference import enumerate as oracle
from poker_knight_ng.reference.cards import _card_id, parse_cards
from poker_knight_ng.reference.evaluator import best_five
from poker_knight_ng.reference.enumerate import enumerate_unknown_opponent


def test_exact_flop_reduced_board_completion_matches_independent_checker(monkeypatch):
    hero_text, board_text = ("As", "Kd"), ("2s", "7h", "9d")
    hero, board = parse_cards(hero_text), parse_cards(board_text)
    remaining = oracle._remaining(hero + board)
    selected_runouts = tuple(independent_combinations(remaining, 2))[:2]

    def bounded_combinations(cards, count):
        options = tuple(cards)
        if count == 2 and len(options) == len(remaining):
            return iter(selected_runouts)
        return independent_combinations(options, count)

    monkeypatch.setattr(oracle, "combinations", bounded_combinations)
    actual = enumerate_unknown_opponent(hero_text, board_text)

    rows = []
    for runout in selected_runouts:
        final_board = board + runout
        available = tuple(card for card in remaining if _card_id(card) not in {_card_id(item) for item in runout})
        for opponent in independent_combinations(available, 2):
            rows.append((best_five(hero + final_board).score, (best_five(opponent + final_board).score,)))
    expected = oracle._equity_from_rows(rows, 1)

    assert actual == expected
    assert actual.completed_trials == len(selected_runouts) * 990
    actual.validate(1)


@pytest.mark.skipif(os.environ.get("RUN_EXACT_FLOP") != "1", reason="release certification: evaluates 1,070,190 complete deals")
def test_exact_flop_has_frozen_integer_result_and_conservation():
    equity = enumerate_unknown_opponent(["As", "Kd"], ["2s", "7h", "9d"])
    assert (equity.completed_trials, equity.unique_wins, equity.tie_by_other_winners, equity.losses, equity.equity_share_units, equity.hero_category_counts) == (1070190, 554910, (11522, 0, 0, 0, 0, 0), 503758, 235481820, (443520, 522720, 89100, 14850, 0, 0, 0, 0, 0))
    equity.validate(1)
