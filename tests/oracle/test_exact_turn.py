import poker_knight_ng.reference.enumerate as oracle
from poker_knight_ng.reference.enumerate import enumerate_unknown_opponent


def test_exact_turn_has_frozen_integer_result_and_conservation():
    equity = enumerate_unknown_opponent(["As", "Kd"], ["2s", "7h", "9d", "Tc"])
    assert (equity.completed_trials, equity.unique_wins, equity.tie_by_other_winners, equity.losses, equity.equity_share_units, equity.hero_category_counts) == (45540, 19317, (396, 0, 0, 0, 0, 0), 25827, 8196300, (27720, 17820, 0, 0, 0, 0, 0, 0, 0))
    equity.validate(1)


def test_unknown_opponent_score_cache_reuses_canonical_seven_card_sets(monkeypatch):
    calls = 0
    original = oracle.best_five

    def counted(cards):
        nonlocal calls
        calls += 1
        return original(cards)

    monkeypatch.setattr(oracle, "best_five", counted)
    equity = oracle.enumerate_unknown_opponent(["As", "Kd"], ["2s", "7h", "9d", "Tc"])
    assert (equity.completed_trials, equity.unique_wins, equity.tie_by_other_winners, equity.losses, equity.equity_share_units, equity.hero_category_counts) == (45540, 19317, (396, 0, 0, 0, 0, 0), 25827, 8196300, (27720, 17820, 0, 0, 0, 0, 0, 0, 0))
    assert calls < equity.completed_trials
