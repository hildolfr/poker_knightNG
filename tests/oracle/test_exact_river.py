from poker_knight_ng.reference.enumerate import enumerate_unknown_opponent


def test_exact_river_is_input_order_invariant_and_has_frozen_integer_result():
    first = enumerate_unknown_opponent(["As", "Kd"], ["2s", "7h", "9d", "Tc", "Jc"])
    second = enumerate_unknown_opponent(["Kd", "As"], ["Jc", "Tc", "9d", "7h", "2s"])
    assert first == second
    assert (first.completed_trials, first.unique_wins, first.tie_by_other_winners, first.losses, first.equity_share_units, first.hero_category_counts) == (990, 268, (9, 0, 0, 0, 0, 0), 713, 114450, (990, 0, 0, 0, 0, 0, 0, 0, 0))
