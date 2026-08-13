from poker_knight_ng.reference.enumerate import enumerate_fixed_holes


def test_fixed_holes_turn_runouts_are_exact_and_order_invariant():
    first = enumerate_fixed_holes(["As", "Kd"], ["2s", "7h", "9d", "Tc"], [["Qh", "Jd"]])
    second = enumerate_fixed_holes(["Kd", "As"], ["Tc", "9d", "7h", "2s"], [["Jd", "Qh"]])
    assert first == second
    assert (first.completed_trials, first.unique_wins, first.tie_by_other_winners, first.losses, first.equity_share_units, first.hero_category_counts) == (44, 31, (0, 0, 0, 0, 0, 0), 13, 13020, (26, 18, 0, 0, 0, 0, 0, 0, 0))


def test_fixed_holes_complete_board_is_one_terminal_runout():
    equity = enumerate_fixed_holes(["As", "Kd"], ["2s", "7h", "9d", "Tc", "Jc"], [["Qh", "Jd"]])
    assert equity.completed_trials == 1
    equity.validate(1)
