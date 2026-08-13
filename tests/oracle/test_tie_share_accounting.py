import pytest

from poker_knight_ng.reference.enumerate import ExactEquity, ExactOracleError, evaluate_terminal


@pytest.mark.parametrize("opponents", [
    [["2c", "3d"]],
    [["2c", "3d"], ["4h", "5c"]],
    [["2c", "3d"], ["4h", "5c"], ["6d", "7h"]],
    [["2c", "3d"], ["4h", "5c"], ["6d", "7h"], ["8c", "9d"]],
    [["2c", "3d"], ["4h", "5c"], ["6d", "7h"], ["8c", "9d"], ["Th", "Jc"]],
    [["2c", "3d"], ["4h", "5c"], ["6d", "7h"], ["8c", "9d"], ["Th", "Jc"], ["Qd", "Kh"]],
])
def test_shared_royal_board_accounts_every_split_size(opponents):
    equity = evaluate_terminal(["Ac", "Ad"], ["As", "Ks", "Qs", "Js", "Ts"], opponents)
    equals = len(opponents)
    assert equity.tie_by_other_winners[equals - 1] == 1
    assert equity.equity_share_units == 420 // (equals + 1)
    equity.validate(equals)


def test_lower_opponent_does_not_change_count_of_tied_maximum():
    equity = evaluate_terminal(["Tc", "3c"], ["As", "Ks", "Qs", "Js", "2d"], [["Th", "4c"], ["8h", "8d"]])
    assert (equity.completed_trials, equity.unique_wins, equity.tie_by_other_winners, equity.losses, equity.equity_share_units, equity.hero_category_counts) == (1, 0, (1, 0, 0, 0, 0, 0), 0, 210, (0, 0, 0, 0, 1, 0, 0, 0, 0))


def test_adr_loss_case_is_accounted_as_a_full_house_loss():
    equity = evaluate_terminal(["Ks", "Kd"], ["As", "Ah", "Ad", "Kc", "Qd"], [["Ac", "Qc"]])
    assert (equity.completed_trials, equity.unique_wins, equity.tie_by_other_winners, equity.losses, equity.equity_share_units, equity.hero_category_counts) == (1, 0, (0, 0, 0, 0, 0, 0), 1, 0, (0, 0, 0, 0, 0, 0, 1, 0, 0))


def test_equity_validation_rejects_boolean_counter_and_impossible_bin():
    with pytest.raises(ExactOracleError):
        ExactEquity(True, 0, (0, 0, 0, 0, 0, 0), 0, 0, (0,) * 9).validate(1)
    with pytest.raises(ExactOracleError):
        ExactEquity(1, 0, (0, 1, 0, 0, 0, 0), 0, 140, (1,) + (0,) * 8).validate(1)


@pytest.mark.parametrize("equity", [
    ExactEquity(True, 0, (0,) * 6, 0, 0, (0,) * 9),
    ExactEquity(1.0, 0, (0,) * 6, 1, 0, (0,) * 9),
    ExactEquity(2**64, 0, (0,) * 6, 1, 0, (0,) * 9),
    ExactEquity(1, 0, [0] * 6, 1, 0, (0,) * 9),
    ExactEquity(1, 0, (0,) * 5, 1, 0, (0,) * 9),
    ExactEquity(1, 0, (0,) * 6, 1, 0, [0] * 9),
    ExactEquity(1, 0, (0,) * 6, 1, 0, (0,) * 8),
])
def test_equity_validation_rejects_invalid_direct_construction(equity):
    with pytest.raises(ExactOracleError):
        equity.validate(1)


def test_equity_validation_rejects_forged_and_malicious_subclass_instances():
    equity = ExactEquity(1, 1, (0,) * 6, 0, 420, (1,) + (0,) * 8)
    object.__setattr__(equity, "unique_wins", True)
    with pytest.raises(ExactOracleError):
        equity.validate(1)

    class MaliciousEquity(ExactEquity):
        def __getattribute__(self, name):
            if name in {"ties", "validate", "unique_wins"}:
                return 1 if name != "validate" else lambda opponent_count: None
            return super().__getattribute__(name)

    malicious = MaliciousEquity(1, 1, (0,) * 6, 0, 420, (1,) + (0,) * 8)
    with pytest.raises(ExactOracleError):
        ExactEquity.validate(malicious, 1)
    with pytest.raises(ExactOracleError):
        _ = ExactEquity.ties.fget(malicious)
