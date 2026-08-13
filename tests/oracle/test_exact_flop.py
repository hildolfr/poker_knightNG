import os

import pytest

from poker_knight_ng.reference.enumerate import enumerate_unknown_opponent


@pytest.mark.skipif(os.environ.get("RUN_EXACT_FLOP") != "1", reason="release certification: evaluates 1,070,190 complete deals")
def test_exact_flop_has_frozen_integer_result_and_conservation():
    equity = enumerate_unknown_opponent(["As", "Kd"], ["2s", "7h", "9d"])
    assert (equity.completed_trials, equity.unique_wins, equity.tie_by_other_winners, equity.losses, equity.equity_share_units, equity.hero_category_counts) == (1070190, 554910, (11522, 0, 0, 0, 0, 0), 503758, 235481820, (443520, 522720, 89100, 14850, 0, 0, 0, 0, 0))
    equity.validate(1)
