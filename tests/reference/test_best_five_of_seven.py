import itertools

import pytest

from poker_knight_ng.reference.evaluator import (
    CATEGORY_NAMES,
    HandScore,
    best_five,
    score_five,
)


def score(cards):
    return score_five(cards).key


@pytest.mark.parametrize(
    ("cards", "expected"),
    [
        ("As 2h 3d 4c 5s", (4, 5, 0, 0, 0, 0)),
        ("2s 3h 4d 5c 6s", (4, 6, 0, 0, 0, 0)),
        ("Ah Kh 9h 5h 2h", (5, 14, 13, 9, 5, 2)),
        ("As Ad Kc Kd Qh", (2, 14, 13, 12, 0, 0)),
        ("As Ah Ad Ac Kd", (7, 14, 13, 0, 0, 0)),
        ("Ts Js Qs Ks As", (8, 14, 0, 0, 0, 0)),
        ("As 2s 3s 4s 5s", (8, 5, 0, 0, 0, 0)),
        ("As Ah Ad Kc Qd", (3, 14, 13, 12, 0, 0)),
        ("As Ad Kc Qd Jh", (1, 14, 13, 12, 11, 0)),
        ("As Kd Qh Jc 9s", (0, 14, 13, 12, 11, 9)),
        ("As Ah Ad Kc Kd", (6, 14, 13, 0, 0, 0)),
    ],
)
def test_adr_rank_vectors(cards, expected):
    assert score(cards.split()) == expected


def test_category_ordering_has_all_adjacent_boundaries():
    hands = [
        "As Kd Qh Jc 9s", "As Ad Kc Qd Jh", "As Ad Kc Kd Qh",
        "As Ah Ad Kc Qd", "2s 3h 4d 5c 6s", "Ah Kh 9h 5h 2h",
        "As Ah Ad Kc Kd", "As Ah Ad Ac Kd", "Ts Js Qs Ks As",
    ]
    scores = [score(hand.split()) for hand in hands]
    assert [key[0] for key in scores] == list(range(9))
    assert all(left < right for left, right in itertools.pairwise(scores))
    assert tuple(CATEGORY_NAMES) == ("high_card", "one_pair", "two_pair", "three_of_a_kind", "straight", "flush", "full_house", "four_of_a_kind", "straight_flush")


def test_rank_details_and_permutations_are_comparable():
    assert score("As Ah Ad Ac Kd".split()) > score("As Ah Ad Ac Qd".split())
    assert score("Ah Kh 9h 5h 2h".split()) > score("Ah Qh Jh 8h 3h".split())
    original = "As Ad Kc Kd Qh".split()
    assert {score(permutation) for permutation in itertools.permutations(original)} == {(2, 14, 13, 12, 0, 0)}


def test_six_high_beats_wheel_for_straights_and_straight_flushes():
    assert score("2s 3h 4d 5c 6s".split()) > score("As 2h 3d 4c 5s".split())
    assert score("2s 3s 4s 5s 6s".split()) > score("As 2s 3s 4s 5s".split())


@pytest.mark.parametrize("cards", ["As Ks Qs Js Ts 2c 3d", "As Ah Ad Kc Kd 2s 2h"])
def test_best_five_selects_deterministically_regardless_of_input_order(cards):
    first = best_five(cards.split())
    second = best_five(tuple(reversed(cards.split())))
    assert first.score == second.score
    assert first.cards == second.cards


def test_best_five_uses_21_subsets_and_board_playing_key_is_equal():
    hero = best_five("As Ks Qs Js Ts 2c 3d".split())
    opponent = best_five("As Ks Qs Js Ts 4h 5c".split())
    assert hero.score.key == opponent.score.key == (8, 14, 0, 0, 0, 0)
    assert hero.subset_count == 21
    assert hero.cards == ("Ts", "Js", "Qs", "Ks", "As")


@pytest.mark.parametrize("cards", ["As Ks Qs Js", "As Ks Qs Js Ts 2c 3d 4h", "As As Kd Qh Jc"])
def test_evaluator_rejects_bad_sizes_and_duplicates(cards):
    with pytest.raises(ValueError):
        best_five(cards.split())


def test_score_five_requires_exactly_five():
    with pytest.raises(ValueError):
        score_five("As Ks Qs Js".split())
