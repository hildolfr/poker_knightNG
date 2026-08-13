from collections import Counter
from itertools import combinations

from poker_knight_ng.reference.cards import CARD_DECK
from poker_knight_ng.reference.evaluator import CATEGORY_NAMES, score_five

EXPECTED_CATEGORY_COUNTS = {
    "high_card": 1_302_540,
    "one_pair": 1_098_240,
    "two_pair": 123_552,
    "three_of_a_kind": 54_912,
    "straight": 10_200,
    "flush": 5_108,
    "full_house": 3_744,
    "four_of_a_kind": 624,
    "straight_flush": 40,
}


def test_all_five_card_hands_have_authoritative_category_counts():
    counts = Counter()
    total = 0
    for hand in combinations(CARD_DECK, 5):
        score = score_five(hand)
        assert score.category in range(len(CATEGORY_NAMES))
        counts[CATEGORY_NAMES[score.category]] += 1
        total += 1
    assert set(counts) == set(EXPECTED_CATEGORY_COUNTS)
    assert counts == EXPECTED_CATEGORY_COUNTS
    assert total == 2_598_960
    assert sum(counts.values()) == 2_598_960
