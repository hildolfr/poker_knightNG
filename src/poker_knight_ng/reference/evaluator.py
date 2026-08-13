"""Direct-combination, transparent five-card poker evaluator."""

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, NamedTuple

from .cards import Card, _card_id, _card_identity, _card_token, parse_cards

CATEGORY_NAMES = (
    "high_card", "one_pair", "two_pair", "three_of_a_kind", "straight",
    "flush", "full_house", "four_of_a_kind", "straight_flush",
)


class HandScore(NamedTuple):
    """Immutable ADR 0002 rank key; ordinary tuple ordering is poker ordering."""

    category: int
    k1: int
    k2: int
    k3: int
    k4: int
    k5: int

    @property
    def key(self) -> tuple[int, int, int, int, int, int]:
        return (self.category, self.k1, self.k2, self.k3, self.k4, self.k5)

    @property
    def tiebreak(self) -> tuple[int, int, int, int, int]:
        return self[1:]


@dataclass(frozen=True)
class BestFive:
    score: HandScore
    cards: tuple[str, ...]
    subset_count: int


def _straight_high(ranks: tuple[int, ...]) -> int | None:
    unique = set(ranks)
    if len(unique) != 5:
        return None
    if unique == {14, 2, 3, 4, 5}:
        return 5
    high = max(unique)
    return high if unique == set(range(high - 4, high + 1)) else None


def score_five(cards: Iterable[Card | str]) -> HandScore:
    """Score exactly five distinct canonical cards with an ADR 0002 key."""
    hand = parse_cards(cards)
    if len(hand) != 5:
        raise ValueError("score_five requires exactly five unique cards")
    identities = tuple(_card_identity(card) for card in hand)
    ranks = tuple(rank for rank, _ in identities)
    descending = tuple(sorted(ranks, reverse=True))
    straight_high = _straight_high(ranks)
    flush = len({suit for _, suit in identities}) == 1
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)

    if flush and straight_high is not None:
        return HandScore(8, straight_high, 0, 0, 0, 0)
    if groups[0][0] == 4:
        quad = groups[0][1]
        kicker = next(rank for rank in ranks if rank != quad)
        return HandScore(7, quad, kicker, 0, 0, 0)
    if [count for count, _ in groups] == [3, 2]:
        return HandScore(6, groups[0][1], groups[1][1], 0, 0, 0)
    if flush:
        return HandScore(5, *descending)
    if straight_high is not None:
        return HandScore(4, straight_high, 0, 0, 0, 0)
    if groups[0][0] == 3:
        trip = groups[0][1]
        kickers = tuple(sorted((rank for rank in ranks if rank != trip), reverse=True))
        return HandScore(3, trip, kickers[0], kickers[1], 0, 0)
    pairs = tuple(sorted((rank for rank, count in counts.items() if count == 2), reverse=True))
    if len(pairs) == 2:
        kicker = next(rank for rank in ranks if rank not in pairs)
        return HandScore(2, pairs[0], pairs[1], kicker, 0, 0)
    if len(pairs) == 1:
        pair = pairs[0]
        kickers = tuple(sorted((rank for rank in ranks if rank != pair), reverse=True))
        return HandScore(1, pair, kickers[0], kickers[1], kickers[2], 0)
    return HandScore(0, *descending)


def best_five(cards: Iterable[Card | str]) -> BestFive:
    """Evaluate every five-card subset of five to seven distinct cards."""
    hand = parse_cards(cards)
    if not 5 <= len(hand) <= 7:
        raise ValueError("best_five requires five to seven unique cards")
    candidates = []
    for subset in combinations(hand, 5):
        canonical = tuple(sorted(subset, key=_card_id))
        candidates.append((score_five(canonical), tuple(_card_token(card) for card in canonical)))
    score, selected = max(candidates, key=lambda item: (item[0], item[1]))
    return BestFive(score=score, cards=selected, subset_count=len(candidates))
