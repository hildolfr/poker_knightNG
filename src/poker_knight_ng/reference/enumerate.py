"""Exact, integer-only Hold'em outcome enumeration using the reference evaluator."""

from dataclasses import dataclass
from itertools import combinations, islice
from typing import Iterable, TypeVar

from .cards import CARD_DECK, Card, _card_id, parse_cards
from .evaluator import best_five

_UNITS = 420
_MAX = (1 << 64) - 1
_T = TypeVar("_T")


class ExactOracleError(ValueError):
    """The Hold'em topology or integer accounting is invalid."""


def _uint(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX:
        raise ExactOracleError(f"{name} must be a non-bool uint64")
    return value


@dataclass(frozen=True)
class ExactEquity:
    completed_trials: int
    unique_wins: int
    tie_by_other_winners: tuple[int, int, int, int, int, int]
    losses: int
    equity_share_units: int
    hero_category_counts: tuple[int, int, int, int, int, int, int, int, int]

    @property
    def ties(self) -> int:
        return sum(_raw_equity(self)[2])

    def validate(self, opponent_count: int) -> None:
        if isinstance(opponent_count, bool) or not isinstance(opponent_count, int) or not 1 <= opponent_count <= 6:
            raise ExactOracleError("opponent_count must be a non-bool integer from 1 through 6")
        completed, wins, ties, losses, units, categories = _raw_equity(self)
        if any(ties[index] for index in range(opponent_count, 6)):
            raise ExactOracleError("impossible tie bin is nonzero")
        if wins + sum(ties) + losses != completed:
            raise ExactOracleError("wins, ties, and losses must equal completed_trials")
        if sum(categories) != completed:
            raise ExactOracleError("hero category counts must equal completed_trials")
        expected_units = _UNITS * wins + sum(_UNITS // (index + 2) * count for index, count in enumerate(ties))
        if units != expected_units:
            raise ExactOracleError("equity share units do not match outcome accounting")


def _raw_equity(result: object) -> tuple[int, int, tuple[int, int, int, int, int, int], int, int, tuple[int, int, int, int, int, int, int, int, int]]:
    """Read public model data without virtual dispatch from an exact instance only."""
    if type(result) is not ExactEquity:
        raise ExactOracleError("equity result must be an exact ExactEquity instance")
    completed, wins, ties, losses, units, categories = tuple(object.__getattribute__(result, name) for name in (
        "completed_trials", "unique_wins", "tie_by_other_winners", "losses", "equity_share_units", "hero_category_counts",
    ))
    for name, value in zip(("completed_trials", "unique_wins", "losses", "equity_share_units"), (completed, wins, losses, units)):
        _uint(value, name)
    if type(ties) is not tuple or len(ties) != 6:
        raise ExactOracleError("tie_by_other_winners must be a tuple of six uint64 values")
    if type(categories) is not tuple or len(categories) != 9:
        raise ExactOracleError("hero_category_counts must be a tuple of nine uint64 values")
    for value in ties + categories:
        _uint(value, "accounting bin")
    return completed, wins, ties, losses, units, categories


def _bounded_tuple(values: Iterable[_T], label: str, maximum: int) -> tuple[_T, ...]:
    try:
        collected = tuple(islice(values, maximum + 1))
    except AssertionError:
        raise
    except Exception as error:
        raise ExactOracleError(f"{label} iteration failed") from error
    if len(collected) > maximum:
        raise ExactOracleError(f"{label} has too many cards")
    return collected


def _as_cards(cards: Iterable[Card | str], label: str, maximum: int) -> tuple[Card, ...]:
    return parse_cards(_bounded_tuple(cards, label, maximum))


def _topology(hero_cards, board_cards, opponent_holes, board_lengths: tuple[int, ...]):
    hero = _as_cards(hero_cards, "hero_cards", 2)
    board = _as_cards(board_cards, "board_cards", 5)
    if len(hero) != 2 or len(board) not in board_lengths:
        raise ExactOracleError("hero must have two cards and board has an unsupported length")
    opponents = _bounded_tuple(opponent_holes, "opponent_holes", 6)
    if not 1 <= len(opponents) <= 6:
        raise ExactOracleError("there must be one through six opponent two-card hands")
    opponents = tuple(_as_cards(hand, "opponent hand", 2) for hand in opponents)
    if any(len(hand) != 2 for hand in opponents):
        raise ExactOracleError("there must be one through six opponent two-card hands")
    all_cards = hero + board + tuple(card for hand in opponents for card in hand)
    if len({_card_id(card) for card in all_cards}) != len(all_cards):
        raise ExactOracleError("cards must be distinct across all players and board")
    return hero, board, opponents


def _equity_from_rows(rows, opponent_count: int) -> ExactEquity:
    wins = losses = units = total = 0
    ties = [0] * 6
    categories = [0] * 9
    for hero_score, opponent_scores in rows:
        total += 1
        categories[hero_score.category] += 1
        maximum = max(opponent_scores)
        if maximum > hero_score:
            losses += 1
        elif maximum == hero_score:
            equal = sum(score == hero_score for score in opponent_scores)
            ties[equal - 1] += 1
            units += _UNITS // (equal + 1)
        else:
            wins += 1
            units += _UNITS
    result = ExactEquity(total, wins, tuple(ties), losses, units, tuple(categories))
    result.validate(opponent_count)
    return result


def evaluate_terminal(hero_cards, board_cards, opponent_holes) -> ExactEquity:
    """Classify one fully specified Hold'em terminal state."""
    hero, board, opponents = _topology(hero_cards, board_cards, opponent_holes, (5,))
    hero_score = best_five(hero + board).score
    return _equity_from_rows(((hero_score, tuple(best_five(hand + board).score for hand in opponents)),), len(opponents))


def _remaining(used: tuple[Card, ...]) -> tuple[Card, ...]:
    ids = {_card_id(card) for card in used}
    return tuple(card for card in CARD_DECK if _card_id(card) not in ids)


def enumerate_unknown_opponent(hero_cards, board_cards) -> ExactEquity:
    """Enumerate every heads-up opponent holding after each board completion."""
    hero = _as_cards(hero_cards, "hero_cards", 2)
    board = _as_cards(board_cards, "board_cards", 5)
    if len(hero) != 2 or len(board) not in (3, 4, 5):
        raise ExactOracleError("hero must have two cards and board must contain three, four, or five cards")
    if len({_card_id(card) for card in hero + board}) != len(hero) + len(board):
        raise ExactOracleError("cards must be distinct across all players and board")
    missing = 5 - len(board)
    remaining = _remaining(hero + board)
    hero_score_cache: dict[tuple[int, ...], object] = {}
    opponent_score_cache: dict[tuple[int, ...], object] = {}

    def score_for(cards: tuple[Card, ...], cache: dict[tuple[int, ...], object]):
        key = tuple(sorted(_card_id(card) for card in cards))
        found = cache.get(key)
        if found is None:
            found = best_five(cards).score
            cache[key] = found
        return found

    def rows():
        for runout in combinations(remaining, missing):
            final_board = board + runout
            hero_score = score_for(hero + final_board, hero_score_cache)
            runout_ids = {_card_id(card) for card in runout}
            available = tuple(card for card in remaining if _card_id(card) not in runout_ids)
            for opponent in combinations(available, 2):
                yield hero_score, (score_for(opponent + final_board, opponent_score_cache),)
    return _equity_from_rows(rows(), 1)


def enumerate_fixed_holes(hero_cards, board_cards, opponent_holes) -> ExactEquity:
    """Enumerate all missing community-card combinations for fixed player holes."""
    hero, board, opponents = _topology(hero_cards, board_cards, opponent_holes, (0, 3, 4, 5))
    remaining = _remaining(hero + board + tuple(card for hand in opponents for card in hand))
    missing = 5 - len(board)
    def rows():
        for runout in combinations(remaining, missing):
            final_board = board + runout
            yield best_five(hero + final_board).score, tuple(best_five(hand + final_board).score for hand in opponents)
    return _equity_from_rows(rows(), len(opponents))
