"""Pure deterministic CPU Monte Carlo accumulation over ADR 0003 simulation IDs."""

from dataclasses import dataclass
from hmac import compare_digest
from typing import Final

from .cards import CARD_DECK
from .dealer import canonical_case_hash, deal_cpu
from .evaluator import best_five

UNITS: Final = 420
UINT64_MAX: Final = (1 << 64) - 1
UINT32_MAX: Final = (1 << 32) - 1
MAX_REQUESTED_TRIALS: Final = UINT64_MAX // UNITS


def _uint(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= UINT64_MAX:
        raise ValueError(f"{name} must be a non-bool uint64")
    return value


def _rejection_count(value: object) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("rejection_count must be a non-bool nonnegative built-in integer")
    return value


def _draw_count(opponent_count: object, board_card_count: object) -> int:
    if isinstance(opponent_count, bool) or not isinstance(opponent_count, int) or not 1 <= opponent_count <= 6:
        raise ValueError("opponent_count must be a non-bool integer from 1 through 6")
    if type(board_card_count) is not int or board_card_count not in (0, 3, 4, 5):
        raise ValueError("board_card_count must be a non-bool built-in integer of 0, 3, 4, or 5")
    return 5 - board_card_count + 2 * opponent_count


def _validate_case(seed: object, hero_card_ids: object, board_card_ids: object, opponent_count: object, requested_trials: object) -> tuple[int, tuple[int, int], tuple[int, ...], int, int, bytes]:
    seed_int = _uint(seed, "seed")
    trials = _uint(requested_trials, "requested_trials")
    if not 1 <= trials <= MAX_REQUESTED_TRIALS:
        raise ValueError(f"requested_trials must be from 1 through {MAX_REQUESTED_TRIALS}")
    # The dealer's canonical boundary owns exact case-ID validation and sorting.
    case_hash = canonical_case_hash(hero_card_ids, board_card_ids, opponent_count)
    if type(hero_card_ids) is not tuple or len(hero_card_ids) != 2:
        raise ValueError("hero_card_ids must be a 2-card tuple")
    if type(board_card_ids) is not tuple or len(board_card_ids) not in (0, 3, 4, 5):
        raise ValueError("board_card_ids must be a 0, 3, 4, or 5-card tuple")
    return seed_int, tuple(sorted(hero_card_ids)), tuple(sorted(board_card_ids)), opponent_count, trials, case_hash


@dataclass(frozen=True)
class MonteCarloResult:
    """Immutable aggregate with uint64 authoritative counters and an unbounded rejection diagnostic."""

    completed_trials: int
    unique_wins: int
    tie_by_other_winners: tuple[int, int, int, int, int, int]
    losses: int
    equity_share_units: int
    hero_category_counts: tuple[int, int, int, int, int, int, int, int, int]
    rejection_count: int

    @property
    def ties(self) -> int:
        return sum(_raw_monte_carlo(self)[2])

    def validate(self, opponent_count: object, requested_trials: object, board_card_count: object) -> None:
        draw_count = _draw_count(opponent_count, board_card_count)
        completed, wins, ties, losses, units, categories, rejection_count = _raw_monte_carlo(self)
        if _uint(requested_trials, "requested_trials") != completed:
            raise ValueError("completed_trials must equal requested_trials")
        if rejection_count > completed * draw_count * UINT32_MAX:
            raise ValueError("rejection_count exceeds completed-trial topology maximum")
        if any(ties[index] for index in range(opponent_count, 6)):
            raise ValueError("impossible tie bin is nonzero")
        if wins + sum(ties) + losses != completed:
            raise ValueError("wins, ties, and losses must equal completed_trials")
        if sum(categories) != completed:
            raise ValueError("hero category counts must equal completed_trials")
        expected_units = UNITS * wins + sum(UNITS // (index + 2) * count for index, count in enumerate(ties))
        if units != expected_units:
            raise ValueError("equity share units do not match outcome accounting")


def _raw_monte_carlo(result: object) -> tuple[int, int, tuple[int, int, int, int, int, int], int, int, tuple[int, int, int, int, int, int, int, int, int], int]:
    """Read public aggregate data without virtual dispatch from an exact instance only."""
    if type(result) is not MonteCarloResult:
        raise ValueError("Monte Carlo result must be an exact MonteCarloResult instance")
    completed, wins, ties, losses, units, categories, rejection_count = tuple(object.__getattribute__(result, name) for name in (
        "completed_trials", "unique_wins", "tie_by_other_winners", "losses", "equity_share_units", "hero_category_counts", "rejection_count",
    ))
    for name, value in zip(
        ("completed_trials", "unique_wins", "losses", "equity_share_units"),
        (completed, wins, losses, units),
    ):
        _uint(value, name)
    _rejection_count(rejection_count)
    if type(ties) is not tuple or len(ties) != 6 or type(categories) is not tuple or len(categories) != 9:
        raise ValueError("accounting bins have invalid shape")
    for value in ties + categories:
        _uint(value, "accounting bin")
    return completed, wins, ties, losses, units, categories, rejection_count


def _run_trial(*, seed: int, hero_card_ids: tuple[int, int], board_card_ids: tuple[int, ...], opponent_count: int, simulation_id: int, case_hash: bytes | None = None) -> MonteCarloResult:
    """Evaluate one simulation ID, retained privately for direct prefix tests."""
    digest = case_hash if case_hash is not None else canonical_case_hash(hero_card_ids, board_card_ids, opponent_count)
    deal = deal_cpu(seed=seed, canonical_case_hash=digest, hero_card_ids=hero_card_ids, board_card_ids=board_card_ids, opponent_count=opponent_count, simulation_id=simulation_id)
    hero = tuple(CARD_DECK[card] for card in hero_card_ids)
    board = tuple(CARD_DECK[card] for card in deal.board_card_ids)
    hero_score = best_five(hero + board).score
    scores = tuple(best_five(tuple(CARD_DECK[card] for card in holes) + board).score for holes in deal.opponent_hole_card_ids)
    ties = [0] * 6
    categories = [0] * 9
    categories[hero_score.category] = 1
    maximum = max(scores)
    if maximum > hero_score:
        wins = losses = units = 0
        losses = 1
    elif maximum == hero_score:
        wins = losses = 0
        equals = sum(score == hero_score for score in scores)
        ties[equals - 1] = 1
        units = UNITS // (equals + 1)
    else:
        wins, losses, units = 1, 0, UNITS
    result = MonteCarloResult(1, wins, tuple(ties), losses, units, tuple(categories), deal.trace.rejection_count)
    result.validate(opponent_count, 1, len(board_card_ids))
    return result


def run_cpu_monte_carlo(*, seed: object, hero_card_ids: object, board_card_ids: object, opponent_count: object, requested_trials: object, replay_case_hash: object | None = None) -> MonteCarloResult:
    """Run every logical ID from zero through ``requested_trials - 1`` exactly once."""
    seed_int, hero, board, opponents, trials, digest = _validate_case(seed, hero_card_ids, board_card_ids, opponent_count, requested_trials)
    if replay_case_hash is not None:
        if type(replay_case_hash) is not bytes or len(replay_case_hash) != 32 or not compare_digest(replay_case_hash, digest):
            raise ValueError("replay hash does not match canonical case hash")
    totals = [0, 0, [0] * 6, 0, 0, [0] * 9, 0]
    for simulation_id in range(trials):
        row = _run_trial(seed=seed_int, hero_card_ids=hero, board_card_ids=board, opponent_count=opponents, simulation_id=simulation_id, case_hash=digest)
        totals[0] += row.completed_trials
        totals[1] += row.unique_wins
        totals[3] += row.losses
        totals[4] += row.equity_share_units
        totals[6] += row.rejection_count
        for index in range(6): totals[2][index] += row.tie_by_other_winners[index]
        for index in range(9): totals[5][index] += row.hero_category_counts[index]
    result = MonteCarloResult(totals[0], totals[1], tuple(totals[2]), totals[3], totals[4], tuple(totals[5]), totals[6])
    result.validate(opponents, trials, len(board))
    return result
