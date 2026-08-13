"""Exact ADR 0003 bounded selection and deterministic CPU dealing.

The public functions here deliberately model one logical simulation only.  They
never shuffle a shared deck: each draw operates on the canonical remaining-card
prefix and uses swap-with-tail removal.  Per-slot trace data records rejection
work so a caller can aggregate it without inferring retry behavior from timing.
"""

from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
from typing import Callable, Final

from .rng import UINT32_MAX, adr0003_candidate

RNG_REJECTION_EXHAUSTED: Final = "RNG_REJECTION_EXHAUSTED"
CANONICAL_CASE_HASH_MISMATCH: Final = "CANONICAL_CASE_HASH_MISMATCH"
_UINT32_DOMAIN: Final = 1 << 32
_CASE_LABEL: Final = b"poker-knight-ng/case/v1"


class RngRejectionExhausted(RuntimeError):
    """The final permitted ADR 0003 candidate rejected."""

    code: Final[str] = RNG_REJECTION_EXHAUSTED

    def __init__(self, simulation_id: int, draw_slot: int) -> None:
        super().__init__(self.code)
        self.simulation_id = simulation_id
        self.draw_slot = draw_slot
        self.final_attempt = UINT32_MAX


class CanonicalCaseHashMismatch(ValueError):
    """A supplied replay hash does not attest to the validated deal inputs."""

    code: Final[str] = CANONICAL_CASE_HASH_MISMATCH

    def __init__(self) -> None:
        super().__init__(self.code)


@dataclass(frozen=True)
class SlotTrace:
    """Secrets-safe deterministic work metadata for one accepted selection."""

    draw_slot: int
    active_range: int
    acceptance_limit: int
    rejection_count: int
    final_attempt: int
    accepted_word: int


@dataclass(frozen=True)
class Selection:
    """One swap-with-tail selection and its active-prefix remainder."""

    card_id: int
    deck_after: tuple[int, ...]
    trace: SlotTrace


@dataclass(frozen=True)
class DealTrace:
    """Ordered per-slot traces; rejected candidates total exactly their sum."""

    slots: tuple[SlotTrace, ...]

    @property
    def rejection_count(self) -> int:
        return sum(slot.rejection_count for slot in self.slots)


@dataclass(frozen=True)
class CpuDeal:
    """One complete logical simulation deal in ADR 0003 slot order."""

    simulation_id: int
    known_card_ids: tuple[int, ...]
    dealt_card_ids: tuple[int, ...]
    board_card_ids: tuple[int, ...]
    opponent_hole_card_ids: tuple[tuple[int, int], ...]
    trace: DealTrace


def _uint(value: object, bits: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < (1 << bits):
        raise ValueError(f"{name} must be an unsigned {bits}-bit integer")
    return value


def _card_id(value: object, name: str) -> int:
    card = _uint(value, 6, name)
    if card > 51:
        raise ValueError(f"{name} must be a card ID from 0 through 51")
    return card


def _card_ids(value: object, expected_length: int, name: str) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != expected_length:
        raise ValueError(f"{name} must be a {expected_length}-card tuple")
    return tuple(_card_id(card, f"{name}[{index}]") for index, card in enumerate(value))


def _validated_case_ids(
    hero_card_ids: object, board_card_ids: object, opponent_count: object
) -> tuple[tuple[int, ...], tuple[int, ...], int]:
    hero = _card_ids(hero_card_ids, 2, "hero_card_ids")
    if type(board_card_ids) is not tuple or len(board_card_ids) not in (0, 3, 4, 5):
        raise ValueError("board_card_ids must be a 0, 3, 4, or 5-card tuple")
    board = tuple(_card_id(card, f"board_card_ids[{index}]") for index, card in enumerate(board_card_ids))
    opponents = _uint(opponent_count, 3, "opponent_count")
    if not 1 <= opponents <= 6:
        raise ValueError("opponent_count must be from 1 through 6")
    if len(set(hero + board)) != len(hero) + len(board):
        raise ValueError("known card IDs must be distinct")
    return tuple(sorted(hero)), tuple(sorted(board)), opponents


def canonical_case_bytes(
    hero_card_ids: object, board_card_ids: object, opponent_count: object
) -> bytes:
    """Return ADR 0003 canonical bytes from validated raw canonical card IDs."""
    hero, board, opponents = _validated_case_ids(hero_card_ids, board_card_ids, opponent_count)
    return bytes([len(_CASE_LABEL)]) + _CASE_LABEL + bytes([2, *hero, len(board), *board, opponents])


def canonical_case_hash(
    hero_card_ids: object, board_card_ids: object, opponent_count: object
) -> bytes:
    """Return the exact SHA-256 replay digest for an ADR 0003 case."""
    return sha256(canonical_case_bytes(hero_card_ids, board_card_ids, opponent_count)).digest()


def _limit(active_range: int) -> int:
    if not 1 <= active_range <= 52:
        raise ValueError("active range must be from 1 through 52")
    return (_UINT32_DOMAIN // active_range) * active_range


def select_unbiased(
    *,
    remaining_card_ids: object,
    simulation_id: object,
    draw_slot: object,
    candidate_lane0: Callable[[int], object],
    initial_attempt: object = 0,
) -> Selection:
    """Select one card with ADR 0003 rejection sampling and swap-with-tail.

    ``candidate_lane0`` is intentionally injected for deterministic unit tests;
    production dealing binds it to Philox lane 0.  A rejection never maps a
    word modulo the active range and never changes the draw slot.
    """
    if type(remaining_card_ids) is not tuple:
        raise ValueError("remaining_card_ids must be a tuple")
    deck = tuple(_card_id(card, f"remaining_card_ids[{index}]") for index, card in enumerate(remaining_card_ids))
    if len(set(deck)) != len(deck):
        raise ValueError("remaining_card_ids must be distinct")
    simulation = _uint(simulation_id, 64, "simulation_id")
    slot = _uint(draw_slot, 32, "draw_slot")
    if slot > 16:
        raise ValueError("draw_slot must be at most 16")
    attempt = _uint(initial_attempt, 32, "initial_attempt")
    if not callable(candidate_lane0):
        raise ValueError("candidate_lane0 must be callable")
    active_range = len(deck)
    limit = _limit(active_range)
    rejections = 0
    while True:
        word = _uint(candidate_lane0(attempt), 32, "candidate lane 0")
        if word < limit:
            index = word % active_range
            card = deck[index]
            # The old tail becomes irrelevant after the active range shrinks.
            after = list(deck)
            after[index] = after[-1]
            after.pop()
            return Selection(card, tuple(after), SlotTrace(slot, active_range, limit, rejections, attempt, word))
        if attempt == UINT32_MAX:
            raise RngRejectionExhausted(simulation, slot)
        rejections += 1
        attempt += 1


def deal_cpu(
    *,
    seed: object,
    canonical_case_hash: object,
    hero_card_ids: object,
    board_card_ids: object,
    opponent_count: object,
    simulation_id: object,
) -> CpuDeal:
    """Deal missing board cards then opponent holes, without replacement."""
    seed_int = _uint(seed, 64, "seed")
    if type(canonical_case_hash) is not bytes or len(canonical_case_hash) != 32:
        raise ValueError("canonical_case_hash must be exactly 32 bytes")
    simulation = _uint(simulation_id, 64, "simulation_id")
    hero, board, opponents = _validated_case_ids(hero_card_ids, board_card_ids, opponent_count)
    derived_case_hash = sha256(
        bytes([len(_CASE_LABEL)]) + _CASE_LABEL + bytes([2, *hero, len(board), *board, opponents])
    ).digest()
    if not compare_digest(canonical_case_hash, derived_case_hash):
        raise CanonicalCaseHashMismatch()
    known = hero + board
    draw_count = 5 - len(board) + 2 * opponents
    if draw_count > 17:
        raise ValueError("deal capacity is unsupported")
    deck = tuple(card for card in range(52) if card not in set(known))
    traces: list[SlotTrace] = []
    dealt: list[int] = []
    for slot in range(draw_count):
        selection = select_unbiased(
            remaining_card_ids=deck,
            simulation_id=simulation,
            draw_slot=slot,
            candidate_lane0=lambda attempt, slot=slot: adr0003_candidate(
                seed_int, derived_case_hash, simulation, slot, attempt
            )[0],
        )
        deck = selection.deck_after
        dealt.append(selection.card_id)
        traces.append(selection.trace)
    complete_board = board + tuple(dealt[: 5 - len(board)])
    holes_start = 5 - len(board)
    holes = tuple((dealt[holes_start + 2 * opponent], dealt[holes_start + 2 * opponent + 1]) for opponent in range(opponents))
    return CpuDeal(simulation, known, tuple(dealt), complete_board, holes, DealTrace(tuple(traces)))
