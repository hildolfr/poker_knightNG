"""Canonical cards for the independent reference evaluator.

Evaluator inputs are restricted to ADR 0002 two-character ASCII tokens or
instances whose exact runtime type is :class:`Card`.  ``Card`` subclasses are
unsupported and rejected at this trust boundary: arbitrary subclasses can
override field access, equality, hashing, or identity properties.  Invalid
tokens, untrusted objects, forged fields, and duplicate card identities raise
:class:`ReferenceCardError`.
"""

from dataclasses import dataclass
from typing import Iterable

RANKS = "23456789TJQKA"
SUITS = "shdc"


class ReferenceCardError(ValueError):
    """A reference-evaluator card is malformed or duplicated."""


def _card_identity(card: "Card") -> tuple[int, int]:
    """Return validated raw ``(rank, suit)`` without virtual Card reads."""
    if type(card) is not Card:
        raise ReferenceCardError("card must be an exact Card instance, not a Card subclass")
    rank = object.__getattribute__(card, "rank")
    suit = object.__getattribute__(card, "suit")
    if isinstance(rank, bool) or not isinstance(rank, int) or not 2 <= rank <= 14:
        raise ReferenceCardError("card rank must be a non-bool integer from 2 through 14")
    if isinstance(suit, bool) or not isinstance(suit, int) or not 0 <= suit <= 3:
        raise ReferenceCardError("card suit must be a non-bool integer from 0 through 3")
    return rank, suit


def _validate_card_fields(card: "Card") -> None:
    """Raise unless ``card`` is an exact Card with canonical raw fields."""
    _card_identity(card)


def _card_id(card: "Card") -> int:
    """Return a validated canonical ID without virtual Card reads."""
    rank, suit = _card_identity(card)
    return suit * 13 + rank - 2


def _card_token(card: "Card") -> str:
    """Return a validated canonical token without virtual Card reads."""
    rank, suit = _card_identity(card)
    return RANKS[rank - 2] + SUITS[suit]


@dataclass(frozen=True, order=True)
class Card:
    rank: int
    suit: int

    def __post_init__(self) -> None:
        _validate_card_fields(self)

    def validate(self) -> None:
        """Raise if this instance is not a canonical reference card."""
        _validate_card_fields(self)

    @classmethod
    def parse(cls, token: object) -> "Card":
        if type(token) is not str or len(token) != 2:
            raise ReferenceCardError("card must be a canonical two-character token")
        rank_index = RANKS.find(token[0])
        suit_index = SUITS.find(token[1])
        if rank_index < 0 or suit_index < 0:
            raise ReferenceCardError("card must use canonical rank and suit characters")
        return cls(rank_index + 2, suit_index)

    @property
    def card_id(self) -> int:
        return _card_id(self)

    @property
    def token(self) -> str:
        return _card_token(self)


CARD_DECK = tuple(Card(rank, suit) for suit in range(4) for rank in range(2, 15))


def parse_cards(cards: Iterable[Card | str]) -> tuple[Card, ...]:
    """Parse trusted exact Cards or canonical strings and reject duplicate identities."""
    parsed: list[Card] = []
    for card in cards:
        if type(card) is Card:
            _validate_card_fields(card)
            parsed.append(card)
        elif type(card) is str:
            parsed.append(Card.parse(card))
        else:
            raise ReferenceCardError("card must be a canonical token or exact Card instance")
    identities = {_card_identity(card) for card in parsed}
    if len(identities) != len(parsed):
        raise ReferenceCardError("duplicate cards are not permitted")
    return tuple(parsed)
