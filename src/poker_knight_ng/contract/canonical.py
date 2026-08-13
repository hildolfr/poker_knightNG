"""ADR 0003 canonical case encoding."""
from hashlib import sha256
from .errors import problem

CARD_RANKS = "23456789TJQKA"
CARD_SUITS = "shdc"
_CASE_LABEL = b"poker-knight-ng/case/v1"


def card_id(card: str) -> int:
    return CARD_SUITS.index(card[1]) * 13 + CARD_RANKS.index(card[0])


def canonical_case_bytes(request: object) -> bytes:
    # Import locally to avoid the models/canonical import cycle.
    from .models import EquityRequest
    if not isinstance(request, EquityRequest):
        raise problem("UNSUPPORTED_REQUEST")
    # Frozen dataclasses can still be altered through object.__setattr__.
    # Validate before card_id so malformed tokens never reach indexing logic.
    request.validate()
    hero = sorted(card_id(card) for card in request.hero_cards)
    board = sorted(card_id(card) for card in request.board_cards)
    return bytes([len(_CASE_LABEL)]) + _CASE_LABEL + bytes([2, *hero, len(board), *board, request.opponent_count])


def canonical_case_hash(request: object) -> str:
    return sha256(canonical_case_bytes(request)).hexdigest()
