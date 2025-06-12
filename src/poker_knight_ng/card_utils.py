"""Card representation utilities for GPU-accelerated poker calculations.

This module provides efficient card representation using integers 0-51 for GPU
operations, with conversion utilities for human-readable formats.

Card Encoding:
    card_value = rank * 4 + suit
    - Ranks: 2=0, 3=1, 4=2, 5=3, 6=4, 7=5, 8=6, 9=7, T=8, J=9, Q=10, K=11, A=12
    - Suits: clubs=0, diamonds=1, hearts=2, spades=3
    
Examples:
    2♣ = 0*4 + 0 = 0
    2♦ = 0*4 + 1 = 1
    A♠ = 12*4 + 3 = 51
"""

import numpy as np
from typing import Union, List, Tuple, Optional

# Unicode suit symbols
SUIT_SYMBOLS = ['♣', '♦', '♥', '♠']
SUIT_CHARS = ['c', 'd', 'h', 's']
RANK_CHARS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

# GPU-friendly constants (will be copied to constant memory)
RANKS_PER_SUIT = 13
SUITS_PER_DECK = 4
CARDS_PER_DECK = 52

# Rank and suit extraction bit operations
RANK_SHIFT = 2  # card >> 2 gives rank
SUIT_MASK = 0x3  # card & 0x3 gives suit


def card_to_int(card_str: str) -> int:
    """Convert card string to integer representation (0-51).
    
    Args:
        card_str: Card as string, e.g., "As", "Kh", "2c", "Td"
                 Case insensitive for suits
    
    Returns:
        Integer 0-51 representing the card
        
    Raises:
        ValueError: If card string is invalid
    """
    if len(card_str) != 2:
        raise ValueError(f"Invalid card string: {card_str}")
    
    rank_char = card_str[0].upper()
    suit_char = card_str[1].lower()
    
    try:
        rank = RANK_CHARS.index(rank_char)
    except ValueError:
        raise ValueError(f"Invalid rank: {rank_char}")
    
    try:
        suit = SUIT_CHARS.index(suit_char)
    except ValueError:
        raise ValueError(f"Invalid suit: {suit_char}")
    
    return rank * 4 + suit


def int_to_card(card_int: int, unicode: bool = True) -> str:
    """Convert integer representation to card string.
    
    Args:
        card_int: Integer 0-51 representing the card
        unicode: If True, use Unicode suit symbols (♠♥♦♣), else use letters (shdc)
    
    Returns:
        Card as string, e.g., "A♠" or "As"
        
    Raises:
        ValueError: If card integer is out of range
    """
    if not 0 <= card_int <= 51:
        raise ValueError(f"Invalid card integer: {card_int}")
    
    rank = card_int >> RANK_SHIFT
    suit = card_int & SUIT_MASK
    
    rank_char = RANK_CHARS[rank]
    suit_char = SUIT_SYMBOLS[suit] if unicode else SUIT_CHARS[suit]
    
    return rank_char + suit_char


def parse_hand(hand_str: str) -> List[int]:
    """Parse a hand string into list of card integers.
    
    Args:
        hand_str: Hand as string, e.g., "AsKs", "AhKd", "2c3c4c5c6c"
                 Cards must be 2 characters each
    
    Returns:
        List of card integers
        
    Raises:
        ValueError: If hand string is invalid or contains duplicates
    """
    if len(hand_str) % 2 != 0:
        raise ValueError(f"Hand string must have even length: {hand_str}")
    
    cards = []
    for i in range(0, len(hand_str), 2):
        card_str = hand_str[i:i+2]
        card_int = card_to_int(card_str)
        if card_int in cards:
            raise ValueError(f"Duplicate card in hand: {card_str}")
        cards.append(card_int)
    
    return cards


def hand_to_string(cards: List[int], unicode: bool = True) -> str:
    """Convert list of card integers to hand string.
    
    Args:
        cards: List of card integers
        unicode: If True, use Unicode suit symbols
    
    Returns:
        Hand as string
    """
    return ''.join(int_to_card(card, unicode) for card in cards)


def get_rank(card_int: int) -> int:
    """Extract rank from card integer (0-12)."""
    return card_int >> RANK_SHIFT


def get_suit(card_int: int) -> int:
    """Extract suit from card integer (0-3)."""
    return card_int & SUIT_MASK


def make_card(rank: int, suit: int) -> int:
    """Create card integer from rank and suit.
    
    Args:
        rank: Rank 0-12 (2 through Ace)
        suit: Suit 0-3 (clubs, diamonds, hearts, spades)
    
    Returns:
        Card integer 0-51
        
    Raises:
        ValueError: If rank or suit out of range
    """
    if not 0 <= rank <= 12:
        raise ValueError(f"Invalid rank: {rank}")
    if not 0 <= suit <= 3:
        raise ValueError(f"Invalid suit: {suit}")
    
    return rank * 4 + suit


def create_deck_mask(excluded_cards: Optional[List[int]] = None) -> np.ndarray:
    """Create a boolean mask for available cards in deck.
    
    This is useful for GPU operations where we need to efficiently
    sample from remaining cards.
    
    Args:
        excluded_cards: List of card integers to exclude (e.g., dealt cards)
    
    Returns:
        Boolean numpy array of length 52, True for available cards
    """
    mask = np.ones(52, dtype=np.bool_)
    if excluded_cards:
        mask[excluded_cards] = False
    return mask


def validate_cards(cards: List[int]) -> None:
    """Validate a list of card integers.
    
    Args:
        cards: List of card integers
        
    Raises:
        ValueError: If any card is invalid or duplicated
    """
    seen = set()
    for card in cards:
        if not 0 <= card <= 51:
            raise ValueError(f"Invalid card integer: {card}")
        if card in seen:
            raise ValueError(f"Duplicate card: {int_to_card(card)}")
        seen.add(card)


# Lookup tables for fast GPU operations (will be copied to texture memory)
# These can be used for fast hand evaluation on GPU
RANK_PRIME = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41], dtype=np.int32)
SUIT_PRIME = np.array([2, 3, 5, 7], dtype=np.int32)

# Bit patterns for each rank (useful for hand evaluation)
RANK_BITS = np.array([
    1 << 0,   # 2
    1 << 1,   # 3
    1 << 2,   # 4
    1 << 3,   # 5
    1 << 4,   # 6
    1 << 5,   # 7
    1 << 6,   # 8
    1 << 7,   # 9
    1 << 8,   # T
    1 << 9,   # J
    1 << 10,  # Q
    1 << 11,  # K
    1 << 12,  # A
], dtype=np.int32)