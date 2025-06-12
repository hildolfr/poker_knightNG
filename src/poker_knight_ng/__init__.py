"""
poker_knight_ng - GPU-accelerated Texas Hold'em poker solver.

A high-performance drop-in replacement for the original Poker Knight,
using CuPy and a monolithic CUDA kernel architecture for maximum speed.
"""

__version__ = "0.1.0"
__author__ = "Hildolfr"

# Main API export
from poker_knight_ng.api import solve_poker_hand

# Card utilities
from poker_knight_ng.card_utils import (
    card_to_int,
    int_to_card,
    parse_hand,
    hand_to_string,
    get_rank,
    get_suit,
    make_card,
)

__all__ = [
    "solve_poker_hand",
    "card_to_int",
    "int_to_card", 
    "parse_hand",
    "hand_to_string",
    "get_rank",
    "get_suit",
    "make_card",
]