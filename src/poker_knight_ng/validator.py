"""
Input validation for poker_knight_ng.

This module provides comprehensive validation for all API parameters
to ensure data integrity before GPU processing.
"""

from typing import Optional, List, Dict, Any, Tuple, Union
from poker_knight_ng.card_utils import parse_hand, validate_cards


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


# Valid options for string parameters
VALID_POSITIONS = {'early', 'middle', 'late', 'button', 'sb', 'bb'}
VALID_SIMULATION_MODES = {'fast', 'default', 'precision'}
VALID_ACTIONS = {'check', 'bet', 'raise', 'reraise'}
VALID_STREETS = {'preflop', 'flop', 'turn', 'river'}
VALID_TOURNAMENT_STAGES = {'early', 'middle', 'bubble', 'final_table'}

# Simulation counts for each mode
SIMULATION_COUNTS = {
    'fast': 10_000,
    'default': 100_000,
    'precision': 500_000
}


def validate_inputs(
    hero_hand: List[str],
    num_opponents: int,
    board_cards: Optional[List[str]] = None,
    simulation_mode: str = 'default',
    hero_position: Optional[str] = None,
    stack_sizes: Optional[List[Union[int, float]]] = None,
    pot_size: Union[int, float] = 0,
    tournament_context: Optional[Dict[str, Any]] = None,
    action_to_hero: Optional[str] = None,
    bet_size: Optional[float] = None,
    street: Optional[str] = None,
    players_to_act: Optional[int] = None,
    tournament_stage: Optional[str] = None,
    blind_level: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validate all inputs for solve_poker_hand.
    
    Returns:
        Dict containing validated and converted inputs:
        - hero_cards: List[int] - Card integers 0-51
        - num_opponents: int
        - board_cards: Optional[List[int]]
        - num_simulations: int
        - All other validated parameters
    
    Raises:
        ValidationError: If any input is invalid
    """
    validated = {}
    
    # Validate hero_hand
    if not isinstance(hero_hand, list) or len(hero_hand) != 2:
        raise ValidationError("hero_hand must be a list of exactly 2 cards")
    
    try:
        # Convert Unicode/string cards to our standard format
        hero_cards_str = ''.join(_normalize_card(card) for card in hero_hand)
        hero_cards = parse_hand(hero_cards_str)
        validate_cards(hero_cards)
        validated['hero_cards'] = hero_cards
    except Exception as e:
        raise ValidationError(f"Invalid hero_hand: {e}")
    
    # Validate num_opponents
    if not isinstance(num_opponents, int) or not 1 <= num_opponents <= 6:
        raise ValidationError("num_opponents must be an integer between 1 and 6")
    validated['num_opponents'] = num_opponents
    
    # Validate board_cards
    if board_cards is not None:
        if not isinstance(board_cards, list):
            raise ValidationError("board_cards must be a list")
        if len(board_cards) not in [0, 3, 4, 5]:
            raise ValidationError("board_cards must contain 0, 3, 4, or 5 cards")
        
        try:
            board_str = ''.join(_normalize_card(card) for card in board_cards)
            board_cards_int = parse_hand(board_str) if board_str else []
            
            # Check for duplicates between hero and board
            all_cards = hero_cards + board_cards_int
            validate_cards(all_cards)
            
            validated['board_cards'] = board_cards_int
        except Exception as e:
            raise ValidationError(f"Invalid board_cards: {e}")
    else:
        validated['board_cards'] = None
    
    # Validate simulation_mode
    if simulation_mode not in VALID_SIMULATION_MODES:
        raise ValidationError(f"simulation_mode must be one of {VALID_SIMULATION_MODES}")
    validated['simulation_mode'] = simulation_mode
    validated['num_simulations'] = SIMULATION_COUNTS[simulation_mode]
    
    # Validate hero_position
    if hero_position is not None:
        if hero_position not in VALID_POSITIONS:
            raise ValidationError(f"hero_position must be one of {VALID_POSITIONS}")
        validated['hero_position'] = hero_position
    
    # Validate stack_sizes
    if stack_sizes is not None:
        if not isinstance(stack_sizes, list):
            raise ValidationError("stack_sizes must be a list")
        if len(stack_sizes) != num_opponents + 1:
            raise ValidationError(
                f"stack_sizes must have {num_opponents + 1} elements "
                f"(hero + {num_opponents} opponents)"
            )
        if not all(isinstance(s, (int, float)) and s > 0 for s in stack_sizes):
            raise ValidationError("All stack sizes must be positive numbers")
        validated['stack_sizes'] = [float(s) for s in stack_sizes]
    
    # Validate pot_size
    if not isinstance(pot_size, (int, float)) or pot_size < 0:
        raise ValidationError("pot_size must be a non-negative number")
    validated['pot_size'] = float(pot_size)
    
    # Validate tournament_context
    if tournament_context is not None:
        if not isinstance(tournament_context, dict):
            raise ValidationError("tournament_context must be a dictionary")
        # Basic structure validation - payouts is the minimum requirement
        if 'payouts' not in tournament_context:
            raise ValidationError("tournament_context must contain 'payouts' key")
        
        # Fill in defaults for missing keys
        context = tournament_context.copy()
        if 'players_remaining' not in context:
            context['players_remaining'] = len(context['payouts'])
        if 'average_stack' not in context:
            context['average_stack'] = 10000  # Default average stack
        
        validated['tournament_context'] = context
    
    # Validate action_to_hero
    if action_to_hero is not None:
        if action_to_hero not in VALID_ACTIONS:
            raise ValidationError(f"action_to_hero must be one of {VALID_ACTIONS}")
        validated['action_to_hero'] = action_to_hero
    
    # Validate bet_size
    if bet_size is not None:
        if not isinstance(bet_size, (int, float)) or bet_size < 0:
            raise ValidationError("bet_size must be a non-negative number")
        validated['bet_size'] = float(bet_size)
    
    # Validate street
    if street is not None:
        if street not in VALID_STREETS:
            raise ValidationError(f"street must be one of {VALID_STREETS}")
        validated['street'] = street
    
    # Validate players_to_act
    if players_to_act is not None:
        if not isinstance(players_to_act, int) or players_to_act < 0:
            raise ValidationError("players_to_act must be a non-negative integer")
        if players_to_act > num_opponents:
            raise ValidationError("players_to_act cannot exceed num_opponents")
        validated['players_to_act'] = players_to_act
    
    # Validate tournament_stage
    if tournament_stage is not None:
        if tournament_stage not in VALID_TOURNAMENT_STAGES:
            raise ValidationError(
                f"tournament_stage must be one of {VALID_TOURNAMENT_STAGES}"
            )
        validated['tournament_stage'] = tournament_stage
    
    # Validate blind_level
    if blind_level is not None:
        if not isinstance(blind_level, int) or blind_level < 1:
            raise ValidationError("blind_level must be a positive integer")
        validated['blind_level'] = blind_level
    
    return validated


def _normalize_card(card: str) -> str:
    """
    Normalize card string to standard format (rank + lowercase suit).
    
    Handles Unicode suit symbols and converts them to letters.
    """
    if len(card) < 2:
        raise ValueError(f"Invalid card: {card}")
    
    rank = card[0].upper()
    suit = card[1:]
    
    # Handle Unicode suits
    unicode_to_letter = {
        '♠': 's', '♤': 's',
        '♥': 'h', '♡': 'h',
        '♦': 'd', '♢': 'd',
        '♣': 'c', '♧': 'c'
    }
    
    if suit in unicode_to_letter:
        suit = unicode_to_letter[suit]
    else:
        suit = suit.lower()
    
    return rank + suit