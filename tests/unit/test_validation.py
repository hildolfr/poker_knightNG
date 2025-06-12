"""Unit tests for input validation."""

import pytest
import sys
sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

from poker_knight_ng.validator import validate_inputs, ValidationError


class TestInputValidation:
    """Test input validation logic."""
    
    def test_valid_basic_input(self):
        """Test validation of basic valid inputs."""
        result = validate_inputs(
            hero_hand=['A♠', 'K♠'],
            num_opponents=2
        )
        
        assert result['hero_cards'] == [51, 47]  # A♠, K♠
        assert result['num_opponents'] == 2
        assert result['num_simulations'] == 100_000  # default mode
        assert result['simulation_mode'] == 'default'
    
    def test_card_format_variations(self):
        """Test different card format inputs."""
        # Unicode suits
        result = validate_inputs(['A♠', 'K♥'], 1)
        assert result['hero_cards'] == [51, 46]  # A♠, K♥
        
        # Letter suits
        result = validate_inputs(['As', 'Kh'], 1)
        assert result['hero_cards'] == [51, 46]  # As, Kh
        
        # Mixed case
        result = validate_inputs(['as', 'KH'], 1)
        assert result['hero_cards'] == [51, 46]
    
    def test_invalid_hero_hand(self):
        """Test invalid hero hand inputs."""
        # Wrong number of cards
        with pytest.raises(ValidationError, match="must be a list of exactly 2 cards"):
            validate_inputs(['A♠'], 1)
        
        with pytest.raises(ValidationError, match="must be a list of exactly 2 cards"):
            validate_inputs(['A♠', 'K♠', 'Q♠'], 1)
        
        # Invalid card format
        with pytest.raises(ValidationError, match="Invalid hero_hand"):
            validate_inputs(['XX', 'YY'], 1)
        
        # Duplicate cards
        with pytest.raises(ValidationError, match="Duplicate card"):
            validate_inputs(['A♠', 'A♠'], 1)
    
    def test_num_opponents_validation(self):
        """Test number of opponents validation."""
        # Valid range
        for n in range(1, 7):
            result = validate_inputs(['A♠', 'K♠'], n)
            assert result['num_opponents'] == n
        
        # Invalid values
        with pytest.raises(ValidationError, match="must be an integer between 1 and 6"):
            validate_inputs(['A♠', 'K♠'], 0)
        
        with pytest.raises(ValidationError, match="must be an integer between 1 and 6"):
            validate_inputs(['A♠', 'K♠'], 7)
    
    def test_board_cards_validation(self):
        """Test board cards validation."""
        # Valid board sizes
        for board in [[], ['A♠', 'K♠', 'Q♠'], ['A♠', 'K♠', 'Q♠', 'J♠'], ['A♠', 'K♠', 'Q♠', 'J♠', 'T♠']]:
            result = validate_inputs(['2♥', '3♦'], 1, board_cards=board if board else None)
            if board:
                assert len(result['board_cards']) == len(board)
        
        # Invalid board size
        with pytest.raises(ValidationError, match="must contain 0, 3, 4, or 5 cards"):
            validate_inputs(['A♠', 'K♠'], 1, board_cards=['Q♠', 'J♠'])
        
        # Duplicate with hero cards
        with pytest.raises(ValidationError, match="Duplicate card"):
            validate_inputs(['A♠', 'K♠'], 1, board_cards=['A♠', 'Q♠', 'J♠'])
    
    def test_simulation_modes(self):
        """Test simulation mode validation."""
        modes = {
            'fast': 10_000,
            'default': 100_000,
            'precision': 500_000
        }
        
        for mode, expected_sims in modes.items():
            result = validate_inputs(['A♠', 'K♠'], 1, simulation_mode=mode)
            assert result['simulation_mode'] == mode
            assert result['num_simulations'] == expected_sims
        
        # Invalid mode
        with pytest.raises(ValidationError, match="simulation_mode must be one of"):
            validate_inputs(['A♠', 'K♠'], 1, simulation_mode='ultra')
    
    def test_position_validation(self):
        """Test position validation."""
        valid_positions = ['early', 'middle', 'late', 'button', 'sb', 'bb']
        
        for pos in valid_positions:
            result = validate_inputs(['A♠', 'K♠'], 1, hero_position=pos)
            assert result['hero_position'] == pos
        
        # Invalid position
        with pytest.raises(ValidationError, match="hero_position must be one of"):
            validate_inputs(['A♠', 'K♠'], 1, hero_position='utg')
    
    def test_stack_sizes_validation(self):
        """Test stack sizes validation."""
        # Valid stack sizes
        result = validate_inputs(['A♠', 'K♠'], 2, stack_sizes=[1000, 800, 1200])
        assert result['stack_sizes'] == [1000.0, 800.0, 1200.0]
        
        # Wrong number of stacks
        with pytest.raises(ValidationError, match="must have 3 elements"):
            validate_inputs(['A♠', 'K♠'], 2, stack_sizes=[1000, 800])
        
        # Negative stack
        with pytest.raises(ValidationError, match="must be positive numbers"):
            validate_inputs(['A♠', 'K♠'], 1, stack_sizes=[1000, -100])
    
    def test_tournament_context_validation(self):
        """Test tournament context validation."""
        # Valid context
        valid_context = {
            'payouts': [50, 30, 20],
            'players_remaining': 10,
            'average_stack': 1000
        }
        result = validate_inputs(['A♠', 'K♠'], 1, tournament_context=valid_context)
        assert result['tournament_context'] == valid_context
        
        # Missing required keys
        with pytest.raises(ValidationError, match="must contain keys"):
            validate_inputs(['A♠', 'K♠'], 1, tournament_context={'payouts': [50, 30, 20]})
    
    def test_advanced_parameters(self):
        """Test advanced parameter validation."""
        result = validate_inputs(
            hero_hand=['A♠', 'K♠'],
            num_opponents=2,
            action_to_hero='bet',
            bet_size=0.5,
            street='flop',
            players_to_act=1,
            tournament_stage='bubble',
            blind_level=10
        )
        
        assert result['action_to_hero'] == 'bet'
        assert result['bet_size'] == 0.5
        assert result['street'] == 'flop'
        assert result['players_to_act'] == 1
        assert result['tournament_stage'] == 'bubble'
        assert result['blind_level'] == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])