#!/usr/bin/env python3
"""
API compliance test to verify implementation matches apiNG.md specifications.
"""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

from poker_knight_ng import solve_poker_hand
from poker_knight_ng.result_builder import SimulationResult
import pytest


class TestAPICompliance:
    """Test that API matches apiNG.md specifications."""
    
    def test_basic_call_signature(self):
        """Test basic function call works as documented."""
        # Minimal call - just required parameters
        result = solve_poker_hand(
            hero_hand=['A♠', 'K♠'],
            num_opponents=2
        )
        
        assert isinstance(result, SimulationResult)
        assert hasattr(result, 'win_probability')
        assert hasattr(result, 'tie_probability')
        assert hasattr(result, 'loss_probability')
    
    def test_all_optional_parameters(self):
        """Test that all optional parameters from apiNG.md are accepted."""
        # Test with all parameters specified
        result = solve_poker_hand(
            hero_hand=['A♠', 'K♠'],
            num_opponents=2,
            board_cards=['Q♠', 'J♠', 'T♥'],
            simulation_mode='default',
            hero_position='button',
            stack_sizes=[1000, 1200, 800],
            pot_size=200,
            tournament_context={
                'payouts': [0.5, 0.3, 0.2],
                'players_remaining': 10,
                'average_stack': 5000
            },
            action_to_hero='bet',
            bet_size=0.5,
            street='flop',
            players_to_act=2,
            tournament_stage='bubble',
            blind_level=15
        )
        
        assert isinstance(result, SimulationResult)
    
    def test_always_present_fields(self):
        """Test fields that should always be present."""
        result = solve_poker_hand(['A♥', 'A♦'], 1)
        
        # Always present fields from apiNG.md
        assert isinstance(result.win_probability, float)
        assert 0 <= result.win_probability <= 1
        
        assert isinstance(result.tie_probability, float)
        assert 0 <= result.tie_probability <= 1
        
        assert isinstance(result.loss_probability, float)
        assert 0 <= result.loss_probability <= 1
        
        assert isinstance(result.execution_time_ms, float)
        assert result.execution_time_ms > 0
        
        assert isinstance(result.execution_time_start, float)
        assert isinstance(result.execution_time_end, float)
        assert result.execution_time_end > result.execution_time_start
        
        # Probabilities should sum to 1
        total = result.win_probability + result.tie_probability + result.loss_probability
        assert abs(total - 1.0) < 0.001
    
    def test_statistical_fields(self):
        """Test statistical fields are present and correct."""
        result = solve_poker_hand(['K♥', 'K♦'], 2, ['Q♣', '7♦', '2♠'])
        
        # Confidence interval
        assert hasattr(result, 'confidence_interval')
        assert isinstance(result.confidence_interval, tuple)
        assert len(result.confidence_interval) == 2
        low, high = result.confidence_interval
        assert 0 <= low <= result.win_probability <= high <= 1
        
        # Hand category frequencies
        assert hasattr(result, 'hand_category_frequencies')
        assert isinstance(result.hand_category_frequencies, dict)
        
        # All categories from apiNG.md should be present
        expected_categories = [
            'high_card', 'pair', 'two_pair', 'three_of_a_kind', 
            'straight', 'flush', 'full_house', 'four_of_a_kind',
            'straight_flush', 'royal_flush'
        ]
        
        for category in expected_categories:
            assert category in result.hand_category_frequencies
            assert 0 <= result.hand_category_frequencies[category] <= 1
        
        # Sum should be 1
        total_freq = sum(result.hand_category_frequencies.values())
        assert abs(total_freq - 1.0) < 0.001
    
    def test_multiway_fields(self):
        """Test multi-way analysis fields (3+ players or position specified)."""
        # Test with 3+ players
        result = solve_poker_hand(
            hero_hand=['Q♠', 'Q♥'],
            num_opponents=3,
            hero_position='middle'
        )
        
        # These fields may be None if not calculated
        multiway_fields = [
            'position_aware_equity',
            'multi_way_statistics',
            'fold_equity_estimates',
            'coordination_effects',
            'defense_frequencies',
            'bluff_catching_frequency',
            'range_coordination_score'
        ]
        
        for field in multiway_fields:
            assert hasattr(result, field)
            # Can be None or appropriate type
            value = getattr(result, field)
            if value is not None:
                if 'frequency' in field or 'score' in field:
                    assert isinstance(value, (int, float))
                else:
                    assert isinstance(value, dict)
    
    def test_tournament_icm_fields(self):
        """Test tournament/ICM fields when stack sizes provided."""
        result = solve_poker_hand(
            hero_hand=['A♣', 'K♣'],
            num_opponents=2,
            stack_sizes=[1000, 1200, 800],
            pot_size=200,
            tournament_context={
                'payouts': [0.5, 0.3, 0.2],
                'players_remaining': 3,
                'average_stack': 1000
            }
        )
        
        # ICM fields
        assert hasattr(result, 'icm_equity')
        if result.icm_equity is not None:
            assert 0 <= result.icm_equity <= 1
        
        assert hasattr(result, 'bubble_factor')
        if result.bubble_factor is not None:
            assert result.bubble_factor >= 0
        
        assert hasattr(result, 'stack_to_pot_ratio')
        if result.stack_to_pot_ratio is not None:
            assert result.stack_to_pot_ratio > 0
        
        assert hasattr(result, 'tournament_pressure')
    
    def test_advanced_analysis_fields(self):
        """Test advanced analysis fields."""
        result = solve_poker_hand(
            hero_hand=['J♥', 'T♥'],
            num_opponents=1,
            board_cards=['9♥', '8♦', '2♣'],
            stack_sizes=[1000, 1000],
            pot_size=100,
            action_to_hero='bet',
            bet_size=0.75
        )
        
        # Advanced fields
        advanced_fields = {
            'spr': float,
            'pot_odds': float,
            'mdf': float,
            'equity_needed': float,
            'commitment_threshold': float,
            'nuts_possible': (list, type(None)),
            'draw_combinations': (dict, type(None)),
            'board_texture_score': (float, type(None)),
            'equity_vs_range_percentiles': (dict, type(None)),
            'positional_advantage_score': (float, type(None)),
            'hand_vulnerability': (float, type(None))
        }
        
        for field, expected_type in advanced_fields.items():
            assert hasattr(result, field)
            value = getattr(result, field)
            if value is not None:
                if isinstance(expected_type, tuple):
                    assert isinstance(value, expected_type)
                else:
                    assert isinstance(value, expected_type)
                
                # Validate ranges for numeric fields
                if field in ['spr', 'pot_odds', 'mdf', 'equity_needed', 
                            'board_texture_score', 'positional_advantage_score', 
                            'hand_vulnerability']:
                    if value is not None:
                        assert value >= 0
                        if field in ['pot_odds', 'mdf', 'equity_needed', 
                                    'board_texture_score', 'hand_vulnerability']:
                            assert value <= 1
    
    def test_simulation_modes(self):
        """Test all simulation modes work as specified."""
        modes = {
            'fast': 10000,
            'default': 100000,
            'precision': 500000
        }
        
        for mode, expected_sims in modes.items():
            result = solve_poker_hand(
                hero_hand=['K♠', 'Q♠'],
                num_opponents=2,
                simulation_mode=mode
            )
            
            # Should have actual_simulations close to expected
            if hasattr(result, 'actual_simulations'):
                # Allow some variance due to grid-stride pattern
                assert abs(result.actual_simulations - expected_sims) / expected_sims < 0.1
    
    def test_position_values(self):
        """Test all position values from apiNG.md are accepted."""
        positions = ["early", "middle", "late", "button", "sb", "bb"]
        
        for position in positions:
            result = solve_poker_hand(
                hero_hand=['9♠', '9♥'],
                num_opponents=3,
                hero_position=position
            )
            assert isinstance(result, SimulationResult)
    
    def test_action_values(self):
        """Test all action values from apiNG.md are accepted."""
        actions = ["check", "bet", "raise", "reraise"]
        
        for action in actions:
            result = solve_poker_hand(
                hero_hand=['A♦', 'Q♦'],
                num_opponents=1,
                action_to_hero=action,
                pot_size=100,
                bet_size=0.5
            )
            assert isinstance(result, SimulationResult)
    
    def test_street_values(self):
        """Test all street values from apiNG.md are accepted."""
        streets_and_boards = [
            ("preflop", None),
            ("flop", ['K♥', 'Q♦', '7♣']),
            ("turn", ['K♥', 'Q♦', '7♣', '2♠']),
            ("river", ['K♥', 'Q♦', '7♣', '2♠', '3♥'])
        ]
        
        for street, board in streets_and_boards:
            result = solve_poker_hand(
                hero_hand=['J♣', 'T♣'],
                num_opponents=2,
                board_cards=board,
                street=street
            )
            assert isinstance(result, SimulationResult)
    
    def test_tournament_stage_values(self):
        """Test all tournament stage values from apiNG.md are accepted."""
        stages = ["early", "middle", "bubble", "final_table"]
        
        for stage in stages:
            result = solve_poker_hand(
                hero_hand=['7♠', '7♥'],
                num_opponents=4,
                tournament_stage=stage,
                tournament_context={'payouts': [0.5, 0.3, 0.2]}
            )
            assert isinstance(result, SimulationResult)
    
    def test_unicode_suits(self):
        """Test Unicode suits work as specified: ♠ ♥ ♦ ♣"""
        result = solve_poker_hand(
            hero_hand=['A♠', 'K♥'],
            num_opponents=1,
            board_cards=['Q♦', 'J♣', 'T♠']
        )
        assert isinstance(result, SimulationResult)
        assert result.win_probability > 0.9  # Straight on board
    
    def test_draw_combinations(self):
        """Test draw_combinations field contains flush/straight draws."""
        # Board with flush draw
        result = solve_poker_hand(
            hero_hand=['A♥', 'K♦'],
            num_opponents=1,
            board_cards=['Q♥', '7♥', '2♣']
        )
        
        if result.draw_combinations:
            assert 'flush_draws' in result.draw_combinations
            assert 'straight_draws' in result.draw_combinations
            assert isinstance(result.draw_combinations['flush_draws'], int)
            assert isinstance(result.draw_combinations['straight_draws'], int)
    
    def test_example_from_apingmd(self):
        """Test the exact example from apiNG.md."""
        result = solve_poker_hand(['A♠', 'K♠'], 2, ['Q♠', 'J♠', 'T♥'])
        
        # Should have a straight (but vs 2 opponents, not invincible)
        assert result.win_probability > 0.90  # AK with straight vs 2 opponents
        
        # Should have flush frequency
        assert 'flush' in result.hand_category_frequencies
        flush_freq = result.hand_category_frequencies['flush']
        assert flush_freq > 0  # Should make flush sometimes with A♠K♠ on Q♠J♠T♥


if __name__ == "__main__":
    pytest.main([__file__, "-v"])