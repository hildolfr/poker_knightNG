#!/usr/bin/env python3
"""
Test batch processing consistency with individual results.

This test ensures that batch processing produces identical results
to individual processing for various hand compositions.
"""

import sys
import time
import numpy as np

sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

import pytest
from poker_knight_ng import create_poker_server, solve_poker_hand


class TestBatchConsistency:
    """Test that batch processing produces consistent results."""
    
    def test_basic_batch_consistency(self):
        """Test batch vs individual results for basic scenarios."""
        server = create_poker_server(keep_alive_seconds=30)
        
        # Define test scenarios
        test_scenarios = [
            # Premium hands
            {'hero_hand': ['A♠', 'A♥'], 'num_opponents': 1},
            {'hero_hand': ['K♠', 'K♥'], 'num_opponents': 2},
            {'hero_hand': ['Q♠', 'Q♥'], 'num_opponents': 3},
            
            # Suited connectors
            {'hero_hand': ['J♠', 'T♠'], 'num_opponents': 1},
            {'hero_hand': ['9♥', '8♥'], 'num_opponents': 2},
            
            # Low pairs
            {'hero_hand': ['2♣', '2♦'], 'num_opponents': 4},
            {'hero_hand': ['5♠', '5♥'], 'num_opponents': 2},
            
            # High cards
            {'hero_hand': ['A♦', 'K♣'], 'num_opponents': 1},
            {'hero_hand': ['K♥', 'Q♦'], 'num_opponents': 3},
            
            # Junk hands
            {'hero_hand': ['7♠', '2♣'], 'num_opponents': 1},
            {'hero_hand': ['9♦', '3♥'], 'num_opponents': 2},
        ]
        
        # Process batch
        print("\nProcessing batch...")
        batch_start = time.time()
        batch_results = server.solve_batch(test_scenarios)
        batch_time = (time.time() - batch_start) * 1000
        
        # Process individually
        print("Processing individually...")
        individual_results = []
        individual_start = time.time()
        for scenario in test_scenarios:
            result = server.solve(**scenario)
            individual_results.append(result)
        individual_time = (time.time() - individual_start) * 1000
        
        # Compare results
        print(f"\nBatch time: {batch_time:.1f}ms")
        print(f"Individual time: {individual_time:.1f}ms")
        print(f"Batch speedup: {individual_time/batch_time:.2f}x")
        
        # Verify consistency
        for i, (batch_res, ind_res) in enumerate(zip(batch_results, individual_results)):
            scenario = test_scenarios[i]
            hand = scenario['hero_hand']
            opponents = scenario['num_opponents']
            
            # Win probability should be very close
            win_diff = abs(batch_res.win_probability - ind_res.win_probability)
            assert win_diff < 0.01, f"Win probability mismatch for {hand} vs {opponents}: {win_diff}"
            
            # Tie probability should be very close
            tie_diff = abs(batch_res.tie_probability - ind_res.tie_probability)
            assert tie_diff < 0.01, f"Tie probability mismatch for {hand} vs {opponents}: {tie_diff}"
            
            # Hand category frequencies should match closely
            hand_categories = ['high_card', 'pair', 'two_pair', 'three_of_a_kind', 
                               'straight', 'flush', 'full_house', 'four_of_a_kind',
                               'straight_flush', 'royal_flush']
            for category in hand_categories:
                batch_freq = batch_res.hand_category_frequencies[category]
                ind_freq = ind_res.hand_category_frequencies[category]
                freq_diff = abs(batch_freq - ind_freq)
                assert freq_diff < 0.01, f"Hand frequency '{category}' mismatch for {hand}"
    
    def test_board_texture_batch_consistency(self):
        """Test batch consistency with board cards and texture analysis."""
        server = create_poker_server(keep_alive_seconds=30)
        
        # Test scenarios with various board textures
        test_scenarios = [
            # Dry flop
            {
                'hero_hand': ['A♠', 'K♠'],
                'num_opponents': 2,
                'board_cards': ['Q♦', '7♣', '2♥']
            },
            # Wet flop with flush draw
            {
                'hero_hand': ['J♥', 'T♥'],
                'num_opponents': 1,
                'board_cards': ['9♥', '8♥', '2♦']
            },
            # Straight draw heavy
            {
                'hero_hand': ['Q♣', 'J♣'],
                'num_opponents': 3,
                'board_cards': ['T♠', '9♦', '8♣']
            },
            # Paired board
            {
                'hero_hand': ['K♥', 'K♦'],
                'num_opponents': 2,
                'board_cards': ['Q♠', 'Q♣', '5♦']
            },
            # Full board - flush possible
            {
                'hero_hand': ['A♣', 'Q♣'],
                'num_opponents': 1,
                'board_cards': ['K♣', 'J♣', '5♣', '7♦', '2♠']
            },
            # Full board - straight on board
            {
                'hero_hand': ['A♥', 'A♦'],
                'num_opponents': 2,
                'board_cards': ['6♠', '7♥', '8♦', '9♣', 'T♠']
            },
        ]
        
        # Process batch
        batch_results = server.solve_batch(test_scenarios)
        
        # Process individually
        individual_results = []
        for scenario in test_scenarios:
            result = solve_poker_hand(**scenario)
            individual_results.append(result)
        
        # Compare results including board texture metrics
        for i, (batch_res, ind_res) in enumerate(zip(batch_results, individual_results)):
            scenario = test_scenarios[i]
            board = scenario.get('board_cards', [])
            
            # Basic probabilities
            assert abs(batch_res.win_probability - ind_res.win_probability) < 0.01
            assert abs(batch_res.tie_probability - ind_res.tie_probability) < 0.01
            
            # Board texture metrics
            if batch_res.board_texture_score is not None:
                assert abs(batch_res.board_texture_score - ind_res.board_texture_score) < 0.001, \
                    f"Board texture score mismatch for board {board}"
            
            # Check draw combinations if present
            if batch_res.draw_combinations is not None:
                batch_flush = batch_res.draw_combinations.get('flush_draws', 0)
                ind_flush = ind_res.draw_combinations.get('flush_draws', 0)
                assert batch_flush == ind_flush, \
                    f"Flush draw count mismatch for board {board}"
                
                batch_straight = batch_res.draw_combinations.get('straight_draws', 0)
                ind_straight = ind_res.draw_combinations.get('straight_draws', 0)
                assert batch_straight == ind_straight, \
                    f"Straight draw count mismatch for board {board}"
            
            # Hand vulnerability
            if batch_res.hand_vulnerability is not None and ind_res.hand_vulnerability is not None:
                assert abs(batch_res.hand_vulnerability - ind_res.hand_vulnerability) < 0.06, \
                    f"Hand vulnerability mismatch for board {board}"  # 6% tolerance for vulnerability
    
    def test_advanced_features_batch_consistency(self):
        """Test batch consistency with ICM and tournament features."""
        server = create_poker_server(keep_alive_seconds=30)
        
        # ICM scenarios
        test_scenarios = [
            # Bubble situation
            {
                'hero_hand': ['J♠', 'J♥'],
                'num_opponents': 1,
                'tournament_context': {
                    'payouts': [0.5, 0.3, 0.2],
                    'stack_sizes': [1500, 1000, 500],
                    'player_index': 0,
                    'tournament_stage': 'bubble'
                }
            },
            # Final table
            {
                'hero_hand': ['A♦', 'Q♦'],
                'num_opponents': 2,
                'tournament_context': {
                    'payouts': [0.4, 0.25, 0.15, 0.1, 0.1],
                    'stack_sizes': [3000, 2500, 2000, 1500, 1000],
                    'player_index': 2,
                    'tournament_stage': 'final_table'
                }
            },
            # Heads up
            {
                'hero_hand': ['K♣', 'T♣'],
                'num_opponents': 1,
                'tournament_context': {
                    'payouts': [0.65, 0.35],
                    'stack_sizes': [7000, 3000],
                    'player_index': 1,
                    'tournament_stage': 'heads_up'
                }
            },
            # Cash game with SPR
            {
                'hero_hand': ['9♥', '9♦'],
                'num_opponents': 3,
                'pot_size': 50,
                'stack_sizes': [200, 200, 200, 200]  # hero + 3 opponents
            },
        ]
        
        # Process batch
        batch_results = server.solve_batch(test_scenarios)
        
        # Process individually
        individual_results = []
        for scenario in test_scenarios:
            result = solve_poker_hand(**scenario)
            individual_results.append(result)
        
        # Compare advanced features
        for i, (batch_res, ind_res) in enumerate(zip(batch_results, individual_results)):
            scenario = test_scenarios[i]
            
            # Basic probabilities
            assert abs(batch_res.win_probability - ind_res.win_probability) < 0.01
            
            # ICM equity (if applicable)
            if 'tournament_context' in scenario and batch_res.icm_equity is not None:
                assert ind_res.icm_equity is not None, "ICM equity missing in individual result"
                assert abs(batch_res.icm_equity - ind_res.icm_equity) < 0.001, \
                    f"ICM equity mismatch in scenario {i}"
            
            # SPR and pot odds (if applicable)
            if 'pot_size' in scenario:
                if batch_res.spr is not None:
                    assert ind_res.spr is not None, "SPR missing in individual result"
                    assert abs(batch_res.spr - ind_res.spr) < 0.001, \
                        f"SPR mismatch in scenario {i}"
                if batch_res.pot_odds is not None:
                    assert ind_res.pot_odds is not None, "Pot odds missing in individual result"
                    assert abs(batch_res.pot_odds - ind_res.pot_odds) < 0.001, \
                        f"Pot odds mismatch in scenario {i}"
                if batch_res.mdf is not None:
                    assert ind_res.mdf is not None, "MDF missing in individual result"
                    assert abs(batch_res.mdf - ind_res.mdf) < 0.001, \
                        f"MDF mismatch in scenario {i}"
            
            # Positional advantage
            if batch_res.positional_advantage_score is not None and ind_res.positional_advantage_score is not None:
                assert abs(batch_res.positional_advantage_score - ind_res.positional_advantage_score) < 0.01
    
    def test_mixed_simulation_modes_batch(self):
        """Test batch with mixed simulation modes."""
        server = create_poker_server(keep_alive_seconds=30)
        
        # Mix of fast, default, and precision modes
        test_scenarios = [
            {'hero_hand': ['A♠', 'A♥'], 'num_opponents': 1, 'simulation_mode': 'fast'},
            {'hero_hand': ['K♠', 'K♥'], 'num_opponents': 2, 'simulation_mode': 'default'},
            {'hero_hand': ['Q♠', 'Q♥'], 'num_opponents': 3, 'simulation_mode': 'precision'},
            {'hero_hand': ['J♠', 'J♥'], 'num_opponents': 1, 'simulation_mode': 'fast'},
            {'hero_hand': ['T♠', 'T♥'], 'num_opponents': 2, 'simulation_mode': 'default'},
        ]
        
        # Process batch
        batch_results = server.solve_batch(test_scenarios)
        
        # Process individually
        individual_results = []
        for scenario in test_scenarios:
            result = solve_poker_hand(**scenario)
            individual_results.append(result)
        
        # Verify results match within tolerance for each mode
        for i, (batch_res, ind_res) in enumerate(zip(batch_results, individual_results)):
            mode = test_scenarios[i].get('simulation_mode', 'default')
            
            # Tolerance based on mode
            if mode == 'fast':
                tolerance = 0.02  # 2% for fast mode
            elif mode == 'default':
                tolerance = 0.01  # 1% for default
            else:  # precision
                tolerance = 0.005  # 0.5% for precision
            
            win_diff = abs(batch_res.win_probability - ind_res.win_probability)
            assert win_diff < tolerance, \
                f"Win probability outside tolerance for {mode} mode: {win_diff}"
            
            # Verify simulation count matches mode
            if mode == 'fast':
                assert 9000 <= batch_res.actual_simulations <= 11000
            elif mode == 'default':
                assert 90000 <= batch_res.actual_simulations <= 110000
            else:  # precision
                assert 450000 <= batch_res.actual_simulations <= 550000
    
    def test_large_batch_consistency(self):
        """Test consistency with larger batches."""
        server = create_poker_server(keep_alive_seconds=60)
        
        # Generate 50 random scenarios
        np.random.seed(42)  # For reproducibility
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suits = ['♠', '♥', '♦', '♣']
        
        test_scenarios = []
        for _ in range(50):
            # Generate random hand
            cards = []
            while len(cards) < 2:
                rank = np.random.choice(ranks)
                suit = np.random.choice(suits)
                card = f"{rank}{suit}"
                if card not in cards:
                    cards.append(card)
            
            # Random number of opponents
            opponents = np.random.randint(1, 5)
            
            # Sometimes add board cards
            scenario = {
                'hero_hand': cards,
                'num_opponents': opponents,
                'simulation_mode': np.random.choice(['fast', 'default'])
            }
            
            if np.random.random() < 0.3:  # 30% chance of board cards
                board_size = np.random.choice([3, 4, 5])
                board_cards = []
                used_cards = cards.copy()
                
                while len(board_cards) < board_size:
                    rank = np.random.choice(ranks)
                    suit = np.random.choice(suits)
                    card = f"{rank}{suit}"
                    if card not in used_cards:
                        board_cards.append(card)
                        used_cards.append(card)
                
                scenario['board_cards'] = board_cards
            
            test_scenarios.append(scenario)
        
        # Process large batch
        print(f"\nProcessing large batch of {len(test_scenarios)} scenarios...")
        batch_start = time.time()
        batch_results = server.solve_batch(test_scenarios)
        batch_time = (time.time() - batch_start) * 1000
        
        # Sample check - verify first 10 individually
        print("Verifying sample results...")
        for i in range(10):
            ind_result = solve_poker_hand(**test_scenarios[i])
            batch_res = batch_results[i]
            
            assert abs(batch_res.win_probability - ind_result.win_probability) < 0.02
            assert abs(batch_res.tie_probability - ind_result.tie_probability) < 0.02
        
        print(f"Large batch processed in {batch_time:.1f}ms")
        print(f"Average time per solve: {batch_time/len(test_scenarios):.1f}ms")
        
        # All results should be valid
        assert all(r is not None for r in batch_results)
        assert all(0 <= r.win_probability <= 1 for r in batch_results)
        assert all(0 <= r.tie_probability <= 1 for r in batch_results)
    
    def test_error_handling_in_batch(self):
        """Test that batch processing handles errors gracefully."""
        server = create_poker_server(keep_alive_seconds=30)
        
        # Mix valid and invalid scenarios
        test_scenarios = [
            {'hero_hand': ['A♠', 'A♥'], 'num_opponents': 1},  # Valid
            {'hero_hand': ['K♠', 'K♠'], 'num_opponents': 2},  # Invalid - duplicate card
            {'hero_hand': ['Q♠', 'Q♥'], 'num_opponents': 3},  # Valid
            {'hero_hand': ['J♠'], 'num_opponents': 1},  # Invalid - only one card
            {'hero_hand': ['T♠', 'T♥'], 'num_opponents': 2},  # Valid
        ]
        
        # Process batch - should handle errors gracefully
        results = server.solve_batch(test_scenarios)
        
        # Check results
        assert results[0] is not None  # Valid scenario
        assert results[1] is None  # Invalid - duplicate
        assert results[2] is not None  # Valid scenario
        assert results[3] is None  # Invalid - one card
        assert results[4] is not None  # Valid scenario
        
        # Valid results should have reasonable values
        assert 0.8 < results[0].win_probability < 0.9  # AA vs 1
        assert 0.5 < results[2].win_probability < 0.6  # QQ vs 3
        assert 0.55 < results[4].win_probability < 0.65  # TT vs 2 (adjusted range)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])