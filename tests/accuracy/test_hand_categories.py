"""
Tests for hand category frequency validation.

This module tests that the solver correctly categorizes poker hands
and that frequencies match expected distributions.
"""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

import pytest
from poker_knight_ng import solve_poker_hand


class TestHandCategoryFrequencies:
    """Test that hand categories are correctly identified and distributed."""
    
    def test_guaranteed_hand_categories(self):
        """Test scenarios where specific hand categories are guaranteed."""
        
        # Test 1: Guaranteed straight flush (full board)
        result = solve_poker_hand(
            hero_hand=['A♠', 'K♠'],
            num_opponents=1,
            board_cards=['Q♠', 'J♠', 'T♠', '3♦', '2♣'],  # Royal flush with full board
            simulation_mode='fast'
        )
        
        # Should have straight flush (royal is a type of straight flush)
        sf_freq = result.hand_category_frequencies.get('straight_flush', 0) + \
                  result.hand_category_frequencies.get('royal_flush', 0)
        assert sf_freq > 0.99, f"Expected straight flush ~100%, got {sf_freq:.1%}"
        
        # Test 2: Guaranteed four of a kind (full board)
        result = solve_poker_hand(
            hero_hand=['A♥', 'A♦'],
            num_opponents=1,
            board_cards=['A♠', 'A♣', '7♥', '3♦', '2♣'],  # Quad aces with full board
            simulation_mode='fast'
        )
        
        quads_freq = result.hand_category_frequencies.get('four_of_a_kind', 0)
        assert quads_freq > 0.99, f"Expected quads ~100%, got {quads_freq:.1%}"
        
        # Test 3: Guaranteed full house (full board)
        result = solve_poker_hand(
            hero_hand=['K♥', 'K♦'],
            num_opponents=1,
            board_cards=['K♠', '7♣', '7♥', '3♦', '2♣'],  # Kings full with full board
            simulation_mode='fast'
        )
        
        fh_freq = result.hand_category_frequencies.get('full_house', 0)
        assert fh_freq > 0.99, f"Expected full house ~100%, got {fh_freq:.1%}"
        
        # Test 4: Guaranteed flush (full board)
        result = solve_poker_hand(
            hero_hand=['9♥', '8♥'],
            num_opponents=1,
            board_cards=['7♥', '6♥', '2♥', '3♦', '4♣'],  # Heart flush with full board
            simulation_mode='fast'
        )
        
        flush_freq = result.hand_category_frequencies.get('flush', 0)
        assert flush_freq > 0.99, f"Expected flush ~100%, got {flush_freq:.1%}"
        
        # Test 5: Guaranteed straight (full board)
        result = solve_poker_hand(
            hero_hand=['8♣', '7♦'],
            num_opponents=1,
            board_cards=['6♥', '5♠', '4♣', '2♦', '2♥'],  # 8-high straight with full board
            simulation_mode='fast'
        )
        
        straight_freq = result.hand_category_frequencies.get('straight', 0)
        assert straight_freq > 0.99, f"Expected straight ~100%, got {straight_freq:.1%}"
    
    def test_hand_category_distributions(self):
        """Test that hand category frequencies match expected distributions."""
        
        # Test random hands (no board) - should see expected distribution
        result = solve_poker_hand(
            hero_hand=['A♠', 'K♥'],
            num_opponents=1,
            simulation_mode='precision'  # Need precision for good distribution
        )
        
        # Verify we get a variety of hands
        categories_seen = sum(1 for freq in result.hand_category_frequencies.values() if freq > 0.01)
        assert categories_seen >= 5, "Should see at least 5 different hand categories"
        
        # High card should be common but AK makes many hands
        high_card_freq = result.hand_category_frequencies.get('high_card', 0)
        assert high_card_freq > 0.15, f"High card should be >15%, got {high_card_freq:.1%}"
        
        # Pairs should be most common
        pair_freq = result.hand_category_frequencies.get('pair', 0)
        assert 0.35 < pair_freq < 0.65, f"Pair frequency unexpected: {pair_freq:.1%}"
    
    def test_pocket_pair_categories(self):
        """Test hand categories with pocket pairs."""
        
        # Test pocket aces - it's actually one pair, not just "pair"
        result = solve_poker_hand(
            hero_hand=['A♠', 'A♥'],
            num_opponents=1,
            board_cards=['K♣', 'Q♦', 'J♥', 'T♠', '2♣'],  # AA makes a straight!
            simulation_mode='default'
        )
        
        # Debug - print frequencies
        print("\nPocket aces frequencies:")
        for cat, freq in result.hand_category_frequencies.items():
            if freq > 0.001:
                print(f"  {cat}: {freq:.3f}")
        
        # AA actually makes a straight on this board (A-K-Q-J-T)
        straight_freq = result.hand_category_frequencies.get('straight', 0)
        assert straight_freq > 0.99, f"AA should make broadway straight, got {straight_freq:.1%}"
        
        # Test pocket pair hitting set (full board)
        result = solve_poker_hand(
            hero_hand=['7♥', '7♦'],
            num_opponents=1,
            board_cards=['7♠', 'K♣', '2♥', '9♣', '4♠'],  # Set of sevens with full board
            simulation_mode='default'
        )
        
        # Should have three of a kind
        trips_freq = result.hand_category_frequencies.get('three_of_a_kind', 0)
        assert trips_freq > 0.99, f"Should have trips ~100%, got {trips_freq:.1%}"
        
        # Test pocket pair that stays as pair
        result = solve_poker_hand(
            hero_hand=['J♠', 'J♥'],
            num_opponents=1,
            board_cards=['8♣', '7♦', '4♥', '3♠', '2♣'],  # No improvement for JJ
            simulation_mode='default'
        )
        
        # Should have pair
        pair_freq = result.hand_category_frequencies.get('pair', 0)
        assert pair_freq > 0.99, f"JJ should be a pair, got {pair_freq:.1%}"
    
    def test_draw_completion_frequencies(self):
        """Test frequencies of completing draws."""
        
        # Test flush draw on flop
        result = solve_poker_hand(
            hero_hand=['A♦', 'K♦'],
            num_opponents=1,
            board_cards=['Q♦', '7♦', '2♣'],  # 4 diamonds
            simulation_mode='precision'
        )
        
        flush_freq = result.hand_category_frequencies.get('flush', 0)
        # With 9 outs and 2 cards to come: ~35%
        assert 0.30 < flush_freq < 0.40, f"Flush draw should complete ~35%, got {flush_freq:.1%}"
        
        # Also check we sometimes make pairs/two pair
        pair_freq = result.hand_category_frequencies.get('pair', 0)
        two_pair_freq = result.hand_category_frequencies.get('two_pair', 0)
        assert pair_freq > 0.2, "Should make pairs sometimes"
        assert two_pair_freq > 0.05, "Should make two pair sometimes"
    
    def test_broadway_hands(self):
        """Test broadway (high card) hand categories."""
        
        # Test broadway cards - KQ on JT9 already makes a straight!
        result = solve_poker_hand(
            hero_hand=['K♠', 'Q♥'],
            num_opponents=1,
            board_cards=['J♦', 'T♣', '9♥'],  # KQ makes K-Q-J-T-9 straight
            simulation_mode='default'
        )
        
        straight_freq = result.hand_category_frequencies.get('straight', 0)
        # KQ with JT9 already has a straight
        assert straight_freq > 0.99, f"KQ should have straight on JT9 board, got {straight_freq:.1%}"
    
    def test_hand_category_sum(self):
        """Test that all hand category frequencies sum to 1.0."""
        
        scenarios = [
            (['A♠', 'K♠'], None),  # Preflop
            (['Q♥', 'Q♦'], ['J♠', 'T♥', '9♣']),  # Flop
            (['7♣', '6♣'], ['5♦', '4♥', '3♠', '2♣']),  # Turn with straight
            (['9♥', '8♦'], ['K♠', 'K♣', 'K♥', 'Q♦', 'J♣'])  # River
        ]
        
        for hand, board in scenarios:
            result = solve_poker_hand(
                hero_hand=hand,
                num_opponents=1,
                board_cards=board,
                simulation_mode='default'
            )
            
            total_freq = sum(result.hand_category_frequencies.values())
            assert abs(total_freq - 1.0) < 0.01, \
                f"Hand frequencies should sum to 1.0, got {total_freq:.3f}"
    
    def test_rare_hands(self):
        """Test that rare hands occur at expected frequencies."""
        
        # Test chance of making quads from pocket pair
        # Pocket pair has ~0.816% chance of making quads by river
        iterations = 0
        total_quads = 0.0
        
        # Run multiple times to get average
        for _ in range(3):
            result = solve_poker_hand(
                hero_hand=['T♥', 'T♦'],
                num_opponents=1,
                simulation_mode='precision'
            )
            quads_freq = result.hand_category_frequencies.get('four_of_a_kind', 0)
            total_quads += quads_freq
            iterations += 1
        
        avg_quads = total_quads / iterations
        # Should be around 0.8% (0.008)
        assert 0.004 < avg_quads < 0.015, \
            f"Pocket pair should make quads ~0.8%, got {avg_quads:.3%}"


class TestHandStrengthConsistency:
    """Test that hand strengths are consistent with categories."""
    
    def test_stronger_categories_win_more(self):
        """Test that stronger hand categories correlate with higher win rates."""
        
        # Create scenarios with known strong hands
        scenarios = [
            # (hand, board, expected_category, min_win_rate)
            (['A♠', 'A♥'], ['A♣', 'K♦', '7♥', '4♠', '2♣'], 'three_of_a_kind', 0.85),
            (['K♥', 'Q♥'], ['J♥', 'T♥', '9♥', '3♦', '2♠'], 'straight_flush', 0.95),
            (['7♣', '7♦'], ['7♥', '7♠', 'K♣', 'Q♦', 'J♠'], 'four_of_a_kind', 0.99),
        ]
        
        for hand, board, expected_cat, min_win in scenarios:
            result = solve_poker_hand(
                hero_hand=hand,
                num_opponents=2,
                board_cards=board,
                simulation_mode='default'
            )
            
            # Check we have the expected category
            cat_freq = result.hand_category_frequencies.get(expected_cat, 0)
            assert cat_freq > 0.95, \
                f"Expected {expected_cat} >95%, got {cat_freq:.1%}"
            
            # Check win rate is high
            assert result.win_probability > min_win, \
                f"{expected_cat} should win >{min_win:.0%}, got {result.win_probability:.1%}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])