"""
Tests for confidence interval statistical accuracy.

This module tests that the confidence intervals produced by the solver
are statistically valid and contain the true value with the expected frequency.
"""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

import pytest
import numpy as np
from poker_knight_ng import solve_poker_hand


class TestConfidenceIntervals:
    """Test that confidence intervals are statistically valid."""
    
    def test_confidence_interval_coverage(self):
        """Test that 95% confidence intervals contain the true value ~95% of the time."""
        
        # Use a scenario with known probability
        # AA vs random hand is ~85% to win
        expected_win_rate = 0.85
        
        # Run multiple simulations to test CI coverage
        num_trials = 20
        contains_true_value = 0
        
        for _ in range(num_trials):
            result = solve_poker_hand(
                hero_hand=['A♠', 'A♥'],
                num_opponents=1,
                simulation_mode='fast'  # Use fast mode for speed
            )
            
            # Check if CI contains expected value
            ci_low, ci_high = result.confidence_interval
            if ci_low <= expected_win_rate <= ci_high:
                contains_true_value += 1
        
        coverage = contains_true_value / num_trials
        # With 20 trials, we expect 19/20 = 95% coverage
        # Allow some variance (80-100% coverage is reasonable for small sample)
        assert 0.80 <= coverage <= 1.0, \
            f"95% CI should contain true value ~95% of time, got {coverage:.0%}"
    
    def test_confidence_interval_width(self):
        """Test that CI width decreases with more simulations."""
        
        modes_and_widths = []
        
        for mode in ['fast', 'default', 'precision']:
            result = solve_poker_hand(
                hero_hand=['K♥', 'K♦'],
                num_opponents=2,
                simulation_mode=mode
            )
            
            ci_low, ci_high = result.confidence_interval
            width = ci_high - ci_low
            modes_and_widths.append((mode, width, result.actual_simulations))
        
        # Check that width decreases with more simulations
        for i in range(len(modes_and_widths) - 1):
            mode1, width1, sims1 = modes_and_widths[i]
            mode2, width2, sims2 = modes_and_widths[i + 1]
            
            assert width2 < width1, \
                f"CI width should decrease: {mode1}({sims1} sims)={width1:.4f} " \
                f"vs {mode2}({sims2} sims)={width2:.4f}"
    
    def test_confidence_interval_validity(self):
        """Test that confidence intervals are valid (low < high, within [0,1])."""
        
        scenarios = [
            (['A♣', 'K♣'], 1, None),  # Preflop
            (['Q♠', 'Q♥'], 2, ['J♦', 'T♣', '9♥']),  # Flop
            (['7♥', '6♥'], 3, ['5♦', '4♣', '3♥', '2♠']),  # Turn straight
        ]
        
        for hand, opps, board in scenarios:
            result = solve_poker_hand(
                hero_hand=hand,
                num_opponents=opps,
                board_cards=board,
                simulation_mode='default'
            )
            
            ci_low, ci_high = result.confidence_interval
            
            # Check validity
            assert 0.0 <= ci_low <= 1.0, f"CI lower bound out of range: {ci_low}"
            assert 0.0 <= ci_high <= 1.0, f"CI upper bound out of range: {ci_high}"
            assert ci_low <= ci_high, f"CI invalid: {ci_low} > {ci_high}"
            
            # Check that win probability is within CI
            assert ci_low <= result.win_probability <= ci_high, \
                f"Win probability {result.win_probability:.3f} outside CI " \
                f"[{ci_low:.3f}, {ci_high:.3f}]"
    
    def test_extreme_probabilities_ci(self):
        """Test CI behavior for extreme probabilities (near 0 or 1)."""
        
        # Test very strong hand (should have narrow CI near 1)
        result = solve_poker_hand(
            hero_hand=['A♠', 'A♥'],
            num_opponents=1,
            board_cards=['A♣', 'A♦', 'K♥', 'Q♦', 'J♣'],  # Quad aces
            simulation_mode='default'
        )
        
        ci_low, ci_high = result.confidence_interval
        
        # Should be very close to 1.0
        assert ci_low > 0.95, f"Quad aces CI lower bound too low: {ci_low}"
        assert ci_high >= 0.99, f"Quad aces CI upper bound too low: {ci_high}"
        assert (ci_high - ci_low) < 0.05, "CI should be narrow for extreme probability"
        
        # Test very weak hand (should have narrow CI near 0)
        result = solve_poker_hand(
            hero_hand=['2♣', '3♦'],
            num_opponents=5,  # Many opponents
            board_cards=['A♠', 'K♥', 'Q♦', 'J♣', 'T♠'],  # Broadway board
            simulation_mode='default'
        )
        
        ci_low, ci_high = result.confidence_interval
        
        # Should be very close to 0.0
        assert ci_high < 0.10, f"23o vs 5 on broadway CI upper bound too high: {ci_high}"
        assert ci_low <= 0.05, f"23o vs 5 on broadway CI lower bound too high: {ci_low}"
    
    def test_ci_formula_correctness(self):
        """Test that CI formula is implemented correctly."""
        
        result = solve_poker_hand(
            hero_hand=['T♥', 'T♣'],
            num_opponents=1,
            simulation_mode='precision'  # Use precision for stable results
        )
        
        # Manually calculate expected CI width
        p = result.win_probability
        n = result.actual_simulations
        
        # Standard error for binomial proportion
        se = np.sqrt(p * (1 - p) / n)
        
        # 95% CI using normal approximation
        z = 1.96
        expected_ci_low = max(0, p - z * se)
        expected_ci_high = min(1, p + z * se)
        
        actual_ci_low, actual_ci_high = result.confidence_interval
        
        # Allow small tolerance for floating point
        tolerance = 0.001
        assert abs(actual_ci_low - expected_ci_low) < tolerance, \
            f"CI lower bound mismatch: {actual_ci_low:.4f} vs {expected_ci_low:.4f}"
        assert abs(actual_ci_high - expected_ci_high) < tolerance, \
            f"CI upper bound mismatch: {actual_ci_high:.4f} vs {expected_ci_high:.4f}"


class TestConfidenceIntervalConsistency:
    """Test CI consistency across different scenarios."""
    
    def test_ci_narrows_with_simulations(self):
        """Test that more simulations produce narrower confidence intervals."""
        
        # Fixed scenario
        hand = ['J♠', 'J♥']
        opponents = 2
        board = ['9♣', '8♦', '7♥']
        
        widths = {}
        for mode in ['fast', 'default', 'precision']:
            result = solve_poker_hand(
                hero_hand=hand,
                num_opponents=opponents,
                board_cards=board,
                simulation_mode=mode
            )
            
            ci_low, ci_high = result.confidence_interval
            widths[mode] = ci_high - ci_low
        
        # Verify narrowing
        assert widths['default'] < widths['fast'], \
            "Default mode should have narrower CI than fast"
        assert widths['precision'] < widths['default'], \
            "Precision mode should have narrower CI than default"
        
        # Check that precision mode has very narrow CI
        assert widths['precision'] < 0.02, \
            f"Precision mode CI too wide: {widths['precision']:.4f}"
    
    def test_ci_stability_across_runs(self):
        """Test that CIs are stable across multiple runs."""
        
        # Run same scenario multiple times
        num_runs = 5
        cis = []
        
        for _ in range(num_runs):
            result = solve_poker_hand(
                hero_hand=['A♦', 'K♦'],
                num_opponents=1,
                board_cards=['Q♦', 'J♦', 'T♣'],  # Flush draw + straight draw
                simulation_mode='default'
            )
            cis.append(result.confidence_interval)
        
        # Check that all CIs overlap
        # Find the intersection of all CIs
        max_low = max(ci[0] for ci in cis)
        min_high = min(ci[1] for ci in cis)
        
        # All CIs should overlap
        assert max_low < min_high, \
            f"CIs don't overlap: max_low={max_low:.4f}, min_high={min_high:.4f}"
        
        # Check that CI widths are similar
        widths = [ci[1] - ci[0] for ci in cis]
        max_width = max(widths)
        min_width = min(widths)
        
        # Widths should be within 20% of each other
        assert (max_width - min_width) / min_width < 0.2, \
            f"CI widths vary too much: {min_width:.4f} to {max_width:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])