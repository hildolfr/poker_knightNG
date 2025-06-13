#!/usr/bin/env python3
"""Test variance reduction from antithetic variates implementation."""

import numpy as np
from poker_knight_ng import solve_poker_hand
import time

def test_variance_reduction():
    """Run multiple iterations to measure variance in results."""
    
    # Test scenario: late-game tournament with ICM pressure
    hero_cards = ["As", "Ks"]
    board = ["Qh", "Jh", "Th"]  # Straight draw board
    
    # Tournament context
    num_opponents = 3
    context = {
        "hero_position": "middle",
        "stack_sizes": [2500, 1800, 3200, 2000],  # Late tournament stacks
        "pot_size": 600,
        "tournament_context": {
            "average_stack": 2375,
            "players_remaining": 4,
            "payouts": [0.5, 0.3, 0.2]  # Top 3 paid
        },
        "simulation_mode": "default"
    }
    
    # Run multiple iterations
    n_iterations = 30
    win_probs = []
    icm_equities = []
    vulnerabilities = []
    bubble_factors = []
    
    print(f"Running {n_iterations} iterations to measure variance...")
    
    for i in range(n_iterations):
        result = solve_poker_hand(hero_cards, num_opponents, board, **context)
        
        win_probs.append(result.win_probability)
        icm_equities.append(result.icm_equity)
        vulnerabilities.append(result.hand_vulnerability)
        bubble_factors.append(result.bubble_factor)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i + 1}/{n_iterations} iterations")
    
    # Calculate statistics
    def calc_stats(values):
        return {
            "mean": np.mean(values),
            "std": np.std(values),
            "cv": np.std(values) / np.mean(values) * 100  # Coefficient of variation %
        }
    
    print("\nVariance Analysis:")
    print("-" * 60)
    
    metrics = [
        ("Win Probability", win_probs),
        ("ICM Equity", icm_equities),
        ("Hand Vulnerability", vulnerabilities),
        ("Bubble Factor", bubble_factors)
    ]
    
    for name, values in metrics:
        stats = calc_stats(values)
        print(f"{name:20} Mean: {stats['mean']:.4f}  Std: {stats['std']:.4f}  CV: {stats['cv']:.2f}%")
    
    # Check if variance is below 1% for critical metrics
    win_cv = calc_stats(win_probs)['cv']
    icm_cv = calc_stats(icm_equities)['cv']
    vuln_cv = calc_stats(vulnerabilities)['cv']
    
    print("\nVariance Targets:")
    print(f"  Win Probability CV:    {win_cv:.2f}% {'✓' if win_cv < 1.0 else '✗'} (target < 1%)")
    print(f"  ICM Equity CV:         {icm_cv:.2f}% {'✓' if icm_cv < 1.0 else '✗'} (target < 1%)")
    print(f"  Hand Vulnerability CV: {vuln_cv:.2f}% {'✓' if vuln_cv < 1.0 else '✗'} (target < 1%)")

if __name__ == "__main__":
    test_variance_reduction()