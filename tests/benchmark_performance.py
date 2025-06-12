#!/usr/bin/env python3
"""Performance benchmark for poker_knight_ng."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

import time
import statistics
from poker_knight_ng import solve_poker_hand


def benchmark_scenario(name, hero_hand, num_opponents, board_cards=None, 
                      simulation_mode='default', iterations=10):
    """Benchmark a specific scenario."""
    times = []
    
    print(f"\n{name}")
    print("-" * 60)
    
    # Warm-up run
    solve_poker_hand(hero_hand, num_opponents, board_cards, simulation_mode)
    
    # Benchmark runs
    for i in range(iterations):
        start = time.time()
        result = solve_poker_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            board_cards=board_cards,
            simulation_mode=simulation_mode
        )
        elapsed = (time.time() - start) * 1000  # ms
        times.append(elapsed)
        
        if i == 0:  # Print results from first run
            print(f"Win probability: {result.win_probability:.2%}")
    
    # Calculate statistics
    avg_time = statistics.mean(times)
    std_dev = statistics.stdev(times) if len(times) > 1 else 0
    min_time = min(times)
    max_time = max(times)
    
    print(f"Average time: {avg_time:.1f}ms (±{std_dev:.1f}ms)")
    print(f"Min/Max: {min_time:.1f}ms / {max_time:.1f}ms")
    print(f"Simulations: {simulation_mode}")
    
    return avg_time


def main():
    """Run performance benchmarks."""
    print("="*60)
    print("Poker Knight NG Performance Benchmark")
    print("="*60)
    
    results = {}
    
    # Benchmark different scenarios
    scenarios = [
        # (name, hero_hand, num_opponents, board_cards, mode)
        ("Preflop - AA vs 1 opponent (fast)", ['A♠', 'A♥'], 1, None, 'fast'),
        ("Preflop - AA vs 1 opponent (default)", ['A♠', 'A♥'], 1, None, 'default'),
        ("Preflop - AA vs 1 opponent (precision)", ['A♠', 'A♥'], 1, None, 'precision'),
        
        ("Flop - Set vs 2 opponents (default)", ['7♥', '7♦'], 2, ['7♠', 'K♣', '2♥'], 'default'),
        ("Turn - Flush draw vs 1 opponent (default)", ['A♠', 'K♠'], 1, ['Q♠', 'J♠', '2♥', '3♦'], 'default'),
        ("River - Full board vs 3 opponents (default)", ['Q♥', 'Q♦'], 3, ['J♠', 'T♥', '9♣', '8♦', '2♠'], 'default'),
        
        ("6-way pot preflop (default)", ['A♣', 'K♣'], 5, None, 'default'),
        ("6-way pot river (default)", ['T♥', 'T♣'], 5, ['9♠', '8♦', '7♥', '6♣', '2♠'], 'default'),
    ]
    
    for scenario in scenarios:
        name, hero, opps, board, mode = scenario
        avg_time = benchmark_scenario(name, hero, opps, board, mode, iterations=5)
        results[name] = avg_time
    
    # Performance summary
    print("\n" + "="*60)
    print("Performance Summary")
    print("="*60)
    
    # Check performance targets
    fast_times = [t for name, t in results.items() if "(fast)" in name]
    default_times = [t for name, t in results.items() if "(default)" in name]
    precision_times = [t for name, t in results.items() if "(precision)" in name]
    
    if fast_times:
        avg_fast = statistics.mean(fast_times)
        print(f"Fast mode (10k sims) average: {avg_fast:.1f}ms")
        print(f"  Target: <2ms {'✓' if avg_fast < 2 else '✗'}")
    
    if default_times:
        avg_default = statistics.mean(default_times)
        print(f"Default mode (100k sims) average: {avg_default:.1f}ms")
        print(f"  Target: <20ms {'✓' if avg_default < 20 else '✗'}")
    
    if precision_times:
        avg_precision = statistics.mean(precision_times)
        print(f"Precision mode (500k sims) average: {avg_precision:.1f}ms")
        print(f"  Target: <100ms {'✓' if avg_precision < 100 else '✗'}")
    
    # Simulations per second
    print(f"\nSimulations per second:")
    print(f"  Fast mode: {10_000 / (avg_fast/1000):,.0f} sims/sec")
    print(f"  Default mode: {100_000 / (avg_default/1000):,.0f} sims/sec")
    print(f"  Precision mode: {500_000 / (avg_precision/1000):,.0f} sims/sec")


if __name__ == "__main__":
    main()