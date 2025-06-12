#!/usr/bin/env python3
"""Integration test for the full poker solver pipeline."""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

import time
import cupy as cp
from poker_knight_ng import solve_poker_hand


def test_basic_solver():
    """Test basic solver functionality."""
    print("Testing basic solver functionality...")
    
    # Test 1: Pocket aces vs one opponent, no board
    print("\n1. Testing AA vs 1 opponent (preflop)...")
    try:
        result = solve_poker_hand(
            hero_hand=['A♠', 'A♥'],
            num_opponents=1,
            simulation_mode='fast'
        )
        
        print(f"   Win probability: {result.win_probability:.2%}")
        print(f"   Tie probability: {result.tie_probability:.2%}")
        print(f"   Loss probability: {result.loss_probability:.2%}")
        print(f"   Execution time: {result.execution_time_ms:.1f}ms")
        
        # Sanity checks
        assert 0.8 < result.win_probability < 0.9, "AA should win ~85% vs random hand"
        assert abs(result.win_probability + result.tie_probability + result.loss_probability - 1.0) < 0.01
        
        print("   ✓ Test passed!")
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return False
    
    # Test 2: AA vs KK preflop (classic matchup)
    print("\n2. Testing AA vs KK equivalent...")
    try:
        # We can't specify opponent's exact cards, but with flop we can test scenarios
        result = solve_poker_hand(
            hero_hand=['A♠', 'A♥'],
            num_opponents=1,
            board_cards=['2♣', '3♦', '4♥'],  # Dry board
            simulation_mode='fast'
        )
        
        print(f"   Win probability: {result.win_probability:.2%}")
        print(f"   Hand frequencies:")
        for category, freq in result.hand_category_frequencies.items():
            if freq > 0.01:
                print(f"     {category}: {freq:.2%}")
        
        print("   ✓ Test passed!")
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return False
    
    # Test 3: Multi-way pot
    print("\n3. Testing multi-way pot (3 players)...")
    try:
        result = solve_poker_hand(
            hero_hand=['Q♠', 'Q♦'],
            num_opponents=2,
            board_cards=['J♠', 'T♥', '9♣'],
            simulation_mode='fast'
        )
        
        print(f"   Win probability: {result.win_probability:.2%}")
        print(f"   Confidence interval: {result.confidence_interval[0]:.2%} - {result.confidence_interval[1]:.2%}")
        
        # Multi-way should have lower win rate
        assert result.win_probability < 0.7, "QQ should win less in 3-way pot"
        
        print("   ✓ Test passed!")
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return False
    
    # Test 4: Flush draw scenario
    print("\n4. Testing flush draw...")
    try:
        result = solve_poker_hand(
            hero_hand=['A♠', 'K♠'],
            num_opponents=1,
            board_cards=['Q♠', 'J♠', '2♥', '3♦'],  # 4 spades total, need 1 more
            simulation_mode='fast'
        )
        
        print(f"   Win probability: {result.win_probability:.2%}")
        flush_freq = result.hand_category_frequencies.get('flush', 0)
        straight_freq = result.hand_category_frequencies.get('straight', 0)
        print(f"   Flush frequency: {flush_freq:.2%}")
        print(f"   Straight frequency: {straight_freq:.2%}")
        
        # We have 4 spades, need 1 more from 9 remaining spades out of 46 cards
        # That's about 9/46 ≈ 19.6% to make flush
        # But we also have straight draw (need a Ten)
        assert 0.15 < flush_freq < 0.25, "Should make flush ~20% of the time"
        
        print("   ✓ Test passed!")
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return False
    
    return True


def test_advanced_features():
    """Test advanced features like position, stacks, etc."""
    print("\n\nTesting advanced features...")
    
    # Test with position and stacks
    print("\n1. Testing with position and stack sizes...")
    try:
        result = solve_poker_hand(
            hero_hand=['T♥', 'T♣'],
            num_opponents=3,
            board_cards=['9♠', '8♦', '7♥'],
            simulation_mode='fast',
            hero_position='button',
            stack_sizes=[1000, 800, 1200, 500],
            pot_size=200
        )
        
        print(f"   Win probability: {result.win_probability:.2%}")
        print(f"   SPR: {result.spr:.2f}" if result.spr else "   SPR: Not calculated")
        
        if result.position_aware_equity:
            print("   Position equity calculated")
        
        print("   ✓ Test passed!")
        
    except Exception as e:
        print(f"   ✗ Test failed: {e}")
        return False
    
    # Test different simulation modes
    print("\n2. Testing simulation modes...")
    modes = ['fast', 'default', 'precision']
    win_probs = []
    
    for mode in modes:
        try:
            start = time.time()
            result = solve_poker_hand(
                hero_hand=['K♥', 'K♦'],
                num_opponents=2,
                simulation_mode=mode
            )
            elapsed = (time.time() - start) * 1000
            
            win_probs.append(result.win_probability)
            print(f"   {mode}: {result.win_probability:.3%} in {elapsed:.0f}ms")
            
        except Exception as e:
            print(f"   ✗ {mode} failed: {e}")
            return False
    
    # Check that probabilities are reasonably close
    if len(win_probs) == 3:
        diff = max(win_probs) - min(win_probs)
        assert diff < 0.05, "Results should be within 5% across modes"
        print("   ✓ All modes consistent!")
    
    return True


def test_gpu_functionality():
    """Test GPU-specific functionality."""
    print("\n\nTesting GPU functionality...")
    
    # Check GPU availability
    print("1. GPU availability:")
    print(f"   CUDA available: {cp.cuda.is_available()}")
    if cp.cuda.is_available():
        device = cp.cuda.Device()
        print(f"   Device: {device.id}")
        print(f"   Compute capability: {device.compute_capability}")
        
        # Get memory info
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        print(f"   Memory: {free_mem/1024**3:.1f}GB free / {total_mem/1024**3:.1f}GB total")
    
    return True


def main():
    """Run all integration tests."""
    print("="*60)
    print("Poker Knight NG Integration Tests")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    if not test_basic_solver():
        all_passed = False
    
    if not test_advanced_features():
        all_passed = False
    
    if not test_gpu_functionality():
        all_passed = False
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ All integration tests passed!")
    else:
        print("❌ Some tests failed!")
    print("="*60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)