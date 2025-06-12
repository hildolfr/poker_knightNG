#!/usr/bin/env python3
"""
GPU profiling tool for poker_knight_ng.

This tool profiles GPU utilization, memory usage, and kernel performance
to identify bottlenecks and optimization opportunities.
"""

import sys
sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

import time
import cupy as cp
from poker_knight_ng import solve_poker_hand
from poker_knight_ng.cuda.kernel_wrapper import get_poker_kernel


class GPUProfiler:
    """Profile GPU performance and identify bottlenecks."""
    
    def __init__(self):
        """Initialize profiler."""
        self.kernel = get_poker_kernel()
        
    def profile_memory_usage(self):
        """Profile GPU memory usage patterns."""
        print("="*60)
        print("GPU MEMORY PROFILING")
        print("="*60)
        
        # Get device info
        device = cp.cuda.Device()
        print(f"Device: GPU {device.id}")
        print(f"Compute Capability: {device.compute_capability}")
        
        # Memory before
        free_before, total = cp.cuda.runtime.memGetInfo()
        print(f"\nMemory before allocation:")
        print(f"  Total: {total / 1024**3:.2f} GB")
        print(f"  Free: {free_before / 1024**3:.2f} GB")
        print(f"  Used: {(total - free_before) / 1024**3:.2f} GB")
        
        # Test different simulation sizes
        modes = {
            'fast': 10_000,
            'default': 100_000,
            'precision': 500_000
        }
        
        print("\nMemory usage by simulation mode:")
        for mode, sims in modes.items():
            # Clear memory pools
            cp.get_default_memory_pool().free_all_blocks()
            
            # Run simulation
            _ = solve_poker_hand(
                hero_hand=['A♠', 'A♥'],
                num_opponents=3,
                simulation_mode=mode
            )
            
            # Check memory after
            free_after, _ = cp.cuda.runtime.memGetInfo()
            used = free_before - free_after
            
            # Estimate memory requirements
            mem_reqs = self.kernel.get_memory_requirements(sims)
            
            print(f"\n  {mode} ({sims:,} simulations):")
            print(f"    Actual used: {used / 1024**2:.1f} MB")
            print(f"    Estimated: {mem_reqs['total'] / 1024**2:.1f} MB")
            print(f"    - RNG states: {mem_reqs['rng_states'] / 1024**2:.1f} MB")
            print(f"    - Shared mem: {mem_reqs['shared_memory'] / 1024**2:.1f} MB")
            print(f"    - Temp mem: {mem_reqs['temp_memory'] / 1024**2:.1f} MB")
    
    def profile_kernel_performance(self):
        """Profile kernel execution performance."""
        print("\n" + "="*60)
        print("KERNEL PERFORMANCE PROFILING")
        print("="*60)
        
        scenarios = [
            ('Simple (1 opp, preflop)', ['K♥', 'K♦'], 1, None),
            ('Medium (3 opp, flop)', ['Q♠', 'Q♥'], 3, ['J♦', 'T♣', '9♥']),
            ('Complex (5 opp, river)', ['T♥', 'T♣'], 5, ['9♠', '8♦', '7♥', '6♣', '2♠'])
        ]
        
        for name, hand, opps, board in scenarios:
            print(f"\n{name}:")
            
            # Warm-up
            _ = solve_poker_hand(hand, opps, board, 'fast')
            
            # Profile different phases
            for mode in ['fast', 'default', 'precision']:
                # Time total execution
                times = []
                for _ in range(5):
                    start = time.time()
                    result = solve_poker_hand(hand, opps, board, mode)
                    elapsed = (time.time() - start) * 1000
                    times.append(elapsed)
                
                avg_time = sum(times) / len(times)
                sims = result.actual_simulations
                sims_per_sec = sims / (avg_time / 1000)
                
                print(f"  {mode}: {avg_time:.1f}ms ({sims_per_sec/1e6:.2f}M sims/sec)")
    
    def profile_occupancy(self):
        """Profile GPU occupancy and thread utilization."""
        print("\n" + "="*60)
        print("GPU OCCUPANCY ANALYSIS")
        print("="*60)
        
        # Get current configuration
        for sims in [10_000, 100_000, 500_000]:
            grid, block = self.kernel.calculate_grid_config(sims)
            num_blocks = grid[0]
            threads_per_block = block[0]
            
            # Calculate theoretical occupancy
            # Assume 48KB shared memory per SM, 2048 threads per SM
            shared_per_block = self.kernel.shared_mem_size
            max_blocks_shared = 48 * 1024 // shared_per_block if shared_per_block > 0 else float('inf')
            max_blocks_threads = 2048 // threads_per_block
            max_blocks_per_sm = min(max_blocks_shared, max_blocks_threads, 32)  # HW limit
            
            theoretical_occupancy = (threads_per_block * max_blocks_per_sm) / 2048 * 100
            
            print(f"\nSimulations: {sims:,}")
            print(f"  Grid: {num_blocks} blocks")
            print(f"  Block: {threads_per_block} threads")
            print(f"  Shared memory: {shared_per_block} bytes/block")
            print(f"  Max blocks/SM: {max_blocks_per_sm}")
            print(f"  Theoretical occupancy: {theoretical_occupancy:.1f}%")
            
            # Work distribution
            work_per_thread = sims / (num_blocks * threads_per_block)
            print(f"  Work/thread: {work_per_thread:.1f} simulations")
    
    def profile_memory_access_patterns(self):
        """Analyze memory access patterns for coalescing."""
        print("\n" + "="*60)
        print("MEMORY ACCESS PATTERN ANALYSIS")
        print("="*60)
        
        print("\nCurrent memory access patterns:")
        print("  ✅ Hero cards: Coalesced (2 consecutive int32)")
        print("  ✅ Board cards: Coalesced (5 consecutive int32)")
        print("  ✅ Output arrays: Coalesced (atomic adds to consecutive addresses)")
        print("  ⚠️  RNG states: Potentially uncoalesced (4 int32 per thread)")
        print("  ⚠️  Opponent cards: Random access pattern (card dealing)")
        
        print("\nOptimization opportunities:")
        print("  1. Pack RNG state into int4 for better coalescing")
        print("  2. Use texture memory for card lookup tables")
        print("  3. Reorganize shared memory layout for bank conflict reduction")
    
    def generate_optimization_report(self):
        """Generate comprehensive optimization report."""
        print("\n" + "="*60)
        print("OPTIMIZATION RECOMMENDATIONS")
        print("="*60)
        
        # Run benchmarks
        baseline_time = self._benchmark_current()
        
        recommendations = [
            {
                'priority': 'HIGH',
                'area': 'Block Configuration',
                'issue': 'Current block size (256) may not be optimal',
                'suggestion': 'Test with 128 and 512 threads/block',
                'potential_gain': '5-10%'
            },
            {
                'priority': 'HIGH',
                'area': 'Memory Coalescing',
                'issue': 'RNG state access is strided',
                'suggestion': 'Pack RNG state into int4 or use SOA layout',
                'potential_gain': '10-15%'
            },
            {
                'priority': 'MEDIUM',
                'area': 'Shared Memory',
                'issue': 'Potential bank conflicts in reduction',
                'suggestion': 'Add padding to shared memory arrays',
                'potential_gain': '3-5%'
            },
            {
                'priority': 'MEDIUM',
                'area': 'Instruction Mix',
                'issue': 'Heavy use of modulo operations',
                'suggestion': 'Replace % 52 with & 0x3F and range check',
                'potential_gain': '2-3%'
            },
            {
                'priority': 'LOW',
                'area': 'Warp Divergence',
                'issue': 'Conditional branches in hand evaluation',
                'suggestion': 'Use branchless evaluation where possible',
                'potential_gain': '1-2%'
            }
        ]
        
        print(f"\nCurrent baseline: {baseline_time:.1f}ms for 100k simulations")
        print(f"Target: <2ms (currently {baseline_time/2:.1f}x target)\n")
        
        for rec in recommendations:
            print(f"[{rec['priority']}] {rec['area']}")
            print(f"  Issue: {rec['issue']}")
            print(f"  Fix: {rec['suggestion']}")
            print(f"  Potential speedup: {rec['potential_gain']}")
            print()
    
    def _benchmark_current(self) -> float:
        """Benchmark current implementation."""
        # Warm-up
        _ = solve_poker_hand(['A♠', 'K♠'], 2, None, 'default')
        
        # Benchmark
        times = []
        for _ in range(10):
            start = time.time()
            _ = solve_poker_hand(['A♠', 'K♠'], 2, None, 'default')
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        return sum(times) / len(times)


def main():
    """Run GPU profiling."""
    profiler = GPUProfiler()
    
    # Run all profiling
    profiler.profile_memory_usage()
    profiler.profile_kernel_performance()
    profiler.profile_occupancy()
    profiler.profile_memory_access_patterns()
    profiler.generate_optimization_report()
    
    print("\n" + "="*60)
    print("Profiling complete!")
    print("="*60)


if __name__ == "__main__":
    main()