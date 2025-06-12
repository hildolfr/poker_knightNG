#!/usr/bin/env python3
"""
Kernel configuration tuning tool.

This script tests different block and grid configurations
to find the optimal settings for maximum GPU occupancy and performance.
"""

import sys
import time
import numpy as np
import cupy as cp
from typing import Dict, List, Tuple

sys.path.insert(0, '/home/user/Documents/poker_knightNG/src')

from poker_knight_ng import solve_poker_hand
from poker_knight_ng.cuda.kernel_wrapper import PokerKernel


class KernelTuner:
    """Tune kernel block and grid dimensions for optimal performance."""
    
    def __init__(self):
        """Initialize tuner with device properties."""
        self.device = cp.cuda.Device()
        self.device_props = self._get_device_properties()
        self.results = []
        
    def _get_device_properties(self) -> Dict:
        """Get relevant GPU properties for tuning."""
        # Get device properties using CuPy's API
        device_id = cp.cuda.runtime.getDevice()
        device_props = cp.cuda.runtime.getDeviceProperties(device_id)
        
        props = {
            'name': device_props['name'].decode() if isinstance(device_props['name'], bytes) else str(device_props['name']),
            'compute_capability': f"{device_props['major']}.{device_props['minor']}",
            'multiprocessor_count': device_props['multiProcessorCount'],
            'max_threads_per_block': device_props['maxThreadsPerBlock'],
            'max_blocks_per_multiprocessor': 32,  # Conservative estimate
            'warp_size': device_props['warpSize'],
            'shared_memory_per_block': device_props['sharedMemPerBlock'],
            'registers_per_block': device_props['regsPerBlock'],
            'max_grid_dim_x': device_props['maxGridSize'][0],
        }
        return props
    
    def test_configuration(
        self,
        threads_per_block: int,
        num_blocks: int,
        num_simulations: int,
        num_runs: int = 3
    ) -> Dict:
        """Test a specific block/grid configuration."""
        # Monkey-patch the kernel configuration
        original_calc_config = PokerKernel.calculate_grid_config
        
        def custom_config(self, num_sims):
            return (num_blocks,), (threads_per_block,)
        
        PokerKernel.calculate_grid_config = custom_config
        
        try:
            # Warm-up run
            _ = solve_poker_hand(
                hero_hand=['A♠', 'A♥'],
                num_opponents=2,
                simulation_mode='fast'
            )
            
            # Time multiple runs
            times = []
            for _ in range(num_runs):
                start = time.time()
                result = solve_poker_hand(
                    hero_hand=['K♥', 'K♦'],
                    num_opponents=2,
                    board_cards=['Q♣', '7♦', '2♠'],
                    simulation_mode='default'  # 100k simulations
                )
                elapsed = (time.time() - start) * 1000
                times.append(elapsed)
            
            # Calculate metrics
            avg_time = np.mean(times)
            std_time = np.std(times)
            
            # Calculate theoretical occupancy
            threads_total = threads_per_block * num_blocks
            max_threads = (self.device_props['multiprocessor_count'] * 
                          self.device_props['max_threads_per_block'])
            occupancy = min(1.0, threads_total / max_threads)
            
            # Simulations per thread
            sims_per_thread = num_simulations / threads_total
            
            return {
                'threads_per_block': threads_per_block,
                'num_blocks': num_blocks,
                'avg_time_ms': avg_time,
                'std_time_ms': std_time,
                'occupancy': occupancy,
                'sims_per_thread': sims_per_thread,
                'throughput': num_simulations / (avg_time / 1000),  # sims/sec
                'win_probability': result.win_probability
            }
            
        finally:
            # Restore original configuration
            PokerKernel.calculate_grid_config = original_calc_config
    
    def find_optimal_configuration(self, num_simulations: int = 100000) -> None:
        """Test various configurations to find optimal settings."""
        print(f"Tuning kernel configuration for {num_simulations} simulations")
        print(f"GPU: {self.device_props['name']}")
        print(f"Compute Capability: {self.device_props['compute_capability']}")
        print(f"Multiprocessors: {self.device_props['multiprocessor_count']}")
        print()
        
        # Test configurations - focus on common optimal values
        thread_configs = [128, 256, 512]
        
        # Filter based on device capabilities
        thread_configs = [t for t in thread_configs 
                         if t <= self.device_props['max_threads_per_block']]
        
        # For each thread configuration, test various block counts
        for threads in thread_configs:
            print(f"\nTesting {threads} threads per block:")
            
            # Calculate block counts to test
            min_blocks = max(1, self.device_props['multiprocessor_count'])
            max_blocks = min(
                self.device_props['max_grid_dim_x'],
                (num_simulations + threads - 1) // threads
            )
            
            # Test a smaller set of block counts
            block_counts = [min_blocks]
            if min_blocks * 2 <= max_blocks:
                block_counts.append(min_blocks * 2)
            if min_blocks * 4 <= max_blocks:
                block_counts.append(min_blocks * 4)
            optimal_blocks = (num_simulations + threads - 1) // threads
            if optimal_blocks not in block_counts and optimal_blocks <= max_blocks:
                block_counts.append(optimal_blocks)
            
            for blocks in block_counts:
                try:
                    config_result = self.test_configuration(
                        threads, blocks, num_simulations
                    )
                    
                    print(f"  Blocks: {blocks:5d} | "
                          f"Time: {config_result['avg_time_ms']:6.1f}ms | "
                          f"Throughput: {config_result['throughput']/1e6:5.1f}M sims/s | "
                          f"Occupancy: {config_result['occupancy']:4.1%}")
                    
                    self.results.append(config_result)
                    
                except Exception as e:
                    print(f"  Blocks: {blocks:5d} | Error: {str(e)}")
        
        # Find best configuration
        self._report_best_configurations()
    
    def _report_best_configurations(self) -> None:
        """Report the best configurations found."""
        if not self.results:
            print("\nNo successful configurations found!")
            return
        
        print("\n" + "="*60)
        print("TOP 5 CONFIGURATIONS BY THROUGHPUT")
        print("="*60)
        
        # Sort by throughput
        sorted_results = sorted(self.results, 
                               key=lambda x: x['throughput'], 
                               reverse=True)
        
        for i, config in enumerate(sorted_results[:5]):
            print(f"\n{i+1}. Threads: {config['threads_per_block']}, "
                  f"Blocks: {config['num_blocks']}")
            print(f"   Throughput: {config['throughput']/1e6:.1f}M sims/s")
            print(f"   Time: {config['avg_time_ms']:.1f}ms "
                  f"(±{config['std_time_ms']:.1f}ms)")
            print(f"   Occupancy: {config['occupancy']:.1%}")
            print(f"   Sims per thread: {config['sims_per_thread']:.1f}")
        
        # Best overall
        best = sorted_results[0]
        print("\n" + "="*60)
        print("RECOMMENDED CONFIGURATION")
        print("="*60)
        print(f"Threads per block: {best['threads_per_block']}")
        print(f"Number of blocks: {best['num_blocks']}")
        print(f"Expected performance: {best['throughput']/1e6:.1f}M simulations/second")
        
        # Generate code snippet
        print("\nSuggested kernel_wrapper.py update:")
        print("-"*40)
        print(f"""
def calculate_grid_config(self, num_simulations: int) -> tuple:
    \"\"\"Calculate optimal grid and block dimensions.\"\"\"
    threads_per_block = {best['threads_per_block']}
    
    # Optimal block count for this GPU
    optimal_blocks = {best['num_blocks']}
    sims_per_thread = num_simulations / (optimal_blocks * threads_per_block)
    
    if sims_per_thread < 10:
        # For small workloads, reduce blocks
        min_blocks = 32
        num_blocks = max(min_blocks, 
                        (num_simulations + threads_per_block - 1) // threads_per_block)
    else:
        # Use optimal configuration
        num_blocks = optimal_blocks
    
    return (num_blocks,), (threads_per_block,)
""")
    
    def test_simulation_modes(self) -> None:
        """Test optimal configurations for each simulation mode."""
        print("\n" + "="*60)
        print("TESTING SIMULATION MODES")
        print("="*60)
        
        modes = {
            'fast': 10000,
            'default': 100000,
            'precision': 500000
        }
        
        # Use current best configuration
        if not self.results:
            print("No configurations to test!")
            return
            
        best = sorted(self.results, 
                     key=lambda x: x['throughput'], 
                     reverse=True)[0]
        
        for mode, num_sims in modes.items():
            print(f"\n{mode.upper()} mode ({num_sims} simulations):")
            
            # Adjust blocks for different simulation counts
            adjusted_blocks = max(32, min(
                best['num_blocks'],
                (num_sims + best['threads_per_block'] - 1) // best['threads_per_block']
            ))
            
            try:
                config_result = self.test_configuration(
                    best['threads_per_block'],
                    adjusted_blocks,
                    num_sims,
                    num_runs=3
                )
                
                print(f"  Time: {config_result['avg_time_ms']:.1f}ms")
                print(f"  Throughput: {config_result['throughput']/1e6:.1f}M sims/s")
                print(f"  Recommended blocks: {adjusted_blocks}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")


def main():
    """Main entry point for tuning tool."""
    tuner = KernelTuner()
    
    # Find optimal configuration for default mode
    tuner.find_optimal_configuration(num_simulations=100000)
    
    # Test different simulation modes
    tuner.test_simulation_modes()
    
    print("\nTuning complete!")


if __name__ == "__main__":
    main()