"""
CuPy RawKernel wrapper for the monolithic CUDA kernel.

This module provides the Python interface to the CUDA kernel,
handling kernel compilation, parameter preparation, and execution.
"""

import cupy as cp
import numpy as np
from typing import Dict, Any, Optional
import os
from pathlib import Path


class PokerKernel:
    """Wrapper for the monolithic poker solver CUDA kernel."""
    
    def __init__(self):
        """Initialize and compile the CUDA kernel."""
        self.kernel = None
        self.shared_mem_size = 0
        self._load_and_compile()
    
    def _load_and_compile(self):
        """Load CUDA source and compile the kernel."""
        # Get the directory containing CUDA files
        cuda_dir = Path(__file__).parent
        
        # Read all required CUDA source files
        sources = []
        
        # Read all source files and combine them
        # First read headers in dependency order
        header_files = ['constants.cuh', 'hand_evaluator.cuh', 'rng.cuh']
        header_contents = []
        
        for header_file in header_files:
            header_path = cuda_dir / header_file
            if header_path.exists():
                with open(header_path, 'r') as f:
                    content = f.read()
                    # Remove #include statements for our own headers
                    lines = content.split('\n')
                    filtered_lines = []
                    for line in lines:
                        if '#include "' in line and any(h in line for h in header_files):
                            continue  # Skip includes of our own headers
                        filtered_lines.append(line)
                    header_contents.append('\n'.join(filtered_lines))
        
        # Read main kernel file
        kernel_path = cuda_dir / 'poker_kernel.cu'
        if kernel_path.exists():
            with open(kernel_path, 'r') as f:
                kernel_content = f.read()
                # Remove #include statements for our headers
                lines = kernel_content.split('\n')
                filtered_lines = []
                for line in lines:
                    if '#include "' in line and any(h in line for h in header_files):
                        continue  # Skip includes of our headers
                    filtered_lines.append(line)
                kernel_content = '\n'.join(filtered_lines)
        else:
            raise FileNotFoundError(f"Kernel source not found: {kernel_path}")
        
        # Combine all sources - headers first, then kernel
        combined_source = '\n'.join(header_contents) + '\n' + kernel_content
        
        # Compile the kernel
        try:
            self.kernel = cp.RawKernel(
                combined_source,
                'solve_poker_hand_kernel',
                backend='nvcc',
                options=('-std=c++14', '-use_fast_math')
            )
            
            # Calculate shared memory requirements
            # SharedMemory struct size = win/tie/loss counts + hand categories + workspace
            threads_per_block = 256
            self.shared_mem_size = (
                3 * threads_per_block * 4 +     # win/tie/loss counts (int)
                10 * threads_per_block * 4 +    # hand categories (int[10])
                2 * threads_per_block * 4       # workspace arrays (float)
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to compile CUDA kernel: {e}")
    
    def calculate_grid_config(self, num_simulations: int) -> tuple:
        """Calculate optimal grid and block dimensions."""
        threads_per_block = 256
        min_blocks = 32
        max_blocks = 1024
        
        # Calculate blocks based on workload
        sims_per_thread = max(1, num_simulations // (min_blocks * threads_per_block))
        ideal_blocks = (num_simulations + threads_per_block - 1) // threads_per_block
        
        # Clamp to reasonable range
        num_blocks = min(max_blocks, max(min_blocks, ideal_blocks))
        
        return (num_blocks,), (threads_per_block,)
    
    def execute(
        self,
        gpu_inputs: 'GPUInputBuffers',
        gpu_outputs: 'GPUOutputBuffers'
    ) -> None:
        """
        Execute the monolithic kernel.
        
        Args:
            gpu_inputs: Input buffers on GPU
            gpu_outputs: Pre-allocated output buffers on GPU
        """
        # Calculate grid configuration
        grid, block = self.calculate_grid_config(gpu_inputs.num_simulations)
        
        # Prepare kernel arguments
        args = (
            # Core inputs
            gpu_inputs.hero_cards,
            gpu_inputs.num_opponents,
            gpu_inputs.board_cards,
            gpu_inputs.board_cards_count,
            gpu_inputs.num_simulations,
            gpu_inputs.random_seed,
            
            # Optional inputs
            gpu_inputs.hero_position_idx,
            gpu_inputs.stack_sizes,
            gpu_inputs.pot_size,
            gpu_inputs.action_to_hero_idx,
            gpu_inputs.bet_size,
            gpu_inputs.street_idx,
            gpu_inputs.players_to_act,
            
            # Tournament context
            gpu_inputs.has_tournament_context,
            gpu_inputs.payouts,
            gpu_inputs.players_remaining,
            gpu_inputs.average_stack,
            gpu_inputs.tournament_stage_idx,
            gpu_inputs.blind_level,
            
            # Outputs
            gpu_outputs.win_probability,
            gpu_outputs.tie_probability,
            gpu_outputs.loss_probability,
            gpu_outputs.hand_frequencies,
            gpu_outputs.confidence_interval_low,
            gpu_outputs.confidence_interval_high,
            gpu_outputs.position_equity,
            gpu_outputs.fold_equity,
            gpu_outputs.icm_equity,
            gpu_outputs.bubble_factor,
            gpu_outputs.spr,
            gpu_outputs.pot_odds,
            gpu_outputs.mdf,
            gpu_outputs.equity_needed,
            gpu_outputs.commitment_threshold,
            gpu_outputs.board_texture_score,
            gpu_outputs.flush_draw_count,
            gpu_outputs.straight_draw_count,
            gpu_outputs.equity_percentiles,
            gpu_outputs.positional_advantage,
            gpu_outputs.hand_vulnerability,
            gpu_outputs.actual_simulations
        )
        
        # Launch kernel
        self.kernel(
            grid=grid,
            block=block,
            args=args,
            shared_mem=self.shared_mem_size
        )
        
        # Synchronize to ensure completion
        cp.cuda.Device().synchronize()
        
        # Post-process results on GPU to ensure proper normalization
        # This is a workaround for the synchronization issue in the kernel
        self._normalize_results(gpu_outputs)
    
    def _normalize_results(self, gpu_outputs):
        """Normalize probability results that are currently raw counts."""
        # Get total simulations from the counts
        total = float(gpu_outputs.win_probability[0] + 
                     gpu_outputs.tie_probability[0] + 
                     gpu_outputs.loss_probability[0])
        
        if total > 0:
            # Normalize probabilities
            gpu_outputs.win_probability[0] /= total
            gpu_outputs.tie_probability[0] /= total
            gpu_outputs.loss_probability[0] /= total
            
            # Normalize hand frequencies
            for i in range(10):
                gpu_outputs.hand_frequencies[i] /= total
            
            # Recalculate confidence intervals with normalized probabilities
            win_p = float(gpu_outputs.win_probability[0])
            std_error = cp.sqrt(win_p * (1.0 - win_p) / total)
            gpu_outputs.confidence_interval_low[0] = max(0.0, win_p - 1.96 * std_error)
            gpu_outputs.confidence_interval_high[0] = min(1.0, win_p + 1.96 * std_error)
    
    def get_memory_requirements(self, num_simulations: int) -> Dict[str, int]:
        """Estimate memory requirements for given simulation count."""
        grid, block = self.calculate_grid_config(num_simulations)
        num_blocks = grid[0]
        threads_per_block = block[0]
        
        # RNG state per thread
        rng_memory = num_blocks * threads_per_block * 16  # 4 * 4 bytes
        
        # Shared memory per block
        shared_per_block = self.shared_mem_size
        total_shared = num_blocks * shared_per_block
        
        # Temporary arrays in kernel (estimate)
        temp_memory = num_simulations * 100  # ~100 bytes per simulation
        
        return {
            'rng_states': rng_memory,
            'shared_memory': total_shared,
            'temp_memory': temp_memory,
            'total': rng_memory + total_shared + temp_memory
        }


# Global kernel instance
_kernel_instance: Optional[PokerKernel] = None


def get_poker_kernel() -> PokerKernel:
    """Get or create the global kernel instance."""
    global _kernel_instance
    if _kernel_instance is None:
        _kernel_instance = PokerKernel()
    return _kernel_instance