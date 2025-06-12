"""
GPU memory management for poker_knight_ng.

This module handles CuPy memory pool management, pinned memory allocation,
and pre-allocated buffer pools for different simulation sizes.
"""

import cupy as cp
import numpy as np
from typing import Dict, Optional, Any
from dataclasses import dataclass
import atexit


@dataclass
class PoolConfig:
    """Configuration for a memory pool."""
    initial_size: int  # Initial pool size in bytes
    maximum_size: int  # Maximum pool size in bytes
    

class MemoryManager:
    """Manages GPU memory allocation and pooling strategies."""
    
    # Memory pool configurations for different simulation modes
    POOL_CONFIGS = {
        'fast': PoolConfig(
            initial_size=100 * 1024 * 1024,      # 100 MB
            maximum_size=500 * 1024 * 1024       # 500 MB
        ),
        'default': PoolConfig(
            initial_size=1024 * 1024 * 1024,     # 1 GB
            maximum_size=2048 * 1024 * 1024      # 2 GB
        ),
        'precision': PoolConfig(
            initial_size=2048 * 1024 * 1024,     # 2 GB
            maximum_size=5120 * 1024 * 1024      # 5 GB
        )
    }
    
    # Estimated memory requirements per simulation
    BYTES_PER_SIMULATION = {
        # Each simulation needs memory for:
        # - Deck state (52 bytes)
        # - Hand evaluations (7 * 4 bytes)
        # - Results storage (4 bytes)
        # - Temporary variables (~100 bytes)
        'workspace': 200  # bytes per simulation
    }
    
    def __init__(self):
        """Initialize memory pools for different simulation modes."""
        self.pools: Dict[str, cp.cuda.MemoryPool] = {}
        self.pinned_allocators: Dict[str, cp.cuda.PinnedMemoryPool] = {}
        self.current_pool: Optional[cp.cuda.MemoryPool] = None
        self.original_pool = cp.get_default_memory_pool()
        self.original_pinned_pool = cp.get_default_pinned_memory_pool()
        
        # Initialize memory pools for each mode
        self._initialize_pools()
        
        # Pre-allocate pinned memory pools
        self._initialize_pinned_pools()
        
        # Register cleanup on exit
        atexit.register(self.cleanup)
    
    def _initialize_pools(self):
        """Initialize CuPy memory pools for different simulation modes."""
        for mode, config in self.POOL_CONFIGS.items():
            # Create a new memory pool
            pool = cp.cuda.MemoryPool()
            
            # Set the pool's initial size (this is advisory)
            # CuPy will grow the pool as needed up to maximum_size
            
            self.pools[mode] = pool
    
    def _initialize_pinned_pools(self):
        """Initialize pinned memory pools for efficient CPU-GPU transfers."""
        # Create pinned memory pools for each mode
        for mode in self.POOL_CONFIGS:
            pinned_pool = cp.cuda.PinnedMemoryPool()
            self.pinned_allocators[mode] = pinned_pool
    
    def select_pool(self, simulation_mode: str) -> cp.cuda.MemoryPool:
        """
        Select appropriate memory pool based on simulation mode.
        
        Args:
            simulation_mode: 'fast', 'default', or 'precision'
            
        Returns:
            CuPy MemoryPool instance
        """
        if simulation_mode not in self.pools:
            raise ValueError(f"Invalid simulation mode: {simulation_mode}")
        
        return self.pools[simulation_mode]
    
    def activate_pool(self, simulation_mode: str):
        """
        Activate memory pool for the given simulation mode.
        
        This sets the pool as the default for all CuPy allocations.
        """
        pool = self.select_pool(simulation_mode)
        pinned_pool = self.pinned_allocators[simulation_mode]
        
        # Set as default pools
        cp.cuda.set_allocator(pool.malloc)
        cp.cuda.set_pinned_memory_allocator(pinned_pool.malloc)
        
        self.current_pool = pool
    
    def deactivate_pool(self):
        """Restore original memory allocators."""
        cp.cuda.set_allocator(self.original_pool.malloc)
        cp.cuda.set_pinned_memory_allocator(self.original_pinned_pool.malloc)
        self.current_pool = None
    
    def estimate_memory_usage(self, num_simulations: int) -> Dict[str, int]:
        """
        Estimate memory requirements for given number of simulations.
        
        Returns:
            Dict with memory estimates in bytes
        """
        workspace = num_simulations * self.BYTES_PER_SIMULATION['workspace']
        
        # Additional memory for:
        # - RNG states (48 bytes per thread, assume 256 threads per block)
        num_blocks = (num_simulations + 255) // 256
        rng_memory = num_blocks * 256 * 48
        
        # Shared memory per block (for hand evaluation tables)
        shared_memory_per_block = 16 * 1024  # 16KB
        total_shared = num_blocks * shared_memory_per_block
        
        return {
            'workspace': workspace,
            'rng_states': rng_memory,
            'shared_memory': total_shared,
            'total': workspace + rng_memory + total_shared
        }
    
    def allocate_workspace(self, num_simulations: int) -> cp.ndarray:
        """
        Allocate workspace memory for simulations.
        
        Args:
            num_simulations: Number of simulations
            
        Returns:
            GPU array for workspace
        """
        # Allocate flat workspace that kernel can partition as needed
        workspace_size = num_simulations * self.BYTES_PER_SIMULATION['workspace']
        return cp.empty(workspace_size, dtype=cp.uint8)
    
    def create_pinned_array(self, data: np.ndarray) -> cp.ndarray:
        """
        Create a pinned memory array for efficient GPU transfer.
        
        Args:
            data: NumPy array to pin
            
        Returns:
            CuPy array in pinned memory
        """
        # Create pinned memory array
        pinned = cp.empty_like(data)
        pinned[:] = cp.asarray(data)
        return pinned
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current GPU memory usage information."""
        if self.current_pool:
            pool_info = {
                'used_bytes': self.current_pool.used_bytes(),
                'total_bytes': self.current_pool.total_bytes(),
                'n_free_blocks': self.current_pool.n_free_blocks()
            }
        else:
            pool_info = None
        
        # Get device memory info
        free_mem, total_mem = cp.cuda.runtime.memGetInfo()
        
        return {
            'device': {
                'free_bytes': free_mem,
                'total_bytes': total_mem,
                'used_bytes': total_mem - free_mem,
                'used_percent': (total_mem - free_mem) / total_mem * 100
            },
            'pool': pool_info
        }
    
    def cleanup(self):
        """Release memory resources."""
        # Free all memory pools
        for pool in self.pools.values():
            pool.free_all_blocks()
        
        for pool in self.pinned_allocators.values():
            pool.free_all_blocks()
        
        # Restore original allocators
        self.deactivate_pool()
        
        # Clear references
        self.pools.clear()
        self.pinned_allocators.clear()


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get or create the global memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager