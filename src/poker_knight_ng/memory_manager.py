"""
GPU memory management for poker_knight_ng.

This module handles CuPy memory pool management, pinned memory allocation,
pre-allocated buffer pools for different simulation sizes, and optional
GPU keep-alive functionality for server scenarios.
"""

import cupy as cp
import numpy as np
import time
import threading
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass
from collections import deque
import atexit
import logging

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Configuration for a memory pool."""
    initial_size: int  # Initial pool size in bytes
    maximum_size: int  # Maximum pool size in bytes


@dataclass
class BufferSet:
    """Pre-allocated GPU buffers for a specific simulation size."""
    input_buffers: Dict[str, cp.ndarray]
    output_buffers: Dict[str, cp.ndarray]
    last_used: float
    size_mode: str


class WarmBufferPool:
    """Maintains pre-allocated GPU buffers for common simulation sizes."""
    
    def __init__(self):
        """Initialize warm buffer pools."""
        self.buffer_sets: Dict[str, BufferSet] = {}
        self._lock = threading.Lock()
        
        # Common buffer sizes for each mode
        self.buffer_configs = {
            'fast': {
                'num_simulations': 10_000,
                'output_sizes': {
                    'win_probability': 1,
                    'tie_probability': 1,
                    'loss_probability': 1,
                    'hand_frequencies': 10,
                    'confidence_interval_low': 1,
                    'confidence_interval_high': 1,
                    'actual_simulations': 1,
                    # Advanced features
                    'icm_equity': 1,
                    'board_texture_score': 1,
                    'flush_draw_count': 1,
                    'straight_draw_count': 1,
                    'spr': 1,
                    'pot_odds': 1,
                    'mdf': 1,
                    'positional_advantage': 1,
                    'hand_vulnerability': 1
                }
            },
            'default': {
                'num_simulations': 100_000,
                'output_sizes': None  # Same as fast
            },
            'precision': {
                'num_simulations': 500_000,
                'output_sizes': None  # Same as fast
            }
        }
        
        # Copy output sizes to other modes
        for mode in ['default', 'precision']:
            self.buffer_configs[mode]['output_sizes'] = \
                self.buffer_configs['fast']['output_sizes'].copy()
    
    def get_or_create_buffers(self, mode: str) -> Tuple[Dict[str, cp.ndarray], Dict[str, cp.ndarray]]:
        """Get pre-warmed buffers for mode, creating if necessary."""
        with self._lock:
            if mode not in self.buffer_sets:
                self._create_buffer_set(mode)
            
            buffer_set = self.buffer_sets[mode]
            buffer_set.last_used = time.time()
            
            return buffer_set.input_buffers, buffer_set.output_buffers
    
    def _create_buffer_set(self, mode: str):
        """Create a new buffer set for the given mode."""
        config = self.buffer_configs[mode]
        
        # Input buffers (relatively small)
        input_buffers = {
            'hero_cards': cp.zeros(2, dtype=cp.int32),
            'board_cards': cp.zeros(5, dtype=cp.int32),
            'stack_sizes': cp.zeros(7, dtype=cp.float32),
            'payouts': cp.zeros(10, dtype=cp.float32)
        }
        
        # Output buffers
        output_buffers = {}
        for name, size in config['output_sizes'].items():
            if size == 1:
                output_buffers[name] = cp.zeros(1, dtype=cp.float32)
            else:
                output_buffers[name] = cp.zeros(size, dtype=cp.float32)
        
        self.buffer_sets[mode] = BufferSet(
            input_buffers=input_buffers,
            output_buffers=output_buffers,
            last_used=time.time(),
            size_mode=mode
        )
        
        logger.debug(f"Created warm buffer set for {mode} mode")
    
    def cleanup_stale_buffers(self, max_age_seconds: float = 60):
        """Release buffers that haven't been used recently."""
        with self._lock:
            current_time = time.time()
            modes_to_remove = []
            
            for mode, buffer_set in self.buffer_sets.items():
                if current_time - buffer_set.last_used > max_age_seconds:
                    modes_to_remove.append(mode)
            
            for mode in modes_to_remove:
                del self.buffer_sets[mode]
                logger.debug(f"Released stale buffer set for {mode} mode")
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Get current memory usage of warm buffers."""
        total_bytes = 0
        
        with self._lock:
            for buffer_set in self.buffer_sets.values():
                # Input buffers
                for buf in buffer_set.input_buffers.values():
                    total_bytes += buf.nbytes
                
                # Output buffers
                for buf in buffer_set.output_buffers.values():
                    total_bytes += buf.nbytes
        
        return {
            'num_buffer_sets': len(self.buffer_sets),
            'total_bytes': total_bytes
        }


class MemoryManager:
    """Manages GPU memory allocation and pooling strategies with optional keep-alive."""
    
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
    
    def __init__(self,
                 keep_alive_seconds: float = 30.0,
                 enable_keep_alive: bool = False,
                 heartbeat_interval: float = 5.0,
                 max_warm_buffer_memory_mb: int = 500):
        """Initialize memory pools for different simulation modes.
        
        Args:
            keep_alive_seconds: Time to keep GPU warm after last activity
            enable_keep_alive: Whether to enable keep-alive functionality
            heartbeat_interval: Seconds between heartbeat GPU operations
            max_warm_buffer_memory_mb: Maximum memory for warm buffers
        """
        self.pools: Dict[str, cp.cuda.MemoryPool] = {}
        self.pinned_allocators: Dict[str, cp.cuda.PinnedMemoryPool] = {}
        self.current_pool: Optional[cp.cuda.MemoryPool] = None
        self.original_pool = cp.get_default_memory_pool()
        self.original_pinned_pool = cp.get_default_pinned_memory_pool()
        
        # Keep-alive configuration
        self.keep_alive_seconds = keep_alive_seconds
        self.enable_keep_alive = enable_keep_alive
        self.heartbeat_interval = heartbeat_interval
        self.max_warm_buffer_memory_mb = max_warm_buffer_memory_mb
        
        # Activity tracking
        self._last_activity_time = time.time()
        self._activity_history = deque(maxlen=100)
        self._lock = threading.Lock()
        
        # Warm buffer pool (only if keep-alive enabled)
        self.warm_buffers = WarmBufferPool() if enable_keep_alive else None
        
        # Keep-alive threading
        self._heartbeat_thread = None
        self._cleanup_timer = None
        self._shutdown_event = threading.Event()
        
        # Initialize memory pools for each mode
        self._initialize_pools()
        
        # Pre-allocate pinned memory pools
        self._initialize_pinned_pools()
        
        # Start heartbeat if enabled
        if self.enable_keep_alive:
            self._start_heartbeat()
        
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
        
        # Mark activity if keep-alive enabled
        if self.enable_keep_alive:
            self.mark_activity()
    
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
        
        # Kernel-specific estimates
        # Each simulation needs:
        # - RNG state: 4 bytes
        # - Deck state: 52 bytes
        # - Temporary arrays: ~200 bytes
        kernel_overhead = num_simulations * 256
        
        # Output arrays (float32 = 4 bytes)
        output_arrays = {
            'win_counts': num_simulations * 4,
            'tie_counts': num_simulations * 4,
            'hand_results': num_simulations * 4,
            # ICM arrays (if enabled)
            'icm_equity': num_simulations * 4,
            # Board analysis (if enabled)
            'board_texture': num_simulations * 4,
        }
        
        total_output = sum(output_arrays.values())
        
        return {
            'workspace': workspace,
            'kernel_overhead': kernel_overhead,
            'output_arrays': total_output,
            'total': workspace + kernel_overhead + total_output,
            'total_mb': (workspace + kernel_overhead + total_output) / (1024 * 1024)
        }
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get current memory usage information."""
        mem_pool = cp.get_default_memory_pool()
        
        # Note: PinnedMemoryPool doesn't have used_bytes/total_bytes methods
        info = {
            'device_memory': {
                'used_bytes': mem_pool.used_bytes(),
                'total_bytes': mem_pool.total_bytes(),
                'free_bytes': mem_pool.total_bytes() - mem_pool.used_bytes(),
            },
            'pools': {
                mode: {
                    'used_bytes': pool.used_bytes(),
                    'total_bytes': pool.total_bytes()
                }
                for mode, pool in self.pools.items()
            }
        }
        
        return info
    
    def mark_activity(self):
        """Mark that GPU activity has occurred."""
        with self._lock:
            current_time = time.time()
            self._last_activity_time = current_time
            self._activity_history.append(current_time)
            
            # Cancel any pending cleanup
            if self._cleanup_timer:
                self._cleanup_timer.cancel()
            
            # Schedule new cleanup
            if self.enable_keep_alive:
                self._schedule_cleanup()
    
    def _schedule_cleanup(self):
        """Schedule GPU resource cleanup after inactivity period."""
        self._cleanup_timer = threading.Timer(
            self.keep_alive_seconds,
            self._perform_cleanup
        )
        self._cleanup_timer.start()
    
    def _perform_cleanup(self):
        """Perform actual cleanup of GPU resources."""
        with self._lock:
            logger.info("GPU keep-alive timeout reached, releasing resources")
            
            # Clean up warm buffers
            if self.warm_buffers:
                self.warm_buffers.cleanup_stale_buffers(0)  # Force cleanup all
            
            # Free memory pools
            for pool in self.pools.values():
                pool.free_all_blocks()
    
    def _start_heartbeat(self):
        """Start the heartbeat thread to keep GPU warm."""
        # Don't start if already running
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            logger.debug("Heartbeat thread already running, skipping start")
            return
            
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_worker,
            daemon=True
        )
        self._heartbeat_thread.start()
        logger.debug("Started GPU heartbeat thread")
    
    def _heartbeat_worker(self):
        """Worker thread that performs minimal GPU work to keep it warm."""
        while not self._shutdown_event.is_set():
            try:
                with self._lock:
                    time_since_activity = time.time() - self._last_activity_time
                    
                    # Only do heartbeat if within keep-alive window
                    if time_since_activity < self.keep_alive_seconds:
                        # Minimal GPU work to keep context active
                        dummy = cp.zeros(1000, dtype=cp.float32)
                        dummy += 1.0
                        result = cp.sum(dummy)
                        
                        # Force synchronization to ensure GPU work happens
                        cp.cuda.Stream.null.synchronize()
                        
                        logger.debug(f"GPU heartbeat performed (idle for {time_since_activity:.1f}s)")
                    else:
                        # Also clean up stale warm buffers periodically
                        if self.warm_buffers:
                            self.warm_buffers.cleanup_stale_buffers(self.keep_alive_seconds)
                
            except Exception as e:
                logger.error(f"Error in GPU heartbeat: {e}")
            
            # Sleep until next heartbeat
            self._shutdown_event.wait(self.heartbeat_interval)
    
    def get_warm_buffers(self, mode: str) -> Tuple[Dict[str, cp.ndarray], Dict[str, cp.ndarray]]:
        """
        Get pre-warmed GPU buffers for the specified mode.
        
        Args:
            mode: Simulation mode ('fast', 'default', 'precision')
            
        Returns:
            Tuple of (input_buffers, output_buffers)
        """
        if not self.warm_buffers:
            raise RuntimeError("Warm buffers not available when keep-alive is disabled")
        
        self.mark_activity()
        return self.warm_buffers.get_or_create_buffers(mode)
    
    def should_use_warm_buffers(self) -> bool:
        """
        Determine if warm buffers should be used based on usage patterns.
        
        Returns:
            True if warm buffers are beneficial
        """
        if not self.enable_keep_alive or not self._activity_history or len(self._activity_history) < 2:
            return False
        
        # Calculate average gap between requests
        gaps = []
        for i in range(1, len(self._activity_history)):
            gap = self._activity_history[i] - self._activity_history[i-1]
            gaps.append(gap)
        
        avg_gap = sum(gaps) / len(gaps)
        
        # Use warm buffers if typical gap is less than warmup cost (~60ms)
        return avg_gap < 0.060  # 60ms threshold
    
    def get_enhanced_memory_info(self) -> Dict[str, Any]:
        """Get enhanced memory information including keep-alive status."""
        base_info = self.get_memory_info()
        
        with self._lock:
            time_since_activity = time.time() - self._last_activity_time
            warm_buffer_info = self.warm_buffers.get_memory_usage() if self.warm_buffers else {}
            
            enhanced_info = {
                **base_info,
                'keep_alive': {
                    'enabled': self.enable_keep_alive,
                    'seconds_since_activity': time_since_activity,
                    'keep_alive_seconds': self.keep_alive_seconds,
                    'is_warm': self.enable_keep_alive and time_since_activity < self.keep_alive_seconds,
                    'activity_count': len(self._activity_history)
                },
                'warm_buffers': warm_buffer_info
            }
        
        return enhanced_info
    
    def force_warmup(self):
        """Force GPU warmup by pre-allocating common buffer sizes."""
        if not self.enable_keep_alive:
            logger.warning("Force warmup called but keep-alive is disabled")
            return
        
        if not self.warm_buffers:
            logger.warning("Force warmup called but warm buffers not available")
            # Still do basic GPU warmup
            test_data = cp.random.random((1000, 1000), dtype=cp.float32)
            result = cp.sum(test_data)
            cp.cuda.Stream.null.synchronize()
            return
            
        logger.info("Forcing GPU warmup...")
        
        # Pre-create buffers for all modes
        for mode in ['fast', 'default', 'precision']:
            _, _ = self.warm_buffers.get_or_create_buffers(mode)
        
        # Do some GPU work to warm up
        test_data = cp.random.random((1000, 1000), dtype=cp.float32)
        result = cp.sum(test_data)
        cp.cuda.Stream.null.synchronize()
        
        self.mark_activity()
        logger.info("GPU warmup complete")
    
    def cleanup(self):
        """Release memory resources."""
        # Stop heartbeat thread if running
        if self.enable_keep_alive:
            self._shutdown_event.set()
            if self._heartbeat_thread:
                self._heartbeat_thread.join(timeout=1.0)
            
            # Cancel any pending cleanup timer
            if self._cleanup_timer:
                self._cleanup_timer.cancel()
        
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
        
        if self.enable_keep_alive:
            logger.info("Memory manager cleanup complete")


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager(keep_alive_seconds: float = 30.0,
                       enable_keep_alive: bool = False,
                       force_new: bool = False) -> MemoryManager:
    """Get or create the global memory manager instance.
    
    Args:
        keep_alive_seconds: Time to keep GPU warm after last activity
        enable_keep_alive: Whether to enable keep-alive functionality
        force_new: Force creation of new instance (for testing)
    
    Returns:
        MemoryManager instance with requested configuration
    """
    global _memory_manager
    
    if force_new and _memory_manager is not None:
        # Clean up existing instance
        _memory_manager.cleanup()
        _memory_manager = None
    
    # Check if we need to create a new instance or if pools were cleared
    if _memory_manager is None or len(_memory_manager.pools) == 0:
        _memory_manager = MemoryManager(
            keep_alive_seconds=keep_alive_seconds,
            enable_keep_alive=enable_keep_alive
        )
    elif (_memory_manager.keep_alive_seconds != keep_alive_seconds or 
          _memory_manager.enable_keep_alive != enable_keep_alive):
        # Handle keep-alive state changes
        old_enable_keep_alive = _memory_manager.enable_keep_alive
        
        # Update configuration
        _memory_manager.keep_alive_seconds = keep_alive_seconds
        _memory_manager.enable_keep_alive = enable_keep_alive
        
        # Handle enabling keep-alive
        if enable_keep_alive and not old_enable_keep_alive:
            logger.info("Enabling keep-alive on existing memory manager")
            
            # Initialize warm buffers if needed
            if _memory_manager.warm_buffers is None:
                _memory_manager.warm_buffers = WarmBufferPool()
            
            # Reset shutdown event if needed
            if not hasattr(_memory_manager, '_shutdown_event') or _memory_manager._shutdown_event is None:
                _memory_manager._shutdown_event = threading.Event()
            elif _memory_manager._shutdown_event.is_set():
                _memory_manager._shutdown_event.clear()
            
            # Initialize heartbeat thread if needed
            if not hasattr(_memory_manager, '_heartbeat_thread'):
                _memory_manager._heartbeat_thread = None
            
            # Start heartbeat thread
            _memory_manager._start_heartbeat()
            
        # Handle disabling keep-alive
        elif not enable_keep_alive and old_enable_keep_alive:
            logger.info("Disabling keep-alive on existing memory manager")
            
            # Stop heartbeat thread
            _memory_manager._shutdown_event.set()
            if _memory_manager._heartbeat_thread:
                _memory_manager._heartbeat_thread.join(timeout=1.0)
                _memory_manager._heartbeat_thread = None
            
            # Cancel cleanup timer
            if _memory_manager._cleanup_timer:
                _memory_manager._cleanup_timer.cancel()
                _memory_manager._cleanup_timer = None
            
            # Clear warm buffers
            _memory_manager.warm_buffers = None
    
    return _memory_manager


# For backward compatibility, keep the enhanced manager function as an alias
def get_enhanced_memory_manager(keep_alive_seconds: float = 30.0,
                                enable_keep_alive: bool = True,
                                force_new: bool = False) -> MemoryManager:
    """Alias for get_memory_manager with keep-alive enabled by default."""
    return get_memory_manager(
        keep_alive_seconds=keep_alive_seconds,
        enable_keep_alive=enable_keep_alive,
        force_new=force_new
    )