"""
Enhanced GPU memory management with keep-alive functionality.

This module extends the basic memory manager with features for server scenarios:
- Time-based GPU keep-alive after last activity
- Pre-warmed buffer pools for common sizes
- Heartbeat mechanism to prevent GPU idle
- Configurable keep-alive duration
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

from .memory_manager import MemoryManager, PoolConfig

logger = logging.getLogger(__name__)


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


class MemoryManagerEnhanced(MemoryManager):
    """Enhanced memory manager with GPU keep-alive functionality."""
    
    def __init__(
        self,
        keep_alive_seconds: float = 30.0,
        enable_keep_alive: bool = True,
        heartbeat_interval: float = 5.0,
        max_warm_buffer_memory_mb: int = 500
    ):
        """
        Initialize enhanced memory manager.
        
        Args:
            keep_alive_seconds: Time to keep GPU warm after last activity
            enable_keep_alive: Whether to enable keep-alive functionality
            heartbeat_interval: Seconds between heartbeat GPU operations
            max_warm_buffer_memory_mb: Maximum memory for warm buffers
        """
        super().__init__()
        
        # Keep-alive configuration
        self.keep_alive_seconds = keep_alive_seconds
        self.enable_keep_alive = enable_keep_alive
        self.heartbeat_interval = heartbeat_interval
        self.max_warm_buffer_memory_mb = max_warm_buffer_memory_mb
        
        # Activity tracking
        self._last_activity_time = time.time()
        self._activity_history = deque(maxlen=100)
        self._lock = threading.Lock()
        
        # Warm buffer pool
        self.warm_buffers = WarmBufferPool()
        
        # Keep-alive threading
        self._heartbeat_thread = None
        self._cleanup_timer = None
        self._shutdown_event = threading.Event()
        
        # Start heartbeat if enabled
        if self.enable_keep_alive:
            self._start_heartbeat()
        
        # Register enhanced cleanup
        atexit.register(self.enhanced_cleanup)
    
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
            self.warm_buffers.cleanup_stale_buffers(0)  # Force cleanup all
            
            # Free memory pools
            for pool in self.pools.values():
                pool.free_all_blocks()
    
    def _start_heartbeat(self):
        """Start the heartbeat thread to keep GPU warm."""
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
        self.mark_activity()
        return self.warm_buffers.get_or_create_buffers(mode)
    
    def should_use_warm_buffers(self) -> bool:
        """
        Determine if warm buffers should be used based on usage patterns.
        
        Returns:
            True if warm buffers are beneficial
        """
        if not self._activity_history or len(self._activity_history) < 2:
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
            warm_buffer_info = self.warm_buffers.get_memory_usage()
            
            enhanced_info = {
                **base_info,
                'keep_alive': {
                    'enabled': self.enable_keep_alive,
                    'seconds_since_activity': time_since_activity,
                    'keep_alive_seconds': self.keep_alive_seconds,
                    'is_warm': time_since_activity < self.keep_alive_seconds,
                    'activity_count': len(self._activity_history)
                },
                'warm_buffers': warm_buffer_info
            }
        
        return enhanced_info
    
    def activate_pool(self, simulation_mode: str):
        """Activate memory pool and mark activity."""
        super().activate_pool(simulation_mode)
        self.mark_activity()
    
    def enhanced_cleanup(self):
        """Enhanced cleanup that handles keep-alive resources."""
        # Stop heartbeat thread
        self._shutdown_event.set()
        if self._heartbeat_thread:
            self._heartbeat_thread.join(timeout=1.0)
        
        # Cancel any pending cleanup timer
        if self._cleanup_timer:
            self._cleanup_timer.cancel()
        
        # Perform base cleanup
        self.cleanup()
        
        logger.info("Enhanced memory manager cleanup complete")
    
    def force_warmup(self):
        """Force GPU warmup by pre-allocating common buffer sizes."""
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


# Global enhanced memory manager instance
_enhanced_memory_manager: Optional[MemoryManagerEnhanced] = None


def get_enhanced_memory_manager(
    keep_alive_seconds: float = 30.0,
    enable_keep_alive: bool = True,
    force_new: bool = False
) -> MemoryManagerEnhanced:
    """
    Get or create the global enhanced memory manager instance.
    
    Args:
        keep_alive_seconds: Time to keep GPU warm after last activity
        enable_keep_alive: Whether to enable keep-alive functionality
        force_new: Force creation of new instance (for testing)
    """
    global _enhanced_memory_manager
    
    if force_new and _enhanced_memory_manager is not None:
        # Clean up existing instance
        _enhanced_memory_manager.enhanced_cleanup()
        _enhanced_memory_manager = None
    
    if _enhanced_memory_manager is None:
        _enhanced_memory_manager = MemoryManagerEnhanced(
            keep_alive_seconds=keep_alive_seconds,
            enable_keep_alive=enable_keep_alive
        )
    elif (_enhanced_memory_manager.keep_alive_seconds != keep_alive_seconds or 
          _enhanced_memory_manager.enable_keep_alive != enable_keep_alive):
        # Update existing instance
        _enhanced_memory_manager.keep_alive_seconds = keep_alive_seconds
        _enhanced_memory_manager.enable_keep_alive = enable_keep_alive
    
    return _enhanced_memory_manager