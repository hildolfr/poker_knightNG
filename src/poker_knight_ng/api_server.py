"""
Server-oriented API for poker_knight_ng with GPU keep-alive.

This module provides a server-friendly interface that maintains GPU warmth
between requests, ideal for handling multiple clients or rapid successive solves.
"""

import time
import logging
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager

from .api import solve_poker_hand
from .memory_manager import get_memory_manager
from .cuda.kernel_wrapper import get_poker_kernel
from .result_builder import SimulationResult

logger = logging.getLogger(__name__)


class PokerSolverServer:
    """
    Server-oriented poker solver with GPU keep-alive functionality.
    
    This class maintains GPU resources warm between solves, reducing
    overhead for server scenarios where multiple solves happen in succession.
    """
    
    def __init__(
        self,
        keep_alive_seconds: float = 30.0,
        enable_keep_alive: bool = True,
        auto_warmup: bool = True,
        use_warm_buffers: bool = True
    ):
        """
        Initialize the poker solver server.
        
        Args:
            keep_alive_seconds: Time to keep GPU warm after last solve
            enable_keep_alive: Whether to enable GPU keep-alive
            auto_warmup: Whether to warm up GPU on initialization
            use_warm_buffers: Whether to use pre-allocated warm buffers
        """
        self.keep_alive_seconds = keep_alive_seconds
        self.enable_keep_alive = enable_keep_alive
        self.use_warm_buffers = use_warm_buffers
        
        # Initialize memory manager with keep-alive
        self.memory_manager = get_memory_manager(
            keep_alive_seconds=keep_alive_seconds,
            enable_keep_alive=enable_keep_alive
        )
        
        # Ensure kernel is loaded
        self.kernel = get_poker_kernel()
        
        # Statistics tracking
        self.solve_count = 0
        self.total_solve_time = 0.0
        self.cold_start_time = None
        self.warm_solve_times = []
        
        # Perform auto warmup if requested
        if auto_warmup:
            self.warmup()
    
    def warmup(self):
        """
        Warm up the GPU by running dummy simulations.
        
        This pre-allocates memory pools and compiles kernels,
        reducing latency for the first real solve.
        """
        logger.info("Warming up GPU...")
        start_time = time.time()
        
        # Force memory manager warmup
        self.memory_manager.force_warmup()
        
        # Run dummy simulations for each mode
        warmup_configs = [
            (['A♠', 'A♥'], 1, 'fast'),
            (['K♥', 'K♦'], 2, 'default'),
            (['Q♣', 'Q♦'], 3, 'precision')
        ]
        
        for hand, opponents, mode in warmup_configs:
            _ = solve_poker_hand(
                hero_hand=hand,
                num_opponents=opponents,
                simulation_mode=mode
            )
        
        warmup_time = (time.time() - start_time) * 1000
        logger.info(f"GPU warmup complete in {warmup_time:.1f}ms")
        
        # Record cold start time
        if self.cold_start_time is None:
            self.cold_start_time = warmup_time
    
    def solve(
        self,
        hero_hand: List[str],
        num_opponents: int,
        board_cards: Optional[List[str]] = None,
        simulation_mode: str = 'default',
        **kwargs
    ) -> SimulationResult:
        """
        Solve a poker hand with GPU keep-alive benefits.
        
        This method maintains GPU warmth between calls when used
        in rapid succession.
        
        Args:
            hero_hand: Hero's hole cards
            num_opponents: Number of opponents (1-6)
            board_cards: Community cards if any
            simulation_mode: 'fast', 'default', or 'precision'
            **kwargs: Additional parameters passed to solve_poker_hand
            
        Returns:
            SimulationResult with all computed metrics
        """
        start_time = time.time()
        
        # Mark activity for keep-alive
        self.memory_manager.mark_activity()
        
        # Solve the hand
        result = solve_poker_hand(
            hero_hand=hero_hand,
            num_opponents=num_opponents,
            board_cards=board_cards,
            simulation_mode=simulation_mode,
            **kwargs
        )
        
        # Track statistics
        solve_time = (time.time() - start_time) * 1000
        self.solve_count += 1
        self.total_solve_time += solve_time
        
        if self.solve_count > 1:  # Not the first solve
            self.warm_solve_times.append(solve_time)
            if len(self.warm_solve_times) > 100:
                self.warm_solve_times.pop(0)
        
        return result
    
    def solve_batch(
        self,
        problems: List[Dict[str, Any]],
        parallel: bool = False
    ) -> List[SimulationResult]:
        """
        Solve multiple poker problems in sequence.
        
        This method is optimized for solving many problems in succession,
        maintaining GPU warmth throughout the batch.
        
        Args:
            problems: List of problem dictionaries with solve parameters
            parallel: Whether to attempt parallel solving (future feature)
            
        Returns:
            List of SimulationResult objects
        """
        if parallel:
            logger.warning("Parallel batch solving not yet implemented, using sequential")
        
        results = []
        batch_start = time.time()
        
        logger.info(f"Starting batch solve of {len(problems)} problems")
        
        for i, problem in enumerate(problems):
            try:
                result = self.solve(**problem)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.debug(f"Completed {i + 1}/{len(problems)} solves")
                    
            except Exception as e:
                logger.error(f"Error solving problem {i}: {e}")
                results.append(None)
        
        batch_time = (time.time() - batch_start) * 1000
        avg_time = batch_time / len(problems) if problems else 0
        
        logger.info(f"Batch solve complete: {len(problems)} problems in {batch_time:.1f}ms "
                   f"(avg: {avg_time:.1f}ms/solve)")
        
        return results
    
    @contextmanager
    def session(self):
        """
        Context manager for a solving session.
        
        Usage:
            with solver.session():
                result1 = solver.solve(hand1, 2)
                result2 = solver.solve(hand2, 3)
                # GPU stays warm throughout session
        """
        logger.debug("Starting solver session")
        session_start = time.time()
        solve_count_start = self.solve_count
        
        try:
            yield self
        finally:
            session_time = time.time() - session_start
            session_solves = self.solve_count - solve_count_start
            
            if session_solves > 0:
                logger.info(f"Session complete: {session_solves} solves in {session_time:.1f}s")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get server performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        avg_solve_time = self.total_solve_time / self.solve_count if self.solve_count > 0 else 0
        avg_warm_time = sum(self.warm_solve_times) / len(self.warm_solve_times) if self.warm_solve_times else 0
        
        # Get memory manager stats
        memory_info = self.memory_manager.get_enhanced_memory_info()
        
        return {
            'solve_count': self.solve_count,
            'total_solve_time_ms': self.total_solve_time,
            'average_solve_time_ms': avg_solve_time,
            'cold_start_time_ms': self.cold_start_time,
            'average_warm_solve_time_ms': avg_warm_time,
            'warmup_benefit_ms': self.cold_start_time - avg_warm_time if self.cold_start_time and avg_warm_time else 0,
            'gpu_is_warm': memory_info['keep_alive']['is_warm'],
            'seconds_since_activity': memory_info['keep_alive']['seconds_since_activity'],
            'memory_info': memory_info
        }
    
    def shutdown(self):
        """
        Gracefully shutdown the server and release GPU resources.
        """
        logger.info("Shutting down poker solver server")
        
        # Get final statistics
        stats = self.get_statistics()
        logger.info(f"Final statistics: {self.solve_count} solves, "
                   f"avg time: {stats['average_solve_time_ms']:.1f}ms")
        
        # Cleanup will happen automatically via atexit, but we can force it
        self.memory_manager.cleanup()


class PokerSolverPool:
    """
    Pool of poker solvers for handling concurrent requests.
    
    This is a placeholder for future enhancement to support
    true concurrent solving with multiple GPU contexts.
    """
    
    def __init__(self, pool_size: int = 1, **solver_kwargs):
        """
        Initialize a pool of solvers.
        
        Args:
            pool_size: Number of solvers in the pool (currently limited to 1)
            **solver_kwargs: Arguments passed to each PokerSolverServer
        """
        if pool_size > 1:
            logger.warning("Multi-solver pools not yet implemented, using size=1")
            pool_size = 1
        
        self.pool_size = pool_size
        self.solvers = [PokerSolverServer(**solver_kwargs) for _ in range(pool_size)]
        self.current_solver = 0
    
    def get_solver(self) -> PokerSolverServer:
        """Get the next available solver from the pool."""
        solver = self.solvers[self.current_solver]
        self.current_solver = (self.current_solver + 1) % self.pool_size
        return solver
    
    def solve(self, **kwargs) -> SimulationResult:
        """Solve using the next available solver in the pool."""
        return self.get_solver().solve(**kwargs)
    
    def shutdown_all(self):
        """Shutdown all solvers in the pool."""
        for solver in self.solvers:
            solver.shutdown()


# Convenience function for simple server usage
def create_poker_server(
    keep_alive_seconds: float = 30.0,
    auto_warmup: bool = True
) -> PokerSolverServer:
    """
    Create a poker solver server with sensible defaults.
    
    Args:
        keep_alive_seconds: Time to keep GPU warm after last solve
        auto_warmup: Whether to warm up GPU on initialization
        
    Returns:
        PokerSolverServer instance ready for use
    """
    return PokerSolverServer(
        keep_alive_seconds=keep_alive_seconds,
        enable_keep_alive=True,
        auto_warmup=auto_warmup,
        use_warm_buffers=True
    )