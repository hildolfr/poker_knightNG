"""
Main API entry point for poker_knight_ng.

This module provides the solve_poker_hand function which serves as the
primary interface for the poker solver, matching the API defined in apiNG.md.
"""

from typing import Optional, List, Union, Dict, Any
import time
import cupy as cp
import logging

from .validator import validate_inputs, ValidationError
from .memory_manager import get_memory_manager
from .cuda.kernel_wrapper import get_poker_kernel
from .gpu_structures import GPUInputBuffers, GPUOutputBuffers
from .result_builder import build_simulation_result

# Configure logging
logger = logging.getLogger(__name__)

# Global configuration for GPU keep-alive
_gpu_keepalive_config = {
    'enabled': False,
    'keep_alive_seconds': 30.0
}


def solve_poker_hand(
    hero_hand: List[str],
    num_opponents: int,
    board_cards: Optional[List[str]] = None,
    simulation_mode: str = 'default',
    hero_position: Optional[str] = None,
    stack_sizes: Optional[List[Union[int, float]]] = None,
    pot_size: Union[int, float] = 0,
    tournament_context: Optional[Dict[str, Any]] = None,
    action_to_hero: Optional[str] = None,
    bet_size: Optional[float] = None,
    street: Optional[str] = None,
    players_to_act: Optional[int] = None,
    tournament_stage: Optional[str] = None,
    blind_level: Optional[int] = None
) -> 'SimulationResult':
    """
    Solve a Texas Hold'em poker hand using GPU-accelerated Monte Carlo simulation.
    
    Args:
        hero_hand: Hero's hole cards as list, e.g., ['A♠', 'K♥']
        num_opponents: Number of opponents (1-6)
        board_cards: Community cards if any, e.g., ['Q♣', 'J♦', 'T♠']
        simulation_mode: 'fast' (10k), 'default' (100k), or 'precision' (500k)
        hero_position: Position at table ('early', 'middle', 'late', 'button', 'sb', 'bb')
        stack_sizes: Stack sizes [hero, opp1, opp2, ...]
        pot_size: Current pot size
        tournament_context: Tournament info with payouts, players_remaining, average_stack
        action_to_hero: Current action facing hero ('check', 'bet', 'raise', 'reraise')
        bet_size: Current bet size relative to pot (e.g., 0.5 for half-pot)
        street: Current street ('preflop', 'flop', 'turn', 'river')
        players_to_act: Number of players still to act after hero
        tournament_stage: Tournament stage ('early', 'middle', 'bubble', 'final_table')
        blind_level: Current blind level for tournament pressure
    
    Returns:
        SimulationResult object with all computed metrics
    """
    # Record start time
    execution_time_start = time.time()
    
    # OOM handling configuration
    oom_reduction_factors = {
        'precision': {'fallback_mode': 'default', 'reduction': 0.2},  # Try 100k
        'default': {'fallback_mode': 'fast', 'reduction': 0.1},       # Try 10k
        'fast': {'fallback_mode': None, 'reduction': 0.5}             # Try 5k
    }
    
    current_mode = simulation_mode
    attempt = 0
    max_oom_attempts = 3
    
    while attempt < max_oom_attempts:
        try:
            # Step 1: Validate inputs
            validated = validate_inputs(
                hero_hand=hero_hand,
                num_opponents=num_opponents,
                board_cards=board_cards,
                simulation_mode=current_mode,
                hero_position=hero_position,
                stack_sizes=stack_sizes,
                pot_size=pot_size,
                tournament_context=tournament_context,
                action_to_hero=action_to_hero,
                bet_size=bet_size,
                street=street,
                players_to_act=players_to_act,
                tournament_stage=tournament_stage,
                blind_level=blind_level
            )
            
            # Step 2: Get memory manager and activate appropriate pool
            memory_manager = get_memory_manager(
                keep_alive_seconds=_gpu_keepalive_config['keep_alive_seconds'],
                enable_keep_alive=_gpu_keepalive_config['enabled']
            )
            if _gpu_keepalive_config['enabled']:
                memory_manager.mark_activity()  # Mark GPU activity
            
            memory_manager.activate_pool(current_mode)
            
            try:
                # Step 3: Create GPU input buffers
                gpu_inputs = GPUInputBuffers.create(validated)
                
                # Step 4: Allocate output buffers
                gpu_outputs = GPUOutputBuffers.allocate(validated['num_simulations'])
                
                # Step 5: Get kernel and execute
                kernel = get_poker_kernel()
                kernel.execute(gpu_inputs, gpu_outputs)
                
                # Step 6: Transfer results back to CPU
                cpu_results = gpu_outputs.to_cpu()
                
                # Record end time
                execution_time_end = time.time()
                execution_time_ms = (execution_time_end - execution_time_start) * 1000
                
                # Log if we had to reduce mode
                if current_mode != simulation_mode:
                    logger.warning(f"OOM: Reduced simulation mode from '{simulation_mode}' to '{current_mode}'")
                
                # Step 7: Build and return result
                return build_simulation_result(
                    cpu_results=cpu_results,
                    validated_inputs=validated,
                    execution_time_ms=execution_time_ms,
                    execution_time_start=execution_time_start,
                    execution_time_end=execution_time_end
                )
                
            finally:
                # Always deactivate memory pool
                memory_manager.deactivate_pool()
                
        except ValidationError as e:
            # Re-raise validation errors with clear message
            raise ValueError(f"Invalid input: {e}")
            
        except (cp.cuda.memory.OutOfMemoryError, RuntimeError) as e:
            # Handle GPU OOM with automatic reduction
            attempt += 1
            
            if 'out of memory' in str(e).lower() or isinstance(e, cp.cuda.memory.OutOfMemoryError):
                logger.warning(f"GPU OOM on attempt {attempt} with mode '{current_mode}': {e}")
                
                # Try to free memory
                try:
                    memory_manager.deactivate_pool()
                    cp.get_default_memory_pool().free_all_blocks()
                    kernel.reset_device() if 'kernel' in locals() else None
                except:
                    pass
                
                # Determine next fallback
                if current_mode in oom_reduction_factors:
                    fallback = oom_reduction_factors[current_mode]
                    if fallback['fallback_mode'] and attempt == 1:
                        # First attempt: try fallback mode
                        current_mode = fallback['fallback_mode']
                        logger.info(f"Retrying with reduced mode: {current_mode}")
                        continue
                    elif attempt == 2:
                        # Second attempt: reduce current mode's simulation count
                        # This requires modifying the SIMULATION_COUNTS temporarily
                        # For now, just try the fallback mode if available
                        if current_mode == 'fast':
                            # Can't reduce further
                            break
                        current_mode = 'fast'
                        logger.info(f"Final retry with fast mode")
                        continue
                
                # If we get here, we've exhausted options
                break
            else:
                # Non-OOM runtime error
                raise RuntimeError(f"GPU kernel error: {e}")
                
        except Exception as e:
            # Log unexpected errors
            raise RuntimeError(f"Unexpected error in solve_poker_hand: {e}")
    
    # If we get here, all OOM attempts failed
    raise RuntimeError(
        f"GPU out of memory even after {attempt} attempts to reduce simulation count. "
        f"Consider using a smaller batch size or upgrading GPU memory."
    )


def enable_gpu_keepalive(keep_alive_seconds: float = 30.0) -> None:
    """
    Enable GPU keep-alive for standalone solve_poker_hand calls.
    
    When enabled, the GPU will stay warm between solve_poker_hand calls,
    dramatically reducing latency for subsequent calculations.
    
    Args:
        keep_alive_seconds: Time to keep GPU warm after last calculation (default: 30.0)
        
    Example:
        >>> enable_gpu_keepalive(60.0)  # Keep GPU warm for 60 seconds
        >>> result1 = solve_poker_hand(['A♠', 'A♥'], 2)  # Cold start
        >>> time.sleep(5)
        >>> result2 = solve_poker_hand(['K♠', 'K♥'], 2)  # Warm! Much faster
    """
    global _gpu_keepalive_config
    _gpu_keepalive_config['enabled'] = True
    _gpu_keepalive_config['keep_alive_seconds'] = keep_alive_seconds
    logger.info(f"GPU keep-alive enabled with {keep_alive_seconds}s timeout")


def disable_gpu_keepalive() -> None:
    """
    Disable GPU keep-alive for standalone solve_poker_hand calls.
    
    This returns to the default behavior where each call starts cold.
    Any active keep-alive resources will be cleaned up.
    """
    global _gpu_keepalive_config
    was_enabled = _gpu_keepalive_config['enabled']
    _gpu_keepalive_config['enabled'] = False
    
    if was_enabled:
        # Clean up memory manager if it exists
        from .memory_manager import _memory_manager
        if _memory_manager is not None and _memory_manager.enable_keep_alive:
            _memory_manager.cleanup()
        logger.info("GPU keep-alive disabled and resources cleaned up")


def is_gpu_keepalive_enabled() -> bool:
    """Check if GPU keep-alive is currently enabled."""
    return _gpu_keepalive_config['enabled']


def get_gpu_keepalive_config() -> Dict[str, Any]:
    """Get current GPU keep-alive configuration."""
    return _gpu_keepalive_config.copy()


# Convenience function for backwards compatibility with string inputs
def solve_poker_hand_string(
    hero_hand: str,
    num_opponents: int,
    board_cards: Optional[str] = None,
    **kwargs
) -> 'SimulationResult':
    """
    Convenience function that accepts string inputs for cards.
    
    Args:
        hero_hand: Hero's hole cards as string, e.g., "AsKh" or "A♠K♥"
        num_opponents: Number of opponents
        board_cards: Community cards as string, e.g., "QcJdTs" or "Q♣J♦T♠"
        **kwargs: Other parameters passed to solve_poker_hand
    
    Returns:
        SimulationResult object
    """
    # Convert string cards to lists
    hero_list = []
    for i in range(0, len(hero_hand), 2):
        if i + 1 < len(hero_hand):
            hero_list.append(hero_hand[i:i+2])
    
    board_list = None
    if board_cards:
        board_list = []
        i = 0
        while i < len(board_cards):
            # Handle Unicode suits (single char) vs letter suits
            if i + 1 < len(board_cards) and board_cards[i+1] in '♠♥♦♣':
                board_list.append(board_cards[i:i+2])
                i += 2
            elif i + 1 < len(board_cards):
                board_list.append(board_cards[i:i+2])
                i += 2
            else:
                break
    
    return solve_poker_hand(
        hero_hand=hero_list,
        num_opponents=num_opponents,
        board_cards=board_list,
        **kwargs
    )