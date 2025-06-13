"""
Result builder for poker_knight_ng.

This module constructs the SimulationResult object from GPU output arrays,
ensuring all fields defined in apiNG.md are properly populated.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Result object matching the API specification in apiNG.md."""
    
    # Core probabilities (always present)
    win_probability: float
    tie_probability: float
    loss_probability: float
    execution_time_ms: float
    execution_time_start: float
    execution_time_end: float
    actual_simulations: int
    
    # Statistical
    confidence_interval: Tuple[float, float]
    hand_category_frequencies: Dict[str, float]
    
    # Multi-way analysis (when 3+ players or position specified)
    position_aware_equity: Optional[Dict[str, float]] = None
    multi_way_statistics: Optional[Dict[str, Any]] = None
    fold_equity_estimates: Optional[Dict[str, float]] = None
    coordination_effects: Optional[Dict[str, float]] = None
    defense_frequencies: Optional[Dict[str, float]] = None
    bluff_catching_frequency: Optional[float] = None
    range_coordination_score: Optional[float] = None
    
    # Tournament/ICM (when stack_sizes provided)
    icm_equity: Optional[float] = None
    bubble_factor: Optional[float] = None
    stack_to_pot_ratio: Optional[float] = None
    tournament_pressure: Optional[Dict[str, float]] = None
    
    # Advanced analysis
    spr: Optional[float] = None
    pot_odds: Optional[float] = None
    mdf: Optional[float] = None
    equity_needed: Optional[float] = None
    commitment_threshold: Optional[float] = None
    nuts_possible: Optional[List[str]] = None
    draw_combinations: Optional[Dict[str, int]] = None
    board_texture_score: Optional[float] = None
    equity_vs_range_percentiles: Optional[Dict[str, float]] = None
    positional_advantage_score: Optional[float] = None
    hand_vulnerability: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {}
        for field, value in self.__dict__.items():
            if value is not None:
                result[field] = value
        return result


def build_simulation_result(
    cpu_results: Dict[str, Any],
    validated_inputs: Dict[str, Any],
    execution_time_ms: float,
    execution_time_start: float,
    execution_time_end: float
) -> SimulationResult:
    """
    Build the complete SimulationResult from GPU outputs.
    
    Args:
        cpu_results: Results transferred from GPU to CPU
        validated_inputs: Validated input parameters
        execution_time_ms: Total execution time in milliseconds
        execution_time_start: Unix timestamp when execution started
        execution_time_end: Unix timestamp when execution ended
        
    Returns:
        SimulationResult object matching apiNG.md specification
    """
    # Hand category names mapping
    hand_categories = [
        'high_card',
        'pair',
        'two_pair',
        'three_of_a_kind',
        'straight',
        'flush',
        'full_house',
        'four_of_a_kind',
        'straight_flush',
        'royal_flush'
    ]
    
    # Build hand category frequencies dictionary
    hand_freq_dict = {}
    for i, category in enumerate(hand_categories):
        if i < len(cpu_results['hand_frequencies']):
            hand_freq_dict[category] = cpu_results['hand_frequencies'][i]
    
    # Create base result with always-present fields
    result = SimulationResult(
        win_probability=cpu_results['win_probability'],
        tie_probability=cpu_results['tie_probability'],
        loss_probability=cpu_results['loss_probability'],
        execution_time_ms=execution_time_ms,
        execution_time_start=execution_time_start,
        execution_time_end=execution_time_end,
        actual_simulations=cpu_results['actual_simulations'],
        confidence_interval=cpu_results['confidence_interval'],
        hand_category_frequencies=hand_freq_dict
    )
    
    # Add multi-way analysis if applicable (even for 2 players if position data exists)
    if validated_inputs['num_opponents'] >= 1 or validated_inputs.get('hero_position'):
        position_names = ['early', 'middle', 'late', 'button', 'sb', 'bb']
        position_equity = {}
        fold_equity = {}
        
        for i, pos in enumerate(position_names):
            if i < len(cpu_results['position_equity']):
                equity = cpu_results['position_equity'][i]
                # Include all values, even zeros, to show what was calculated
                position_equity[pos] = equity
                    
            if i < len(cpu_results['fold_equity']):
                fold = cpu_results['fold_equity'][i]
                fold_equity[pos] = fold
        
        # Always include if we have any position data
        if position_equity:
            result.position_aware_equity = position_equity
        if fold_equity:
            result.fold_equity_estimates = fold_equity
            
        # Add multi-way statistics
        result.multi_way_statistics = {
            'players': validated_inputs['num_opponents'] + 1,
            'simulation_mode': validated_inputs['simulation_mode']
        }
    
    # Add tournament/ICM metrics if stack sizes provided
    if validated_inputs.get('stack_sizes'):
        # Include ICM values even if 0 to show calculation was attempted
        result.icm_equity = cpu_results.get('icm_equity', 0.0)
        result.bubble_factor = cpu_results.get('bubble_factor', 1.0)
        
        # Stack to pot ratio can be calculated even without ICM
        if cpu_results.get('spr', 0) > 0:
            result.stack_to_pot_ratio = cpu_results['spr']
        
        # Add tournament pressure if we have stack data (even without full tournament context)
        result.tournament_pressure = {
            'icm_pressure': cpu_results.get('bubble_factor', 1.0),
            'stage': validated_inputs.get('tournament_stage', 'early'),
            'effective_stacks': min([s for s in validated_inputs['stack_sizes'] if s > 0], default=0)
        }
    
    # Add advanced analysis fields (include zeros to show what was calculated)
    result.spr = cpu_results.get('spr', 0.0)
    result.pot_odds = cpu_results.get('pot_odds', 0.0)
    result.mdf = cpu_results.get('mdf', 0.0)
    result.equity_needed = cpu_results.get('equity_needed', 0.0)
    result.commitment_threshold = cpu_results.get('commitment_threshold', 0.0)
    
    # Board texture and draws
    result.board_texture_score = cpu_results.get('board_texture_score', 0.0)
    
    # Always include draw combinations if board analysis was attempted
    if validated_inputs.get('board_cards') and len(validated_inputs['board_cards']) >= 3:
        result.draw_combinations = {
            'flush_draws': cpu_results.get('flush_draw_count', 0),
            'straight_draws': cpu_results.get('straight_draw_count', 0)
        }
    
    # Range analysis - include all percentiles even if 0
    if 'equity_percentiles' in cpu_results:
        percentiles = ['10', '20', '30', '50', '100']
        equity_dict = {}
        for i, pct in enumerate(percentiles):
            if i < len(cpu_results['equity_percentiles']):
                value = cpu_results['equity_percentiles'][i]
                equity_dict[f'top_{pct}_percent'] = value
        result.equity_vs_range_percentiles = equity_dict
    
    # Include positional advantage and vulnerability even if 0
    result.positional_advantage_score = cpu_results.get('positional_advantage', 0.0)
    result.hand_vulnerability = cpu_results.get('hand_vulnerability', 0.0)
    
    return result