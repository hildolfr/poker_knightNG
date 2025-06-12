"""
GPU data structures for poker_knight_ng.

This module defines the input and output buffer structures for the monolithic
CUDA kernel, ensuring proper memory alignment and efficient GPU access patterns.
"""

import numpy as np
import cupy as cp
from typing import Optional, Dict, Any, NamedTuple
from dataclasses import dataclass


# GPU-friendly data types
FLOAT32 = np.float32
INT32 = np.int32
UINT32 = np.uint32
BOOL = np.bool_


@dataclass
class GPUInputBuffers:
    """Input buffers for the monolithic CUDA kernel.
    
    All arrays are pre-allocated and pinned for efficient GPU transfer.
    Memory layout is optimized for coalesced access patterns.
    """
    # Core inputs (always required)
    hero_cards: cp.ndarray          # shape: (2,), dtype: int32
    num_opponents: int              # scalar
    num_simulations: int            # scalar
    
    # Board cards (optional, -1 for unused slots)
    board_cards: cp.ndarray         # shape: (5,), dtype: int32
    board_cards_count: int          # 0, 3, 4, or 5
    
    # Advanced parameters (all optional, use defaults if not provided)
    hero_position_idx: int          # -1 for none, 0-5 for positions
    stack_sizes: cp.ndarray         # shape: (7,), dtype: float32, -1 for unused
    pot_size: float                 # float32
    
    # Action context
    action_to_hero_idx: int         # -1 for none, 0-3 for actions
    bet_size: float                 # float32, relative to pot
    street_idx: int                 # -1 for none, 0-3 for streets
    players_to_act: int             # int32
    
    # Tournament context
    has_tournament_context: bool
    payouts: cp.ndarray             # shape: (10,), dtype: float32, tournament payouts
    players_remaining: int          # int32
    average_stack: float            # float32
    tournament_stage_idx: int       # -1 for none, 0-3 for stages
    blind_level: int                # int32
    
    # Random seed for reproducibility
    random_seed: int                # uint32
    
    @classmethod
    def create(cls, validated_inputs: Dict[str, Any], random_seed: Optional[int] = None) -> 'GPUInputBuffers':
        """Create GPU input buffers from validated inputs."""
        # Initialize with defaults
        hero_cards = cp.array(validated_inputs['hero_cards'], dtype=INT32)
        
        # Board cards with padding
        board_cards = cp.full(5, -1, dtype=INT32)
        if validated_inputs['board_cards']:
            board_cards_np = np.array(validated_inputs['board_cards'], dtype=INT32)
            board_cards[:len(board_cards_np)] = cp.asarray(board_cards_np)
            board_cards_count = len(validated_inputs['board_cards'])
        else:
            board_cards_count = 0
        
        # Position mapping
        position_map = {'early': 0, 'middle': 1, 'late': 2, 'button': 3, 'sb': 4, 'bb': 5}
        hero_position_idx = position_map.get(validated_inputs.get('hero_position'), -1)
        
        # Stack sizes with padding
        stack_sizes = cp.full(7, -1.0, dtype=FLOAT32)
        if 'stack_sizes' in validated_inputs and validated_inputs['stack_sizes']:
            stack_sizes_np = np.array(validated_inputs['stack_sizes'], dtype=FLOAT32)
            stack_sizes[:len(stack_sizes_np)] = cp.asarray(stack_sizes_np)
        
        # Action mapping
        action_map = {'check': 0, 'bet': 1, 'raise': 2, 'reraise': 3}
        action_to_hero_idx = action_map.get(validated_inputs.get('action_to_hero'), -1)
        
        # Street mapping
        street_map = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
        street_idx = street_map.get(validated_inputs.get('street'), -1)
        
        # Tournament context
        has_tournament_context = 'tournament_context' in validated_inputs and validated_inputs['tournament_context']
        payouts = cp.zeros(10, dtype=FLOAT32)
        players_remaining = 0
        average_stack = 0.0
        
        if has_tournament_context:
            tc = validated_inputs['tournament_context']
            if 'payouts' in tc and tc['payouts']:
                payouts_np = np.array(tc['payouts'][:10], dtype=FLOAT32)
                payouts[:len(payouts_np)] = cp.asarray(payouts_np)
            players_remaining = tc.get('players_remaining', 0)
            average_stack = float(tc.get('average_stack', 0))
        
        # Tournament stage mapping
        stage_map = {'early': 0, 'middle': 1, 'bubble': 2, 'final_table': 3}
        tournament_stage_idx = stage_map.get(validated_inputs.get('tournament_stage'), -1)
        
        # Random seed
        if random_seed is None:
            random_seed = np.random.randint(0, 2**32 - 1)
        
        return cls(
            hero_cards=hero_cards,
            num_opponents=validated_inputs['num_opponents'],
            num_simulations=validated_inputs['num_simulations'],
            board_cards=board_cards,
            board_cards_count=board_cards_count,
            hero_position_idx=hero_position_idx,
            stack_sizes=stack_sizes,
            pot_size=float(validated_inputs.get('pot_size', 0)),
            action_to_hero_idx=action_to_hero_idx,
            bet_size=float(validated_inputs.get('bet_size', 0)),
            street_idx=street_idx,
            players_to_act=validated_inputs.get('players_to_act', 0),
            has_tournament_context=has_tournament_context,
            payouts=payouts,
            players_remaining=players_remaining,
            average_stack=average_stack,
            tournament_stage_idx=tournament_stage_idx,
            blind_level=validated_inputs.get('blind_level', 0),
            random_seed=random_seed
        )


@dataclass
class GPUOutputBuffers:
    """Output buffers for the monolithic CUDA kernel.
    
    Pre-allocated buffers for all possible outputs. The kernel fills
    only the relevant buffers based on input parameters.
    """
    # Core probabilities
    win_probability: cp.ndarray         # shape: (1,), dtype: float32
    tie_probability: cp.ndarray         # shape: (1,), dtype: float32
    loss_probability: cp.ndarray        # shape: (1,), dtype: float32
    
    # Hand category frequencies (10 categories)
    hand_frequencies: cp.ndarray        # shape: (10,), dtype: float32
    
    # Statistical measures
    confidence_interval_low: cp.ndarray  # shape: (1,), dtype: float32
    confidence_interval_high: cp.ndarray # shape: (1,), dtype: float32
    
    # Multi-way analysis (when applicable)
    position_equity: cp.ndarray         # shape: (6,), dtype: float32
    fold_equity: cp.ndarray             # shape: (6,), dtype: float32
    
    # Tournament/ICM (when applicable)
    icm_equity: cp.ndarray              # shape: (1,), dtype: float32
    bubble_factor: cp.ndarray           # shape: (1,), dtype: float32
    
    # Advanced metrics
    spr: cp.ndarray                     # shape: (1,), dtype: float32
    pot_odds: cp.ndarray                # shape: (1,), dtype: float32
    mdf: cp.ndarray                     # shape: (1,), dtype: float32
    equity_needed: cp.ndarray           # shape: (1,), dtype: float32
    commitment_threshold: cp.ndarray    # shape: (1,), dtype: float32
    
    # Board analysis
    board_texture_score: cp.ndarray     # shape: (1,), dtype: float32
    flush_draw_count: cp.ndarray        # shape: (1,), dtype: int32
    straight_draw_count: cp.ndarray     # shape: (1,), dtype: int32
    
    # Range analysis
    equity_percentiles: cp.ndarray      # shape: (5,), dtype: float32 (vs top 10%, 20%, 30%, 50%, 100%)
    positional_advantage: cp.ndarray    # shape: (1,), dtype: float32
    hand_vulnerability: cp.ndarray      # shape: (1,), dtype: float32
    
    # Execution metadata
    actual_simulations: cp.ndarray      # shape: (1,), dtype: int32
    
    @classmethod
    def allocate(cls, max_simulations: int) -> 'GPUOutputBuffers':
        """Allocate output buffers on GPU."""
        return cls(
            # Core probabilities
            win_probability=cp.zeros(1, dtype=FLOAT32),
            tie_probability=cp.zeros(1, dtype=FLOAT32),
            loss_probability=cp.zeros(1, dtype=FLOAT32),
            
            # Hand frequencies
            hand_frequencies=cp.zeros(10, dtype=FLOAT32),
            
            # Statistical
            confidence_interval_low=cp.zeros(1, dtype=FLOAT32),
            confidence_interval_high=cp.zeros(1, dtype=FLOAT32),
            
            # Multi-way
            position_equity=cp.zeros(6, dtype=FLOAT32),
            fold_equity=cp.zeros(6, dtype=FLOAT32),
            
            # Tournament
            icm_equity=cp.zeros(1, dtype=FLOAT32),
            bubble_factor=cp.zeros(1, dtype=FLOAT32),
            
            # Advanced
            spr=cp.zeros(1, dtype=FLOAT32),
            pot_odds=cp.zeros(1, dtype=FLOAT32),
            mdf=cp.zeros(1, dtype=FLOAT32),
            equity_needed=cp.zeros(1, dtype=FLOAT32),
            commitment_threshold=cp.zeros(1, dtype=FLOAT32),
            
            # Board analysis
            board_texture_score=cp.zeros(1, dtype=FLOAT32),
            flush_draw_count=cp.zeros(1, dtype=INT32),
            straight_draw_count=cp.zeros(1, dtype=INT32),
            
            # Range analysis
            equity_percentiles=cp.zeros(5, dtype=FLOAT32),
            positional_advantage=cp.zeros(1, dtype=FLOAT32),
            hand_vulnerability=cp.zeros(1, dtype=FLOAT32),
            
            # Metadata
            actual_simulations=cp.zeros(1, dtype=INT32)
        )
    
    def to_cpu(self) -> Dict[str, Any]:
        """Transfer results back to CPU memory."""
        return {
            # Core probabilities
            'win_probability': float(self.win_probability.get()[0]),
            'tie_probability': float(self.tie_probability.get()[0]),
            'loss_probability': float(self.loss_probability.get()[0]),
            
            # Hand frequencies
            'hand_frequencies': self.hand_frequencies.get().tolist(),
            
            # Statistical
            'confidence_interval': (
                float(self.confidence_interval_low.get()[0]),
                float(self.confidence_interval_high.get()[0])
            ),
            
            # Multi-way
            'position_equity': self.position_equity.get().tolist(),
            'fold_equity': self.fold_equity.get().tolist(),
            
            # Tournament
            'icm_equity': float(self.icm_equity.get()[0]),
            'bubble_factor': float(self.bubble_factor.get()[0]),
            
            # Advanced
            'spr': float(self.spr.get()[0]),
            'pot_odds': float(self.pot_odds.get()[0]),
            'mdf': float(self.mdf.get()[0]),
            'equity_needed': float(self.equity_needed.get()[0]),
            'commitment_threshold': float(self.commitment_threshold.get()[0]),
            
            # Board analysis
            'board_texture_score': float(self.board_texture_score.get()[0]),
            'flush_draw_count': int(self.flush_draw_count.get()[0]),
            'straight_draw_count': int(self.straight_draw_count.get()[0]),
            
            # Range analysis
            'equity_percentiles': self.equity_percentiles.get().tolist(),
            'positional_advantage': float(self.positional_advantage.get()[0]),
            'hand_vulnerability': float(self.hand_vulnerability.get()[0]),
            
            # Metadata
            'actual_simulations': int(self.actual_simulations.get()[0])
        }


# Hand ranking constants for GPU
HAND_CATEGORIES = [
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

# Position indices
POSITION_INDICES = {
    'early': 0,
    'middle': 1,
    'late': 2,
    'button': 3,
    'sb': 4,
    'bb': 5
}