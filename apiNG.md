# Poker Knight API (apiNG)

## Function

```python
from poker_knight import solve_poker_hand

result = solve_poker_hand(hero_hand, num_opponents, board_cards, simulation_mode, 
                         hero_position, stack_sizes, pot_size, tournament_context)
```

## Inputs

### Required
- `hero_hand` (List[str]): Your 2 cards. Use Unicode suits: ♠ ♥ ♦ ♣
  - Example: `['A♠', 'K♠']`
- `num_opponents` (int): Number of opponents (1-6)

### Optional
- `board_cards` (List[str]): Community cards (0, 3, 4, or 5 cards)
  - Example: `['Q♠', 'J♠', 'T♥']`
- `simulation_mode` (str): `"fast"` (10k), `"default"` (100k), `"precision"` (500k)
- `hero_position` (str): `"early"`, `"middle"`, `"late"`, `"button"`, `"sb"`, `"bb"`
- `stack_sizes` (List[int]): `[hero_stack, opp1_stack, opp2_stack, ...]`
- `pot_size` (int): Current pot size
- `tournament_context` (Dict): Tournament info for ICM
- `action_to_hero` (str): `"check"`, `"bet"`, `"raise"`, `"reraise"` - Current action facing hero
- `bet_size` (float): Current bet size relative to pot (e.g., 0.5 for half-pot)
- `street` (str): `"preflop"`, `"flop"`, `"turn"`, `"river"`
- `players_to_act` (int): Number of players still to act after hero
- `tournament_stage` (str): `"early"`, `"middle"`, `"bubble"`, `"final_table"`
- `blind_level` (int): Current blind level for tournament pressure

## Returns

`SimulationResult` object with fields:

### Always Present
- `win_probability` (float): 0.0-1.0
- `tie_probability` (float): 0.0-1.0  
- `loss_probability` (float): 0.0-1.0
- `execution_time_ms` (float): Execution time
- `execution_time_start` (float): Unix timestamp when execution started
- `execution_time_end` (float): Unix timestamp when execution completed

### Statistical
- `confidence_interval` (Tuple[float, float]): 95% confidence range
- `hand_category_frequencies` (Dict[str, float]): Frequencies for:
  - `'high_card'`, `'pair'`, `'two_pair'`, `'three_of_a_kind'`, `'straight'`,
  - `'flush'`, `'full_house'`, `'four_of_a_kind'`, `'straight_flush'`, `'royal_flush'`

### Multi-Way Analysis (when 3+ players or position specified)
- `position_aware_equity` (Dict[str, float])
- `multi_way_statistics` (Dict[str, Any])
- `fold_equity_estimates` (Dict[str, float])
- `coordination_effects` (Dict[str, float])
- `defense_frequencies` (Dict[str, float])
- `bluff_catching_frequency` (float)
- `range_coordination_score` (float)

### Tournament/ICM (when stack_sizes provided)
- `icm_equity` (float)
- `bubble_factor` (float)
- `stack_to_pot_ratio` (float)
- `tournament_pressure` (Dict[str, float])

### Advanced Analysis
- `spr` (float): Stack-to-pot ratio for commitment decisions
- `pot_odds` (float): Odds being offered by current bet
- `mdf` (float): Minimum defense frequency against bet size
- `equity_needed` (float): Breakeven equity required to call
- `commitment_threshold` (float): SPR where hero is pot-committed
- `nuts_possible` (List[str]): Possible nut hands on current board
- `draw_combinations` (Dict[str, int]): Count of flush/straight draws
- `board_texture_score` (float): 0.0-1.0 (dry to wet)
- `equity_vs_range_percentiles` (Dict[str, float]): Hero equity vs top X% of hands
- `positional_advantage_score` (float): Quantified positional value
- `hand_vulnerability` (float): Likelihood of being outdrawn

## Example

```python
result = solve_poker_hand(['A♠', 'K♠'], 2, ['Q♠', 'J♠', 'T♥'])
print(f"Win: {result.win_probability:.1%}")
print(f"Flush frequency: {result.hand_category_frequencies['flush']:.1%}")
```
