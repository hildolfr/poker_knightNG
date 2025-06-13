# Poker Knight API v2 (apiNG v2)

**Updated**: Now provides more data with less restrictive conditions. Many fields that previously required specific inputs are now always calculated when possible.

## Functions

### Standard API

```python
from poker_knight_ng import solve_poker_hand

result = solve_poker_hand(hero_hand, num_opponents, board_cards, simulation_mode, 
                         hero_position, stack_sizes, pot_size, tournament_context)
```

### Standard API with GPU Keep-Alive (New)

```python
from poker_knight_ng import solve_poker_hand, enable_gpu_keepalive, disable_gpu_keepalive

# Enable GPU keep-alive for standalone calls
enable_gpu_keepalive(keep_alive_seconds=30.0)

# Now solve_poker_hand calls benefit from GPU warmth
result1 = solve_poker_hand(['A♠', 'A♥'], 2)  # First call: ~200ms
result2 = solve_poker_hand(['K♠', 'K♥'], 2)  # Subsequent calls: ~3ms

# Check if keep-alive is enabled
from poker_knight_ng import is_gpu_keepalive_enabled, get_gpu_keepalive_config
if is_gpu_keepalive_enabled():
    config = get_gpu_keepalive_config()
    print(f"Keep-alive timeout: {config['keep_alive_seconds']}s")

# Disable when done
disable_gpu_keepalive()
```

### Server API (New)

```python
from poker_knight_ng import create_poker_server

# Create server with GPU keep-alive
server = create_poker_server(keep_alive_seconds=30.0, auto_warmup=True)

# Single solve (same interface as solve_poker_hand)
result = server.solve(hero_hand, num_opponents, board_cards, simulation_mode,
                     hero_position, stack_sizes, pot_size, tournament_context)

# Batch solving (processed sequentially for now)
results = server.solve_batch(problems)

# Session context
with server.session():
    result1 = server.solve(...)
    result2 = server.solve(...)
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
- `actual_simulations` (int): Number of simulations actually performed

### Statistical
- `confidence_interval` (Tuple[float, float]): 95% confidence range
- `hand_category_frequencies` (Dict[str, float]): Frequencies for:
  - `'high_card'`, `'pair'`, `'two_pair'`, `'three_of_a_kind'`, `'straight'`,
  - `'flush'`, `'full_house'`, `'four_of_a_kind'`, `'straight_flush'`, `'royal_flush'`

### Multi-Way Analysis (when 2+ players or position specified)
- `position_aware_equity` (Dict[str, float]): Position-based equity values (includes 0 values)
- `multi_way_statistics` (Dict[str, Any])
- `fold_equity_estimates` (Dict[str, float]): All positions shown, even if 0
- `coordination_effects` (Dict[str, float])
- `defense_frequencies` (Dict[str, float])
- `bluff_catching_frequency` (float)
- `range_coordination_score` (float)

### Tournament/ICM (when stack_sizes provided)
- `icm_equity` (float): ICM value or chip EV if no tournament context
- `bubble_factor` (float): Always calculated from stack distributions
- `stack_to_pot_ratio` (float): SPR when pot > 0
- `tournament_pressure` (Dict[str, float]): Includes effective stacks

### Advanced Analysis (always present, may be 0)
- `spr` (float): Stack-to-pot ratio (or stack in BBs if pot=0)
- `pot_odds` (float): Odds being offered by current bet (0 if no bet)
- `mdf` (float): Minimum defense frequency against bet size (0 if no bet)
- `equity_needed` (float): Breakeven equity required to call (0 if no bet)
- `commitment_threshold` (float): SPR where hero is pot-committed (always calculated)
- `nuts_possible` (List[str]): Possible nut hands on current board
- `draw_combinations` (Dict[str, int]): Count of flush/straight draws (requires 3+ board cards)
- `board_texture_score` (float): 0.0-1.0 (dry to wet, 0 if < 3 board cards)
- `equity_vs_range_percentiles` (Dict[str, float]): Hero equity vs top X% of hands
- `positional_advantage_score` (float): Quantified positional value (0 if no position)
- `hand_vulnerability` (float): Likelihood of being outdrawn (always calculated)

## Examples

### Basic Usage

```python
from poker_knight_ng import solve_poker_hand

result = solve_poker_hand(['A♠', 'K♠'], 2, ['Q♠', 'J♠', 'T♥'])
print(f"Win: {result.win_probability:.1%}")
print(f"Flush frequency: {result.hand_category_frequencies['flush']:.1%}")
```

### Server Usage (Recommended for Multiple Calculations)

```python
from poker_knight_ng import create_poker_server

# Create server - GPU warmup happens once
server = create_poker_server(keep_alive_seconds=30.0)

# First solve is slower due to GPU initialization
result1 = server.solve(['A♠', 'A♥'], 2)
print(f"AA vs 2: {result1.win_probability:.1%}")

# Subsequent solves are much faster (GPU already warm)
result2 = server.solve(['K♥', 'K♦'], 3, board_cards=['Q♠', '7♣', '2♥'])
print(f"KK vs 3 on Q72: {result2.win_probability:.1%}")

# Get performance stats
stats = server.get_statistics()
print(f"Average solve time: {stats['average_solve_time_ms']:.1f}ms")
```

### Batch Processing

```python
# Process multiple hands efficiently
problems = [
    {'hero_hand': ['A♠', 'A♥'], 'num_opponents': 1},
    {'hero_hand': ['K♠', 'K♥'], 'num_opponents': 2, 'board_cards': ['Q♦', '7♣', '2♥']},
    {'hero_hand': ['J♠', 'T♠'], 'num_opponents': 1, 'simulation_mode': 'precision'},
    {
        'hero_hand': ['9♥', '9♦'],
        'num_opponents': 2,
        'tournament_context': {
            'payouts': [0.5, 0.3, 0.2],
            'stack_sizes': [1500, 1000, 500],
            'player_index': 0,
            'tournament_stage': 'bubble'
        }
    }
]

results = server.solve_batch(problems)
for i, result in enumerate(results):
    if result is not None:  # None indicates invalid input for that problem
        print(f"Problem {i}: Win={result.win_probability:.1%}")
    else:
        print(f"Problem {i}: Invalid input")
```

### Session Context

```python
# Use session for grouped calculations
with server.session():
    # Analyze a specific hand through streets
    preflop = server.solve(['A♣', 'K♣'], 2)
    flop = server.solve(['A♣', 'K♣'], 2, board_cards=['Q♣', 'J♦', '5♠'])
    turn = server.solve(['A♣', 'K♣'], 2, board_cards=['Q♣', 'J♦', '5♠', '2♥'])
    river = server.solve(['A♣', 'K♣'], 2, board_cards=['Q♣', 'J♦', '5♠', '2♥', '9♣'])
    
    print(f"Equity by street: {preflop.win_probability:.1%} → "
          f"{flop.win_probability:.1%} → {turn.win_probability:.1%} → "
          f"{river.win_probability:.1%}")
```

## Server API Configuration

### create_poker_server Parameters

- `keep_alive_seconds` (float): Time to keep GPU warm after last activity (default: 30.0)
- `auto_warmup` (bool): Automatically warm up GPU on creation (default: True)

### Performance Characteristics

- **Cold start**: ~200ms (first calculation)
- **Warm solve**: ~1-3ms (subsequent calculations)
- **Keep-alive benefit**: ~99.5% reduction in latency
- **Batch efficiency**: ~2-3ms per problem
- **Memory overhead**: ~500MB for warm buffer pools

### When to Use Server API

Use the server API when:
- Making multiple poker calculations in succession
- Building a web service or API
- Analyzing many hands for study/training
- Response time is critical
- Working with bursty workloads

Use the standard API when:
- Making only a single calculation
- Memory usage is constrained
- Running in a one-off script

### Resource Management

```python
# Always shutdown when done to free GPU resources
server.shutdown()

# The server will also automatically clean up on program exit via atexit
```

## Error Handling

Both APIs handle errors gracefully:

```python
# Invalid input returns clear error
try:
    result = solve_poker_hand(['A♠', 'A♠'], 2)  # Duplicate card
except ValueError as e:
    print(f"Error: {e}")  # "Invalid input: Duplicate card in hand: A♠"

# Batch processing continues despite individual errors
results = server.solve_batch([
    {'hero_hand': ['A♠', 'A♥'], 'num_opponents': 1},  # Valid
    {'hero_hand': ['K♠', 'K♠'], 'num_opponents': 2},  # Invalid - duplicate
    {'hero_hand': ['Q♠', 'Q♥'], 'num_opponents': 3},  # Valid
])
# results = [SimulationResult(...), None, SimulationResult(...)]
```

## Performance Tips

1. **Use Server API for Multiple Calculations**: The performance benefit is significant
2. **Batch Similar Problems**: Group calculations to maximize GPU efficiency
3. **Choose Appropriate Simulation Mode**: 
   - `fast` for real-time decisions
   - `default` for analysis
   - `precision` for critical spots
4. **Set Reasonable Keep-Alive**: 30-60 seconds works well for most use cases
5. **Monitor GPU Memory**: Use `server.get_statistics()` to track resource usage