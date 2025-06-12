# Poker Knight NG - Implementation TODO List

## Phase 1: Project Setup
- [x] Create project directory structure as defined in ARCHITECTURE.md
- [x] Create setup.py and pyproject.toml for package configuration
- [x] Install CuPy for CUDA 12.x and verify GPU access
- [x] Create all __init__.py files and basic module structure

## Phase 2: Core Components
- [ ] Define card representation system (0-51 integers) and Unicode conversion utilities
- [ ] Implement validator.py with input validation for all API parameters
- [ ] Create data structures for GPU input/output buffers

## Phase 3: Memory Management
- [ ] Implement memory_manager.py with CuPy memory pool management
- [ ] Create pinned memory allocation strategy for CPU-GPU transfers
- [ ] Implement pre-allocated buffer pools for 10k, 100k, 500k simulations

## Phase 4: CUDA Kernel Development
- [ ] Create constants.cuh with poker hand rankings and evaluation constants
- [ ] Implement fast 7-card hand evaluator in CUDA
- [ ] Implement CUDA RNG initialization (cuRAND or similar)
- [ ] Create monolithic kernel structure with 5 phases as per ARCHITECTURE.md
- [ ] Implement Phase 2: Monte Carlo simulation loop with grid-stride pattern
- [ ] Implement Phase 3: Statistical reduction for win/tie/loss probabilities
- [ ] Implement Phase 4: ICM equity calculations in specialized warp
- [ ] Implement Phase 4: Board texture analysis in specialized warp
- [ ] Implement Phase 4: SPR, pot odds, and MDF calculations
- [ ] Implement Phase 4: Range equity and positional scoring
- [ ] Implement Phase 5: Final result aggregation and output

## Phase 5: Python Wrapper
- [ ] Create kernel_wrapper.py with CuPy RawKernel integration
- [ ] Implement GPU error handling and retry logic
- [ ] Add OOM handling with automatic batch size reduction

## Phase 6: API Implementation
- [ ] Implement api.py with solve_poker_hand() main entry point
- [ ] Create result_builder.py for SimulationResult construction
- [ ] Ensure all fields from apiNG.md are populated correctly
- [ ] Add execution timing with millisecond precision

## Phase 7: Testing Infrastructure
- [ ] Create unit tests for input validation
- [ ] Implement accuracy tests for known poker probabilities (AA vs KK, etc.)
- [ ] Add hand category frequency validation tests
- [ ] Create ICM calculation accuracy tests
- [ ] Implement confidence interval statistical tests
- [ ] Create performance benchmark framework with timestamped results
- [ ] Add test result comparison tool for regression detection

## Phase 8: Performance Optimization
- [ ] Profile GPU utilization and identify bottlenecks
- [ ] Optimize memory access patterns for coalescing
- [ ] Tune block and grid dimensions for maximum occupancy

## Phase 9: Documentation & Polish
- [ ] Add comprehensive docstrings to all public functions
- [ ] Create usage examples demonstrating all API features
- [ ] Update CLAUDE.md with any build/test commands discovered

## Phase 10: Final Validation
- [ ] Run full API compliance test suite
- [ ] Verify performance meets targets (<2ms for 100k simulations)
- [ ] Package for pip installation and test install process

## Implementation Order

The recommended implementation order follows the phases above, with these key dependencies:

1. **Setup must be complete first** - Can't proceed without project structure and CuPy
2. **Core components before CUDA** - Need data structures defined before kernel
3. **Memory management early** - Required for both kernel and wrapper
4. **CUDA kernel is the heart** - Most complex component, needs careful implementation
5. **Python wrapper depends on kernel** - Can't wrap what doesn't exist
6. **API pulls it all together** - Integrates all components
7. **Testing throughout** - Add tests as each component is completed
8. **Optimization after correctness** - Don't optimize until it works
9. **Documentation as you go** - Document while implementation is fresh
10. **Final validation** - Ensure everything meets requirements

## Critical Path Items

These items block the most other work:
1. CuPy installation and GPU verification
2. Card representation and data structures
3. Memory pool implementation
4. Basic kernel structure
5. API entry point

## Risk Items

These items have the highest implementation risk:
1. 7-card hand evaluator performance
2. ICM calculations accuracy
3. Memory management for large simulations
4. Achieving <2ms target for 100k simulations
5. Monolithic kernel complexity

## Notes

- Each completed item should be checked off in this file
- When items are completed, remove any related TODO comments from source
- If new tasks are discovered, add them to the appropriate phase
- Performance targets are aggressive - may need iteration