# Poker Knight NG Architecture

## Overview

Poker Knight NG is a GPU-accelerated Texas Hold'em poker solver designed as a high-performance drop-in replacement for the original Poker Knight. The architecture prioritizes speed through a monolithic CUDA kernel design while maintaining the exact API compatibility defined in `apiNG.md`.

## System Requirements

- CUDA 12.8+ (current development system)
- CuPy 13.x (compatible with CUDA 12.x)
- Python 3.8+
- NVIDIA GPU with compute capability 7.0+ (for advanced features)

## Directory Structure

```
poker_knight_ng/
├── src/
│   ├── poker_knight_ng/
│   │   ├── __init__.py          # Package exports
│   │   ├── api.py               # Main API entry point
│   │   ├── validator.py         # Input validation
│   │   ├── memory_manager.py    # GPU memory management
│   │   ├── result_builder.py    # SimulationResult construction
│   │   └── cuda/
│   │       ├── kernel.cu        # Monolithic CUDA kernel
│   │       ├── kernel_wrapper.py # CuPy RawKernel wrapper
│   │       └── constants.cuh    # CUDA constants
├── tests/
│   ├── unit/                    # Individual component tests
│   ├── accuracy/                # Statistical accuracy tests
│   ├── performance/             # Benchmark tests
│   └── results/                 # Timestamped test results
├── benchmarks/
│   └── profiling/              # GPU profiling scripts
├── docs/
│   └── implementation/         # Technical documentation
└── setup.py                    # Package configuration
```

## Core Design Principles

### 1. Monolithic CUDA Kernel
- Single kernel execution for ALL computations
- No CPU fallback except for unavoidable operations (input parsing, result object creation)
- Persistent threads throughout computation lifecycle
- Warp-level primitives for efficient communication

### 2. Memory Architecture

**Host Memory Strategy:**
- Pinned memory for all CPU-GPU transfers
- Pre-allocated buffers for common simulation sizes (10k, 100k, 500k)

**Device Memory Layout:**
```cuda
// Global Memory Regions
- Input buffer (pinned): ~1KB
- Simulation state: num_simulations * state_size
- Result accumulators: fixed size based on metrics
- Scratch space: for intermediate calculations

// Shared Memory Usage (per thread block)
- Card lookup tables: 416 bytes (52 cards * 8 bytes)
- Local hand evaluator cache: 1KB
- Reduction buffers: 2KB
```

**Memory Pool Management:**
- CuPy memory pool with custom allocator
- Pre-allocated pools: 100MB (fast), 1GB (default), 5GB (precision)
- Automatic pool selection based on simulation count

### 3. API Implementation

**Entry Point:**
```python
def solve_poker_hand(hero_hand, num_opponents, board_cards=None, 
                    simulation_mode='default', **kwargs):
    # 1. CPU: Validate inputs
    # 2. CPU: Select memory pool
    # 3. CPU: Transfer inputs to GPU (pinned memory)
    # 4. GPU: Execute monolithic kernel
    # 5. CPU: Build SimulationResult from GPU output
    return SimulationResult(...)
```

**Error Handling:**
- Input validation on CPU (fail fast)
- GPU OOM: Automatic fallback to smaller batch sizes
- Invalid GPU state: Reset and retry once
- All errors wrapped in descriptive exceptions

### 4. Kernel Organization

```cuda
__global__ void solve_poker_hand_kernel(...) {
    // Phase 1: Initialization (all threads)
    setup_shared_memory();
    initialize_rng();
    
    // Phase 2: Simulation Loop (grid-stride)
    for (int sim = blockIdx.x * blockDim.x + threadIdx.x; 
         sim < num_simulations; 
         sim += blockDim.x * gridDim.x) {
        simulate_hand();
        accumulate_results();
    }
    
    // Phase 3: Statistical Reduction (cooperative groups)
    reduce_win_probabilities();
    compute_confidence_intervals();
    
    // Phase 4: Advanced Metrics (specialized warps)
    if (warp_id == 0) compute_icm_equity();
    if (warp_id == 1) analyze_board_texture();
    // ... etc
    
    // Phase 5: Final Output (block 0 only)
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        finalize_results();
    }
}
```

### 5. Testing Strategy

**Phase 1: API Compliance**
- Verify all return fields present
- Type checking and structure validation
- Edge case handling

**Phase 2: Statistical Accuracy**
- Compare against known poker probabilities
- Validate hand rankings and frequencies
- Test ICM calculations against reference implementation

**Phase 3: Performance Benchmarking**
- Baseline establishment
- Regression testing with timestamped results
- GPU utilization metrics

**Test Result Storage:**
```
tests/results/
├── 2024_03_15_142305_accuracy.json
├── 2024_03_15_143012_performance.json
└── comparison_reports/
```

### 6. Build and Packaging

**Dependencies:**
- Runtime: `cupy-cuda12x>=13.0`, `numpy>=1.20`
- Build: `nvcc` (ships with CUDA toolkit)

**Installation:**
```bash
pip install -e .  # Development
pip install poker-knight-ng  # Production
```

**CUDA Compilation:**
- JIT compilation via CuPy RawKernel
- Optional: Pre-compiled PTX for common architectures
- Compute capability 7.0+ required for advanced features

### 7. Performance Targets

| Mode | Simulations | Target Time | Memory Usage |
|------|-------------|-------------|--------------|
| Fast | 10,000 | < 1ms | ~50MB |
| Default | 100,000 | < 2ms | ~500MB |
| Precision | 500,000 | < 10ms | ~2.5GB |

### 8. Integration with Daemon Service

- Stateless design for concurrent requests
- Thread-safe GPU context management
- Request queuing handled by daemon (not this module)

## Future Considerations

- FP16 computation for non-critical paths
- Tensor Core utilization for matrix operations

## Development Workflow

1. Make changes to Python wrapper or CUDA kernel
2. Run accuracy tests to ensure correctness
3. Run performance benchmarks and compare with baseline
4. Update timestamped test results
5. Document any TODO items in source or TODO.md
6. Submit PR with test results included