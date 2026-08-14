"""Source-level conformance for the replacement deterministic CUDA two-stage kernels.

These tests deliberately do not compile or launch CUDA: checkpoint E supplies inert native
sources only, with no compiler/runtime/engine wiring.  CUDA compilation belongs to later
qualification work on a CUDA-capable runner.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "poker_knight_ng" / "cuda-sources" / "deterministic_kernels.cu"


def test_replacement_kernels_fuse_trials_to_per_block_partials_then_parallel_fixed_final_reduce() -> None:
    text = SOURCE.read_text()
    # Stage one fuses simulation and local reduction: no TrialResult array is materialized.
    assert 'extern "C" __global__ void pkng_simulate_block_partials_kernel' in text
    assert "AggregateResult* block_partials" in text
    assert "TrialResult trial{};" in text
    assert "reduce_trial(trial, simulation_id, &local);" in text
    assert "block_partials[blockIdx.x] = shared[0];" in text
    assert "TrialResult* results" not in text
    assert "pkng_simulate_trials_kernel" not in text

    # The final reducer has a fixed-width cooperative tree, not a thread-zero loop.
    assert 'extern "C" __global__ void pkng_reduce_block_partials_kernel' in text
    assert "constexpr unsigned BLOCK_PARTIAL_THREADS = 128U;" in text
    assert "constexpr unsigned FINAL_REDUCER_THREADS = 128U;" in text
    assert "blockDim.y != 1U || blockDim.z != 1U" in text
    assert "gridDim.y != 1U || gridDim.z != 1U" in text
    assert "block_partials[blockIdx.x] = invalid;" in text
    assert "if (count - index <= stride) break;" in text
    assert "blockDim.x != FINAL_REDUCER_THREADS" in text
    assert "gridDim.x != 1U" in text
    assert "*aggregate = invalid;" in text
    assert "blockIdx.x != 0U" in text
    assert "!aggregate" in text
    assert "for (std::uint64_t i = threadIdx.x; i < partial_count; i += FINAL_REDUCER_THREADS)" in text
    assert "for (unsigned stride = FINAL_REDUCER_THREADS / 2U; stride > 0U; stride /= 2U)" in text
    assert "merge_aggregate(shared[threadIdx.x + stride], &shared[threadIdx.x]);" in text
    assert "if (threadIdx.x == 0U) *aggregate = shared[0];" in text
    assert "reduce_range(" not in text
    assert "atomic" not in text.lower()
    assert "curand" not in text.lower()
    assert "float" not in text.lower()


def test_replacement_source_consumes_only_approved_simulation_and_reduction_headers() -> None:
    text = SOURCE.read_text()
    assert '#include "simulate.cuh"' in text
    assert '#include "reduce.cuh"' in text
    for forbidden in ("constants.cuh", "rng.cuh", "hand_evaluator.cuh", "icm_calculator.cuh", "board_analyzer.cuh", "cooperative_groups"):
        assert forbidden not in text
