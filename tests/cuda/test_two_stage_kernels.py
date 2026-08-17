"""Source-level conformance for the replacement deterministic CUDA two-stage kernels.

These tests deliberately do not compile or launch CUDA: checkpoint E supplies inert native
sources only, with no compiler/runtime/engine wiring.  CUDA compilation belongs to later
qualification work on a CUDA-capable runner.
"""
from __future__ import annotations

from pathlib import Path
import random

from poker_knight_ng import _cuda_runtime


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


def _two_stage_host_reduce(records: list[dict[str, object]], width: int = 128) -> dict[str, object]:
    """Executable host mirror of each CUDA lane fold plus the fixed tree result."""
    lanes = [_cuda_runtime.empty_aggregate_record() for _ in range(width)]
    for index, record in enumerate(records):
        lanes[index % width] = _cuda_runtime._merge(record, lanes[index % width])
    stride = width // 2
    while stride:
        for index in range(stride):
            lanes[index] = _cuda_runtime._merge(lanes[index + stride], lanes[index])
        stride //= 2
    return lanes[0]


def _naive_host_reduce(records: list[dict[str, object]]) -> dict[str, object]:
    result = _cuda_runtime.empty_aggregate_record()
    for record in records:
        result = _cuda_runtime._merge(record, result)
    return result


def _record(value: int) -> dict[str, object]:
    record = _cuda_runtime.empty_aggregate_record()
    record.update({
        "completed_trials": value,
        "unique_wins": value % 2,
        "losses": value % 3,
        "equity_share_units": value * 420,
        "tie_by_other_winners": [value % 5] + [0] * 5,
        "hero_category_counts": [value] + [0] * 8,
        "rejection_lo": value,
    })
    return record


def test_two_stage_shared_reduction_matches_naive_host_oracle_for_fixed_and_random_counts() -> None:
    randomizer = random.Random(0xC0FFEE)
    for count in (0, 1, 2, 127, 128, 129, 257, 513):
        values = [_record(randomizer.randrange(1, 1000)) for _ in range(count)]
        assert _two_stage_host_reduce(values) == _naive_host_reduce(values)
