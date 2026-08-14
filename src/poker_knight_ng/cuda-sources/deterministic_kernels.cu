// Replacement CUDA checkpoint E: fused block partials and a fixed final reducer.
//
// This translation unit is intentionally inert: it supplies device kernels only. It
// contains no launch wrapper, compiler integration, allocation, runtime selection, or
// legacy-kernel dependency. Host admission owns pointer/topology/key/range validation.

#include <cstdint>

#include "simulate.cuh"
#include "reduce.cuh"

namespace poker_knight_ng::cuda {
namespace {

// Both kernels require this launch width. Keeping their trees fixed makes the merge
// topology independent of scheduling; the host launcher must reject other widths.
constexpr unsigned BLOCK_PARTIAL_THREADS = 128U;
constexpr unsigned FINAL_REDUCER_THREADS = 128U;

}  // namespace

// Stage one: each block reduces its grid-stride subset directly into one exact partial.
// No global TrialResult array is materialized. Logical IDs remain first_simulation_id+i,
// regardless of the launch geometry; only their ownership by a block changes.
extern "C" __global__ void pkng_simulate_block_partials_kernel(
    const std::uint8_t hero[2], const std::uint8_t* board,
    const std::uint32_t board_count, const std::uint32_t opponents,
    const std::uint32_t k0, const std::uint32_t k1,
    const std::uint64_t first_simulation_id, const std::uint64_t count,
    AggregateResult* block_partials) {
  if (!block_partials) return;
  const bool invalid_geometry = blockDim.x != BLOCK_PARTIAL_THREADS ||
      blockDim.y != 1U || blockDim.z != 1U || gridDim.y != 1U || gridDim.z != 1U;
  if (invalid_geometry) {
    if (threadIdx.x == 0U && threadIdx.y == 0U && threadIdx.z == 0U &&
        blockIdx.y == 0U && blockIdx.z == 0U) {
      AggregateResult invalid{};
      invalidate_aggregate(&invalid);
      block_partials[blockIdx.x] = invalid;
    }
    return;
  }

  AggregateResult local{};
  const std::uint64_t maximum = NO_FAILURE_SIMULATION_ID;
  const bool valid_range = first_simulation_id != maximum &&
      (!count || count - 1U < maximum - first_simulation_id);
  const std::uint64_t start = static_cast<std::uint64_t>(blockIdx.x) * BLOCK_PARTIAL_THREADS + threadIdx.x;
  const std::uint64_t stride = static_cast<std::uint64_t>(gridDim.x) * BLOCK_PARTIAL_THREADS;
  if (!valid_range || stride == 0U) {
    invalidate_aggregate(&local);
  } else {
    for (std::uint64_t index = start; index < count;) {
      TrialResult trial{};
      const std::uint64_t simulation_id = first_simulation_id + index;
      simulate_one_trial(hero, board, board_count, opponents, k0, k1, simulation_id, &trial);
      reduce_trial(trial, simulation_id, &local);
      if (count - index <= stride) break;
      index += stride;
    }
  }

  __shared__ AggregateResult shared[BLOCK_PARTIAL_THREADS];
  shared[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride_width = BLOCK_PARTIAL_THREADS / 2U; stride_width > 0U; stride_width /= 2U) {
    if (threadIdx.x < stride_width) merge_aggregate(shared[threadIdx.x + stride_width], &shared[threadIdx.x]);
    __syncthreads();
  }
  if (threadIdx.x == 0U) block_partials[blockIdx.x] = shared[0];
}

// Stage two: one fixed-width block folds the block partials with the same checked,
// associative algebra. Each lane owns a fixed strided subsequence; the shared-memory
// binary tree is fixed at 128 lanes, so no thread-zero serial reducer remains.
extern "C" __global__ void pkng_reduce_block_partials_kernel(
    const AggregateResult* block_partials, const std::uint64_t partial_count,
    AggregateResult* aggregate) {
  if (!aggregate) return;
  const bool invalid_geometry = blockDim.x != FINAL_REDUCER_THREADS ||
      blockDim.y != 1U || blockDim.z != 1U || gridDim.x != 1U ||
      gridDim.y != 1U || gridDim.z != 1U || blockIdx.x != 0U;
  if (invalid_geometry) {
    if (blockIdx.x == 0U && blockIdx.y == 0U && blockIdx.z == 0U &&
        threadIdx.x == 0U && threadIdx.y == 0U && threadIdx.z == 0U) {
      AggregateResult invalid{};
      invalidate_aggregate(&invalid);
      *aggregate = invalid;
    }
    return;
  }

  AggregateResult local{};
  if (!block_partials && partial_count) {
    invalidate_aggregate(&local);
  } else {
    for (std::uint64_t i = threadIdx.x; i < partial_count; i += FINAL_REDUCER_THREADS) {
      merge_aggregate(block_partials[i], &local);
    }
  }

  __shared__ AggregateResult shared[FINAL_REDUCER_THREADS];
  shared[threadIdx.x] = local;
  __syncthreads();
  for (unsigned stride = FINAL_REDUCER_THREADS / 2U; stride > 0U; stride /= 2U) {
    if (threadIdx.x < stride) merge_aggregate(shared[threadIdx.x + stride], &shared[threadIdx.x]);
    __syncthreads();
  }
  if (threadIdx.x == 0U) *aggregate = shared[0];
}

}  // namespace poker_knight_ng::cuda
