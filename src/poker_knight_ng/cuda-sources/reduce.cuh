#ifndef POKER_KNIGHT_NG_CUDA_SOURCES_REDUCE_CUH
#define POKER_KNIGHT_NG_CUDA_SOURCES_REDUCE_CUH

#include <cstdint>
#include "simulate.cuh"

namespace poker_knight_ng::cuda {

// This header is a CUDA-host/device-safe algebra only: no launches, atomics, allocation,
// STL, floating point, or compiler-specific 128-bit ABI.  Counter overflows saturate the
// affected field (u128 saturates both lanes) and promote status to COUNTER_OVERFLOW.
enum class AggregateStatus : std::uint8_t {
  OK = 0, RNG_REJECTION_EXHAUSTED = 1, INVALID_INPUT = 2, COUNTER_OVERFLOW = 3,
};
constexpr std::uint64_t NO_FAILURE_SIMULATION_ID = ~std::uint64_t{0};

struct Unsigned128 { std::uint64_t lo = 0; std::uint64_t hi = 0; };
struct AggregateResult {
  AggregateStatus status = AggregateStatus::OK;
  std::uint64_t completed_trials = 0;
  std::uint64_t unique_wins = 0;
  std::uint64_t tie_by_other_winners[6]{};
  std::uint64_t losses = 0;
  std::uint64_t equity_share_units = 0;
  std::uint64_t hero_category_counts[9]{};
  Unsigned128 rejection_count{};
  std::uint64_t failure_simulation_id = NO_FAILURE_SIMULATION_ID;
  std::uint32_t failure_draw_slot = NO_FAILURE_DRAW_SLOT;
};

PKNG_HD constexpr bool known_aggregate_status(const AggregateStatus status) {
  return status == AggregateStatus::OK || status == AggregateStatus::RNG_REJECTION_EXHAUSTED ||
      status == AggregateStatus::INVALID_INPUT || status == AggregateStatus::COUNTER_OVERFLOW;
}
PKNG_HD constexpr unsigned aggregate_rank(const AggregateStatus status) {
  return status == AggregateStatus::OK ? 0U : status == AggregateStatus::RNG_REJECTION_EXHAUSTED ? 1U :
      status == AggregateStatus::INVALID_INPUT ? 2U : status == AggregateStatus::COUNTER_OVERFLOW ? 3U : 0U;
}
PKNG_HD inline void promote(AggregateResult* a, const AggregateStatus status) {
  if (known_aggregate_status(status) && (!known_aggregate_status(a->status) || aggregate_rank(status) > aggregate_rank(a->status))) a->status = status;
}
PKNG_HD inline void checked_add(std::uint64_t* destination, const std::uint64_t value, AggregateResult* a) {
  const std::uint64_t maximum = NO_FAILURE_SIMULATION_ID;
  if (maximum - *destination < value) { *destination = maximum; promote(a, AggregateStatus::COUNTER_OVERFLOW); }
  else *destination += value;
}
PKNG_HD inline void checked_add_u128(Unsigned128* destination, const Unsigned128 value, AggregateResult* a) {
  const std::uint64_t maximum = NO_FAILURE_SIMULATION_ID;
  const std::uint64_t old_lo = destination->lo;
  const std::uint64_t lo = old_lo + value.lo;
  const std::uint64_t carry = lo < old_lo ? 1U : 0U;
  if (maximum - destination->hi < value.hi || maximum - (destination->hi + value.hi) < carry) {
    destination->lo = maximum; destination->hi = maximum; promote(a, AggregateStatus::COUNTER_OVERFLOW);
  } else { destination->lo = lo; destination->hi += value.hi + carry; }
}
PKNG_HD inline bool valid_exhaustion_slot(const std::uint32_t slot) { return slot <= 16U; }
PKNG_HD inline bool no_aggregate_failure(const AggregateResult& a) {
  return a.failure_simulation_id == NO_FAILURE_SIMULATION_ID && a.failure_draw_slot == NO_FAILURE_DRAW_SLOT;
}
PKNG_HD inline bool valid_aggregate_failure(const AggregateResult& a) {
  return a.failure_simulation_id != NO_FAILURE_SIMULATION_ID && valid_exhaustion_slot(a.failure_draw_slot);
}
PKNG_HD inline bool checked_sum(std::uint64_t* total, const std::uint64_t value) {
  const std::uint64_t maximum = NO_FAILURE_SIMULATION_ID;
  if (maximum - *total < value) return false;
  *total += value; return true;
}
PKNG_HD inline bool valid_aggregate_partial(const AggregateResult& a) {
  if (!known_aggregate_status(a.status) || (!no_aggregate_failure(a) && !valid_aggregate_failure(a))) return false;
  if (a.status == AggregateStatus::OK && !no_aggregate_failure(a)) return false;
  if (a.status == AggregateStatus::RNG_REJECTION_EXHAUSTED && !valid_aggregate_failure(a)) return false;
  if (a.status == AggregateStatus::COUNTER_OVERFLOW) return true;
  std::uint64_t outcomes = 0, categories = 0, units = 0;
  if (!checked_sum(&outcomes, a.unique_wins)) return false;
  for (unsigned i=0;i<6;++i) if (!checked_sum(&outcomes, a.tie_by_other_winners[i])) return false;
  if (!checked_sum(&outcomes, a.losses) || outcomes != a.completed_trials) return false;
  for (unsigned i=0;i<9;++i) if (!checked_sum(&categories, a.hero_category_counts[i])) return false;
  if (categories != a.completed_trials) return false;
  const std::uint64_t maximum = NO_FAILURE_SIMULATION_ID;
  if (a.unique_wins > maximum / 420U || !checked_sum(&units, 420U * a.unique_wins)) return false;
  for (unsigned i=0;i<6;++i) {
    const std::uint64_t share = 420U / (i + 2U);
    if (a.tie_by_other_winners[i] > maximum / share || !checked_sum(&units, share * a.tie_by_other_winners[i])) return false;
  }
  return units == a.equity_share_units;
}
PKNG_HD inline void invalidate_aggregate(AggregateResult* a) {
  promote(a, AggregateStatus::INVALID_INPUT);
  a->failure_simulation_id = NO_FAILURE_SIMULATION_ID;
  a->failure_draw_slot = NO_FAILURE_DRAW_SLOT;
}
PKNG_HD inline void record_failure(AggregateResult* a, const std::uint64_t id, const std::uint32_t slot) {
  if (id == NO_FAILURE_SIMULATION_ID || !valid_exhaustion_slot(slot)) { promote(a, AggregateStatus::INVALID_INPUT); return; }
  if (a->failure_simulation_id == NO_FAILURE_SIMULATION_ID || id < a->failure_simulation_id ||
      (id == a->failure_simulation_id && slot < a->failure_draw_slot)) { a->failure_simulation_id = id; a->failure_draw_slot = slot; }
}
PKNG_HD inline bool valid_ok_trial(const TrialResult& r) {
  if (r.failure_draw_slot != NO_FAILURE_DRAW_SLOT || r.hero_category > 8U) return false;
  if (r.outcome == TrialOutcome::UNIQUE_WIN) return r.tie_other_winners == 0U && r.equity_share_units == 420U;
  if (r.outcome == TrialOutcome::LOSS) return r.tie_other_winners == 0U && r.equity_share_units == 0U;
  return r.outcome == TrialOutcome::TIE && r.tie_other_winners >= 1U && r.tie_other_winners <= 6U &&
      r.equity_share_units == static_cast<std::uint16_t>(420U / (r.tie_other_winners + 1U));
}

// Leaves reject UINT64_MAX because it is the public no-failure sentinel.  This is outside
// the normative 0..N-1 range (N <= UINT64_MAX/420), avoiding ambiguous diagnostics.
PKNG_HD inline void reduce_trial(const TrialResult& r, const std::uint64_t simulation_id, AggregateResult* a) {
  if (!a) return;
  if (!valid_aggregate_partial(*a)) invalidate_aggregate(a);
  if (simulation_id == NO_FAILURE_SIMULATION_ID) { promote(a, AggregateStatus::INVALID_INPUT); return; }
  if (r.status == TrialStatus::RNG_REJECTION_EXHAUSTED) {
    if (!valid_exhaustion_slot(r.failure_draw_slot)) promote(a, AggregateStatus::INVALID_INPUT);
    else { promote(a, AggregateStatus::RNG_REJECTION_EXHAUSTED); record_failure(a, simulation_id, r.failure_draw_slot); }
    return;
  }
  if (r.status != TrialStatus::OK || !valid_ok_trial(r)) { promote(a, AggregateStatus::INVALID_INPUT); return; }
  checked_add(&a->completed_trials, 1U, a);
  checked_add_u128(&a->rejection_count, Unsigned128{r.rejection_count, 0}, a);
  checked_add(&a->hero_category_counts[r.hero_category], 1U, a);
  if (r.outcome == TrialOutcome::UNIQUE_WIN) checked_add(&a->unique_wins, 1U, a);
  else if (r.outcome == TrialOutcome::LOSS) checked_add(&a->losses, 1U, a);
  else checked_add(&a->tie_by_other_winners[r.tie_other_winners - 1U], 1U, a);
  checked_add(&a->equity_share_units, r.equity_share_units, a);
}

PKNG_HD inline void merge_aggregate(const AggregateResult& source, AggregateResult* a) {
  if (!a) return;
  if (!valid_aggregate_partial(*a)) invalidate_aggregate(a);
  // A malformed source contributes neither counters nor identity.  Preserve any
  // already-valid exhaustion minimum in the destination while promoting the
  // public status, so malformed-input ordering cannot change diagnostics.
  if (!valid_aggregate_partial(source)) { promote(a, AggregateStatus::INVALID_INPUT); return; }
  promote(a, source.status);
  checked_add(&a->completed_trials, source.completed_trials, a); checked_add(&a->unique_wins, source.unique_wins, a);
  for (unsigned i=0;i<6;++i) checked_add(&a->tie_by_other_winners[i], source.tie_by_other_winners[i], a);
  checked_add(&a->losses, source.losses, a); checked_add(&a->equity_share_units, source.equity_share_units, a);
  for (unsigned i=0;i<9;++i) checked_add(&a->hero_category_counts[i], source.hero_category_counts[i], a);
  checked_add_u128(&a->rejection_count, source.rejection_count, a);
  if (valid_aggregate_failure(source)) record_failure(a, source.failure_simulation_id, source.failure_draw_slot);
}
PKNG_HD inline void reduce_range(const TrialResult* results, const std::uint64_t first_simulation_id, const std::uint64_t count, AggregateResult* a) {
  if (!a) return;
  if (!valid_aggregate_partial(*a)) invalidate_aggregate(a);
  const std::uint64_t maximum = NO_FAILURE_SIMULATION_ID;
  if ((!results && count) || first_simulation_id == maximum || (count && (count - 1U >= maximum - first_simulation_id))) { promote(a, AggregateStatus::INVALID_INPUT); return; }
  for (std::uint64_t i=0;i<count;++i) reduce_trial(results[i], first_simulation_id + i, a);
}
} // namespace poker_knight_ng::cuda
#endif
