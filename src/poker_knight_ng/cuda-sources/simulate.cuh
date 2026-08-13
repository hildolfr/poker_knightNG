#ifndef POKER_KNIGHT_NG_CUDA_SOURCES_SIMULATE_CUH
#define POKER_KNIGHT_NG_CUDA_SOURCES_SIMULATE_CUH

#include <cstdint>

#include "dealer.cuh"
#include "evaluator.cuh"

namespace poker_knight_ng::cuda {

// A result is deliberately a single fixed, trivially-copyable trial record.  A later
// deterministic reduction owns aggregate counters and ignores fields if status != OK.
enum class TrialStatus : std::uint8_t { OK = 0, RNG_REJECTION_EXHAUSTED = 1, INVALID_INPUT = 2 };
enum class TrialOutcome : std::uint8_t { UNIQUE_WIN = 0, TIE = 1, LOSS = 2 };

struct TrialResult {
  TrialStatus status = TrialStatus::INVALID_INPUT;
  TrialOutcome outcome = TrialOutcome::LOSS;
  std::uint8_t tie_other_winners = 0;
  std::uint8_t hero_category = 0;
  std::uint16_t equity_share_units = 0;
  std::uint64_t rejection_count = 0;
};

PKNG_HD constexpr TrialStatus trial_status_from_dealer(const DealerStatus status) {
  return status == DealerStatus::OK ? TrialStatus::OK :
      (status == DealerStatus::RNG_REJECTION_EXHAUSTED ? TrialStatus::RNG_REJECTION_EXHAUSTED : TrialStatus::INVALID_INPUT);
}

PKNG_HD constexpr bool hand_score_equal(const HandScore& left, const HandScore& right) {
  return !(left < right) && !(right < left);
}

// One deterministic Hold'em trial.  Host validation is authoritative; exact preconditions
// here are sorted, distinct card IDs in 0..51; exactly two hero IDs; board_count in
// {0,3,4,5}; 1..6 opponents; and pre-derived ADR 0003 k0/k1.  The sole random input is
// simulation_id.  Dealt slots are missing board cards then two hole cards per opponent.
// No thread/grid state, atomics, dynamic allocation, float, curand, or aggregate storage.
PKNG_HD inline TrialStatus simulate_one_trial(const std::uint8_t hero[2], const std::uint8_t* board,
    const std::uint32_t board_count, const std::uint32_t opponents, const std::uint32_t k0,
    const std::uint32_t k1, const std::uint64_t simulation_id, TrialResult* result) {
  if (!result) return TrialStatus::INVALID_INPUT;
  result->status = TrialStatus::INVALID_INPUT;
  if (!hero || (board_count != 0 && !board) || !(board_count == 0 || board_count == 3 || board_count == 4 || board_count == 5) || opponents < 1 || opponents > 6) return result->status;
  if (hero[0] >= 52 || hero[1] >= 52 || hero[0] >= hero[1]) return result->status;
  for (std::uint32_t index = 0; index < board_count; ++index) {
    if (board[index] >= 52 || (index && board[index - 1] >= board[index])) return result->status;
    if (board[index] == hero[0] || board[index] == hero[1]) return result->status;
  }

  std::uint8_t known[7]{};
  known[0] = hero[0]; known[1] = hero[1];
  for (std::uint32_t index = 0; index < board_count; ++index) known[2 + index] = board[index];

  DealerResult deal{};
  const DealerStatus dealer_status = deal_adr0003(known, 2 + board_count, opponents, k0, k1, simulation_id, &deal);
  result->status = trial_status_from_dealer(dealer_status);
  if (result->status != TrialStatus::OK) return result->status;

  std::uint8_t hero_hand[7]{};
  hero_hand[0] = hero[0]; hero_hand[1] = hero[1];
  for (std::uint32_t index = 0; index < board_count; ++index) hero_hand[2 + index] = board[index];
  const std::uint32_t missing_board = 5 - board_count;
  for (std::uint32_t index = 0; index < missing_board; ++index) hero_hand[2 + board_count + index] = deal.dealt_card_ids[index];
  const HandScore hero_score = best_five_score(hero_hand, 7);

  bool has_loss = false;
  std::uint8_t equal_opponents = 0;
  for (std::uint32_t opponent = 0; opponent < opponents; ++opponent) {
    std::uint8_t opponent_hand[7]{};
    for (std::uint32_t index = 0; index < 5; ++index) opponent_hand[index] = hero_hand[2 + index];
    opponent_hand[5] = deal.dealt_card_ids[missing_board + 2 * opponent];
    opponent_hand[6] = deal.dealt_card_ids[missing_board + 2 * opponent + 1];
    const HandScore opponent_score = best_five_score(opponent_hand, 7);
    if (hero_score < opponent_score) has_loss = true;
    else if (hand_score_equal(hero_score, opponent_score)) ++equal_opponents;
  }

  result->hero_category = hero_score.category;
  result->rejection_count = 0;
  for (std::uint32_t slot = 0; slot < deal.draw_count; ++slot) result->rejection_count += deal.rejection_counts[slot];
  if (has_loss) {
    result->outcome = TrialOutcome::LOSS;
    result->tie_other_winners = 0;
    result->equity_share_units = 0;
  } else if (equal_opponents) {
    result->outcome = TrialOutcome::TIE;
    result->tie_other_winners = equal_opponents;
    result->equity_share_units = static_cast<std::uint16_t>(420 / (equal_opponents + 1));
  } else {
    result->outcome = TrialOutcome::UNIQUE_WIN;
    result->tie_other_winners = 0;
    result->equity_share_units = 420;
  }
  return result->status;
}

}  // namespace poker_knight_ng::cuda

#endif  // POKER_KNIGHT_NG_CUDA_SOURCES_SIMULATE_CUH
