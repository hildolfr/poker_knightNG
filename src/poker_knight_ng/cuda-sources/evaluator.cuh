#ifndef POKER_KNIGHT_NG_CUDA_EVALUATOR_CUH
#define POKER_KNIGHT_NG_CUDA_EVALUATOR_CUH

#include <cstdint>

#include "cards.cuh"

namespace poker_knight_ng::cuda {

struct HandScore {
  std::uint8_t category;
  std::uint8_t k1;
  std::uint8_t k2;
  std::uint8_t k3;
  std::uint8_t k4;
  std::uint8_t k5;

  PKNG_HD constexpr bool operator<(const HandScore& other) const {
    if (category != other.category) return category < other.category;
    if (k1 != other.k1) return k1 < other.k1;
    if (k2 != other.k2) return k2 < other.k2;
    if (k3 != other.k3) return k3 < other.k3;
    if (k4 != other.k4) return k4 < other.k4;
    return k5 < other.k5;
  }
};

PKNG_HD constexpr std::uint8_t straight_high(const std::uint16_t rank_mask) {
  // Ace may only play low in A-2-3-4-5, whose key high card is five.
  constexpr std::uint16_t kWheel = (1U << 12U) | 0x000FU;
  for (int high = 14; high >= 6; --high) {
    const std::uint16_t sequence = static_cast<std::uint16_t>(0x1FU << (high - 6));
    if ((rank_mask & sequence) == sequence) return static_cast<std::uint8_t>(high);
  }
  if ((rank_mask & kWheel) == kWheel) return 5;
  return 0;
}

PKNG_HD inline void descending_ranks(const std::uint16_t mask, std::uint8_t out[5]) {
  int next = 0;
  for (int rank = 14; rank >= 2 && next < 5; --rank) {
    if ((mask & (1U << (rank - 2))) != 0U) out[next++] = static_cast<std::uint8_t>(rank);
  }
}

PKNG_HD inline HandScore score_five(const std::uint8_t cards[5]) {
  std::uint8_t counts[15] = {};
  std::uint16_t rank_mask = 0;
  const std::uint8_t suit = card_suit(cards[0]);
  bool flush = true;
  for (int i = 0; i < 5; ++i) {
    const std::uint8_t rank = card_rank(cards[i]);
    ++counts[rank];
    rank_mask = static_cast<std::uint16_t>(rank_mask | (1U << (rank - 2U)));
    flush = flush && card_suit(cards[i]) == suit;
  }
  const std::uint8_t straight = straight_high(rank_mask);
  if (flush && straight) return {8, straight, 0, 0, 0, 0};

  std::uint8_t quad = 0;
  std::uint8_t trip = 0;
  std::uint8_t pairs[2] = {};
  int pair_count = 0;
  for (int rank = 14; rank >= 2; --rank) {
    if (counts[rank] == 4) quad = static_cast<std::uint8_t>(rank);
    else if (counts[rank] == 3) trip = static_cast<std::uint8_t>(rank);
    else if (counts[rank] == 2 && pair_count < 2) pairs[pair_count++] = static_cast<std::uint8_t>(rank);
  }
  if (quad) {
    for (int rank = 14; rank >= 2; --rank) {
      if (rank != quad && counts[rank] != 0) return {7, quad, static_cast<std::uint8_t>(rank), 0, 0, 0};
    }
  }
  if (trip && pair_count == 1) return {6, trip, pairs[0], 0, 0, 0};
  std::uint8_t ranks[5] = {};
  descending_ranks(rank_mask, ranks);
  if (flush) return {5, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]};
  if (straight) return {4, straight, 0, 0, 0, 0};
  if (trip) {
    int next = 0;
    for (int rank = 14; rank >= 2; --rank) if (rank != trip && counts[rank] != 0) ranks[next++] = static_cast<std::uint8_t>(rank);
    return {3, trip, ranks[0], ranks[1], 0, 0};
  }
  if (pair_count == 2) {
    for (int rank = 14; rank >= 2; --rank) if (rank != pairs[0] && rank != pairs[1] && counts[rank] != 0) return {2, pairs[0], pairs[1], static_cast<std::uint8_t>(rank), 0, 0};
  }
  if (pair_count == 1) {
    int next = 0;
    for (int rank = 14; rank >= 2; --rank) if (rank != pairs[0] && counts[rank] != 0) ranks[next++] = static_cast<std::uint8_t>(rank);
    return {1, pairs[0], ranks[0], ranks[1], ranks[2], 0};
  }
  return {0, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]};
}

// Exact reference path retained for conformance tests and non-seven-card inputs.
PKNG_HD inline HandScore best_five_score_combinatorial_oracle(const std::uint8_t* cards, const int count) {
  HandScore best{0, 0, 0, 0, 0, 0};
  bool have_best = false;
  for (int a = 0; a < count - 4; ++a) for (int b = a + 1; b < count - 3; ++b)
  for (int c = b + 1; c < count - 2; ++c) for (int d = c + 1; d < count - 1; ++d)
  for (int e = d + 1; e < count; ++e) {
    const std::uint8_t subset[5] = {cards[a], cards[b], cards[c], cards[d], cards[e]};
    const HandScore candidate = score_five(subset);
    if (!have_best || best < candidate) { best = candidate; have_best = true; }
  }
  return best;
}

// Direct seven-card scorer: rank histogram plus per-suit rank masks, no subset enumeration.
PKNG_HD inline HandScore score_seven_direct(const std::uint8_t* cards) {
  std::uint8_t counts[15] = {};
  std::uint8_t suit_counts[4] = {};
  std::uint16_t rank_mask = 0;
  std::uint16_t suit_masks[4] = {};
  for (int i = 0; i < 7; ++i) {
    const std::uint8_t rank = card_rank(cards[i]);
    const std::uint8_t suit = card_suit(cards[i]);
    const std::uint16_t bit = static_cast<std::uint16_t>(1U << (rank - 2U));
    ++counts[rank]; ++suit_counts[suit];
    rank_mask = static_cast<std::uint16_t>(rank_mask | bit);
    suit_masks[suit] = static_cast<std::uint16_t>(suit_masks[suit] | bit);
  }
  for (int suit = 0; suit < 4; ++suit) if (suit_counts[suit] >= 5) {
    const std::uint8_t straight_flush = straight_high(suit_masks[suit]);
    if (straight_flush) return {8, straight_flush, 0, 0, 0, 0};
  }
  std::uint8_t quad = 0, trip = 0, second_trip = 0, pair = 0, second_pair = 0;
  for (int rank = 14; rank >= 2; --rank) {
    if (counts[rank] == 4) quad = static_cast<std::uint8_t>(rank);
    else if (counts[rank] == 3) { if (!trip) trip = static_cast<std::uint8_t>(rank); else second_trip = static_cast<std::uint8_t>(rank); }
    else if (counts[rank] == 2) { if (!pair) pair = static_cast<std::uint8_t>(rank); else if (!second_pair) second_pair = static_cast<std::uint8_t>(rank); }
  }
  if (quad) { std::uint8_t ranks[5] = {}; descending_ranks(static_cast<std::uint16_t>(rank_mask & ~(1U << (quad - 2U))), ranks); return {7, quad, ranks[0], 0, 0, 0}; }
  if (trip && (second_trip || pair)) return {6, trip, second_trip ? second_trip : pair, 0, 0, 0};
  for (int suit = 0; suit < 4; ++suit) if (suit_counts[suit] >= 5) { std::uint8_t ranks[5] = {}; descending_ranks(suit_masks[suit], ranks); return {5, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]}; }
  const std::uint8_t straight = straight_high(rank_mask);
  if (straight) return {4, straight, 0, 0, 0, 0};
  std::uint8_t ranks[5] = {};
  if (trip) { descending_ranks(static_cast<std::uint16_t>(rank_mask & ~(1U << (trip - 2U))), ranks); return {3, trip, ranks[0], ranks[1], 0, 0}; }
  if (pair && second_pair) { descending_ranks(static_cast<std::uint16_t>(rank_mask & ~(1U << (pair - 2U)) & ~(1U << (second_pair - 2U))), ranks); return {2, pair, second_pair, ranks[0], 0, 0}; }
  if (pair) { descending_ranks(static_cast<std::uint16_t>(rank_mask & ~(1U << (pair - 2U))), ranks); return {1, pair, ranks[0], ranks[1], ranks[2], 0}; }
  descending_ranks(rank_mask, ranks);
  return {0, ranks[0], ranks[1], ranks[2], ranks[3], ranks[4]};
}

PKNG_HD inline HandScore best_five_score(const std::uint8_t* cards, const int count) {
  if (count == 7) return score_seven_direct(cards);
  return best_five_score_combinatorial_oracle(cards, count);
}

}  // namespace poker_knight_ng::cuda

#endif  // POKER_KNIGHT_NG_CUDA_EVALUATOR_CUH
