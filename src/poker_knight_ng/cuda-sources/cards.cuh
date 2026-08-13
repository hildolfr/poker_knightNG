#ifndef POKER_KNIGHT_NG_CUDA_CARDS_CUH
#define POKER_KNIGHT_NG_CUDA_CARDS_CUH

#include <cstdint>

#if defined(__CUDACC__)
#define PKNG_HD __host__ __device__
#else
#define PKNG_HD
#endif

namespace poker_knight_ng::cuda {

constexpr std::uint8_t kDeckSize = 52;
constexpr std::uint8_t kMinRank = 2;
constexpr std::uint8_t kMaxRank = 14;

// ADR 0002: suit * 13 + rank, rank 2..A maps to 0..12, suits shdc map to 0..3.
PKNG_HD constexpr std::uint8_t card_rank(const std::uint8_t card) {
  return static_cast<std::uint8_t>(card % 13U + kMinRank);
}

PKNG_HD constexpr std::uint8_t card_suit(const std::uint8_t card) {
  return static_cast<std::uint8_t>(card / 13U);
}

}  // namespace poker_knight_ng::cuda

#endif  // POKER_KNIGHT_NG_CUDA_CARDS_CUH
