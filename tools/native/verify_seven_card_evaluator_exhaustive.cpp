// Exhaustively certify score_seven_direct against the retained combinatorial oracle.
// Build with C++17, optimization, strict warnings, OpenMP, and the CUDA-source include directory.
#include <array>
#include <atomic>
#include <cstdint>
#include <iostream>

#include "evaluator.cuh"

namespace pkng = poker_knight_ng::cuda;

constexpr bool equal_score(const pkng::HandScore& left, const pkng::HandScore& right) {
  return left.category == right.category && left.k1 == right.k1 && left.k2 == right.k2 &&
         left.k3 == right.k3 && left.k4 == right.k4 && left.k5 == right.k5;
}

int main() {
  std::atomic<std::uint64_t> checked{0};
  std::atomic<std::uint64_t> mismatches{0};
#pragma omp parallel for schedule(dynamic, 1)
  for (int a = 0; a < 46; ++a) {
    std::array<std::uint8_t, 7> cards{};
    cards[0] = static_cast<std::uint8_t>(a);
    std::uint64_t local_checked = 0;
    std::uint64_t local_mismatches = 0;
    for (int b = a + 1; b < 47; ++b) {
      cards[1] = static_cast<std::uint8_t>(b);
      for (int c = b + 1; c < 48; ++c) {
        cards[2] = static_cast<std::uint8_t>(c);
        for (int d = c + 1; d < 49; ++d) {
          cards[3] = static_cast<std::uint8_t>(d);
          for (int e = d + 1; e < 50; ++e) {
            cards[4] = static_cast<std::uint8_t>(e);
            for (int f = e + 1; f < 51; ++f) {
              cards[5] = static_cast<std::uint8_t>(f);
              for (int g = f + 1; g < 52; ++g) {
                cards[6] = static_cast<std::uint8_t>(g);
                const auto direct = pkng::score_seven_direct(cards.data());
                const auto oracle = pkng::best_five_score_combinatorial_oracle(cards.data(), 7);
                ++local_checked;
                if (!equal_score(direct, oracle)) ++local_mismatches;
              }
            }
          }
        }
      }
    }
    checked.fetch_add(local_checked, std::memory_order_relaxed);
    mismatches.fetch_add(local_mismatches, std::memory_order_relaxed);
  }
  constexpr std::uint64_t expected = 133784560ULL;
  std::cout << "checked=" << checked.load() << " mismatches=" << mismatches.load() << '\n';
  return checked.load() == expected && mismatches.load() == 0 ? 0 : 1;
}
