#ifndef POKER_KNIGHT_NG_CUDA_SOURCES_PHILOX_CUH
#define POKER_KNIGHT_NG_CUDA_SOURCES_PHILOX_CUH

#include <cstdint>

#if defined(__CUDACC__)
#define POKER_KNIGHT_NG_HD __host__ __device__
#else
#define POKER_KNIGHT_NG_HD
#endif

namespace poker_knight_ng { namespace cuda {

// Exact ADR 0003 Philox4x32-10. Counter lane order is (sim low, sim high,
// draw slot, rejection attempt); callers consume output lane zero only.
POKER_KNIGHT_NG_HD inline void philox4x32_10(const std::uint32_t counter[4], const std::uint32_t key[2], std::uint32_t output[4]) {
  std::uint32_t c0=counter[0], c1=counter[1], c2=counter[2], c3=counter[3];
  std::uint32_t k0=key[0], k1=key[1];
  for (std::uint32_t round=0; round<10; ++round) {
    const std::uint64_t p0=std::uint64_t(0xD2511F53u)*c0;
    const std::uint64_t p1=std::uint64_t(0xCD9E8D57u)*c2;
    const std::uint32_t hi0=static_cast<std::uint32_t>(p0>>32), lo0=static_cast<std::uint32_t>(p0);
    const std::uint32_t hi1=static_cast<std::uint32_t>(p1>>32), lo1=static_cast<std::uint32_t>(p1);
    const std::uint32_t n0=hi1^c1^k0, n2=hi0^c3^k1;
    c0=n0; c1=lo1; c2=n2; c3=lo0;
    if (round != 9) { k0 += 0x9E3779B9u; k1 += 0xBB67AE85u; }
  }
  output[0]=c0; output[1]=c1; output[2]=c2; output[3]=c3;
}

}} // namespace poker_knight_ng::cuda
#endif
