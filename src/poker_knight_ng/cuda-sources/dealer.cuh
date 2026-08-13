#ifndef POKER_KNIGHT_NG_CUDA_SOURCES_DEALER_CUH
#define POKER_KNIGHT_NG_CUDA_SOURCES_DEALER_CUH

#include <cstdint>
#include "philox.cuh"

namespace poker_knight_ng { namespace cuda {

enum class DealerStatus : std::uint32_t { OK=0, RNG_REJECTION_EXHAUSTED=1, INVALID_INPUT=2 };
struct DealerResult {
  std::uint8_t dealt_card_ids[17]{};
  std::uint32_t final_attempts[17]{};
  std::uint32_t rejection_counts[17]{};
  std::uint8_t draw_count=0;
  DealerStatus status=DealerStatus::INVALID_INPUT;
};

// Advance one rejection-sampling candidate. From zero, the dealer preserves the
// invariant rejection_count == attempt: accepted candidates leave both unchanged;
// a nonterminal rejection increments both; the terminal candidate does neither.
POKER_KNIGHT_NG_HD inline DealerStatus dealer_candidate_step(std::uint32_t candidate, std::uint32_t range,
    std::uint32_t* attempt, std::uint32_t* rejection_count, std::uint32_t* index, bool* accepted) {
  if (!range || !attempt || !rejection_count || !index || !accepted) return DealerStatus::INVALID_INPUT;
  *accepted=false;
  const std::uint64_t limit=((std::uint64_t(1)<<32)/range)*range;
  if (std::uint64_t(candidate)<limit) { *index=candidate%range; *accepted=true; return DealerStatus::OK; }
  if (*attempt==UINT32_MAX) return DealerStatus::RNG_REJECTION_EXHAUSTED;
  ++*attempt; ++*rejection_count;
  return DealerStatus::OK;
}

// Host validation is authoritative. Preconditions here: known ids are distinct 0..51,
// known_count=2+B with B in {0,3,4,5}, opponents 1..6; key words are pre-derived;
// simulation_id is uint64. It returns ordered slots: missing board then opponent holes.
POKER_KNIGHT_NG_HD inline DealerStatus deal_adr0003(const std::uint8_t* known, std::uint32_t known_count,
    std::uint32_t opponents, std::uint32_t k0, std::uint32_t k1, std::uint64_t simulation_id, DealerResult* result) {
  if (!result) return DealerStatus::INVALID_INPUT;
  if (!known || !(known_count==2 || known_count==5 || known_count==6 || known_count==7) || opponents<1 || opponents>6) { result->status=DealerStatus::INVALID_INPUT; return result->status; }
  const std::uint32_t board_count=known_count-2, draws=5-board_count+2*opponents;
  if (draws<2 || draws>17) { result->status=DealerStatus::INVALID_INPUT; return result->status; }
  std::uint8_t deck[52]; std::uint32_t range=0;
  for (std::uint32_t card=0; card<52; ++card) { bool present=false; for(std::uint32_t j=0;j<known_count;++j) if(known[j]==card) present=true; if(!present) deck[range++]=static_cast<std::uint8_t>(card); }
  if (range != 52-known_count) { result->status=DealerStatus::INVALID_INPUT; return result->status; }
  for(std::uint32_t slot=0;slot<draws;++slot) {
    std::uint32_t attempt=0, rejects=0;
    for(;;) { std::uint32_t counter[4]={static_cast<std::uint32_t>(simulation_id),static_cast<std::uint32_t>(simulation_id>>32),slot,attempt}; std::uint32_t key[2]={k0,k1}, out[4], index=0; bool accepted=false; philox4x32_10(counter,key,out);
      const DealerStatus step=dealer_candidate_step(out[0], range, &attempt, &rejects, &index, &accepted);
      if (step != DealerStatus::OK) { result->status=step; return result->status; }
      if (accepted) { result->dealt_card_ids[slot]=deck[index]; deck[index]=deck[range-1]; --range; result->final_attempts[slot]=attempt; result->rejection_counts[slot]=rejects; break; }
    }
  }
  result->draw_count=static_cast<std::uint8_t>(draws); result->status=DealerStatus::OK; return result->status;
}

}} // namespace poker_knight_ng::cuda
#endif
