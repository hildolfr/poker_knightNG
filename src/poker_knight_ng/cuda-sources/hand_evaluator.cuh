/*
 * Fast 7-card hand evaluator for CUDA.
 * 
 * Uses bit manipulation and lookup tables for maximum performance.
 * Evaluates poker hands and returns a 32-bit value where higher values beat lower values.
 */

#ifndef HAND_EVALUATOR_CUH
#define HAND_EVALUATOR_CUH

#include "constants.cuh"

// Device function to count set bits (population count)
__device__ __forceinline__ int popcnt(int x) {
    return __popc(x);
}

// Device function to find the position of the highest set bit
__device__ __forceinline__ int find_highest_bit(int x) {
    return 31 - __clz(x);
}

// Device function to extract the top N set bits from a bit pattern
__device__ __forceinline__ int extract_top_bits(int bits, int n) {
    int result = 0;
    for (int i = 0; i < n && bits; i++) {
        int highest = 31 - __clz(bits);
        result |= (1 << highest);
        bits &= ~(1 << highest);
    }
    return result;
}

// Device function to evaluate a 7-card poker hand
__device__ int evaluate_7card_hand(const int cards[7]) {
    // Initialize rank and suit counters
    int rank_counts[13] = {0};
    int suit_counts[4] = {0};
    int rank_bits = 0;
    
    // Count ranks and suits
    #pragma unroll
    for (int i = 0; i < 7; i++) {
        int rank = GET_RANK(cards[i]);
        int suit = GET_SUIT(cards[i]);
        rank_counts[rank]++;
        suit_counts[suit]++;
        rank_bits |= RANK_BITS[rank];
    }
    
    // Check for flush
    int flush_suit = -1;
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        if (suit_counts[s] >= 5) {
            flush_suit = s;
            break;
        }
    }
    
    // If flush exists, check for straight flush
    if (flush_suit >= 0) {
        int flush_ranks = 0;
        #pragma unroll
        for (int i = 0; i < 7; i++) {
            if (GET_SUIT(cards[i]) == flush_suit) {
                flush_ranks |= RANK_BITS[GET_RANK(cards[i])];
            }
        }
        
        // Check for straight flush patterns
        #pragma unroll
        for (int p = 0; p < 10; p++) {
            if ((flush_ranks & STRAIGHT_PATTERNS[p]) == STRAIGHT_PATTERNS[p]) {
                // Found straight flush
                if (p == 0) {
                    // Royal flush
                    return MAKE_HAND_VALUE(HAND_ROYAL_FLUSH, 0, 0, 0, 0, 0);
                } else {
                    // Regular straight flush, high card is the highest card in pattern
                    int high_card = find_highest_bit(STRAIGHT_PATTERNS[p]);
                    return MAKE_HAND_VALUE(HAND_STRAIGHT_FLUSH, high_card, 0, 0, 0, 0);
                }
            }
        }
    }
    
    // Count pairs, trips, quads
    int quads = 0, trips = 0, pairs = 0;
    int quad_rank = -1, trip_rank = -1, pair_ranks = 0;
    
    #pragma unroll
    for (int r = 12; r >= 0; r--) {  // Check from Ace down
        if (rank_counts[r] == 4) {
            quads++;
            quad_rank = r;
        } else if (rank_counts[r] == 3) {
            trips++;
            if (trip_rank < 0) trip_rank = r;
        } else if (rank_counts[r] == 2) {
            pairs++;
            pair_ranks |= RANK_BITS[r];
        }
    }
    
    // Four of a kind
    if (quads > 0) {
        // Find best kicker from remaining cards
        int kicker_bits = rank_bits & ~RANK_BITS[quad_rank];
        int kicker = find_highest_bit(kicker_bits);
        return MAKE_HAND_VALUE(HAND_FOUR_OF_KIND, quad_rank, kicker, 0, 0, 0);
    }
    
    // Full house
    if (trips > 0 && (pairs > 0 || trips > 1)) {
        int pair_rank = -1;
        if (trips > 1) {
            // Two sets of trips, use lower as pair
            #pragma unroll
            for (int r = 12; r >= 0; r--) {
                if (rank_counts[r] == 3 && r != trip_rank) {
                    pair_rank = r;
                    break;
                }
            }
        } else {
            // Use highest pair
            pair_rank = find_highest_bit(pair_ranks);
        }
        return MAKE_HAND_VALUE(HAND_FULL_HOUSE, trip_rank, pair_rank, 0, 0, 0);
    }
    
    // Flush (already checked for straight flush)
    if (flush_suit >= 0) {
        int flush_ranks = 0;
        int flush_count = 0;
        #pragma unroll
        for (int i = 0; i < 7; i++) {
            if (GET_SUIT(cards[i]) == flush_suit) {
                flush_ranks |= RANK_BITS[GET_RANK(cards[i])];
                flush_count++;
            }
        }
        
        // Extract top 5 cards
        int k1 = find_highest_bit(flush_ranks);
        flush_ranks &= ~RANK_BITS[k1];
        int k2 = find_highest_bit(flush_ranks);
        flush_ranks &= ~RANK_BITS[k2];
        int k3 = find_highest_bit(flush_ranks);
        flush_ranks &= ~RANK_BITS[k3];
        int k4 = find_highest_bit(flush_ranks);
        flush_ranks &= ~RANK_BITS[k4];
        int k5 = find_highest_bit(flush_ranks);
        
        return MAKE_HAND_VALUE(HAND_FLUSH, k1, k2, k3, k4, k5);
    }
    
    // Check for straight
    #pragma unroll
    for (int p = 0; p < 10; p++) {
        if ((rank_bits & STRAIGHT_PATTERNS[p]) == STRAIGHT_PATTERNS[p]) {
            int high_card = find_highest_bit(STRAIGHT_PATTERNS[p]);
            return MAKE_HAND_VALUE(HAND_STRAIGHT, high_card, 0, 0, 0, 0);
        }
    }
    
    // Three of a kind
    if (trips > 0) {
        int kicker_bits = rank_bits & ~RANK_BITS[trip_rank];
        int k1 = find_highest_bit(kicker_bits);
        kicker_bits &= ~RANK_BITS[k1];
        int k2 = find_highest_bit(kicker_bits);
        return MAKE_HAND_VALUE(HAND_THREE_OF_KIND, trip_rank, k1, k2, 0, 0);
    }
    
    // Two pair
    if (pairs >= 2) {
        int high_pair = find_highest_bit(pair_ranks);
        pair_ranks &= ~RANK_BITS[high_pair];
        int low_pair = find_highest_bit(pair_ranks);
        
        int kicker_bits = rank_bits & ~RANK_BITS[high_pair] & ~RANK_BITS[low_pair];
        int kicker = find_highest_bit(kicker_bits);
        
        return MAKE_HAND_VALUE(HAND_TWO_PAIR, high_pair, low_pair, kicker, 0, 0);
    }
    
    // One pair
    if (pairs == 1) {
        int pair_rank = find_highest_bit(pair_ranks);
        int kicker_bits = rank_bits & ~RANK_BITS[pair_rank];
        
        int k1 = find_highest_bit(kicker_bits);
        kicker_bits &= ~RANK_BITS[k1];
        int k2 = find_highest_bit(kicker_bits);
        kicker_bits &= ~RANK_BITS[k2];
        int k3 = find_highest_bit(kicker_bits);
        
        return MAKE_HAND_VALUE(HAND_PAIR, pair_rank, k1, k2, k3, 0);
    }
    
    // High card
    int k1 = find_highest_bit(rank_bits);
    rank_bits &= ~RANK_BITS[k1];
    int k2 = find_highest_bit(rank_bits);
    rank_bits &= ~RANK_BITS[k2];
    int k3 = find_highest_bit(rank_bits);
    rank_bits &= ~RANK_BITS[k3];
    int k4 = find_highest_bit(rank_bits);
    rank_bits &= ~RANK_BITS[k4];
    int k5 = find_highest_bit(rank_bits);
    
    return MAKE_HAND_VALUE(HAND_HIGH_CARD, k1, k2, k3, k4, k5);
}

// Device function to get hand category from evaluation value
__device__ __forceinline__ int get_hand_category(int eval_value) {
    return (eval_value >> HAND_RANK_SHIFT) & 0xF;
}

#endif // HAND_EVALUATOR_CUH