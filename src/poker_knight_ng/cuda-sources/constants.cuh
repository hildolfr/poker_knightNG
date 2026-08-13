/*
 * Poker constants and lookup tables for GPU evaluation.
 * 
 * This file contains all constants needed for fast 7-card hand evaluation
 * on the GPU using bit manipulation and lookup tables.
 */

#ifndef POKER_CONSTANTS_CUH
#define POKER_CONSTANTS_CUH

#include <cuda_runtime.h>

// Card representation constants
#define CARDS_PER_DECK 52
#define RANKS_PER_SUIT 13
#define SUITS_PER_DECK 4
#define CARDS_PER_HAND 7
#define HOLE_CARDS 2
#define MAX_BOARD_CARDS 5

// Rank and suit extraction
#define GET_RANK(card) ((card) >> 2)
#define GET_SUIT(card) ((card) & 0x3)
#define MAKE_CARD(rank, suit) (((rank) << 2) | (suit))

// Hand ranking categories (0-9)
#define HAND_HIGH_CARD       0
#define HAND_PAIR           1
#define HAND_TWO_PAIR       2
#define HAND_THREE_OF_KIND  3
#define HAND_STRAIGHT       4
#define HAND_FLUSH          5
#define HAND_FULL_HOUSE     6
#define HAND_FOUR_OF_KIND   7
#define HAND_STRAIGHT_FLUSH 8
#define HAND_ROYAL_FLUSH    9

// Bit masks for ranks (A=12, K=11, Q=10, J=9, T=8, 9=7, 8=6, 7=5, 6=4, 5=3, 4=2, 3=1, 2=0)
#define RANK_2 (1 << 0)
#define RANK_3 (1 << 1)
#define RANK_4 (1 << 2)
#define RANK_5 (1 << 3)
#define RANK_6 (1 << 4)
#define RANK_7 (1 << 5)
#define RANK_8 (1 << 6)
#define RANK_9 (1 << 7)
#define RANK_T (1 << 8)
#define RANK_J (1 << 9)
#define RANK_Q (1 << 10)
#define RANK_K (1 << 11)
#define RANK_A (1 << 12)

// Special case for A-2-3-4-5 straight
#define WHEEL_STRAIGHT 0x100F  // Binary: 1 0000 0000 1111 (A + 5-4-3-2)

// Lookup tables for fast evaluation
// Prime numbers for rank hashing (2-A)
__constant__ int RANK_PRIMES[13] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41};

// Prime numbers for suit hashing
__constant__ int SUIT_PRIMES[4] = {2, 3, 5, 7};

// Bit patterns for each rank
__constant__ int RANK_BITS[13] = {
    1 << 0,   // 2
    1 << 1,   // 3
    1 << 2,   // 4
    1 << 3,   // 5
    1 << 4,   // 6
    1 << 5,   // 7
    1 << 6,   // 8
    1 << 7,   // 9
    1 << 8,   // T
    1 << 9,   // J
    1 << 10,  // Q
    1 << 11,  // K
    1 << 12   // A
};

// Straight patterns (ordered from highest to lowest)
__constant__ int STRAIGHT_PATTERNS[10] = {
    0x1F00,  // A-K-Q-J-T (Royal)
    0x0F80,  // K-Q-J-T-9
    0x07C0,  // Q-J-T-9-8
    0x03E0,  // J-T-9-8-7
    0x01F0,  // T-9-8-7-6
    0x00F8,  // 9-8-7-6-5
    0x007C,  // 8-7-6-5-4
    0x003E,  // 7-6-5-4-3
    0x001F,  // 6-5-4-3-2
    0x100F   // A-5-4-3-2 (Wheel)
};

// Hand evaluation constants
#define HAND_RANK_SHIFT 20      // Bits for hand category
#define KICKER_SHIFT 16         // Bits for primary kicker
#define KICKER2_SHIFT 12        // Bits for secondary kicker
#define KICKER3_SHIFT 8         // Bits for tertiary kicker
#define KICKER4_SHIFT 4         // Bits for quaternary kicker
#define KICKER5_SHIFT 0         // Bits for quinary kicker

// Macro to create hand value
#define MAKE_HAND_VALUE(category, k1, k2, k3, k4, k5) \
    (((category) << HAND_RANK_SHIFT) | \
     ((k1) << KICKER_SHIFT) | \
     ((k2) << KICKER2_SHIFT) | \
     ((k3) << KICKER3_SHIFT) | \
     ((k4) << KICKER4_SHIFT) | \
     ((k5) << KICKER5_SHIFT))

// Position indices
#define POS_EARLY  0
#define POS_MIDDLE 1
#define POS_LATE   2
#define POS_BUTTON 3
#define POS_SB     4
#define POS_BB     5

// Action indices
#define ACTION_CHECK   0
#define ACTION_BET     1
#define ACTION_RAISE   2
#define ACTION_RERAISE 3

// Street indices
#define STREET_PREFLOP 0
#define STREET_FLOP    1
#define STREET_TURN    2
#define STREET_RIVER   3

// Tournament stage indices
#define STAGE_EARLY       0
#define STAGE_MIDDLE      1
#define STAGE_BUBBLE      2
#define STAGE_FINAL_TABLE 3

// Simulation parameters
#define THREADS_PER_BLOCK 256
#define MIN_BLOCKS 32
#define MAX_BLOCKS 1024

// RNG constants (for xorshift128+)
#define RNG_STATE_SIZE 4  // 4 * 32-bit = 128-bit state

// Shared memory sizes
#define SHARED_EVAL_TABLES_SIZE 4096  // 4KB for evaluation tables
#define SHARED_RESULTS_SIZE 2048       // 2KB for reduction results

// Mathematical constants
#define CONFIDENCE_Z_SCORE 1.96f  // 95% confidence interval

#endif // POKER_CONSTANTS_CUH