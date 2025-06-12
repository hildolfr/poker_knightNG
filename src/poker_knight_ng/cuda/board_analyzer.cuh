/*
 * Board texture analyzer for poker hands.
 * 
 * Analyzes board texture to identify draws, coordination,
 * and other strategic considerations.
 */

#ifndef BOARD_ANALYZER_CUH
#define BOARD_ANALYZER_CUH

#include <cuda_runtime.h>
#include "constants.cuh"

/*
 * Analyze board texture and count possible draws.
 * 
 * @param board 5 community cards
 * @param flush_draws Output: number of flush draw possibilities
 * @param straight_draws Output: number of straight draw possibilities
 * @return Board texture score (0.0 = dry, 1.0 = very wet/coordinated)
 */
__device__ inline float analyze_board_texture(
    const int* board,
    int* flush_draws,
    int* straight_draws
) {
    // Count cards by suit
    int suit_counts[4] = {0, 0, 0, 0};
    int rank_counts[13] = {0};
    int ranks_present = 0;
    int board_size = 0;
    
    // Analyze board cards
    for (int i = 0; i < 5; i++) {
        if (board[i] >= 0 && board[i] < 52) {
            int suit = board[i] & 3;  // Get suit (0-3)
            int rank = board[i] >> 2; // Get rank (0-12)
            
            suit_counts[suit]++;
            if (rank_counts[rank] == 0) {
                ranks_present++;
            }
            rank_counts[rank]++;
            board_size++;
        }
    }
    
    // Count flush draws (need exactly 3 or 4 of same suit on board)
    *flush_draws = 0;
    for (int s = 0; s < 4; s++) {
        if (suit_counts[s] == 3) {
            *flush_draws += 1;  // One flush draw possible
        } else if (suit_counts[s] == 4) {
            *flush_draws += 2;  // Two different flush draws possible
        }
    }
    
    // Count straight draws
    *straight_draws = 0;
    
    // Check for connected cards (within 4 ranks)
    for (int start_rank = 0; start_rank <= 9; start_rank++) {
        int cards_in_range = 0;
        int gaps = 0;
        
        for (int r = start_rank; r < start_rank + 5 && r < 13; r++) {
            if (rank_counts[r] > 0) {
                cards_in_range++;
            } else {
                gaps++;
            }
        }
        
        // Open-ended straight draw (4 cards with 1 gap)
        if (cards_in_range == 4 && gaps == 1) {
            *straight_draws += 1;
        }
        // Gutshot (4 cards with specific gap pattern)
        else if (cards_in_range == 3 && gaps == 2) {
            *straight_draws += 1;
        }
    }
    
    // Special case: Ace can be low (A-2-3-4-5)
    int low_straight_cards = rank_counts[12]; // Ace
    for (int r = 0; r < 4; r++) {
        if (rank_counts[r] > 0) low_straight_cards++;
    }
    if (low_straight_cards >= 3) {
        *straight_draws += 1;
    }
    
    // Calculate board texture score
    float texture_score = 0.0f;
    
    // Flush texture (0-0.4)
    // Adjust for board size - monotone flop (3 cards) is extremely wet
    int max_suit_count = 0;
    for (int s = 0; s < 4; s++) {
        if (suit_counts[s] > max_suit_count) {
            max_suit_count = suit_counts[s];
        }
    }
    
    if (board_size >= 3) {
        float flush_ratio = (float)max_suit_count / (float)board_size;
        if (flush_ratio >= 1.0f) {
            // Monotone board
            texture_score += 0.4f;
        } else if (flush_ratio >= 0.6f) {
            // Very flushy
            texture_score += 0.3f;
        } else if (max_suit_count >= 3) {
            // Flush draw possible
            texture_score += 0.2f;
        }
    }
    
    // Straight texture (0-0.4)
    float connectedness = 0.0f;
    int consecutive = 0;
    int max_consecutive = 0;
    
    for (int r = 0; r < 13; r++) {
        if (rank_counts[r] > 0) {
            consecutive++;
            if (consecutive > max_consecutive) {
                max_consecutive = consecutive;
            }
        } else {
            consecutive = 0;
        }
    }
    
    // Check wrap-around (K-A-2)
    if (rank_counts[12] > 0 && rank_counts[0] > 0) {
        if (rank_counts[11] > 0) max_consecutive = max(max_consecutive, 3);
        else if (rank_counts[1] > 0) max_consecutive = max(max_consecutive, 3);
    }
    
    connectedness = (float)max_consecutive / 5.0f;
    texture_score += connectedness * 0.4f;
    
    // Pair texture (0-0.2)
    int pairs = 0;
    int trips = 0;
    for (int r = 0; r < 13; r++) {
        if (rank_counts[r] == 2) pairs++;
        else if (rank_counts[r] >= 3) trips++;
    }
    
    if (pairs > 0 || trips > 0) {
        texture_score += 0.1f;
        if (pairs > 1 || trips > 0) {
            texture_score += 0.1f;  // Very paired board
        }
    }
    
    return min(texture_score, 1.0f);
}

/*
 * Calculate hand vulnerability based on board texture.
 * 
 * @param hand_strength Current hand strength (0-1)
 * @param board_texture Board texture score
 * @param num_opponents Number of opponents
 * @return Vulnerability score (0 = invulnerable, 1 = very vulnerable)
 */
__device__ inline float calculate_hand_vulnerability(
    float hand_strength,
    float board_texture,
    int num_opponents
) {
    // Strong hands on dry boards are less vulnerable
    if (hand_strength > 0.8f && board_texture < 0.3f) {
        return 0.1f;
    }
    
    // Medium hands on wet boards are very vulnerable
    if (hand_strength < 0.6f && board_texture > 0.7f) {
        return 0.8f + 0.1f * num_opponents;
    }
    
    // Linear interpolation based on board texture and opponents
    float base_vulnerability = (1.0f - hand_strength) * board_texture;
    float opponent_factor = 1.0f + (num_opponents - 1) * 0.1f;
    
    return min(base_vulnerability * opponent_factor, 1.0f);
}

/*
 * Identify possible nut hands on the board.
 * Returns a bitmask of possible nut hand types.
 */
__device__ inline int identify_nuts_possible(const int* board) {
    int nuts_mask = 0;
    
    // Check for straight flush possibility
    int suit_counts[4] = {0};
    int suit_ranks[4][5];
    int suit_rank_count[4] = {0};
    
    for (int i = 0; i < 5; i++) {
        if (board[i] >= 0) {
            int suit = board[i] & 3;
            int rank = board[i] >> 2;
            suit_counts[suit]++;
            if (suit_rank_count[suit] < 5) {
                suit_ranks[suit][suit_rank_count[suit]++] = rank;
            }
        }
    }
    
    // Need 3+ of same suit for flush
    for (int s = 0; s < 4; s++) {
        if (suit_counts[s] >= 3) {
            nuts_mask |= (1 << 5); // FLUSH possible
            
            // Check for straight flush
            if (suit_counts[s] >= 3) {
                // Simple check - if suited cards span 4 or less ranks
                int min_rank = 12, max_rank = 0;
                for (int i = 0; i < suit_rank_count[s]; i++) {
                    min_rank = min(min_rank, suit_ranks[s][i]);
                    max_rank = max(max_rank, suit_ranks[s][i]);
                }
                if (max_rank - min_rank <= 4) {
                    nuts_mask |= (1 << 8); // STRAIGHT_FLUSH possible
                }
            }
        }
    }
    
    // Check for quads possibility (need pair on board)
    int rank_counts[13] = {0};
    for (int i = 0; i < 5; i++) {
        if (board[i] >= 0) {
            rank_counts[board[i] >> 2]++;
        }
    }
    
    for (int r = 0; r < 13; r++) {
        if (rank_counts[r] >= 2) {
            nuts_mask |= (1 << 7); // FOUR_OF_A_KIND possible
        }
        if (rank_counts[r] >= 3) {
            nuts_mask |= (1 << 6); // FULL_HOUSE definitely possible
        }
    }
    
    // Check for straight possibility
    for (int start = 0; start <= 9; start++) {
        int cards_for_straight = 0;
        for (int r = start; r < start + 5 && r < 13; r++) {
            if (rank_counts[r] > 0) cards_for_straight++;
        }
        if (cards_for_straight >= 3) {
            nuts_mask |= (1 << 4); // STRAIGHT possible
            break;
        }
    }
    
    return nuts_mask;
}

#endif // BOARD_ANALYZER_CUH