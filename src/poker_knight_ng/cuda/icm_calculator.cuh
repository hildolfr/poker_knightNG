/*
 * ICM (Independent Chip Model) calculator for tournament equity.
 * 
 * Implements ICM calculations on GPU for tournament scenarios,
 * computing equity based on stack sizes and payout structure.
 */

#ifndef ICM_CALCULATOR_CUH
#define ICM_CALCULATOR_CUH

#include <cuda_runtime.h>

// Maximum players for ICM calculation (limited by computation complexity)
#define MAX_ICM_PLAYERS 10

/*
 * Calculate ICM equity for a given stack distribution.
 * 
 * ICM calculates the expected value of a player's stack in terms of
 * tournament prize money, not just chip count.
 * 
 * @param stack_sizes Array of stack sizes (up to MAX_ICM_PLAYERS)
 * @param num_players Number of active players
 * @param payouts Tournament payout structure (percentage of prize pool)
 * @param num_payouts Number of paid positions
 * @param player_idx Index of player to calculate equity for
 * @return ICM equity as a fraction of total prize pool
 */
__device__ inline double calculate_icm_equity(
    const float* stack_sizes,
    int num_players,
    const float* payouts,
    int num_payouts,
    int player_idx
) {
    // Quick validation
    if (num_players < 2 || num_players > MAX_ICM_PLAYERS || 
        player_idx >= num_players || stack_sizes[player_idx] <= 0) {
        return 0.0;
    }
    
    // Calculate total chips
    double total_chips = 0.0;
    for (int i = 0; i < num_players; i++) {
        if (stack_sizes[i] > 0) {
            total_chips += (double)stack_sizes[i];
        }
    }
    
    if (total_chips <= 0) return 0.0;
    
    // For 2 players, it's simple - winner takes the difference in payouts
    if (num_players == 2) {
        double p_win = (double)stack_sizes[player_idx] / total_chips;
        return p_win * (double)payouts[0] + (1.0 - p_win) * (double)payouts[1];
    }
    
    // For 3+ players, use recursive ICM formula
    // This is computationally intensive, so we'll use an approximation
    // for larger fields
    
    double equity = 0.0;
    double p_finish[MAX_ICM_PLAYERS];
    
    // Calculate probability of finishing in each position
    // Simplified approach: probability proportional to stack size
    for (int finish_pos = 0; finish_pos < min(num_players, num_payouts); finish_pos++) {
        if (finish_pos == 0) {
            // Probability of finishing first
            p_finish[0] = (double)stack_sizes[player_idx] / total_chips;
        } else {
            // Approximate probability of finishing in position 'finish_pos'
            // This is simplified - true ICM uses recursive calculation
            double p_not_bust = 1.0;
            for (int j = 0; j < finish_pos; j++) {
                p_not_bust *= (1.0 - (double)stack_sizes[player_idx] / total_chips);
            }
            p_finish[finish_pos] = p_not_bust * ((double)stack_sizes[player_idx] / total_chips);
        }
        
        equity += p_finish[finish_pos] * (double)payouts[finish_pos];
    }
    
    return equity;
}

/*
 * Calculate bubble factor for tournament play.
 * 
 * The bubble factor represents how much more valuable chips become
 * as we approach the money bubble or pay jumps.
 * 
 * @param num_players Current number of players
 * @param num_payouts Number of paid positions
 * @param avg_stack Average stack size
 * @param hero_stack Hero's stack size
 * @return Bubble factor (1.0 = normal, >1.0 = increased pressure)
 */
__device__ inline double calculate_bubble_factor(
    int num_players,
    int num_payouts,
    float avg_stack,
    float hero_stack
) {
    // No bubble pressure if already in the money
    if (num_players <= num_payouts) {
        return 1.0;
    }
    
    // Calculate distance from bubble
    int players_to_bubble = num_players - num_payouts;
    
    // Base bubble factor increases as we approach the bubble
    double base_factor = 1.0 + (0.5 / (double)(players_to_bubble + 1));
    
    // Adjust based on stack size relative to average
    // Short stacks feel more pressure, big stacks less
    double stack_ratio = (double)hero_stack / (double)avg_stack;
    double stack_adjustment = 1.0;
    
    if (stack_ratio < 0.5) {
        // Very short stack - high pressure
        stack_adjustment = 1.5;
    } else if (stack_ratio < 1.0) {
        // Below average - some pressure
        stack_adjustment = 1.0 + (1.0 - stack_ratio) * 0.5;
    } else if (stack_ratio > 2.0) {
        // Big stack - can apply pressure
        stack_adjustment = 0.8;
    } else {
        // Above average - slight reduction
        stack_adjustment = 1.0 - (stack_ratio - 1.0) * 0.2;
    }
    
    return base_factor * stack_adjustment;
}

/*
 * Simplified ICM calculation for large fields (10+ players).
 * Uses Malmuth-Harville formula approximation.
 */
__device__ inline double calculate_icm_equity_large_field(
    const float* stack_sizes,
    int num_players,
    const float* payouts,
    int num_payouts,
    int player_idx
) {
    double total_chips = 0.0;
    for (int i = 0; i < num_players; i++) {
        total_chips += (double)stack_sizes[i];
    }
    
    if (total_chips <= 0) return 0.0;
    
    double my_stack = (double)stack_sizes[player_idx];
    double equity = 0.0;
    
    // Use Malmuth-Harville approximation
    for (int place = 0; place < min(num_payouts, num_players); place++) {
        double p_finish = my_stack / total_chips;
        
        // Adjust probability for each position
        for (int i = 0; i < place; i++) {
            p_finish *= (1.0 - my_stack / (total_chips - i * (total_chips / (double)num_players)));
        }
        
        equity += p_finish * (double)payouts[place];
    }
    
    return equity;
}

#endif // ICM_CALCULATOR_CUH