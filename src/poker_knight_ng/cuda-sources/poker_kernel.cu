/*
 * Monolithic CUDA kernel for poker hand evaluation.
 * 
 * This kernel performs ALL computations in a single launch:
 * - Monte Carlo simulations
 * - Statistical analysis
 * - Advanced metrics
 * - Result aggregation
 */

#include "constants.cuh"
#include "hand_evaluator.cuh"
#include "rng.cuh"
#include "icm_calculator.cuh"
#include "board_analyzer.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// Shared memory structure for block-level cooperation
struct SharedMemory {
    // Win/tie/loss counters for reduction
    int win_count[THREADS_PER_BLOCK];
    int tie_count[THREADS_PER_BLOCK];
    int loss_count[THREADS_PER_BLOCK];
    
    // Hand category counters
    int hand_categories[10][THREADS_PER_BLOCK];
    
    // Advanced metrics workspace
    float icm_workspace[THREADS_PER_BLOCK];
    float equity_workspace[THREADS_PER_BLOCK];
};

// Main monolithic kernel
extern "C" __global__ void solve_poker_hand_kernel(
    // Inputs
    const int* hero_cards,           // 2 cards
    const int num_opponents,         // 1-6
    const int* board_cards,          // 5 cards (-1 for unknown)
    const int board_cards_count,     // 0, 3, 4, or 5
    const int num_simulations,       // Total simulations to run
    const unsigned int random_seed,  // RNG seed
    
    // Optional inputs
    const int hero_position_idx,     // -1 or 0-5
    const float* stack_sizes,        // 7 floats (-1 for unused)
    const float pot_size,            // Current pot
    const int action_to_hero_idx,    // -1 or 0-3
    const float bet_size,            // Relative to pot
    const int street_idx,            // -1 or 0-3
    const int players_to_act,        // 0+
    
    // Tournament context
    const bool has_tournament_context,
    const float* payouts,            // 10 floats
    const int players_remaining,     // For ICM
    const float average_stack,       // For ICM
    const int tournament_stage_idx,  // -1 or 0-3
    const int blind_level,           // 1+
    
    // Outputs
    float* win_probability,          // Single value
    float* tie_probability,          // Single value
    float* loss_probability,         // Single value
    float* hand_frequencies,         // 10 values
    float* confidence_interval_low,  // Single value
    float* confidence_interval_high, // Single value
    float* position_equity,          // 6 values
    float* fold_equity,              // 6 values
    float* icm_equity,               // Single value
    float* bubble_factor,            // Single value
    float* spr,                      // Single value
    float* pot_odds,                 // Single value
    float* mdf,                      // Single value
    float* equity_needed,            // Single value
    float* commitment_threshold,     // Single value
    float* board_texture_score,      // Single value
    int* flush_draw_count,           // Single value
    int* straight_draw_count,        // Single value
    float* equity_percentiles,       // 5 values
    float* positional_advantage,     // Single value
    float* hand_vulnerability,       // Single value
    int* actual_simulations          // Single value
) {
    // Get thread and block information
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int global_tid = bid * blockDim.x + tid;
    const int total_threads = gridDim.x * blockDim.x;
    
    // Initialize shared memory
    extern __shared__ char shared_mem[];
    SharedMemory* shared = reinterpret_cast<SharedMemory*>(shared_mem);
    
    // Initialize RNG state
    RNGState rng_state;
    init_rng(&rng_state, random_seed, global_tid);
    
    // Phase 1: Setup and initialization
    // Copy hero and board cards to local memory
    int known_cards[7];
    int num_known = 2;  // Hero cards always known
    known_cards[0] = hero_cards[0];
    known_cards[1] = hero_cards[1];
    
    // Add known board cards
    for (int i = 0; i < board_cards_count; i++) {
        known_cards[num_known++] = board_cards[i];
    }
    
    // Initialize thread-local counters
    int thread_wins = 0;
    int thread_ties = 0;
    int thread_losses = 0;
    int thread_hand_counts[10] = {0};
    
    // Phase 2: Monte Carlo simulation loop (grid-stride pattern)
    // Process simulations in pairs for antithetic variates
    int num_pairs = (num_simulations + 1) / 2;  // Round up
    for (int pair_idx = global_tid; pair_idx < num_pairs; pair_idx += total_threads) {
        // Each pair consists of one normal and one antithetic simulation
        for (int is_antithetic = 0; is_antithetic < 2; is_antithetic++) {
            // Skip second simulation if we've reached the total count
            int sim = pair_idx * 2 + is_antithetic;
            if (sim >= num_simulations) break;
        // Deal remaining board cards if needed
        int simulation_board[5];
        for (int i = 0; i < 5; i++) {
            if (i < board_cards_count) {
                simulation_board[i] = board_cards[i];
            } else {
                simulation_board[i] = -1;  // To be dealt
            }
        }
        
        // Deal missing board cards
        if (board_cards_count < 5) {
            int cards_to_deal = 5 - board_cards_count;
            int dealt_cards[5];
            deal_cards(dealt_cards, cards_to_deal, known_cards, num_known, &rng_state, is_antithetic);
            
            // Fill in missing board cards
            int dealt_idx = 0;
            for (int i = board_cards_count; i < 5; i++) {
                simulation_board[i] = dealt_cards[dealt_idx++];
            }
        }
        
        // Create hero's 7-card hand
        int hero_hand[7];
        hero_hand[0] = hero_cards[0];
        hero_hand[1] = hero_cards[1];
        for (int i = 0; i < 5; i++) {
            hero_hand[i + 2] = simulation_board[i];
        }
        
        // Evaluate hero's hand
        int hero_value = evaluate_7card_hand(hero_hand);
        int hero_category = get_hand_category(hero_value);
        thread_hand_counts[hero_category]++;
        
        // Deal and evaluate opponent hands
        int best_opponent_value = -1;
        int num_ties = 0;
        
        // Build list of cards to exclude when dealing opponent cards
        int excluded_cards[7];
        int num_excluded = 0;
        excluded_cards[num_excluded++] = hero_cards[0];
        excluded_cards[num_excluded++] = hero_cards[1];
        for (int i = 0; i < 5; i++) {
            excluded_cards[num_excluded++] = simulation_board[i];
        }
        
        // Simulate each opponent
        for (int opp = 0; opp < num_opponents; opp++) {
            // Deal opponent hole cards
            int opp_hole[2];
            deal_cards(opp_hole, 2, excluded_cards, num_excluded, &rng_state, is_antithetic);
            
            // Create opponent's 7-card hand
            int opp_hand[7];
            opp_hand[0] = opp_hole[0];
            opp_hand[1] = opp_hole[1];
            for (int i = 0; i < 5; i++) {
                opp_hand[i + 2] = simulation_board[i];
            }
            
            // Evaluate opponent's hand
            int opp_value = evaluate_7card_hand(opp_hand);
            
            // Track best opponent
            if (opp_value > best_opponent_value) {
                best_opponent_value = opp_value;
                num_ties = 0;
            } else if (opp_value == best_opponent_value) {
                num_ties++;
            }
            
            // Add opponent cards to excluded list for next opponent
            excluded_cards[num_excluded++] = opp_hole[0];
            excluded_cards[num_excluded++] = opp_hole[1];
        }
        
        // Determine outcome
        if (hero_value > best_opponent_value) {
            thread_wins++;
        } else if (hero_value == best_opponent_value) {
            thread_ties++;
        } else {
            thread_losses++;
        }
        }  // End of antithetic loop
    }
    
    // Phase 3: Statistical reduction within block
    __syncthreads();
    
    // Store thread results in shared memory
    shared->win_count[tid] = thread_wins;
    shared->tie_count[tid] = thread_ties;
    shared->loss_count[tid] = thread_losses;
    
    for (int i = 0; i < 10; i++) {
        shared->hand_categories[i][tid] = thread_hand_counts[i];
    }
    
    __syncthreads();
    
    // Parallel reduction within block
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared->win_count[tid] += shared->win_count[tid + stride];
            shared->tie_count[tid] += shared->tie_count[tid + stride];
            shared->loss_count[tid] += shared->loss_count[tid + stride];
            
            for (int i = 0; i < 10; i++) {
                shared->hand_categories[i][tid] += shared->hand_categories[i][tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Phase 4: Advanced metrics calculation (specialized warps)
    // Only first warp of first block computes these
    if (bid == 0 && tid < 32) {
        __syncwarp();
        
        // Thread 0: ICM calculations and tournament metrics
        if (tid == 0 && stack_sizes[0] > 0) {
            // Count active players
            int active_players = 0;
            float total_chips = 0.0f;
            for (int i = 0; i < 7; i++) {
                if (stack_sizes[i] > 0) {
                    active_players++;
                    total_chips += stack_sizes[i];
                }
            }
            
            if (active_players >= 2) {
                // Calculate average stack
                float calc_avg_stack = (average_stack > 0) ? average_stack : (total_chips / active_players);
                
                // Always calculate bubble factor based on stack sizes
                *bubble_factor = (float)calculate_bubble_factor(
                    (players_remaining > 0) ? players_remaining : active_players,
                    10,  // Assume top 10 paid
                    calc_avg_stack,
                    stack_sizes[0]
                );
                
                // Full ICM calculations only with tournament context and payouts
                if (has_tournament_context && players_remaining > 0) {
                    // Calculate ICM equity
                    if (active_players <= MAX_ICM_PLAYERS) {
                        *icm_equity = (float)calculate_icm_equity(
                            stack_sizes, 
                            active_players, 
                            payouts, 
                            10,  // max payouts
                            0    // hero is always index 0
                        );
                    } else {
                        *icm_equity = (float)calculate_icm_equity_large_field(
                            stack_sizes, 
                            active_players, 
                            payouts, 
                            10, 
                            0
                        );
                    }
                } else {
                    // Without full tournament context, estimate ICM based on chip distribution
                    // Simple chip EV model: your chips / total chips
                    *icm_equity = stack_sizes[0] / total_chips;
                }
            }
        }
        
        // Thread 1: Board texture analysis
        if (tid == 1 && board_cards_count >= 3) {
            int flush_draws = 0, straight_draws = 0;
            float texture = analyze_board_texture(
                board_cards,  // Use the provided board cards
                &flush_draws,
                &straight_draws
            );
            
            *board_texture_score = texture;
            *flush_draw_count = flush_draws;
            *straight_draw_count = straight_draws;
        }
        
        // Thread 2: SPR and pot odds calculations
        if (tid == 2 && stack_sizes[0] > 0) {
            if (pot_size > 0) {
                *spr = stack_sizes[0] / pot_size;
                
                if (bet_size > 0) {
                    // bet_size is relative to pot (e.g., 0.5 = half pot)
                    float actual_bet = bet_size * pot_size;
                    *pot_odds = actual_bet / (pot_size + actual_bet);
                    *mdf = 1.0f - (actual_bet / (pot_size + 2.0f * actual_bet));
                    *equity_needed = *pot_odds;
                }
            } else {
                // When pot is 0, calculate effective stack in big blinds
                // Assume big blind = 1.0 if not specified
                float big_blind = 1.0f;
                *spr = stack_sizes[0] / big_blind;  // Effective stack depth
            }
            
            // Commitment threshold based on SPR (works for both cases)
            if (*spr < 1.0f) {
                *commitment_threshold = 0.5f;  // Very committed
            } else if (*spr < 3.0f) {
                *commitment_threshold = 2.0f;  // Somewhat committed
            } else {
                *commitment_threshold = 4.0f;  // Not committed
            }
        }
        
        // Thread 3: Hand vulnerability
        if (tid == 3) {
            // Simple vulnerability calculation based on win probability
            float win_prob = (float)shared->win_count[0] / 
                            (float)(shared->win_count[0] + shared->tie_count[0] + shared->loss_count[0]);
            
            float texture = *board_texture_score;
            *hand_vulnerability = (float)calculate_hand_vulnerability(
                win_prob,
                texture,
                num_opponents
            );
        }
        
        // Thread 4: Positional advantage
        if (tid == 4 && hero_position_idx >= 0) {
            // Simple positional scoring
            // Late position = higher score
            float pos_scores[6] = {0.0f, 0.2f, 0.4f, 0.8f, 0.6f, 0.3f}; // EP, MP, LP, BTN, SB, BB
            *positional_advantage = pos_scores[hero_position_idx];
            
            // Adjust for number of players to act
            if (players_to_act > 0) {
                *positional_advantage *= (1.0f - 0.1f * players_to_act);
            }
        }
        
        __syncwarp();
    }
    
    // Phase 5: Final result aggregation (thread 0 of each block)
    if (tid == 0) {
        // Use atomic operations to accumulate across blocks
        atomicAdd(&win_probability[0], (float)shared->win_count[0]);
        atomicAdd(&tie_probability[0], (float)shared->tie_count[0]);
        atomicAdd(&loss_probability[0], (float)shared->loss_count[0]);
        
        for (int i = 0; i < 10; i++) {
            atomicAdd(&hand_frequencies[i], (float)shared->hand_categories[i][0]);
        }
        
        // Track actual simulations
        atomicAdd(actual_simulations, shared->win_count[0] + shared->tie_count[0] + shared->loss_count[0]);
    }
}

// Kernel launch configuration calculator
extern "C" void get_kernel_config(int num_simulations, int* blocks, int* threads) {
    *threads = THREADS_PER_BLOCK;
    
    // Calculate optimal number of blocks
    int sims_per_block = num_simulations / MIN_BLOCKS;
    if (sims_per_block < THREADS_PER_BLOCK) {
        *blocks = MIN_BLOCKS;
    } else {
        *blocks = (num_simulations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        *blocks = min(*blocks, MAX_BLOCKS);
        *blocks = max(*blocks, MIN_BLOCKS);
    }
}