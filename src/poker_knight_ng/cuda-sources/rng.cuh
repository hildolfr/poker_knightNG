/*
 * Random number generation for CUDA poker simulations.
 * 
 * Uses xorshift128+ for fast, high-quality random numbers.
 * Each thread maintains its own RNG state.
 */

#ifndef RNG_CUH
#define RNG_CUH

#include "constants.cuh"
#include <curand_kernel.h>

// RNG state structure for xorshift128+
struct RNGState {
    unsigned int s[4];  // 128-bit state
};

// Initialize RNG state from seed and thread ID
__device__ __forceinline__ void init_rng(RNGState* state, unsigned int seed, unsigned int tid) {
    // Mix seed with thread ID to ensure different sequences per thread
    state->s[0] = seed ^ tid;
    state->s[1] = (seed << 16) ^ (tid << 8) ^ 0x9E3779B9;
    state->s[2] = (seed >> 16) ^ (tid >> 8) ^ 0x85EBCA6B;
    state->s[3] = seed ^ (tid * 0xC2B2AE35);
    
    // Warm up the generator
    for (int i = 0; i < 10; i++) {
        // Xorshift operations
        unsigned int t = state->s[0] ^ (state->s[0] << 11);
        state->s[0] = state->s[1];
        state->s[1] = state->s[2];
        state->s[2] = state->s[3];
        state->s[3] = state->s[3] ^ (state->s[3] >> 19) ^ t ^ (t >> 8);
    }
}

// Generate next random 32-bit integer
__device__ __forceinline__ unsigned int rand_uint(RNGState* state) {
    unsigned int t = state->s[0] ^ (state->s[0] << 11);
    state->s[0] = state->s[1];
    state->s[1] = state->s[2];
    state->s[2] = state->s[3];
    state->s[3] = state->s[3] ^ (state->s[3] >> 19) ^ t ^ (t >> 8);
    return state->s[3];
}

// Generate random integer in range [0, max)
__device__ __forceinline__ int rand_int(RNGState* state, int max) {
    // Use rejection sampling for uniform distribution
    unsigned int threshold = (0xFFFFFFFF / max) * max;
    unsigned int r;
    do {
        r = rand_uint(state);
    } while (r >= threshold);
    return r % max;
}

// Generate antithetic random integer in range [0, max)
__device__ __forceinline__ int rand_int_antithetic(RNGState* state, int max) {
    // Generate normal random value first
    unsigned int threshold = (0xFFFFFFFF / max) * max;
    unsigned int r;
    do {
        r = rand_uint(state);
    } while (r >= threshold);
    // Apply antithetic transformation: (max - 1) - (r % max)
    return (max - 1) - (r % max);
}

// Generate random float in range [0, 1)
__device__ __forceinline__ float rand_float(RNGState* state) {
    return rand_uint(state) * 2.3283064365386963e-10f;  // 1.0f / (1ULL << 32)
}

// Generate antithetic random float in range [0, 1)
__device__ __forceinline__ float rand_float_antithetic(RNGState* state) {
    // Generate U, return 1-U for antithetic
    float u = rand_uint(state) * 2.3283064365386963e-10f;
    return 1.0f - u;
}

// Shuffle array using Fisher-Yates algorithm
template <typename T>
__device__ void shuffle_array(T* array, int n, RNGState* state) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand_int(state, i + 1);
        // Swap array[i] and array[j]
        T temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Sample k items from n without replacement
// Results are stored in the first k positions of array
__device__ void sample_without_replacement(int* array, int n, int k, RNGState* state) {
    // Initialize array with 0..n-1
    for (int i = 0; i < n; i++) {
        array[i] = i;
    }
    
    // Partial Fisher-Yates shuffle
    for (int i = 0; i < k; i++) {
        int j = i + rand_int(state, n - i);
        // Swap array[i] and array[j]
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

// Deal cards from deck excluding known cards
__device__ void deal_cards(int* dest, int num_cards, const int* excluded, int num_excluded, RNGState* state, bool antithetic = false) {
    // Create available cards array
    int available[52];
    int num_available = 0;
    
    // Build list of available cards
    for (int card = 0; card < 52; card++) {
        bool is_excluded = false;
        for (int i = 0; i < num_excluded; i++) {
            if (excluded[i] == card) {
                is_excluded = true;
                break;
            }
        }
        if (!is_excluded) {
            available[num_available++] = card;
        }
    }
    
    // Shuffle and deal
    for (int i = 0; i < num_cards; i++) {
        int idx = i + (antithetic ? rand_int_antithetic(state, num_available - i) : rand_int(state, num_available - i));
        dest[i] = available[idx];
        // Move dealt card to used portion
        available[idx] = available[i];
    }
}

// Alternative: Using cuRAND for comparison (optional)
struct CurandState {
    curandState_t state;
};

__device__ __forceinline__ void init_curand(CurandState* state, unsigned int seed, unsigned int tid) {
    curand_init(seed, tid, 0, &state->state);
}

__device__ __forceinline__ unsigned int curand_uint(CurandState* state) {
    return curand(&state->state);
}

__device__ __forceinline__ float curand_float(CurandState* state) {
    return curand_uniform(&state->state);
}

#endif // RNG_CUH