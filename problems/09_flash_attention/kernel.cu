#include "kernel.cuh"
#include <cfloat>
#include <cmath>

// ==============================================================================
// MILESTONE 1: Standard Attention (Global Memory Baseline)
// ==============================================================================
// Materializes the full N x N attention matrix S in global memory:
// 1. Compute logits: S = scale * (Q @ K^T)  [N x N]
// 2. Row-wise Softmax: P = softmax(S)       [N x N]
// 3. Output projection: O = P @ V           [N x d]
//
// Memory Complexity: O(N^2) DRAM traffic. For large N, memory traffic
// quickly overwhelms the GPU memory bus.
// ==============================================================================

__global__ void attention_naive_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       int seq_len, int d,
                                       float scale, bool is_causal) {
    // TODO: Implement Milestone 1
    // Each thread (or thread block) computes one row of the output O.
    // 1. Identify row: int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= seq_len) return;
    // 2. Compute logits against all K rows: S[j] = scale * dot(Q[i], K[j])
    // 3. Find row max m_i = max_j S[j] (apply causal mask: j > i -> -inf)
    // 4. Compute sum of exp: l_i = sum_j exp(S[j] - m_i)
    // 5. Compute P[j] = exp(S[j] - m_i) / l_i
    // 6. Accumulate O[i, k] = sum_j (P[j] * V[j, k])
}

void launch_attention_naive(const float* d_Q, const float* d_K, const float* d_V,
                            float* d_O, int seq_len, int d, float scale, bool is_causal) {
    dim3 block(128);
    dim3 grid((seq_len + block.x - 1) / block.x);
    attention_naive_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d, scale, is_causal);
}

bool is_attention_naive_implemented() {
    return false;
}

// ==============================================================================
// MILESTONE 2: Tiled FlashAttention (SRAM Tiling + Online Softmax)
// ==============================================================================
// Key Idea (Dao et al., 2022):
// Never write the N x N attention matrix S to DRAM!
// Load tiles of Q (Br x d) and tiles of K, V (Bc x d) into fast __shared__ memory (SRAM).
//
// Use the Online Softmax recurrence to maintain running row max m_i and running sum l_i:
//   m_new = max(m_old, m_tile)
//   l_new = l_old * exp(m_old - m_new) + l_tile * exp(m_tile - m_new)
//   O_new = O_old * exp(m_old - m_new) + P_tile @ V_tile
//
// Memory Complexity: O(N) DRAM traffic (only reads Q, K, V and writes O).
// ==============================================================================

__global__ void flash_attention_tiled_kernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V,
                                             float* __restrict__ O,
                                             int seq_len, int d,
                                             float scale, bool is_causal) {
    // TODO: Implement Milestone 2
    // Allocate shared memory for tiles:
    // __shared__ float s_Q[TILE_BR][HEAD_DIM];
    // __shared__ float s_K[TILE_BC][HEAD_DIM];
    // __shared__ float s_V[TILE_BC][HEAD_DIM];
    //
    // Outer loop over K, V blocks (or Q blocks).
    // Cooperatively load tiles from DRAM to SRAM.
    // Compute local tile dot products in SRAM.
    // Apply online softmax update.
}

void launch_flash_attention_tiled(const float* d_Q, const float* d_K, const float* d_V,
                                  float* d_O, int seq_len, int d, float scale, bool is_causal) {
    dim3 block(TILE_BR);
    dim3 grid((seq_len + TILE_BR - 1) / TILE_BR);
    flash_attention_tiled_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d, scale, is_causal);
}

bool is_flash_attention_tiled_implemented() {
    return false;
}

// ==============================================================================
// MILESTONE 3: FlashAttention-2 (Outer-Loop over Q, Causal Masking, Register Rescaling)
// ==============================================================================
// FlashAttention-2 Innovations (Dao, 2023):
// 1. Invert loop order: Outer loop over Q tiles (parallelized across thread blocks).
//    Inner loop over K, V tiles. This keeps output accumulators O entirely in
//    fast thread registers until the block finishes!
// 2. Rescale O only by exp(m_old - m_new) without dividing by l_new until the very end:
//    O_final = O_accum / l_final.
// 3. Efficient Causal Masking: Skip entire K, V tiles where j_start > i_end!
// ==============================================================================

__global__ void flash_attention_2_kernel(const float* __restrict__ Q,
                                         const float* __restrict__ K,
                                         const float* __restrict__ V,
                                         float* __restrict__ O,
                                         int seq_len, int d,
                                         float scale, bool is_causal) {
    // TODO: Implement Milestone 3
}

void launch_flash_attention_2(const float* d_Q, const float* d_K, const float* d_V,
                              float* d_O, int seq_len, int d, float scale, bool is_causal) {
    dim3 block(TILE_BR);
    dim3 grid((seq_len + TILE_BR - 1) / TILE_BR);
    flash_attention_2_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d, scale, is_causal);
}

bool is_flash_attention_2_implemented() {
    return false;
}
