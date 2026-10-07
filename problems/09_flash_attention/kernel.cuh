#pragma once

#include <cuda_runtime.h>

// Standard tile sizes for FlashAttention forward pass
constexpr int HEAD_DIM = 64;   // Default head dimension (d)
constexpr int TILE_BR  = 32;   // Tile size along query sequence length (Q chunk)
constexpr int TILE_BC  = 32;   // Tile size along key/value sequence length (K/V chunk)

// Milestone 1: Standard Attention (Materializes S and P in DRAM)
void launch_attention_naive(const float* d_Q, const float* d_K, const float* d_V,
                            float* d_O, int seq_len, int d, float scale, bool is_causal = false);

// Milestone 2: Tiled FlashAttention (SRAM Tiling + Online Softmax)
void launch_flash_attention_tiled(const float* d_Q, const float* d_K, const float* d_V,
                                  float* d_O, int seq_len, int d, float scale, bool is_causal = false);

// Milestone 3: FlashAttention-2 (Outer-Loop over Q, Causal Masking, Register Rescaling)
void launch_flash_attention_2(const float* d_Q, const float* d_K, const float* d_V,
                              float* d_O, int seq_len, int d, float scale, bool is_causal = false);

// Status queries for the benchmark harness
bool is_attention_naive_implemented();
bool is_flash_attention_tiled_implemented();
bool is_flash_attention_2_implemented();
