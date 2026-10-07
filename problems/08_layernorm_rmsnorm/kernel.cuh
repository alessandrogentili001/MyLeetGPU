#pragma once

#include <cuda_runtime.h>

// Standard block size for row-wise normalization kernels
constexpr int LAYERNORM_BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// Milestone 1: Multi-Pass LayerNorm (Block-level Shared Memory Reduction)
void launch_layernorm_twopass(const float* d_x, const float* d_gamma, const float* d_beta,
                              float* d_y, int m, int n, float eps = 1e-5f);

// Milestone 2: One-Pass Welford LayerNorm (Warp Shuffle Reduction)
void launch_layernorm_welford(const float* d_x, const float* d_gamma, const float* d_beta,
                              float* d_y, int m, int n, float eps = 1e-5f);

// Milestone 3: Fused Vectorized RMSNorm (float4 + Warp Shuffle)
void launch_rmsnorm_vectorized(const float* d_x, const float* d_gamma,
                               float* d_y, int m, int n, float eps = 1e-5f);

// Status queries for the benchmark harness
bool is_layernorm_twopass_implemented();
bool is_layernorm_welford_implemented();
bool is_rmsnorm_vectorized_implemented();
