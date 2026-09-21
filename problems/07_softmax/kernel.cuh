#pragma once

#include <cuda_runtime.h>

// Standard block size for row-wise softmax kernels
constexpr int SOFTMAX_BLOCK_SIZE = 256;
constexpr int WARP_SIZE = 32;

// Milestone 1: Two-Pass Safe Softmax (Block-level Shared Memory Reduction)
void launch_softmax_block_twopass(const float* d_in, float* d_out, int m, int n);

// Milestone 2: Warp-Accelerated Safe Softmax (__shfl_down_sync)
void launch_softmax_warp_shuffle(const float* d_in, float* d_out, int m, int n);

// Milestone 3: Online Safe Softmax (FlashSoftmax / Single-Pass)
void launch_softmax_online(const float* d_in, float* d_out, int m, int n);

// Status queries for the benchmark harness
bool is_block_twopass_implemented();
bool is_warp_shuffle_implemented();
bool is_online_implemented();
