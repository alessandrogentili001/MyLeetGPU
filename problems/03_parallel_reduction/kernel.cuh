#pragma once

#include <cuda_runtime.h>

// Block size for reduction kernels (256 threads = 8 warps)
constexpr int REDUCTION_BLOCK_SIZE = 256;

// Milestone 1: Interleaved Addressing with Warp Divergence
void launch_reduction_divergent(const float* d_in, float* d_out, int n);

// Milestone 2: Interleaved Addressing without Divergence (Strided Indexing, Bank Conflicts)
void launch_reduction_interleaved(const float* d_in, float* d_out, int n);

// Milestone 3: Sequential Addressing (Conflict-Free Shared Memory Reduction)
void launch_reduction_sequential(const float* d_in, float* d_out, int n);

// Milestone 4: Warp Shuffle Reduction (__shfl_down_sync)
void launch_reduction_warp_shuffle(const float* d_in, float* d_out, int n);

// Status queries for the benchmark harness
bool is_interleaved_implemented();
bool is_sequential_implemented();
bool is_warp_shuffle_implemented();
