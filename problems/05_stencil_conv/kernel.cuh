#pragma once

#include <cuda_runtime.h>

// Filter / Stencil parameters (5x5 filter, radius = 2)
constexpr int KERNEL_RADIUS = 2;
constexpr int KERNEL_DIAMETER = 2 * KERNEL_RADIUS + 1; // 5
constexpr int KERNEL_SIZE = KERNEL_DIAMETER * KERNEL_DIAMETER; // 25

// Output tile dimensions for shared memory tiling
constexpr int TILE_DIM = 16;
constexpr int APRON_DIM = TILE_DIM + 2 * KERNEL_RADIUS; // 16 + 4 = 20

// Milestone 1: Naive 2D Convolution (Global Memory for both input and mask)
void launch_conv2d_naive(const float* d_in, const float* d_mask, float* d_out, int height, int width);

// Milestone 2: Constant Memory Filter Mask
void launch_conv2d_constant_mask(const float* d_in, const float* h_mask, float* d_out, int height, int width);

// Milestone 3: Shared Memory Apron Tiling (Cooperative halo loading)
void launch_conv2d_shared_tiled(const float* d_in, const float* h_mask, float* d_out, int height, int width);

// Milestone 4: Read-Only Streaming Cache (const __restrict__)
void launch_conv2d_readonly_cached(const float* d_in, const float* h_mask, float* d_out, int height, int width);

// Status queries for the benchmark harness
bool is_constant_mask_implemented();
bool is_shared_tiled_implemented();
bool is_readonly_cached_implemented();
