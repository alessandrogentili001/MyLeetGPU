#pragma once

#include <cuda_runtime.h>

// Tile dimensions for shared memory implementations
constexpr int TILE_DIM = 32;
constexpr int BLOCK_ROWS = 8; // For coarse-grained tiling (Milestone 4)

// Milestone 1: Naive (Coalesced Read, Strided Uncoalesced Write)
void launch_matrix_transpose_naive(const float* d_in, float* d_out, int rows, int cols);

// Milestone 2: Shared Memory Tiling (Coalesced Global Access, but 32-way Bank Conflicts)
void launch_matrix_transpose_shared_conflict(const float* d_in, float* d_out, int rows, int cols);

// Milestone 3: Bank-Conflict-Free Shared Memory Tiling (Padded [TILE_DIM][TILE_DIM + 1])
void launch_matrix_transpose_shared_padded(const float* d_in, float* d_out, int rows, int cols);

// Milestone 4 (Bonus): Coarse-Grained Tiling (32x8 threads per 32x32 tile, increased ILP)
void launch_matrix_transpose_coarse(const float* d_in, float* d_out, int rows, int cols);

// Status queries for the benchmark harness
bool is_shared_conflict_implemented();
bool is_shared_padded_implemented();
bool is_coarse_implemented();
