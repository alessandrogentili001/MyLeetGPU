#pragma once

#include <cublas_v2.h>

// Shared Memory Tile size for Milestone 2
#define TILE_SIZE 32

// Block and Thread dimensions for Milestone 3 (2D Register Tiling)
#define BM 64
#define BN 64
#define BK 8
#define TM 8
#define TN 8

// Note: A is M x K
// Note: B is K x N
// Note: C is M x N
// All matrices are row-major.

void launch_sgemm_naive(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
void launch_sgemm_shared_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);
void launch_sgemm_2d_register_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K);

// Provided Baseline
void launch_sgemm_cublas(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C, int M, int N, int K);

bool is_naive_implemented();
bool is_shared_tiled_implemented();
bool is_2d_register_tiled_implemented();
