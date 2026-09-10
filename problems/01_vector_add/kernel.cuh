#pragma once

#include <cuda_runtime.h>

// Version 1: Naive (1 thread per element)
void launch_vector_add_naive(const float* d_a, const float* d_b, float* d_c, int n);

// Version 2: Grid-Stride Loop
void launch_vector_add_grid_stride(const float* d_a, const float* d_b, float* d_c, int n);

// Version 3: Vectorized Memory Access (float4 / 128-bit transactions)
void launch_vector_add_vectorized(const float* d_a, const float* d_b, float* d_c, int n);

// Status queries for the benchmark harness
bool is_grid_stride_implemented();
bool is_vectorized_implemented();
