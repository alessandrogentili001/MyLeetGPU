#pragma once

#include <vector>

// Simple CPU reference for SGEMM: C = alpha * A * B + beta * C
// Assuming alpha = 1.0, beta = 0.0 for simplicity in this exercise.
// A is M x K
// B is K x N
// C is M x N
// All matrices are in row-major order.
inline void sgemm_cpu_reference(const float* A, const float* B, float* C, 
                                int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
