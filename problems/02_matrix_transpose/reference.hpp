#pragma once

#include <vector>
#include <random>

// CPU Ground Truth: Rectangular Matrix Transpose (M x N -> N x M)
// Input 'in' has dimensions: rows x cols (M x N)
// Output 'out' has dimensions: cols x rows (N x M)
inline void matrix_transpose_cpu_reference(const float* in, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            out[c * rows + r] = in[r * cols + c];
        }
    }
}

// Deterministic random matrix initializer
inline void init_random_matrix(std::vector<float>& mat, float min_val = -10.0f, float max_val = 10.0f, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dis(gen);
    }
}
