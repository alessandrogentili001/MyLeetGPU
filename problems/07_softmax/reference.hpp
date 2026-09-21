#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

inline void init_random_matrix(std::vector<float>& vec, int m, int n, 
                               float min_val = -5.0f, float max_val = 5.0f, 
                               unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    size_t total = static_cast<size_t>(m) * n;
    vec.resize(total);
    for (size_t i = 0; i < total; ++i) {
        vec[i] = dis(gen);
    }
}

// Numerically stable CPU reference for row-wise Softmax
inline void softmax_cpu_reference(const float* in, float* out, int m, int n) {
    for (int i = 0; i < m; ++i) {
        const float* row_in = in + i * n;
        float* row_out = out + i * n;

        // Pass 1: Find maximum value in the row to prevent exp() overflow
        float max_val = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < n; ++j) {
            max_val = std::max(max_val, row_in[j]);
        }

        // Pass 2: Compute sum of exponentials
        float sum_exp = 0.0f;
        for (int j = 0; j < n; ++j) {
            sum_exp += std::exp(row_in[j] - max_val);
        }

        // Pass 3: Normalize
        float inv_sum = 1.0f / sum_exp;
        for (int j = 0; j < n; ++j) {
            row_out[j] = std::exp(row_in[j] - max_val) * inv_sum;
        }
    }
}
