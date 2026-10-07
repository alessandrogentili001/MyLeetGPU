#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

inline void init_random_attention_matrix(std::vector<float>& vec, size_t count,
                                         float min_val = -1.5f, float max_val = 1.5f,
                                         unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    vec.resize(count);
    for (size_t i = 0; i < count; ++i) {
        vec[i] = dis(gen);
    }
}

// Exact CPU Reference for Scaled Dot-Product Attention:
// S = scale * (Q @ K^T)
// P = softmax(S, dim=-1)   [with optional causal masking]
// O = P @ V
inline void attention_cpu_reference(const float* Q, const float* K, const float* V,
                                    float* O, int seq_len, int d,
                                    float scale, bool is_causal = false) {
    std::vector<float> S_row(seq_len, 0.0f);
    std::vector<float> P_row(seq_len, 0.0f);

    for (int i = 0; i < seq_len; ++i) {
        const float* q_vec = Q + i * d;

        // 1. Compute dot product with all K vectors: S[i, j] = scale * dot(Q[i], K[j])
        float row_max = -std::numeric_limits<float>::infinity();
        for (int j = 0; j < seq_len; ++j) {
            if (is_causal && j > i) {
                S_row[j] = -std::numeric_limits<float>::infinity();
                continue;
            }

            const float* k_vec = K + j * d;
            float dot = 0.0f;
            for (int k = 0; k < d; ++k) {
                dot += q_vec[k] * k_vec[k];
            }
            float val = dot * scale;
            S_row[j] = val;
            if (val > row_max) {
                row_max = val;
            }
        }

        // 2. Safe Softmax: compute sum of exponentials
        float sum_exp = 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            if (is_causal && j > i) {
                P_row[j] = 0.0f;
            } else {
                float p = std::exp(S_row[j] - row_max);
                P_row[j] = p;
                sum_exp += p;
            }
        }

        float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
        for (int j = 0; j < seq_len; ++j) {
            P_row[j] *= inv_sum;
        }

        // 3. Multiply by V: O[i, k] = sum_j (P[i, j] * V[j, k])
        float* o_vec = O + i * d;
        for (int k = 0; k < d; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < seq_len; ++j) {
                if (P_row[j] > 0.0f) {
                    sum += P_row[j] * V[j * d + k];
                }
            }
            o_vec[k] = sum;
        }
    }
}
