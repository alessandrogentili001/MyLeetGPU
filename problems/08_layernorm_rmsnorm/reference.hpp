#pragma once

#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

inline void init_random_tensor(std::vector<float>& vec, size_t count,
                               float min_val = -2.0f, float max_val = 2.0f,
                               unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    vec.resize(count);
    for (size_t i = 0; i < count; ++i) {
        vec[i] = dis(gen);
    }
}

// Numerically accurate CPU reference for Layer Normalization (LayerNorm)
// Y[i, j] = ((X[i, j] - mean) / sqrt(var + eps)) * gamma[j] + beta[j]
inline void layernorm_cpu_reference(const float* x, const float* gamma, const float* beta,
                                    float* y, int m, int n, float eps = 1e-5f) {
    for (int i = 0; i < m; ++i) {
        const float* row_x = x + i * n;
        float* row_y = y + i * n;

        // Compute mean
        double sum = 0.0;
        for (int j = 0; j < n; ++j) {
            sum += static_cast<double>(row_x[j]);
        }
        float mean = static_cast<float>(sum / n);

        // Compute variance
        double var_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            double diff = static_cast<double>(row_x[j]) - mean;
            var_sum += diff * diff;
        }
        float var = static_cast<float>(var_sum / n);
        float inv_std = 1.0f / std::sqrt(var + eps);

        // Scale and shift
        for (int j = 0; j < n; ++j) {
            float norm = (row_x[j] - mean) * inv_std;
            float val = norm;
            if (gamma != nullptr) val *= gamma[j];
            if (beta != nullptr) val += beta[j];
            row_y[j] = val;
        }
    }
}

// Numerically accurate CPU reference for RMS Normalization (RMSNorm)
// Y[i, j] = (X[i, j] / sqrt(mean(X^2) + eps)) * gamma[j]
inline void rmsnorm_cpu_reference(const float* x, const float* gamma,
                                  float* y, int m, int n, float eps = 1e-5f) {
    for (int i = 0; i < m; ++i) {
        const float* row_x = x + i * n;
        float* row_y = y + i * n;

        // Compute sum of squares
        double sq_sum = 0.0;
        for (int j = 0; j < n; ++j) {
            double val = static_cast<double>(row_x[j]);
            sq_sum += val * val;
        }
        float rms = static_cast<float>(std::sqrt(sq_sum / n + static_cast<double>(eps)));
        float inv_rms = 1.0f / rms;

        // Scale
        for (int j = 0; j < n; ++j) {
            float val = row_x[j] * inv_rms;
            if (gamma != nullptr) val *= gamma[j];
            row_y[j] = val;
        }
    }
}
