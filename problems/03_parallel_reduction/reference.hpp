#pragma once

#include <vector>
#include <random>

// CPU Ground Truth: High-precision accumulation using double
inline float reduction_cpu_reference(const float* in, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += static_cast<double>(in[i]);
    }
    return static_cast<float>(sum);
}

// Random vector initializer with uniform range [-1.0, 1.0]
inline void init_random_vector(std::vector<float>& vec, float min_val = -1.0f, float max_val = 1.0f, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = dis(gen);
    }
}
