#pragma once

#include <vector>
#include <random>

// CPU Ground Truth: Expected Output Generator
inline void vector_add_cpu_reference(const float* a, const float* b, float* expected_c, int n) {
    for (int i = 0; i < n; ++i) {
        expected_c[i] = a[i] + b[i];
    }
}

// Input data generator
inline void init_random_vector(std::vector<float>& vec, float min_val = -10.0f, float max_val = 10.0f, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = dis(gen);
    }
}
