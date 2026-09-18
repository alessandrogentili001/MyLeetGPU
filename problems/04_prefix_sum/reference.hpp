#pragma once
#include <vector>
#include <random>

inline void init_random_vector(std::vector<float>& vec, float min_val = -10.0f, float max_val = 10.0f, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] = dis(gen);
    }
}
inline void scan_cpu_reference(const float* in, float* out, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        out[i] = sum;
        sum += in[i];
    }
}
