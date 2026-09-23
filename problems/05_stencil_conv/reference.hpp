#pragma once

#include <vector>
#include <random>
#include <cmath>

// Initialize matrix with uniform random floats
inline void init_random_matrix(std::vector<float>& mat, float min_val = -1.0f, float max_val = 1.0f, unsigned int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(min_val, max_val);
    for (size_t i = 0; i < mat.size(); ++i) {
        mat[i] = dis(gen);
    }
}

// Initialize a normalized 2D Gaussian blur filter mask
inline void init_gaussian_mask(std::vector<float>& mask, int radius, float sigma = 1.0f) {
    int diameter = 2 * radius + 1;
    mask.resize(diameter * diameter);
    float sum = 0.0f;

    for (int r = -radius; r <= radius; ++r) {
        for (int c = -radius; c <= radius; ++c) {
            float val = std::exp(-(r * r + c * c) / (2.0f * sigma * sigma));
            mask[(r + radius) * diameter + (c + radius)] = val;
            sum += val;
        }
    }

    // Normalize so sum of weights equals 1.0
    for (size_t i = 0; i < mask.size(); ++i) {
        mask[i] /= sum;
    }
}

// Deterministic CPU reference for 2D convolution with zero-padding boundary condition
inline void conv2d_cpu_reference(const float* in, const float* mask, float* out, 
                                 int height, int width, int radius) {
    int diameter = 2 * radius + 1;

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            float sum = 0.0f;
            for (int kr = -radius; kr <= radius; ++kr) {
                for (int kc = -radius; kc <= radius; ++kc) {
                    int in_r = r + kr;
                    int in_c = c + kc;
                    if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
                        float pixel = in[in_r * width + in_c];
                        float weight = mask[(kr + radius) * diameter + (kc + radius)];
                        sum += pixel * weight;
                    }
                }
            }
            out[r * width + c] = sum;
        }
    }
}
