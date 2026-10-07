#include "kernel.cuh"
#include "reference.hpp"
#include "cuda_utils.cuh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

// Test LayerNorm correctness for a given shape (m x n)
bool test_layernorm_correctness(const std::string& name,
                                void (*launch_fn)(const float*, const float*, const float*, float*, int, int, float),
                                int m, int n, float eps = 1e-5f) {
    size_t total_elements = static_cast<size_t>(m) * n;
    std::vector<float> h_x(total_elements);
    std::vector<float> h_gamma(n);
    std::vector<float> h_beta(n);
    std::vector<float> h_ref(total_elements, 0.0f);
    std::vector<float> h_gpu(total_elements, 0.0f);

    init_random_tensor(h_x, total_elements, -3.0f, 3.0f, 1001);
    init_random_tensor(h_gamma, n, 0.5f, 1.5f, 2002);
    init_random_tensor(h_beta, n, -0.5f, 0.5f, 3003);

    layernorm_cpu_reference(h_x.data(), h_gamma.data(), h_beta.data(), h_ref.data(), m, n, eps);

    float *d_x, *d_gamma, *d_beta, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, total_elements * sizeof(float)));

    launch_fn(d_x, d_gamma, d_beta, d_y, m, n, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_y, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "  Testing " << name << " (" << m << "x" << n << ")...\n";
    bool passed = BenchmarkReport::verifyCorrectness(h_ref.data(), h_gpu.data(), total_elements, 1e-4f, 1e-4f);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_y));

    return passed;
}

// Test RMSNorm correctness for a given shape (m x n)
bool test_rmsnorm_correctness(const std::string& name,
                              void (*launch_fn)(const float*, const float*, float*, int, int, float),
                              int m, int n, float eps = 1e-5f) {
    size_t total_elements = static_cast<size_t>(m) * n;
    std::vector<float> h_x(total_elements);
    std::vector<float> h_gamma(n);
    std::vector<float> h_ref(total_elements, 0.0f);
    std::vector<float> h_gpu(total_elements, 0.0f);

    init_random_tensor(h_x, total_elements, -3.0f, 3.0f, 4004);
    init_random_tensor(h_gamma, n, 0.5f, 1.5f, 5005);

    rmsnorm_cpu_reference(h_x.data(), h_gamma.data(), h_ref.data(), m, n, eps);

    float *d_x, *d_gamma, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x, h_x.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_y, 0, total_elements * sizeof(float)));

    launch_fn(d_x, d_gamma, d_y, m, n, eps);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_y, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "  Testing " << name << " (" << m << "x" << n << ")...\n";
    bool passed = BenchmarkReport::verifyCorrectness(h_ref.data(), h_gpu.data(), total_elements, 1e-4f, 1e-4f);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_y));

    return passed;
}

// Benchmark LayerNorm kernel
void benchmark_layernorm(const std::string& name,
                         void (*launch_fn)(const float*, const float*, const float*, float*, int, int, float),
                         int m, int n, float eps = 1e-5f,
                         int warmup_iters = 5, int benchmark_iters = 20) {
    size_t total_elements = static_cast<size_t>(m) * n;
    double traffic_mb = (2.0 * total_elements * sizeof(float)) / (1024.0 * 1024.0);

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " 🚀 Benchmarking: " << name << " (M=" << m << ", N=" << n
              << ", Total=" << total_elements / (1024 * 1024) << "M floats, "
              << std::fixed << std::setprecision(1) << traffic_mb << " MB DRAM traffic)\n";
    std::cout << "--------------------------------------------------------\n";

    float *d_x, *d_gamma, *d_beta, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_beta, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_x, 1, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gamma, 1, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_beta, 0, n * sizeof(float)));

    for (int i = 0; i < warmup_iters; ++i) {
        launch_fn(d_x, d_gamma, d_beta, d_y, m, n, eps);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < benchmark_iters; ++i) {
        launch_fn(d_x, d_gamma, d_beta, d_y, m, n, eps);
    }
    float avg_ms = timer.stop() / benchmark_iters;

    double total_bytes = 2.0 * total_elements * sizeof(float);
    double total_flops = 7.0 * total_elements; // ~7 FLOPs per element in LayerNorm

    BenchmarkReport::printMetrics(avg_ms, total_bytes, total_flops);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_beta));
    CUDA_CHECK(cudaFree(d_y));
}

// Benchmark RMSNorm kernel
void benchmark_rmsnorm(const std::string& name,
                       void (*launch_fn)(const float*, const float*, float*, int, int, float),
                       int m, int n, float eps = 1e-5f,
                       int warmup_iters = 5, int benchmark_iters = 20) {
    size_t total_elements = static_cast<size_t>(m) * n;
    double traffic_mb = (2.0 * total_elements * sizeof(float)) / (1024.0 * 1024.0);

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " 🚀 Benchmarking: " << name << " (M=" << m << ", N=" << n
              << ", Total=" << total_elements / (1024 * 1024) << "M floats, "
              << std::fixed << std::setprecision(1) << traffic_mb << " MB DRAM traffic)\n";
    std::cout << "--------------------------------------------------------\n";

    float *d_x, *d_gamma, *d_y;
    CUDA_CHECK(cudaMalloc(&d_x, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gamma, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_x, 1, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_gamma, 1, n * sizeof(float)));

    for (int i = 0; i < warmup_iters; ++i) {
        launch_fn(d_x, d_gamma, d_y, m, n, eps);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < benchmark_iters; ++i) {
        launch_fn(d_x, d_gamma, d_y, m, n, eps);
    }
    float avg_ms = timer.stop() / benchmark_iters;

    double total_bytes = 2.0 * total_elements * sizeof(float);
    double total_flops = 3.0 * total_elements; // ~3 FLOPs per element in RMSNorm

    BenchmarkReport::printMetrics(avg_ms, total_bytes, total_flops);

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_gamma));
    CUDA_CHECK(cudaFree(d_y));
}

int main() {
    std::cout << "\n========================================================\n";
    std::cout << "    LeetGPU Problem 08: LayerNorm & RMSNorm Kernels     \n";
    std::cout << "    Platform: NVIDIA A100-SXM4-64GB (sm_80)             \n";
    std::cout << "========================================================\n\n";

    std::vector<std::pair<int, int>> test_shapes = {
        {1, 1},          // Scalar edge case
        {4, 32},         // Single warp row size
        {16, 127},       // Odd length, not multiple of 4 or 32
        {37, 777},       // Arbitrary non-power-of-2 dimensions
        {64, 512},       // Moderate sequence length
        {128, 1024},     // Standard transformer dimension
        {256, 4096}      // Large hidden size (LLaMA-7B)
    };

    bool all_passed = true;

    std::cout << "[TEST SUITE] Running edge-case & numerical correctness tests...\n\n";

    // Milestone 1
    if (is_layernorm_twopass_implemented()) {
        std::cout << "========================================================\n";
        std::cout << "Testing Milestone 1: Multi-Pass Block LayerNorm\n";
        std::cout << "========================================================\n";
        for (const auto& shape : test_shapes) {
            if (!test_layernorm_correctness("Milestone 1", launch_layernorm_twopass, shape.first, shape.second)) {
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    // Milestone 2
    if (is_layernorm_welford_implemented()) {
        std::cout << "========================================================\n";
        std::cout << "Testing Milestone 2: One-Pass Welford LayerNorm\n";
        std::cout << "========================================================\n";
        for (const auto& shape : test_shapes) {
            if (!test_layernorm_correctness("Milestone 2", launch_layernorm_welford, shape.first, shape.second)) {
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    // Milestone 3
    if (is_rmsnorm_vectorized_implemented()) {
        std::cout << "========================================================\n";
        std::cout << "Testing Milestone 3: Fused Vectorized RMSNorm\n";
        std::cout << "========================================================\n";
        for (const auto& shape : test_shapes) {
            if (!test_rmsnorm_correctness("Milestone 3", launch_rmsnorm_vectorized, shape.first, shape.second)) {
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    if (!is_layernorm_twopass_implemented() &&
        !is_layernorm_welford_implemented() &&
        !is_rmsnorm_vectorized_implemented()) {
        std::cout << "⚠️  No milestones implemented yet! Open `kernel.cu` to begin.\n";
        return 0;
    }

    if (!all_passed) {
        std::cout << "❌ Some correctness tests failed! Please fix before benchmarking.\n";
        return 1;
    }

    std::cout << "🎉 ALL CORRECTNESS TESTS PASSED! Proceeding to performance benchmarks...\n";

    // Benchmark on M=8192, N=4096 (33.5M floats, 268 MB DRAM traffic -> exceeds 32 MB A100 L2 cache)
    int bench_m = 8192;
    int bench_n = 4096;

    if (is_layernorm_twopass_implemented()) {
        benchmark_layernorm("Milestone 1: Multi-Pass Block LayerNorm", launch_layernorm_twopass, bench_m, bench_n);
    }

    if (is_layernorm_welford_implemented()) {
        benchmark_layernorm("Milestone 2: One-Pass Welford LayerNorm", launch_layernorm_welford, bench_m, bench_n);
    }

    if (is_rmsnorm_vectorized_implemented()) {
        benchmark_rmsnorm("Milestone 3: Fused Vectorized RMSNorm", launch_rmsnorm_vectorized, bench_m, bench_n);
    }

    if (is_layernorm_twopass_implemented() &&
        is_layernorm_welford_implemented() &&
        is_rmsnorm_vectorized_implemented()) {
        std::cout << "\n========================================================\n";
        std::cout << " ✅ Problem 08 Complete! Ready for Problem 09 (FlashAttention-2). \n";
        std::cout << "========================================================\n";
    }

    return 0;
}
