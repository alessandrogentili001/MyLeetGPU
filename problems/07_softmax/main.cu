#include "kernel.cuh"
#include "reference.hpp"
#include "cuda_utils.cuh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

// Run correctness test on a given matrix shape (M x N)
bool test_correctness(const std::string& name, 
                      void (*launch_fn)(const float*, float*, int, int), 
                      int m, int n,
                      float min_val = -5.0f, float max_val = 5.0f) {
    size_t total_elements = static_cast<size_t>(m) * n;
    std::vector<float> h_in(total_elements);
    std::vector<float> h_ref(total_elements, 0.0f);
    std::vector<float> h_gpu(total_elements, 0.0f);

    init_random_matrix(h_in, m, n, min_val, max_val, 1234);

    // Compute ground-truth output on CPU
    softmax_cpu_reference(h_in.data(), h_ref.data(), m, n);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, total_elements * sizeof(float)));

    // Launch user kernel
    launch_fn(d_in, d_out, m, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "  Testing " << name << " (" << m << "x" << n 
              << ", range [" << min_val << ", " << max_val << "])...\n";

    // FP32 softmax has slight variance due to reduction order (associativity of floating point additions)
    bool passed = BenchmarkReport::verifyCorrectness(h_ref.data(), h_gpu.data(), total_elements, 1e-4f, 1e-4f);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return passed;
}

// Benchmark kernel performance on a large size that exceeds L2 cache (32MB)
void benchmark_kernel(const std::string& name,
                      void (*launch_fn)(const float*, float*, int, int),
                      int m, int n,
                      int warmup_iters = 5,
                      int benchmark_iters = 20) {
    size_t total_elements = static_cast<size_t>(m) * n;
    double traffic_mb = (2.0 * total_elements * sizeof(float)) / (1024.0 * 1024.0);

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " 🚀 Benchmarking: " << name << " (M=" << m << ", N=" << n 
              << ", Total=" << total_elements / (1024 * 1024) << "M floats, " 
              << std::fixed << std::setprecision(1) << traffic_mb << " MB DRAM traffic)\n";
    std::cout << "--------------------------------------------------------\n";

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_in, 1, total_elements * sizeof(float)));

    for (int i = 0; i < warmup_iters; ++i) {
        launch_fn(d_in, d_out, m, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < benchmark_iters; ++i) {
        launch_fn(d_in, d_out, m, n);
    }
    float total_ms = timer.stop();
    float avg_ms = total_ms / benchmark_iters;

    // Minimum ideal traffic: read input once (4 bytes), write output once (4 bytes)
    double total_bytes = 2.0 * total_elements * sizeof(float);
    // FLOPs: ~5 per element (max compare, subtract, exp, sum add, normalize multiply)
    double total_flops = 5.0 * total_elements;

    BenchmarkReport::printMetrics(avg_ms, total_bytes, total_flops);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    std::cout << "\n========================================================\n";
    std::cout << "        LeetGPU Problem 07: Softmax Kernel              \n";
    std::cout << "        Platform: NVIDIA A100-SXM4-64GB                 \n";
    std::cout << "========================================================\n\n";

    // Test matrix shapes: (M, N)
    std::vector<std::pair<int, int>> test_shapes = {
        {1, 1},          // Extreme edge case: 1x1 scalar
        {4, 32},         // Single warp row size
        {16, 127},       // Odd length, less than block size
        {37, 777},       // Arbitrary non-power-of-2 dimensions
        {64, 512},       // Moderate sequence length
        {128, 1024},     // Standard transformer hidden dimension
        {256, 2048}      // Large vocabulary / context length
    };

    bool all_passed = true;

    std::cout << "[TEST SUITE] Running edge-case & numerical correctness tests...\n\n";

    // Milestone 1
    if (is_block_twopass_implemented()) {
        std::cout << "========================================================\n";
        std::cout << "Testing Milestone 1: Two-Pass Block Softmax\n";
        std::cout << "========================================================\n";
        for (const auto& shape : test_shapes) {
            if (!test_correctness("Milestone 1", launch_softmax_block_twopass, shape.first, shape.second)) {
                all_passed = false;
            }
        }
        // Numerical stability test with large values that would overflow naive softmax (exp(89) overflows FP32)
        std::cout << "  Testing Numerical Overflow Safety [50.0, 100.0]...\n";
        if (!test_correctness("Milestone 1 (Overflow Safety)", launch_softmax_block_twopass, 8, 1024, 50.0f, 100.0f)) {
            all_passed = false;
        }
        std::cout << "\n";
    }

    // Milestone 2
    if (is_warp_shuffle_implemented()) {
        std::cout << "========================================================\n";
        std::cout << "Testing Milestone 2: Warp-Accelerated Softmax\n";
        std::cout << "========================================================\n";
        for (const auto& shape : test_shapes) {
            if (!test_correctness("Milestone 2", launch_softmax_warp_shuffle, shape.first, shape.second)) {
                all_passed = false;
            }
        }
        std::cout << "  Testing Numerical Overflow Safety [50.0, 100.0]...\n";
        if (!test_correctness("Milestone 2 (Overflow Safety)", launch_softmax_warp_shuffle, 8, 1024, 50.0f, 100.0f)) {
            all_passed = false;
        }
        std::cout << "\n";
    }

    // Milestone 3
    if (is_online_implemented()) {
        std::cout << "========================================================\n";
        std::cout << "Testing Milestone 3: Online FlashSoftmax\n";
        std::cout << "========================================================\n";
        for (const auto& shape : test_shapes) {
            if (!test_correctness("Milestone 3", launch_softmax_online, shape.first, shape.second)) {
                all_passed = false;
            }
        }
        std::cout << "  Testing Numerical Overflow Safety [50.0, 100.0]...\n";
        if (!test_correctness("Milestone 3 (Overflow Safety)", launch_softmax_online, 8, 1024, 50.0f, 100.0f)) {
            all_passed = false;
        }
        std::cout << "\n";
    }

    if (!is_block_twopass_implemented() && !is_warp_shuffle_implemented() && !is_online_implemented()) {
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

    if (is_block_twopass_implemented()) {
        benchmark_kernel("Milestone 1: Two-Pass Block Softmax", launch_softmax_block_twopass, bench_m, bench_n);
    }

    if (is_warp_shuffle_implemented()) {
        benchmark_kernel("Milestone 2: Warp-Accelerated Softmax", launch_softmax_warp_shuffle, bench_m, bench_n);
    }

    if (is_online_implemented()) {
        benchmark_kernel("Milestone 3: Online FlashSoftmax", launch_softmax_online, bench_m, bench_n);
    }

    if (is_block_twopass_implemented() && is_warp_shuffle_implemented() && is_online_implemented()) {
        std::cout << "\n========================================================\n";
        std::cout << " ✅ Problem 07 Complete! Ready for the next challenge.   \n";
        std::cout << "========================================================\n";
    }

    return 0;
}
