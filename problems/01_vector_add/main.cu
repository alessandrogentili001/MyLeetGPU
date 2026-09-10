#include "kernel.cuh"
#include "reference.hpp"
#include "cuda_utils.cuh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

// Run correctness test on a given size
bool test_correctness(const std::string& name, 
                      void (*launch_fn)(const float*, const float*, float*, int), 
                      int n) {
    std::vector<float> h_a(n), h_b(n), h_ref(n), h_gpu(n);
    init_random_vector(h_a, -100.0f, 100.0f, 1234);
    init_random_vector(h_b, -100.0f, 100.0f, 5678);

    // Compute expected output on CPU
    vector_add_cpu_reference(h_a.data(), h_b.data(), h_ref.data(), n);

    // Device memory allocation
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    // Launch user kernel
    launch_fn(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    bool passed = BenchmarkReport::verifyCorrectness(h_ref.data(), h_gpu.data(), n);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));

    return passed;
}

// Benchmark kernel performance on a large size (saturates HBM2e)
void benchmark_kernel(const std::string& name,
                      void (*launch_fn)(const float*, const float*, float*, int),
                      int n,
                      int warmup_iters = 5,
                      int benchmark_iters = 20) {
    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " 🚀 Benchmarking: " << name << " (N = " << n / (1024 * 1024) << "M floats, " 
              << (3.0 * n * sizeof(float)) / (1024.0 * 1024.0) << " MB traffic)\n";
    std::cout << "--------------------------------------------------------\n";

    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    // Initialize with 1.0f on device
    CUDA_CHECK(cudaMemset(d_a, 1, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_b, 1, n * sizeof(float)));

    // Warm-up
    for (int i = 0; i < warmup_iters; ++i) {
        launch_fn(d_a, d_b, d_c, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark loop with CUDA Events
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < benchmark_iters; ++i) {
        launch_fn(d_a, d_b, d_c, n);
    }
    float total_ms = timer.stop();
    float avg_ms = total_ms / benchmark_iters;

    // Total bytes transferred: read A (4B) + read B (4B) + write C (4B) = 12B per element
    double total_bytes = 3.0 * n * sizeof(float);
    double total_flops = 1.0 * n; // 1 addition per element

    BenchmarkReport::printMetrics(avg_ms, total_bytes, total_flops);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
}

int main() {
    std::cout << "\n========================================================\n";
    std::cout << "        LeetGPU Problem 01: Vector Addition             \n";
    std::cout << "        Platform: NVIDIA A100-SXM4-64GB                 \n";
    std::cout << "========================================================\n";

    // -------------------------------------------------------------------------
    // PART 1: Edge-Case Correctness Suite
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST SUITE] Running edge-case correctness tests...\n";
    const std::vector<int> test_sizes = {
        1,          // Edge case: single element
        37,         // Edge case: small non-power-of-2
        1023,       // Edge case: 1 less than 1024 (boundary check)
        1024,       // Exact block multiple
        1000003     // Medium prime number
    };

    std::vector<std::pair<std::string, void (*)(const float*, const float*, float*, int)>> kernels = {
        {"Naive (1-Thread/Element)", launch_vector_add_naive}
    };

    if (is_grid_stride_implemented()) {
        kernels.push_back({"Grid-Stride Loop", launch_vector_add_grid_stride});
    }
    if (is_vectorized_implemented()) {
        kernels.push_back({"Vectorized (float4)", launch_vector_add_vectorized});
    }

    bool all_passed = true;
    for (const auto& [name, fn] : kernels) {
        std::cout << "\nTesting: " << name << "\n";
        for (int size : test_sizes) {
            std::cout << "  • Size N = " << std::setw(8) << size << " : ";
            bool passed = test_correctness(name, fn, size);
            if (!passed) all_passed = false;
        }
    }

    if (!all_passed) {
        std::cerr << "\n❌ Some correctness tests failed! Please fix before benchmarking.\n";
        return 1;
    }

    std::cout << "\n🎉 ALL CORRECTNESS TESTS PASSED! Proceeding to performance benchmarks...\n";

    // -------------------------------------------------------------------------
    // PART 2: Memory Bandwidth Roofline Benchmark
    // -------------------------------------------------------------------------
    // N = 64M floats (256 MB per array, 768 MB total memory traffic)
    // Exceeds A100's 32 MB L2 cache, measuring pure HBM2e memory bus bandwidth
    constexpr int BENCHMARK_N = 67108864; // 64 * 1024 * 1024

    for (const auto& [name, fn] : kernels) {
        benchmark_kernel(name, fn, BENCHMARK_N);
    }

    std::cout << "\n========================================================\n";
    std::cout << " ✅ Problem 01 Complete! Ready for the next challenge.   \n";
    std::cout << "========================================================\n\n";

    return 0;
}
