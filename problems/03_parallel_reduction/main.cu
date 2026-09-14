#include "kernel.cuh"
#include "reference.hpp"
#include "cuda_utils.cuh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

// Run correctness test on a given size
bool test_correctness(const std::string& name, 
                      void (*launch_fn)(const float*, float*, int), 
                      int n) {
    std::vector<float> h_in(n);
    float h_ref = 0.0f;
    float h_gpu = 0.0f;

    // Initialize with small values to prevent massive floating point accumulation errors
    init_random_vector(h_in, -1.0f, 1.0f, 1234);

    // Compute expected output on CPU
    h_ref = reduction_cpu_reference(h_in.data(), n);

    // Device memory allocation
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Clear the output accumulator before launching
    CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));

    // Launch user kernel
    launch_fn(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(&h_gpu, d_out, sizeof(float), cudaMemcpyDeviceToHost));

    // Verify. Because we sum up to 16M floats, we use a slightly relaxed tolerance
    // to account for the non-associativity of floating point addition.
    bool passed = BenchmarkReport::verifyCorrectness(&h_ref, &h_gpu, 1, 1e-2f, 1e-3f);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return passed;
}

// Benchmark kernel performance on a large size (saturates HBM2e)
void benchmark_kernel(const std::string& name,
                      void (*launch_fn)(const float*, float*, int),
                      int n,
                      int warmup_iters = 5,
                      int benchmark_iters = 20) {
    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " 🚀 Benchmarking: " << name << " (N = " << n / (1024 * 1024) << "M floats, " 
              << (1.0 * n * sizeof(float)) / (1024.0 * 1024.0) << " MB traffic)\n";
    std::cout << "--------------------------------------------------------\n";

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, sizeof(float)));

    // Initialize with 1.0f on device
    CUDA_CHECK(cudaMemset(d_in, 1, n * sizeof(float)));

    // Warm-up
    for (int i = 0; i < warmup_iters; ++i) {
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        launch_fn(d_in, d_out, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark loop with CUDA Events
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < benchmark_iters; ++i) {
        // Note: we must clear the accumulator each iteration!
        // cudaMemset is extremely fast for 4 bytes, but to be strictly fair
        // we include it in the timing (it's negligible compared to 64M reads).
        CUDA_CHECK(cudaMemset(d_out, 0, sizeof(float)));
        launch_fn(d_in, d_out, n);
    }
    float total_ms = timer.stop();
    float avg_ms = total_ms / benchmark_iters;

    // Total bytes transferred: read input (4B) + write output (negligible)
    double total_bytes = 1.0 * n * sizeof(float);
    double total_flops = 1.0 * n; // 1 addition per element

    BenchmarkReport::printMetrics(avg_ms, total_bytes, total_flops);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    std::cout << "\n========================================================\n";
    std::cout << "        LeetGPU Problem 03: Parallel Reduction          \n";
    std::cout << "        Platform: NVIDIA A100-SXM4-64GB                 \n";
    std::cout << "========================================================\n";

    // -------------------------------------------------------------------------
    // PART 1: Edge-Case Correctness Suite
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST SUITE] Running edge-case correctness tests...\n";
    const std::vector<int> test_sizes = {
        1,          // Edge case: single element
        37,         // Edge case: small prime
        1023,       // Edge case: 1 less than 1024
        1024,       // Exact block multiple (e.g. 4 blocks of 256)
        2048,       // Multiple blocks
        1000003,    // Medium prime number
        16777216    // 16M elements
    };

    std::vector<std::pair<std::string, void (*)(const float*, float*, int)>> kernels = {
        {"Milestone 1: Interleaved Divergent", launch_reduction_divergent}
    };

    if (is_interleaved_implemented()) {
        kernels.push_back({"Milestone 2: Interleaved Strided (Bank Conflicts)", launch_reduction_interleaved});
    }
    if (is_sequential_implemented()) {
        kernels.push_back({"Milestone 3: Sequential (Conflict-Free)", launch_reduction_sequential});
    }
    if (is_warp_shuffle_implemented()) {
        kernels.push_back({"Milestone 4: Warp Shuffle (__shfl_down_sync)", launch_reduction_warp_shuffle});
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
    // N = 64M floats (256 MB memory traffic)
    // Exceeds A100's 32 MB L2 cache, measuring pure HBM2e memory bus bandwidth
    constexpr int BENCHMARK_N = 67108864; // 64 * 1024 * 1024

    for (const auto& [name, fn] : kernels) {
        benchmark_kernel(name, fn, BENCHMARK_N);
    }

    std::cout << "\n========================================================\n";
    std::cout << " ✅ Problem 03 Complete! Ready for the next challenge.   \n";
    std::cout << "========================================================\n\n";

    return 0;
}
