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
    std::vector<float> h_ref(n, 0.0f);
    std::vector<float> h_gpu(n, 0.0f);

    init_random_vector(h_in, 0.0f, 1.0f, 1234);

    // Compute expected output on CPU
    scan_cpu_reference(h_in.data(), h_ref.data(), n);

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, n * sizeof(float)));

    // Launch user kernel
    launch_fn(d_in, d_out, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Because prefix sum accumulates values, the max diff can grow with N.
    // For 16M floats summing values in [0, 1], the final sum is ~8M. 
    // Float32 mantissa has ~7 digits of precision.
    // We allow a slightly larger relative error tolerance for very large arrays.
    bool passed = BenchmarkReport::verifyCorrectness(h_ref.data(), h_gpu.data(), n, 1e-1f, 1e-2f);

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
              << (1.0 * n * sizeof(float) * 2) / (1024.0 * 1024.0) << " MB traffic)\n";
    std::cout << "--------------------------------------------------------\n";

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_in, 1, n * sizeof(float)));

    for (int i = 0; i < warmup_iters; ++i) {
        launch_fn(d_in, d_out, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < benchmark_iters; ++i) {
        launch_fn(d_in, d_out, n);
    }
    float total_ms = timer.stop();
    float avg_ms = total_ms / benchmark_iters;

    double total_bytes = 2.0 * n * sizeof(float); // Read input, write output
    double total_flops = 1.0 * n; 

    BenchmarkReport::printMetrics(avg_ms, total_bytes, total_flops);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    std::cout << "\n========================================================\n";
    std::cout << "        LeetGPU Problem 04: Prefix Sum (Scan)           \n";
    std::cout << "        Platform: NVIDIA A100-SXM4-64GB                 \n";
    std::cout << "========================================================\n\n";

    std::vector<int> test_sizes = {1, 37, 1023, 1024, 2048, 1000003, 16777216};
    bool all_passed = true;

    std::cout << "[TEST SUITE] Running edge-case correctness tests...\n\n";

    // Milestone 1
    if (is_hillis_steele_implemented()) {
        std::cout << "Testing: Milestone 1: Hillis-Steele\n";
        for (int sz : test_sizes) {
            if (!test_correctness("Hillis-Steele", launch_scan_hillis_steele, sz)) {
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    // Milestone 2
    if (is_blelloch_implemented()) {
        std::cout << "Testing: Milestone 2: Blelloch\n";
        for (int sz : test_sizes) {
            if (!test_correctness("Blelloch", launch_scan_blelloch, sz)) {
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    // Milestone 3
    if (is_blelloch_padded_implemented()) {
        std::cout << "Testing: Milestone 3: Blelloch Padded\n";
        for (int sz : test_sizes) {
            if (!test_correctness("Blelloch Padded", launch_scan_blelloch_padded, sz)) {
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    if (!is_hillis_steele_implemented() && !is_blelloch_implemented() && !is_blelloch_padded_implemented()) {
        std::cout << "⚠️  No milestones implemented yet! Open `kernel.cu` to begin.\n";
        return 0;
    }

    if (!all_passed) {
        std::cout << "❌ Some correctness tests failed! Please fix before benchmarking.\n";
        return 1;
    }

    std::cout << "🎉 ALL CORRECTNESS TESTS PASSED! Proceeding to performance benchmarks...\n";

    int benchmark_n = 64 * 1024 * 1024; // 64M floats = 256MB

    if (is_hillis_steele_implemented()) {
        benchmark_kernel("Milestone 1: Hillis-Steele", launch_scan_hillis_steele, benchmark_n);
    }
    
    if (is_blelloch_implemented()) {
        benchmark_kernel("Milestone 2: Blelloch", launch_scan_blelloch, benchmark_n);
    }
    
    if (is_blelloch_padded_implemented()) {
        benchmark_kernel("Milestone 3: Blelloch Padded", launch_scan_blelloch_padded, benchmark_n);
    }

    if (is_hillis_steele_implemented() && is_blelloch_implemented() && is_blelloch_padded_implemented()) {
        std::cout << "\n========================================================\n";
        std::cout << " ✅ Problem 04 Complete! Ready for the next challenge.   \n";
        std::cout << "========================================================\n";
    }

    return 0;
}
