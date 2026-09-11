#include "kernel.cuh"
#include "reference.hpp"
#include "cuda_utils.cuh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

// Run correctness test on a matrix of size (rows x cols)
bool test_correctness(const std::string& name, 
                      void (*launch_fn)(const float*, float*, int, int), 
                      int rows, int cols) {
    size_t total_elements = static_cast<size_t>(rows) * cols;
    std::vector<float> h_in(total_elements);
    std::vector<float> h_ref(total_elements);
    std::vector<float> h_gpu(total_elements);

    init_random_matrix(h_in, -50.0f, 50.0f, 4242);

    // Compute expected output on CPU: (rows x cols) -> (cols x rows)
    matrix_transpose_cpu_reference(h_in.data(), h_ref.data(), rows, cols);

    // Allocate device buffers
    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_elements * sizeof(float)));

    // Copy input to device, zero out output buffer to catch partial writes
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, total_elements * sizeof(float)));

    // Launch kernel under test
    launch_fn(d_in, d_out, rows, cols);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify
    bool passed = BenchmarkReport::verifyCorrectness(h_ref.data(), h_gpu.data(), total_elements);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return passed;
}

// Benchmark kernel performance on large matrices (saturates A100 HBM2e memory bus)
void benchmark_kernel(const std::string& name,
                      void (*launch_fn)(const float*, float*, int, int),
                      int rows, int cols,
                      int warmup_iters = 5,
                      int benchmark_iters = 25) {
    size_t total_elements = static_cast<size_t>(rows) * cols;
    double traffic_mb = (2.0 * total_elements * sizeof(float)) / (1024.0 * 1024.0);

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " 🚀 Benchmarking: " << name << "\n";
    std::cout << "    Dimensions  : " << rows << " x " << cols << " (" 
              << total_elements / (1024 * 1024) << "M floats, " 
              << traffic_mb << " MB total memory traffic)\n";
    std::cout << "--------------------------------------------------------\n";

    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_in, 1, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, total_elements * sizeof(float)));

    // Warm-up runs
    for (int i = 0; i < warmup_iters; ++i) {
        launch_fn(d_in, d_out, rows, cols);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed benchmark loop
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < benchmark_iters; ++i) {
        launch_fn(d_in, d_out, rows, cols);
    }
    float total_ms = timer.stop();
    float avg_ms = total_ms / benchmark_iters;

    // Total memory traffic: read input (4B) + write output (4B) = 8 bytes per element
    double total_bytes = 2.0 * total_elements * sizeof(float);
    BenchmarkReport::printMetrics(avg_ms, total_bytes, 0.0);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    std::cout << "\n========================================================\n";
    std::cout << "        LeetGPU Problem 02: Matrix Transpose            \n";
    std::cout << "        Platform: NVIDIA A100-SXM4-64GB                 \n";
    std::cout << "========================================================\n";

    // -------------------------------------------------------------------------
    // PART 1: Edge-Case Correctness Suite
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST SUITE] Running edge-case correctness tests...\n";

    struct TestCase {
        int rows;
        int cols;
        std::string desc;
    };

    const std::vector<TestCase> test_cases = {
        {1, 1, "Single element (1x1)"},
        {31, 31, "Small non-power-of-2 (smaller than 1 tile)"},
        {32, 32, "Exact single tile (32x32)"},
        {64, 128, "Rectangular power-of-2 (M < N)"},
        {128, 64, "Rectangular power-of-2 (M > N)"},
        {127, 513, "Odd rectangular non-power-of-2"},
        {1, 1000, "Row vector to column vector (1x1000)"},
        {1000, 1, "Column vector to row vector (1000x1)"},
        {1024, 1024, "Medium square matrix (1024x1024)"}
    };

    std::vector<std::pair<std::string, void (*)(const float*, float*, int, int)>> kernels = {
        {"Milestone 1: Naive (Uncoalesced Write)", launch_matrix_transpose_naive}
    };

    if (is_shared_conflict_implemented()) {
        kernels.push_back({"Milestone 2: Shared Memory (Bank Conflicts)", launch_matrix_transpose_shared_conflict});
    }
    if (is_shared_padded_implemented()) {
        kernels.push_back({"Milestone 3: Shared Memory (Padded [32][33])", launch_matrix_transpose_shared_padded});
    }
    if (is_coarse_implemented()) {
        kernels.push_back({"Milestone 4: Coarse-Grained (32x8 block)", launch_matrix_transpose_coarse});
    }

    bool all_passed = true;
    for (const auto& [name, fn] : kernels) {
        std::cout << "\nTesting: " << name << "\n";
        for (const auto& tc : test_cases) {
            std::cout << "  • " << std::left << std::setw(42) 
                      << (tc.desc + " [" + std::to_string(tc.rows) + "x" + std::to_string(tc.cols) + "]") 
                      << " : ";
            bool passed = test_correctness(name, fn, tc.rows, tc.cols);
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
    // Matrix: 8192 x 8192 floats = 67.1M floats = 268.4 MB input + 268.4 MB output = 536.8 MB traffic
    // Exceeds A100's 32 MB L2 cache by >16x, measuring genuine HBM2e memory bandwidth!
    constexpr int BENCHMARK_ROWS = 8192;
    constexpr int BENCHMARK_COLS = 8192;

    for (const auto& [name, fn] : kernels) {
        benchmark_kernel(name, fn, BENCHMARK_ROWS, BENCHMARK_COLS);
    }

    std::cout << "\n========================================================\n";
    std::cout << " ✅ Problem 02 Evaluation Complete!                      \n";
    std::cout << "========================================================\n\n";

    return 0;
}
