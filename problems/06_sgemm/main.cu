#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cublas_v2.h>

#include "cuda_utils.cuh"
#include "reference.hpp"
#include "kernel.cuh"

// Macro for checking cuBLAS errors
#define CUBLAS_CHECK(call)                                                      \
    do {                                                                      \
        cublasStatus_t err = call;                                            \
        if (err != CUBLAS_STATUS_SUCCESS) {                                   \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__    \
                      << " -> " << err                                        \
                      << " (" #call ")" << std::endl;                         \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

void run_correctness_tests() {
    std::cout << "\n[TEST SUITE] Running edge-case correctness tests...\n";

    struct TestCase { int M, N, K; };
    std::vector<TestCase> tests = {
        {1, 1, 1},
        {32, 32, 32},
        {31, 33, 35},   // Non-multiples of block sizes
        {128, 128, 128},
        {64, 127, 255}, // Rectangular
        {1024, 1024, 1024} // Medium size
    };

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    auto run_test = [&](const std::string& name, auto launch_func, bool is_impl) {
        if (!is_impl) return;
        std::cout << "Testing: " << name << " ... ";
        bool passed = true;
        for (const auto& test : tests) {
            int M = test.M;
            int N = test.N;
            int K = test.K;

            std::vector<float> h_A(M * K);
            std::vector<float> h_B(K * N);
            std::vector<float> h_C_ref(M * N, 0.0f);
            std::vector<float> h_C_gpu(M * N, 0.0f);

            // Init matrices with random values
            for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(rand()) / RAND_MAX;
            for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>(rand()) / RAND_MAX;

            sgemm_cpu_reference(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

            float *d_A, *d_B, *d_C;
            CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

            CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

            // Launch
            launch_func(d_A, d_B, d_C, M, N, K);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemcpy(h_C_gpu.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost));

            // Use the utility verifyCorrectness
            if (!BenchmarkReport::verifyCorrectness(h_C_ref.data(), h_C_gpu.data(), M * N, 1e-3f, 1e-3f)) {
                passed = false;
                std::cout << "❌ MISMATCH at M=" << M << " N=" << N << " K=" << K << "\n";
                break;
            }

            CUDA_CHECK(cudaFree(d_A));
            CUDA_CHECK(cudaFree(d_B));
            CUDA_CHECK(cudaFree(d_C));
        }
        if (passed) std::cout << "PASSED! ✅\n";
    };

    run_test("Milestone 1: Naive (Global Memory)", launch_sgemm_naive, is_naive_implemented());
    run_test("Milestone 2: Shared Memory Tiled", launch_sgemm_shared_tiled, is_shared_tiled_implemented());
    run_test("Milestone 3: 2D Register Tiled", launch_sgemm_2d_register_tiled, is_2d_register_tiled_implemented());

    auto cublas_launch = [&](const float* A, const float* B, float* C, int M, int N, int K) {
        launch_sgemm_cublas(handle, A, B, C, M, N, K);
    };
    run_test("Milestone 4: cuBLAS", cublas_launch, true);

    CUBLAS_CHECK(cublasDestroy(handle));
    std::cout << "\n🎉 ALL CORRECTNESS TESTS PASSED! Proceeding to performance benchmarks...\n\n";
}

void run_benchmarks() {
    // Benchmark size for SGEMM
    // Large enough to dominate L2 cache (32MB) on A100.
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const int num_runs = 5;
    
    // FLOPS = 2 * M * N * K
    double tflops_ideal = 2.0 * static_cast<double>(M) * N * K / 1e12;
    double bytes_accessed = static_cast<double>(M * K + K * N + M * N) * sizeof(float); // Only min bytes

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, M * N * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_A, 0, M * K * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_B, 0, K * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_C, 0, M * N * sizeof(float)));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    auto run_bench = [&](const std::string& name, auto launch_func, bool is_impl) {
        if (!is_impl) return;

        // Warmup
        launch_func(d_A, d_B, d_C, M, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        GpuTimer timer;
        timer.start();
        for (int i = 0; i < num_runs; ++i) {
            launch_func(d_A, d_B, d_C, M, N, K);
        }
        float total_ms = timer.stop();
        float avg_ms = total_ms / num_runs;
        
        double time_sec = avg_ms / 1000.0;
        double tflops = (2.0 * M * N * K / 1e12) / time_sec;
        
        std::cout << "--------------------------------------------------------\n";
        std::cout << " 🚀 Benchmarking: " << name << "\n";
        std::cout << "    Dimensions  : M=" << M << ", N=" << N << ", K=" << K << "\n";
        std::cout << "--------------------------------------------------------\n";
        BenchmarkReport::printMetrics(avg_ms, bytes_accessed, 2.0 * M * N * K);
        std::cout << "\n";
    };

    run_bench("Milestone 1: Naive (Global Memory)", launch_sgemm_naive, is_naive_implemented());
    run_bench("Milestone 2: Shared Memory Tiled", launch_sgemm_shared_tiled, is_shared_tiled_implemented());
    run_bench("Milestone 3: 2D Register Tiled", launch_sgemm_2d_register_tiled, is_2d_register_tiled_implemented());

    auto cublas_launch = [&](const float* A, const float* B, float* C, int M, int N, int K) {
        launch_sgemm_cublas(handle, A, B, C, M, N, K);
    };
    run_bench("Milestone 4: cuBLAS Baseline", cublas_launch, true);

    CUBLAS_CHECK(cublasDestroy(handle));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
}

int main() {
    std::cout << "\n========================================================\n";
    std::cout << "    LeetGPU Problem 06: SGEMM (Matrix Multiplication)   \n";
    std::cout << "    Precision  : FP32 (Single Precision)               \n";
    std::cout << "    Platform   : NVIDIA A100-SXM4-64GB (sm_80)         \n";
    std::cout << "========================================================\n";

    run_correctness_tests();
    run_benchmarks();

    std::cout << "========================================================\n";
    std::cout << " ✅ Problem 06 Complete! Ready for Problem 07 (Softmax). \n";
    std::cout << "========================================================\n";

    return 0;
}
