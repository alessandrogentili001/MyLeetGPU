#pragma once

#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>
#include <cuda_runtime.h>

// CUDA Error Checking Macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__      \
                      << " -> " << cudaGetErrorString(err)                    \
                      << " (" #call ")" << std::endl;                         \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// High-precision GPU Timer using CUDA Events
struct GpuTimer {
    cudaEvent_t start_event;
    cudaEvent_t stop_event;

    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~GpuTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_event, stream));
    }

    float stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event));
        return elapsed_ms; // milliseconds
    }
};

// Pretty reporting for MyLeetGPU benchmarks
namespace BenchmarkReport {
    // NVIDIA A100-SXM4-64GB Theoretical Peaks (Leonardo Booster)
    constexpr double A100_PEAK_FP32_TFLOPS = 19.49;    // 19.5 TFLOPS
    constexpr double A100_PEAK_FP16_TC_TFLOPS = 312.0; // 312 TFLOPS with Tensor Cores
    constexpr double A100_PEAK_BANDWIDTH_GBS = 1555.0; // ~1555 - 2039 GB/s HBM2e

    inline void printHeader(const std::string& kernel_name) {
        std::cout << "\n========================================================\n";
        std::cout << "  Kernel: " << kernel_name << "\n";
        std::cout << "========================================================\n";
    }

    inline void printMetrics(float time_ms, 
                             double bytes_accessed = 0.0, 
                             double flops = 0.0) {
        double time_sec = time_ms / 1000.0;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << " ⏱️  Avg Latency  : " << time_ms << " ms (" << time_ms * 1000.0 << " us)\n";

        if (bytes_accessed > 0.0) {
            double bandwidth_gbs = (bytes_accessed / 1e9) / time_sec;
            double bw_pct = (bandwidth_gbs / A100_PEAK_BANDWIDTH_GBS) * 100.0;
            std::cout << " 🚀 Bandwidth    : " << bandwidth_gbs << " GB/s ("
                      << bw_pct << "% of A100 peak " << A100_PEAK_BANDWIDTH_GBS << " GB/s)\n";
        }

        if (flops > 0.0) {
            double tflops = (flops / 1e12) / time_sec;
            double fp32_pct = (tflops / A100_PEAK_FP32_TFLOPS) * 100.0;
            std::cout << " ⚡ Compute      : " << tflops << " TFLOPS ("
                      << fp32_pct << "% of A100 FP32 peak)\n";
        }
        std::cout << "--------------------------------------------------------\n";
    }

    inline bool verifyCorrectness(const float* ref, const float* out, size_t n, float atol = 1e-4f, float rtol = 1e-4f) {
        float max_diff = 0.0f;
        size_t error_count = 0;
        size_t first_err_idx = 0;

        for (size_t i = 0; i < n; ++i) {
            float diff = std::abs(ref[i] - out[i]);
            float tol = atol + rtol * std::abs(ref[i]);
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > tol) {
                if (error_count == 0) first_err_idx = i;
                error_count++;
            }
        }

        if (error_count == 0) {
            std::cout << " ✅ Correctness  : PASSED! (Max diff: " << max_diff << ")\n";
            return true;
        } else {
            std::cout << " ❌ Correctness  : FAILED! (" << error_count << " mismatches out of " << n << ")\n";
            std::cout << "    First mismatch at index " << first_err_idx 
                      << ": Expected " << ref[first_err_idx] 
                      << ", Got " << out[first_err_idx] 
                      << " (diff: " << std::abs(ref[first_err_idx] - out[first_err_idx]) << ")\n";
            return false;
        }
    }
}
