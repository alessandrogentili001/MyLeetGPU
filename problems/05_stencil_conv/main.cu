#include "kernel.cuh"
#include "reference.hpp"
#include "cuda_utils.cuh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

// Wrapper for Milestone 1 to match the uniform benchmark signature
void wrapper_naive(const float* d_in, const float* h_mask, float* d_out, int height, int width) {
    static float* d_mask = nullptr;
    if (!d_mask) {
        CUDA_CHECK(cudaMalloc(&d_mask, KERNEL_SIZE * sizeof(float)));
    }
    CUDA_CHECK(cudaMemcpy(d_mask, h_mask, KERNEL_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    launch_conv2d_naive(d_in, d_mask, d_out, height, width);
}

// Correctness test on a given matrix dimension (height x width)
bool test_correctness(const std::string& name, 
                      void (*launch_fn)(const float*, const float*, float*, int, int), 
                      int height, int width) {
    size_t total_elements = static_cast<size_t>(height) * width;
    std::vector<float> h_in(total_elements);
    std::vector<float> h_mask(KERNEL_SIZE);
    std::vector<float> h_ref(total_elements, 0.0f);
    std::vector<float> h_gpu(total_elements, 0.0f);

    init_random_matrix(h_in, -10.0f, 10.0f, 1337);
    init_gaussian_mask(h_mask, KERNEL_RADIUS, 1.2f);

    // Compute ground truth on CPU
    conv2d_cpu_reference(h_in.data(), h_mask.data(), h_ref.data(), height, width, KERNEL_RADIUS);

    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0, total_elements * sizeof(float)));

    // Launch GPU kernel
    launch_fn(d_in, h_mask.data(), d_out, height, width);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_out, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify against CPU reference
    bool passed = BenchmarkReport::verifyCorrectness(h_ref.data(), h_gpu.data(), total_elements, 1e-4f, 1e-4f);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return passed;
}

// Benchmark kernel performance on large matrices (blowing past A100 L2 cache)
void benchmark_kernel(const std::string& name,
                      void (*launch_fn)(const float*, const float*, float*, int, int),
                      int height, int width,
                      int warmup_iters = 5,
                      int benchmark_iters = 25) {
    size_t total_elements = static_cast<size_t>(height) * width;
    // Ideal memory traffic: read input (4B) + write output (4B) = 8 bytes per pixel
    double total_bytes = 2.0 * total_elements * sizeof(float);
    // 25 FMAs per output element = 50 FLOPs per pixel
    double total_flops = 2.0 * KERNEL_SIZE * total_elements;

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " 🚀 Benchmarking: " << name << "\n";
    std::cout << "    Dimensions  : " << height << " x " << width << " (" 
              << total_elements / (1024 * 1024) << "M pixels, " 
              << (total_bytes / (1024.0 * 1024.0)) << " MB DRAM traffic)\n";
    std::cout << "--------------------------------------------------------\n";

    std::vector<float> h_mask(KERNEL_SIZE);
    init_gaussian_mask(h_mask, KERNEL_RADIUS, 1.2f);

    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_in, 1, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, total_elements * sizeof(float)));

    // Warm-up runs
    for (int i = 0; i < warmup_iters; ++i) {
        launch_fn(d_in, h_mask.data(), d_out, height, width);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Timed benchmark loop
    GpuTimer timer;
    timer.start();
    for (int i = 0; i < benchmark_iters; ++i) {
        launch_fn(d_in, h_mask.data(), d_out, height, width);
    }
    float total_ms = timer.stop();
    float avg_ms = total_ms / benchmark_iters;

    BenchmarkReport::printMetrics(avg_ms, total_bytes, total_flops);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main() {
    std::cout << "\n========================================================\n";
    std::cout << "    LeetGPU Problem 05: 2D Stencil & Convolution        \n";
    std::cout << "    Filter Size: 5x5 (Radius = 2, 25 Weights)          \n";
    std::cout << "    Platform   : NVIDIA A100-SXM4-64GB (sm_80)         \n";
    std::cout << "========================================================\n";

    // -------------------------------------------------------------------------
    // PART 1: Edge-Case Correctness Suite
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST SUITE] Running edge-case correctness tests...\n";

    struct TestCase {
        int height;
        int width;
        std::string desc;
    };

    const std::vector<TestCase> test_cases = {
        {1, 1, "Single element (1x1, smaller than filter)"},
        {5, 5, "Exact filter size (5x5)"},
        {16, 16, "Exact single tile (16x16)"},
        {17, 23, "Small non-multiple of tile size"},
        {32, 32, "Multiple tiles (32x32)"},
        {64, 128, "Rectangular (H < W)"},
        {128, 64, "Rectangular (H > W)"},
        {127, 513, "Odd rectangular dimensions"},
        {1024, 1024, "Medium square image (1024x1024)"}
    };

    std::vector<std::pair<std::string, void (*)(const float*, const float*, float*, int, int)>> kernels = {
        {"Milestone 1: Naive (Global Memory)", wrapper_naive}
    };

    if (is_constant_mask_implemented()) {
        kernels.push_back({"Milestone 2: Constant Memory Mask", launch_conv2d_constant_mask});
    }
    if (is_shared_tiled_implemented()) {
        kernels.push_back({"Milestone 3: Shared Memory Apron Tiling", launch_conv2d_shared_tiled});
    }
    if (is_readonly_cached_implemented()) {
        kernels.push_back({"Milestone 4: Read-Only Cache Streaming", launch_conv2d_readonly_cached});
    }

    bool all_passed = true;
    for (const auto& [name, fn] : kernels) {
        std::cout << "Testing: " << name << " ... ";
        bool kernel_passed = true;
        for (const auto& tc : test_cases) {
            if (!test_correctness(name, fn, tc.height, tc.width)) {
                std::cout << "\n   ❌ FAILED on test case: " << tc.desc 
                          << " (" << tc.height << "x" << tc.width << ")\n";
                kernel_passed = false;
                all_passed = false;
                break;
            }
        }
        if (kernel_passed) {
            std::cout << "PASSED! ✅\n";
        }
    }

    if (!all_passed) {
        std::cout << "\n❌ Some correctness tests failed! Please fix before benchmarking.\n";
        return 1;
    }

    std::cout << "\n🎉 ALL CORRECTNESS TESTS PASSED! Proceeding to performance benchmarks...\n";

    // -------------------------------------------------------------------------
    // PART 2: Roofline Performance Benchmark (8192 x 8192 = 64M floats = 512 MB traffic)
    // -------------------------------------------------------------------------
    int bench_h = 8192;
    int bench_w = 8192;

    for (const auto& [name, fn] : kernels) {
        benchmark_kernel(name, fn, bench_h, bench_w);
    }

    std::cout << "\n========================================================\n";
    if (is_constant_mask_implemented() && is_shared_tiled_implemented() && is_readonly_cached_implemented()) {
        std::cout << " ✅ Problem 05 Complete! Ready for Problem 06 (SGEMM). \n";
    } else {
        std::cout << " 💡 Next step: Open `kernel.cu` to implement milestones:\n";
        if (!is_constant_mask_implemented()) std::cout << "    • Milestone 2: Constant Memory Mask (__constant__)\n";
        if (!is_shared_tiled_implemented())  std::cout << "    • Milestone 3: Shared Memory Apron Tiling\n";
        if (!is_readonly_cached_implemented()) std::cout << "    • Milestone 4: Read-Only Cache Streaming\n";
    }
    std::cout << "========================================================\n";

    return 0;
}
