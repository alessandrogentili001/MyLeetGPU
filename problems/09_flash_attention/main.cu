#include "kernel.cuh"
#include "reference.hpp"
#include "cuda_utils.cuh"
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>

// Test correctness for a given sequence length and causal setting
bool test_attention_correctness(const std::string& name,
                                void (*launch_fn)(const float*, const float*, const float*, float*, int, int, float, bool),
                                int seq_len, int d, float scale, bool is_causal) {
    size_t total_elements = static_cast<size_t>(seq_len) * d;
    std::vector<float> h_Q(total_elements);
    std::vector<float> h_K(total_elements);
    std::vector<float> h_V(total_elements);
    std::vector<float> h_ref(total_elements, 0.0f);
    std::vector<float> h_gpu(total_elements, 0.0f);

    init_random_attention_matrix(h_Q, total_elements, -1.0f, 1.0f, 101);
    init_random_attention_matrix(h_K, total_elements, -1.0f, 1.0f, 202);
    init_random_attention_matrix(h_V, total_elements, -1.0f, 1.0f, 303);

    // CPU ground truth
    attention_cpu_reference(h_Q.data(), h_K.data(), h_V.data(), h_ref.data(),
                            seq_len, d, scale, is_causal);

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_Q, h_Q.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V.data(), total_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_O, 0, total_elements * sizeof(float)));

    launch_fn(d_Q, d_K, d_V, d_O, seq_len, d, scale, is_causal);
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_gpu.data(), d_O, total_elements * sizeof(float), cudaMemcpyDeviceToHost));

    std::string causal_str = is_causal ? "Causal" : "Non-Causal";
    std::cout << "  Testing " << name << " (" << causal_str << ", N=" << seq_len << ", d=" << d << ")...\n";

    // Attention involves many floating point additions in reduction tree; allow 1e-3 tolerance
    bool passed = BenchmarkReport::verifyCorrectness(h_ref.data(), h_gpu.data(), total_elements, 1e-3f, 1e-3f);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));

    return passed;
}

// Benchmark attention kernel
void benchmark_attention(const std::string& name,
                         void (*launch_fn)(const float*, const float*, const float*, float*, int, int, float, bool),
                         int seq_len, int d, float scale, bool is_causal,
                         int warmup_iters = 5, int benchmark_iters = 20) {
    size_t total_elements = static_cast<size_t>(seq_len) * d;

    // FLOPs: 2 * N^2 * d (Q @ K^T) + 2 * N^2 * d (P @ V) = 4 * N^2 * d
    double total_flops = 4.0 * static_cast<double>(seq_len) * seq_len * d;
    if (is_causal) {
        total_flops *= 0.5; // Causal attention computes half the attention matrix
    }

    // Minimum ideal DRAM traffic: Read Q, K, V once + Write O once
    double min_traffic_bytes = 4.0 * total_elements * sizeof(float);

    std::cout << "\n--------------------------------------------------------\n";
    std::cout << " 🚀 Benchmarking: " << name << " (N=" << seq_len << ", d=" << d
              << ", " << (is_causal ? "Causal" : "Non-Causal") << ")\n";
    std::cout << "--------------------------------------------------------\n";

    float *d_Q, *d_K, *d_V, *d_O;
    CUDA_CHECK(cudaMalloc(&d_Q, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_K, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_V, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_O, total_elements * sizeof(float)));

    CUDA_CHECK(cudaMemset(d_Q, 1, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_K, 1, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_V, 1, total_elements * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_O, 0, total_elements * sizeof(float)));

    for (int i = 0; i < warmup_iters; ++i) {
        launch_fn(d_Q, d_K, d_V, d_O, seq_len, d, scale, is_causal);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int i = 0; i < benchmark_iters; ++i) {
        launch_fn(d_Q, d_K, d_V, d_O, seq_len, d, scale, is_causal);
    }
    float avg_ms = timer.stop() / benchmark_iters;

    BenchmarkReport::printMetrics(avg_ms, min_traffic_bytes, total_flops);

    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_O));
}

int main() {
    std::cout << "\n========================================================\n";
    std::cout << "    LeetGPU Problem 09: FlashAttention Forward Pass     \n";
    std::cout << "    Platform: NVIDIA A100-SXM4-64GB (sm_80)             \n";
    std::cout << "========================================================\n\n";

    int d = HEAD_DIM; // 64
    float scale = 1.0f / std::sqrt(static_cast<float>(d));

    std::vector<int> test_lengths = {16, 32, 63, 128, 256};

    bool all_passed = true;

    std::cout << "[TEST SUITE] Running edge-case & causal correctness tests...\n\n";

    // Milestone 1
    if (is_attention_naive_implemented()) {
        std::cout << "========================================================\n";
        std::cout << "Testing Milestone 1: Standard Attention (Global Memory)\n";
        std::cout << "========================================================\n";
        for (int seq_len : test_lengths) {
            if (!test_attention_correctness("Milestone 1", launch_attention_naive, seq_len, d, scale, false)) {
                all_passed = false;
            }
            if (!test_attention_correctness("Milestone 1", launch_attention_naive, seq_len, d, scale, true)) {
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    // Milestone 2
    if (is_flash_attention_tiled_implemented()) {
        std::cout << "========================================================\n";
        std::cout << "Testing Milestone 2: Tiled FlashAttention (SRAM Tiling)\n";
        std::cout << "========================================================\n";
        for (int seq_len : test_lengths) {
            if (!test_attention_correctness("Milestone 2", launch_flash_attention_tiled, seq_len, d, scale, false)) {
                all_passed = false;
            }
            if (!test_attention_correctness("Milestone 2", launch_flash_attention_tiled, seq_len, d, scale, true)) {
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    // Milestone 3
    if (is_flash_attention_2_implemented()) {
        std::cout << "========================================================\n";
        std::cout << "Testing Milestone 3: FlashAttention-2\n";
        std::cout << "========================================================\n";
        for (int seq_len : test_lengths) {
            if (!test_attention_correctness("Milestone 3", launch_flash_attention_2, seq_len, d, scale, false)) {
                all_passed = false;
            }
            if (!test_attention_correctness("Milestone 3", launch_flash_attention_2, seq_len, d, scale, true)) {
                all_passed = false;
            }
        }
        std::cout << "\n";
    }

    if (!is_attention_naive_implemented() &&
        !is_flash_attention_tiled_implemented() &&
        !is_flash_attention_2_implemented()) {
        std::cout << "⚠️  No milestones implemented yet! Open `kernel.cu` to begin.\n";
        return 0;
    }

    if (!all_passed) {
        std::cout << "❌ Some correctness tests failed! Please fix before benchmarking.\n";
        return 1;
    }

    std::cout << "🎉 ALL CORRECTNESS TESTS PASSED! Proceeding to performance benchmarks...\n";

    // Benchmark on N=4096, d=64
    int bench_seq = 4096;

    if (is_attention_naive_implemented()) {
        benchmark_attention("Milestone 1: Standard Attention", launch_attention_naive, bench_seq, d, scale, true);
    }

    if (is_flash_attention_tiled_implemented()) {
        benchmark_attention("Milestone 2: Tiled FlashAttention", launch_flash_attention_tiled, bench_seq, d, scale, true);
    }

    if (is_flash_attention_2_implemented()) {
        benchmark_attention("Milestone 3: FlashAttention-2", launch_flash_attention_2, bench_seq, d, scale, true);
    }

    if (is_attention_naive_implemented() &&
        is_flash_attention_tiled_implemented() &&
        is_flash_attention_2_implemented()) {
        std::cout << "\n========================================================\n";
        std::cout << " 🏆 CONGRATULATIONS! LeetGPU Curriculum 100% Completed! \n";
        std::cout << "========================================================\n";
    }

    return 0;
}
