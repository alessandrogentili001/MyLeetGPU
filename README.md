# My LeetGPU 🚀

> *"I am not willing to pay the bill to learn GPU kernels, therefore I created this repo."*  
> An open-source, hands-on GPU kernel learning curriculum powered by **CINECA Leonardo Booster** (**NVIDIA A100-SXM4-64GB**, Ampere `sm_80`).

---

## ⚡ Quick Start

### 1. Activate Environment
Every session on Leonardo, source `.env` to load the exact compiler stack and SLURM profile:
```bash
source .env
# or: source env.sh
```
This loads `gcc/12.2.0`, `cuda/12.2`, `cmake/3.27.9`, sets target architecture to `sm_80`, and configures the `PHD_gentili` allocation.

### 2. Build Kernels
```bash
leetgpu-build
```

### 3. Run on A100 GPU
Execute any problem through the automated test judge in seconds:
```bash
# Problem 01: Vector Addition
leetgpu-run ./build/problems/01_vector_add/vector_add

# Problem 02: Matrix Transpose
leetgpu-run ./build/problems/02_matrix_transpose/matrix_transpose

# Device inspector
leetgpu-run ./build/device_info
```

### 4. Interactive GPU Shell (Optional)
To attach an interactive bash shell with an A100 GPU for 30 minutes:
```bash
leetgpu-alloc
```

---

## 🏗️ How Each Problem Works (The LeetGPU Pattern)

Every problem in `problems/` is 100% self-contained and follows the LeetCode / LeetGPU paradigm:

```text
problems/XX_problem_name/
├── README.md         <- Problem statement, arithmetic intensity & roofline math
├── kernel.cuh        <- Function prototypes & launch signatures
├── kernel.cu         <- 🎯 YOUR PLAYGROUND (only your CUDA kernels and launch functions)
├── reference.hpp     <- 🧠 EXPECTED OUTPUT (CPU ground truth generator)
├── main.cu           <- 🧪 THE JUDGE (runs edge cases + A100 roofline benchmarks)
└── CMakeLists.txt    <- Builds the problem target
```

### The Separation of Concerns
1. **`kernel.cu` (Your Workspace)**: You write **only** the GPU kernels and grid/block launch logic. No test boilerplate.
2. **`reference.hpp` (Ground Truth)**: Computes deterministic reference outputs on CPU.
3. **`main.cu` (The Judge)**:
   - **Phase 1: Edge-Case Test Suite**: Validates boundary conditions ($N=1$, odd lengths, $N=1023$, non-multiples of block size) against ground truth.
   - **Phase 2: Roofline Performance Benchmark**: Measures execution time using `cudaEvent_t` on large data sizes (blowing past the 32 MB L2 cache) and calculates **achieved GB/s** or **TFLOPS** against A100 theoretical limits.

---

## 🎯 Hardware Specifications (Leonardo Booster)

Our code runs on Leonardo's custom SXM4 supercomputing partition:

| Metric | Specification |
| :--- | :--- |
| **GPU Model** | NVIDIA A100-SXM4-64GB |
| **Architecture** | Ampere (`sm_80`) |
| **Streaming Multiprocessors (SMs)** | **124 SMs** *(custom Leonardo SKU, vs 108 on standard A100)* |
| **Peak HBM2e Memory Bandwidth** | **1,555 GB/s** (~1.55 TB/s) |
| **Peak FP32 Compute** | **19.49 TFLOPS** |
| **Peak Tensor Core (FP16/BF16)** | **312 TFLOPS** |
| **L2 Cache Size** | 32 MB |
| **Max Threads per Block** | **1,024** |
| **Max Threads per SM** | **2,048** (up to 2 blocks of 1024, or 4 of 512) |
| **Shared Memory per SM** | Up to 164 KB |

---

## 🗺️ Problem Curriculum & Progress

| # | Problem | Focus / Optimization Concepts | Status | Best A100 Performance |
| :-: | :--- | :--- | :--- | :--- |
| **01** | [**Vector Addition**](problems/01_vector_add/) | 1D indexing, Grid-stride loop, `float4` vectorized loads | ✅ **Completed** | **1,395 GB/s** (89.7% HBM2e Peak) |
| **02** | [**Matrix Transpose**](problems/02_matrix_transpose/) | Coalesced writes, Shared Memory tiling, Bank Conflict padding | ✅ **Completed** | **1,187 GB/s** (76.3% HBM2e Peak) |
| **03** | [**Parallel Reduction**](problems/03_parallel_reduction/) | Warp divergence, Interleaved vs Sequential addressing, Warp shuffles (`__shfl_down_sync`) | ✅ **Completed** | **796 GB/s** (51.2% HBM2e Peak) |
| **04** | [**Prefix Sum (Scan)**](problems/04_prefix_sum/) | Work-efficiency, Hillis-Steele vs Blelloch, Bank conflicts | 📝 Prepared | — |
| **05** | **1D/2D Stencil & Conv** | Constant memory, Halo cell caching, Shared memory apron | ⏳ Upcoming | — |
| **06** | **SGEMM (Matrix Mult)** | Naive $\to$ Shared Memory Tiling $\to$ 2D Register Tiling $\to$ Tensor Cores (`wmma`) | ⏳ Upcoming | — |
| **07** | [**Softmax**](problems/07_softmax/) | Two-pass vs Online Safe Softmax (FlashSoftmax), Warp reductions | 📝 Prepared | — |
| **08** | **LayerNorm / RMSNorm** | Welford's algorithm, Fused elementwise operations | ⏳ Upcoming | — |
| **09** | **FlashAttention-2** | Tiling Q, K, V in SRAM, causal masking, online softmax rescaling | ⏳ Upcoming | — |