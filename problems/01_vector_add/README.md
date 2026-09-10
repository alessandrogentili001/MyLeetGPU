# Problem 01: Vector Addition ($C = A + B$)

> **Difficulty**: Warmup / Easy  
> **Type**: Memory-Bound  
> **Key Concepts**: 1D Thread Indexing, Grid-Stride Loops, Vectorized Loads (`float4`), Memory Bandwidth Roofline.

---

## 📖 Problem Description

Given two 1D vectors $A$ and $B$ of length $N$ containing single-precision floating point numbers (`float`), compute their element-wise sum and store the result in vector $C$:

$$C[i] = A[i] + B[i], \quad \forall i \in [0, N-1]$$

---

## 🔬 Hardware & Roofline Analysis (NVIDIA A100-SXM4-64GB)

Vector addition performs **1 floating point operation (FLOP)** for every **3 memory accesses** (reading $A[i]$ (4 bytes), reading $B[i]$ (4 bytes), writing $C[i]$ (4 bytes)).

- **Arithmetic Intensity**:
  $$\text{Arithmetic Intensity} = \frac{1 \text{ FLOP}}{12 \text{ bytes}} \approx 0.083 \text{ FLOPs/Byte}$$
- **A100 Peak Memory Bandwidth**: $\sim 1,555 \text{ GB/s}$ ($1.55 \text{ TB/s}$)
- **A100 FP32 Peak Compute**: $19.49 \text{ TFLOPS}$
- **Roofline Knee Point**:
  $$\text{Machine Balance} = \frac{19,490 \text{ GFLOPS}}{1,555 \text{ GB/s}} \approx 12.5 \text{ FLOPs/Byte}$$

Because $0.083 \ll 12.5$, **Vector Addition is strictly memory-bandwidth bound**. Compute speed does not matter here; your only goal is to maximize the throughput of the A100 HBM2e memory bus!

For a benchmark size of $N = 67,108,864$ elements ($64\text{M}$ floats, $256\text{ MB}$ per vector):
- Total bytes transferred: $3 \times 64\text{M} \times 4\text{ bytes} = 768\text{ MB} = 0.805 \times 10^9\text{ bytes}$.
- Theoretical minimum latency on A100:
  $$t_{\min} = \frac{0.768 \text{ GB}}{1,555 \text{ GB/s}} \approx 0.49 \text{ ms}$$

---

## 🎯 Milestones

### Milestone 1: Naive (1 Thread per Element)
- Calculate global thread ID: `int idx = blockIdx.x * blockDim.x + threadIdx.x;`
- Boundary check: `if (idx < n)`
- Experiment with block sizes: 64, 128, 256, 512, 1024.
- Expected Bandwidth: $\approx 800 - 1,100\text{ GB/s}$.

### Milestone 2: Grid-Stride Loop
- Instead of mapping 1 thread to 1 element, use a grid-stride loop:
  ```cuda
  for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += gridDim.x * blockDim.x) {
      c[idx] = a[idx] + b[idx];
  }
  ```
- **Why?** Decouples the grid size from the vector size $N$. You can launch a fixed number of blocks (e.g., $32 \times \text{SMs}$) and let threads work through the array in coalesced strides.
- Expected Bandwidth: $\approx 1,000 - 1,250\text{ GB/s}$.

### Milestone 3: Vectorized Memory Access (`float4`)
- Load and store 128 bits (4 floats) at a time using CUDA's built-in `float4` vector type.
- Generates `LDG.E.128` and `STG.E.128` PTX instructions.
- Increases Instruction-Level Parallelism (ILP) and saturates memory controllers with fewer total instructions.
- Expected Bandwidth: **$\approx 1,300 - 1,500+\text{ GB/s}$ (Near Peak A100 Roofline!)**.

---

## 🚀 How to Run

```bash
# 1. Source environment
source .env

# 2. Build
leetgpu-build

# 3. Run test runner & benchmark on A100 GPU
leetgpu-run ./build/problems/01_vector_add/vector_add
```
