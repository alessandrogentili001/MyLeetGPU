# Problem 06: SGEMM (Single Precision General Matrix Multiplication)

> **Difficulty**: Hard  
> **Type**: Compute-Bound / Memory Hierarchy  
> **Key Concepts**: Shared Memory Tiling, Cache Blocking, 2D Register Tiling, Thread Coarsening, Arithmetic Intensity, cuBLAS.

---

## 📖 Problem Description

Given two matrices $A \in \mathbb{R}^{M \times K}$ and $B \in \mathbb{R}^{K \times N}$, compute their product $C \in \mathbb{R}^{M \times N}$:
$$C = \alpha AB + \beta C$$
For this exercise, we assume $\alpha = 1$ and $\beta = 0$, so $C = AB$.

All matrices are stored in **row-major** order.

SGEMM is the cornerstone of Deep Learning and High-Performance Computing. 

---

## 🔬 Hardware & Roofline Analysis (NVIDIA A100-SXM4-64GB)

For a matrix multiplication of size $M \times N \times K$:
- **Total FLOPs**: $2 \cdot M \cdot N \cdot K$
- **Total Bytes** (ideal, one-time read/write): $(M \cdot K + K \cdot N + M \cdot N) \times 4$ bytes.
- **Arithmetic Intensity (Ideal)**: $\approx \frac{2 \cdot K}{3 \cdot 4} \text{ FLOPs/Byte} = \frac{K}{6}$.

As $K$ grows, the Arithmetic Intensity increases. This means SGEMM can shift from being memory-bound to **compute-bound**.

### Leonardo A100 Roofline Limits:
- **Peak FP32 Compute**: $19.49 \text{ TFLOPS}$
- **Peak HBM2e Bandwidth**: $1,555 \text{ GB/s}$
- **Ridge Point (Knee)**: $12.53 \text{ FLOPs/Byte}$

If $K / 6 \ge 12.53 \implies K \ge 75$, the problem is theoretically compute-bound! However, reaching the theoretical $19.5 \text{ TFLOPS}$ requires meticulous utilization of the memory hierarchy to prevent the GPU from waiting on DRAM.

---

## 🧠 The 4-Stage Optimization Progression

### Milestone 1: Naive (Global Memory)
- Each thread computes one element of $C$.
- A thread loops over the $K$ dimension, reading elements from $A$ and $B$ directly from global memory.
- **Problem**: Terrible memory reuse. Matrix $A$ is loaded $N$ times, and Matrix $B$ is loaded $M$ times from DRAM.

### Milestone 2: Shared Memory Tiling (Cache Blocking)
- Instead of reading elements one-by-one from global memory, threads cooperatively load a $32 \times 32$ block of $A$ and a $32 \times 32$ block of $B$ into `__shared__` memory.
- The block computes a $32 \times 32$ tile of $C$ by iterating over the $K$ dimension in steps of $32$.
- **Result**: Global memory traffic is reduced by a factor of 32!

### Milestone 3: 2D Register Tiling (Thread Coarsening)
- To push performance to the limit, we must increase the arithmetic intensity per thread.
- Instead of each thread computing $1$ element of $C$, each thread computing an $8 \times 8$ tile of $C$ using registers.
- Registers are the fastest memory on the GPU. By keeping the $C$ accumulators and chunks of $A$ and $B$ in registers, we maximize the FLOP/s executed per instruction.

### Milestone 4: cuBLAS Baseline
- A provided call to NVIDIA's cuBLAS library to see the true ceiling of the A100.

---

## 🎯 Milestones & A100 Achieved Results

Benchmark size: $M=4096, N=4096, K=4096$ on Leonardo Booster NVIDIA A100-SXM4-64GB:

| Milestone | Strategy | Latency | Achieved Bandwidth | Compute Throughput | Peak FP32 % |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Naive** | Direct DRAM | ⏳ Pending | — | — | — |
| **2. Shared Tiled** | 32x32 `__shared__` | ⏳ Pending | — | — | — |
| **3. 2D Reg Tiled** | 8x8 registers/thread | ⏳ Pending | — | — | — |
| **4. cuBLAS** | Optimized Baseline | ⏳ Pending | — | — | — |


---

## 🚀 How to Run

```bash
# 1. Activate Leonardo environment
source .env

# 2. Build target
leetgpu-build

# 3. Run test suite & A100 roofline benchmark
leetgpu-run ./build/problems/06_sgemm/sgemm
```
