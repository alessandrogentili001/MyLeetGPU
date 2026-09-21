# Problem 07: Softmax (Safe Softmax & FlashSoftmax)

> **Difficulty**: Medium / Hard  
> **Type**: Memory-Bound / Deep Learning Kernel Engineering  
> **Key Concepts**: Numerical Stability (Safe Softmax), Multi-Pass vs Single-Pass, Warp Reductions (`__shfl_down_sync`), Online Softmax (FlashSoftmax trick), Vectorized Loads (`float4`).

---

## 📖 Problem Description

Given a 2D matrix $X \in \mathbb{R}^{M \times N}$ containing single-precision floating point numbers (`float`), compute the row-wise **Softmax** and store the result in matrix $Y \in \mathbb{R}^{M \times N}$:

For each row $i \in [0, M-1]$ and column $j \in [0, N-1]$:

$$Y[i, j] = \frac{\exp(X[i, j] - m_i)}{\sum_{k=0}^{N-1} \exp(X[i, k] - m_i)}$$

where $m_i = \max_{k=0}^{N-1} X[i, k]$ is the row maximum.

Softmax is the core activation function in transformer attention mechanisms (Self-Attention, Cross-Attention, FlashAttention), classifier heads, and mixture-of-experts (MoE) routing. Because every output element $Y[i, j]$ depends on all elements in row $i$, an efficient GPU implementation requires coordinating reductions and memory traffic across parallel threads.

---

## ⚠️ Why "Safe" Softmax? (The Floating-Point Trap)

The textbook definition of Softmax is:

$$Y[i, j] = \frac{\exp(X[i, j])}{\sum_k \exp(X[i, k])}$$

In IEEE 754 single-precision (FP32), the maximum representable finite number is $\approx 3.4 \times 10^{38}$. The natural exponential function $\exp(x)$ overflows when:

$$x > \ln(3.4028 \times 10^{38}) \approx 88.7228$$

If an activation element $X[i, j] = 90.0$, evaluating $\exp(90.0)$ produces `+inf`. In the sum, $\sum \exp = \text{inf}$, and evaluating $\frac{\infty}{\infty}$ yields **`NaN`**, poisoning gradients and crashing training runs!

By subtracting the row maximum $m_i = \max_k X[i, k]$:
1. The exponent $(X[i, j] - m_i) \le 0$ for all $j$.
2. Therefore, $\exp(X[i, j] - m_i) \in (0, 1.0]$.
3. **Overflow is mathematically impossible!**

---

## 🔬 Hardware & Roofline Analysis (NVIDIA A100-SXM4-64GB)

Let $X$ be an $M \times N$ matrix.
- **Floating-point Operations (FLOPs)** per element:
  - 1 maximum comparison ($x > m$)
  - 1 subtraction ($x - m$)
  - 1 exponential ($\exp$)
  - 1 sum addition ($s + \exp$)
  - 1 normalizer multiply/division ($y = \exp \cdot \frac{1}{s}$)
  - Total $\approx 5$ FLOPs per element.
- **Ideal DRAM Traffic**:
  - Read input $X$: $4$ bytes per element
  - Write output $Y$: $4$ bytes per element
  - Total $\approx 8$ bytes per element.
- **Arithmetic Intensity**:
  $$\text{AI} = \frac{5 \text{ FLOPs}}{8 \text{ Bytes}} = 0.625 \text{ FLOPs/Byte}$$

On the NVIDIA A100 (Peak Compute: $19.49 \text{ TFLOPS}$, Peak HBM2e: $1,555 \text{ GB/s}$), the ridge point is:
$$\text{Knee Point} = \frac{19.49 \times 10^{12} \text{ FLOPs/s}}{1555 \times 10^9 \text{ Bytes/s}} \approx 12.53 \text{ FLOPs/Byte}$$

Because $0.625 \ll 12.53$, **Softmax is strictly memory-bandwidth bound**!

### Memory Traffic Comparison
| Implementation | Global Reads | Global Writes | DRAM Bytes / Element |
| :--- | :---: | :---: | :---: |
| **Traditional Multi-Pass** | 2 reads (Pass 1 max, Pass 2 sum) | 1 write | **12 bytes** |
| **Online FlashSoftmax** | 1 read (fused max + sum) | 1 write | **8 bytes** (33% less DRAM traffic!) |

---

## 🧠 The 3-Stage Optimization Progression

### Milestone 1: Two-Pass Safe Softmax (Block Shared Memory)
Each thread block handles one row of the matrix.
1. **Find Max**: Threads loop over row columns, compute thread-local max, and perform a shared-memory reduction tree to find $m_i$.
2. **Compute Exp-Sum**: Threads loop over columns again, compute $\exp(x - m_i)$, and perform a shared-memory reduction tree to find $d_i = \sum \exp$.
3. **Normalize**: Threads compute $Y[i, j] = \exp(X[i, j] - m_i) / d_i$ and write back to global memory.

### Milestone 2: Warp-Accelerated Safe Softmax (`__shfl_down_sync`)
Shared-memory tree reductions require explicit `__syncthreads()` at every level and suffer from bank conflicts if not padded.
Milestone 2 replaces shared memory tree reductions with **warp shuffle primitives**:
1. Within each 32-thread warp, reduction uses `__shfl_down_sync` in register space.
2. Only the warp leaders (`lane == 0`) write their partial results into a tiny shared memory buffer (size $\le 32$).
3. Warp 0 performs the final reduction over warp leaders.
4. Latency drops significantly and memory bank conflicts are eliminated.

### Milestone 3: Online Safe Softmax (FlashSoftmax / Single-Pass)
In modern LLMs and FlashAttention (Dao et al., 2022; Milakov & Gimelshein, 2018), we cannot afford multiple passes over activations.
Online Softmax computes running max and running sum in a single pass using the recurrence:

Given previous state $(m_A, d_A)$ and new chunk $(m_B, d_B)$:
$$m_{\text{new}} = \max(m_A, m_B)$$
$$d_{\text{new}} = d_A \cdot \exp(m_A - m_{\text{new}}) + d_B \cdot \exp(m_B - m_{\text{new}})$$

Combined with `float4` (128-bit) vectorized loads, this minimizes HBM roundtrips and maximizes DRAM throughput!

---

## 🎯 Milestones & A100 Achieved Results

Benchmark size: $M = 8,192 \times N = 4,096$ matrix ($32\text{M}$ floats, $256.0\text{ MB}$ DRAM traffic) on Leonardo Booster NVIDIA A100-SXM4-64GB:

| Milestone | Architecture Strategy | Reduction Primitives | Vectorization | Achieved Bandwidth | Latency | Peak HBM2e % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Block Two-Pass** | Block Shared Memory | `s_max`, `s_sum` Tree Reductions | Scalar (`float`) | **1,094.32 GB/s** | 0.245 ms | 70.4% |
| **2. Warp-Accelerated** | Intra-warp register shuffles | `__shfl_down_sync` + Warp 0 | Scalar (`float`) | **1,115.03 GB/s** | **0.241 ms** | **71.7%** 🚀 |
| **3. Online FlashSoftmax** | Single-pass running stats | Online recurrence + Warp shuffles | Vectorized (`float4`) | **1,088.64 GB/s** | 0.247 ms | 70.0% |

### Key Takeaways from the Numbers:
1. **Warp Shuffles Cut Latency and Shared Memory Contention**:
   Moving from block shared memory tree reductions (Milestone 1) to intra-warp register shuffle instructions (`__shfl_down_sync` in Milestone 2) reduces block synchronizations (`__syncthreads()`) and completely eliminates shared memory bank conflicts, achieving the fastest execution time (**240.7 µs, 1,115 GB/s**).
2. **L2 Cache Dynamics in Multi-Pass vs. Single-Pass**:
   For standalone row-wise softmax with $N=4,096$, each row is only $16\text{ KB}$, easily fitting in the A100's $32\text{ MB}$ L2 cache and SM L1/SRAM. Passes 2 and 3 hit high-speed L2 cache (>3 TB/s) rather than re-reading from DRAM.
3. **Chunked `float4` Vectorization Powers Online Softmax**:
   Without vectorization, online softmax is bottlenecked by Special Function Unit (SFU) transcendental operations (`__expf`). Finding the `local_max` of 4 elements via fast 1-cycle ALU comparisons and issuing 128-bit `float4` memory instructions boosts Online FlashSoftmax throughput to **1,088.64 GB/s**.

---

## 🚀 How to Run

```bash
# 1. Activate environment
source .env

# 2. Build target
leetgpu-build

# 3. Run correctness test suite & roofline benchmark
leetgpu-run ./build/problems/07_softmax/softmax
```
