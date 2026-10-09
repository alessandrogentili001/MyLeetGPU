# Problem 09: FlashAttention Forward Pass

> **Difficulty**: Hard / Expert  
> **Type**: Compute & Memory Hierarchy Co-Design / Transformer Kernel Engineering  
> **Key Concepts**: Scaled Dot-Product Attention, SRAM Tiling, Online Softmax Rescaling, Causal Masking, Register Accumulation, FlashAttention-2.

---

## 📖 Problem Description

Given Query $Q \in \mathbb{R}^{N \times d}$, Key $K \in \mathbb{R}^{N \times d}$, and Value $V \in \mathbb{R}^{N \times d}$ matrices, compute the Scaled Dot-Product Attention output $O \in \mathbb{R}^{N \times d}$:

$$S = \frac{Q K^T}{\sqrt{d}} \in \mathbb{R}^{N \times N}$$
$$P = \text{softmax}(S) \in \mathbb{R}^{N \times N}$$
$$O = P V \in \mathbb{R}^{N \times d}$$

Where $N$ is the sequence length and $d$ is the head dimension (typically $d = 64$ or $128$).
When causal masking is enabled, $S_{i, j} = -\infty$ for $j > i$ (preventing tokens from attending to future tokens).

---

## ⚠️ The Memory Bottleneck of Standard Attention

In standard attention (Vaswani et al., 2017), the $N \times N$ matrix $S$ and attention probabilities $P$ are materialized directly in GPU High-Bandwidth Memory (DRAM):
- For sequence length $N = 4096$, storing $S$ and $P$ takes $2 \times 4096^2 \times 4\text{ B} = \mathbf{128\text{ MB}}$ per attention head!
- As $N \to 32\text{k}, 128\text{k}$, memory requirements scale quadratically as $\mathcal{O}(N^2)$, causing GPU Out-Of-Memory (OOM) errors and severe memory bandwidth throttling.

---

## 💡 The FlashAttention Breakthrough (Dao et al., 2022; 2023)

FlashAttention solves this by **never materializing the $N \times N$ attention matrix $S$ in DRAM**:
1. **Tiling in SRAM**: Divides $Q, K, V$ into tiles small enough to fit inside on-chip Shared Memory (SRAM, ~164 KB per SM on A100).
2. **Online Softmax**: Exploits the running-max and running-sum softmax property (from Problem 07) to update output accumulators incrementally without needing the entire row of $S$ at once.
3. **IO Complexity**: Reduces global memory access from $\mathcal{O}(N^2)$ down to $\mathbf{\mathcal{O}(N)}$!

```
 Standard Attention:              FlashAttention:
   Q, K ──► [ DRAM S (NxN) ]        Q, K, V ──► [ Fast On-Chip SRAM ]
                │                                       │
                ▼                                       ▼
            [ DRAM P (NxN) ]                   Compute tile & Rescale
                │                                       │
                ▼                                       ▼
            Output O (Nxd)                       Output O (Nxd)
   (Heavy O(N^2) DRAM Traffic)             (IO-Aware O(N) DRAM Traffic)
```

---

## 🔬 Hardware & Roofline Analysis (NVIDIA A100-SXM4-64GB)

For $N = 4096, d = 64$ with causal masking:
- **Total FLOPs**: $\approx 2 \cdot N^2 \cdot d = 2 \cdot 4096^2 \cdot 64 \approx \mathbf{2.15\text{ GFLOPs}}$ per head.
- **Ideal DRAM Traffic**:
  - Read $Q, K, V$: $3 \times N \times d \times 4\text{ B} \approx 3.14\text{ MB}$
  - Write $O$: $N \times d \times 4\text{ B} \approx 1.05\text{ MB}$
  - Total Traffic: $\approx \mathbf{4.19\text{ MB}}$
- **Arithmetic Intensity**:
  $$\text{AI} = \frac{2.15 \times 10^9 \text{ FLOPs}}{4.19 \times 10^6 \text{ Bytes}} \approx \mathbf{512\text{ FLOPs/Byte}}$$

Because $512 \gg 12.53$ (A100 ridge point), **FlashAttention is strongly compute-bound**! It transforms an otherwise memory-choked operation into high-throughput tensor compute!

---

## 🧠 The 3-Stage Optimization Progression

### Milestone 1: Standard Attention (Global Memory Baseline)
- Computes $S = \tau Q K^T$ row-by-row into global memory.
- Performs row-wise safe softmax to obtain $P$.
- Multiplies $P V$ to write $O$.

### Milestone 2: Tiled FlashAttention (SRAM Tiling + Online Softmax)
- Tiles $Q$ ($B_r \times d$) and tiles $K, V$ ($B_c \times d$) into `__shared__` memory.
- Uses online softmax: maintains running max $m$ and running sum $\ell$ per row.
- Incrementally updates $O$ using the recurrence:
  $$\tilde{m} = \max(m, m_{\text{tile}})$$
  $$O_{\text{new}} = O \cdot \exp(m - \tilde{m}) + P_{\text{tile}} V_{\text{tile}}$$
  $$\ell_{\text{new}} = \ell \cdot \exp(m - \tilde{m}) + \ell_{\text{tile}}$$

### Milestone 3: FlashAttention-2 (Outer-Loop over Q, Causal Masking, Register Accumulation)
- Inverts loop nesting: Outer loop tiles $Q$ (mapped to thread blocks), inner loop tiles $K, V$.
- Keeps output accumulators $O$ purely in thread registers throughout the entire sequence!
- Skips causal tiles where $j_{\text{start}} > i_{\text{end}}$.
- Divides by $\ell$ only once at the very end.

---

## 🎯 Milestones & A100 Achieved Results

Benchmark size: $N=4096, d=64$ (Causal Attention) on Leonardo Booster NVIDIA A100-SXM4-64GB:

| Milestone | Strategy | Latency | Compute Throughput | Peak FP32 % |
| :--- | :--- | :---: | :---: | :---: |
| **1. Standard Attention** | Global Memory (DRAM) | **42.48 ms** | 0.051 TFLOPS | 0.26% |
| **2. Tiled FlashAttention** | SRAM Tiling + Online Softmax | **12.10 ms** | 0.177 TFLOPS | 0.91% |
| **3. FlashAttention-2** | Register Rescaling + Causal | **4.51 ms** | 0.476 TFLOPS | 2.44% |

---

## 🚀 How to Run

```bash
# 1. Activate Leonardo environment
source .env

# 2. Build target
leetgpu-build

# 3. Run test suite & A100 benchmark
leetgpu-run ./build/problems/09_flash_attention/flash_attention
```
