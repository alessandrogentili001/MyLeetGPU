# Problem 08: LayerNorm & RMSNorm

> **Difficulty**: Medium / Hard  
> **Type**: Memory-Bound / Deep Learning Kernel Engineering  
> **Key Concepts**: Layer Normalization, Root Mean Square Normalization (LLaMA style), Welford's Algorithm (One-Pass Variance), Warp Shuffle Reductions (`__shfl_down_sync`), Vectorized Memory Access (`float4`).

---

## 📖 Problem Description

Given a 2D activation matrix $X \in \mathbb{R}^{M \times N}$, a scale parameter vector $\gamma \in \mathbb{R}^N$ (`weight`), and a shift parameter vector $\beta \in \mathbb{R}^N$ (`bias`), compute row-wise normalization into $Y \in \mathbb{R}^{M \times N}$.

### 1. Layer Normalization (LayerNorm - Ba et al., 2016)
Used in classic Transformers (BERT, GPT-2, GPT-3, ViT):
For each row $i \in [0, M-1]$:
$$\mu_i = \frac{1}{N} \sum_{j=0}^{N-1} X[i, j]$$
$$\sigma_i^2 = \frac{1}{N} \sum_{j=0}^{N-1} (X[i, j] - \mu_i)^2$$
$$Y[i, j] = \frac{X[i, j] - \mu_i}{\sqrt{\sigma_i^2 + \epsilon}} \cdot \gamma[j] + \beta[j]$$

### 2. Root Mean Square Normalization (RMSNorm - Zhang & Sennrich, 2019)
Used in modern frontier LLMs (**LLaMA 1/2/3**, **Mistral**, **Gemma**, **Qwen**, **DeepSeek**):
RMSNorm hypothesizes that the re-centering invariance (subtracting $\mu$) is unnecessary. By enforcing only root-mean-square scale invariance, it saves 30–50% of the normalization overhead:
$$\text{RMS}_i = \sqrt{\frac{1}{N} \sum_{j=0}^{N-1} X[i, j]^2 + \epsilon}$$
$$Y[i, j] = \frac{X[i, j]}{\text{RMS}_i} \cdot \gamma[j]$$

---

## 🔬 Hardware & Roofline Analysis (NVIDIA A100-SXM4-64GB)

Let $X$ have shape $M \times N$:
- **Arithmetic Intensity**:
  - Ideal DRAM traffic: Read $X$ once ($4$ bytes) + Write $Y$ once ($4$ bytes) $\approx 8$ bytes/element (assuming $\gamma, \beta$ are cached in L1/L2).
  - Floating-point Operations (FLOPs):
    - LayerNorm: $\approx 7$ FLOPs per element ($\text{AI} = \frac{7}{8} \approx 0.875\text{ FLOPs/Byte}$).
    - RMSNorm: $\approx 3$ FLOPs per element ($\text{AI} = \frac{3}{8} \approx 0.375\text{ FLOPs/Byte}$).
- **Leonardo A100 SXM4 Ridge Point**:
  $$\text{Knee Point} = \frac{19.49\text{ TFLOPS}}{1555\text{ GB/s}} \approx 12.53\text{ FLOPs/Byte}$$

Because $\text{AI} \ll 12.53$, **LayerNorm and RMSNorm are strictly memory-bandwidth bound**! Reaching peak performance requires minimizing global memory roundtrips and maximizing DRAM transaction efficiency.

| Algorithm | DRAM Passes over $X$ | Global Memory Traffic |
| :--- | :---: | :---: |
| **Naive Multi-Pass LayerNorm** | 3 passes (mean, var, write) | $3 \times 4\text{B} + 4\text{B} = \mathbf{16\text{ bytes/elem}}$ |
| **One-Pass Welford LayerNorm** | 2 passes (Welford, write) | $2 \times 4\text{B} + 4\text{B} = \mathbf{12\text{ bytes/elem}}$ |
| **Fused Vectorized RMSNorm** | 2 passes ($x^2$, write) | $2 \times 4\text{B} + 4\text{B} = \mathbf{12\text{ bytes/elem}}$ (with 128-bit `float4`) |

---

## 🧠 The 3-Stage Optimization Progression

### Milestone 1: Multi-Pass LayerNorm (Shared Memory Reduction)
- Each thread block processes one row.
- **Pass 1**: Grid-stride loop computes thread-local sum of $X$, then reduces in `__shared__` memory to find row mean $\mu$.
- **Pass 2**: Grid-stride loop computes thread-local sum of $(X - \mu)^2$, then reduces in `__shared__` memory to find variance $\sigma^2$.
- **Pass 3**: Reads $X, \gamma, \beta$ and writes normalized output to $Y$.

### Milestone 2: One-Pass Welford LayerNorm (`__shfl_down_sync`)
- Traditional two-pass variance calculation can suffer from catastrophic cancellation ($E[X^2] - (E[X])^2$) or requires two separate passes over $X$.
- **Welford's Algorithm** computes the mean and variance simultaneously in a single pass:
  $$\text{Given } A=(n_A, \mu_A, M_{2,A}) \text{ and } B=(n_B, \mu_B, M_{2,B}):$$
  $$n = n_A + n_B$$
  $$\delta = \mu_B - \mu_A$$
  $$\mu = \mu_A + \delta \cdot \frac{n_B}{n}$$
  $$M_2 = M_{2,A} + M_{2,B} + \delta^2 \cdot \frac{n_A \cdot n_B}{n}$$
- Using **warp shuffle instructions** (`__shfl_down_sync`), threads merge Welford states directly in register space without shared memory synchronization overhead!

### Milestone 3: Fused Vectorized RMSNorm (`float4` + Warp Shuffle)
- RMSNorm simplifies the reduction to a single sum of squares $\sum X^2$.
- We employ **128-bit vectorized memory operations (`float4`)**:
  - Each thread reads and writes 4 `float` elements per instruction.
  - Generates full 128-byte DRAM transactions, saturating the A100 memory controllers.
  - Warp shuffle tree reduction performs lightning-fast row-wise aggregation.

---

## 🎯 Milestones & A100 Achieved Results

Benchmark size: $M=8192, N=4096$ on Leonardo Booster NVIDIA A100-SXM4-64GB:

| Milestone | Strategy | Latency | Achieved Bandwidth | Peak HBM2e % |
| :--- | :--- | :---: | :---: | :---: |
| **1. Multi-Pass LayerNorm** | Block Shared Memory | 0.314 ms | 855.6 GB/s | 55.0% |
| **2. Welford LayerNorm** | One-Pass Warp Shuffles | 0.303 ms | 885.9 GB/s | 57.0% |
| **3. Vectorized RMSNorm** | `float4` + Warp Shuffle | **0.262 ms** | **1,023.2 GB/s** | **65.8%** |

---

## 🚀 How to Run

```bash
# 1. Activate Leonardo environment
source .env

# 2. Build target
leetgpu-build

# 3. Run test suite & A100 roofline benchmark
leetgpu-run ./build/problems/08_layernorm_rmsnorm/layernorm_rmsnorm
```
