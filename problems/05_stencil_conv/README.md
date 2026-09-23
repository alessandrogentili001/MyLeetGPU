# Problem 05: 2D Stencil & Convolution

> **Difficulty**: Medium  
> **Type**: Memory-Bound / Spatial Locality & Stencil Engineering  
> **Key Concepts**: 2D Stencils, Constant Memory broadcast (`__constant__`), Halo / Ghost Cells, Shared Memory Apron Tiling, Cooperative Loading, Ampere Read-Only Cache (`ld.global.nc`).

---

## 📖 Problem Description

Given a 2D input matrix (or image) $X \in \mathbb{R}^{H \times W}$ and a square 2D filter mask $M \in \mathbb{R}^{K \times K}$ with odd diameter $K$ and radius $R = \lfloor K / 2 \rfloor$, compute the 2D convolution $Y \in \mathbb{R}^{H \times W}$:

For each row $r \in [0, H-1]$ and column $c \in [0, W-1]$:

$$Y[r, c] = \sum_{i=-R}^{R} \sum_{j=-R}^{R} X[r+i, c+j] \cdot M[R+i, R+j]$$

### Boundary Condition (Zero-Padding)
When a filter tap reaches outside the matrix boundaries ($r+i < 0$, $r+i \ge H$, $c+j < 0$, or $c+j \ge W$), the boundary value is defined as zero:
$$X[r+i, c+j] = 0.0f$$

In this problem, we use a standard **$5 \times 5$ filter** ($K=5$, $R=2$, $25$ weights).

2D Stencils and Convolutions are foundational throughout scientific computing and machine learning:
- **Computer Vision & CNNs**: Convolutional layers, feature extraction, Gaussian blurring, Sobel edge detectors, Laplacian sharpening.
- **Partial Differential Equations (PDEs)**: Finite difference methods for heat diffusion, wave propagation, Navier-Stokes fluid dynamics.

---

## ⚠️ The Memory Wall: Why Naive Stencils Hurt Performance

In a $5 \times 5$ stencil, calculating the value of a single output pixel $Y[r, c]$ requires reading $25$ neighborhood pixels from $X$:

```text
       c-2   c-1    c    c+1   c+2
r-2  [ ( )   ( )   ( )   ( )   ( ) ]
r-1  [ ( )   ( )   ( )   ( )   ( ) ]
 r   [ ( )   ( )  Y[r,c] ( )   ( ) ]
r+1  [ ( )   ( )   ( )   ( )   ( ) ]
r+2  [ ( )   ( )   ( )   ( )   ( ) ]
```

Notice that adjacent output pixels $Y[r, c]$ and $Y[r, c+1]$ share **20 out of 25 input pixels**!
- If every thread independently reads its $5 \times 5$ window from global DRAM, every single input element is fetched from global memory up to **25 separate times**.
- This multiplies global DRAM traffic by $25\times$, causing severe bus saturation and poor utilization of SM compute cores.

---

## 🔬 Hardware & Roofline Analysis (NVIDIA A100-SXM4-64GB)

Let $H \times W$ be the image dimensions and $K \times K$ be the filter size ($K = 5$, $25$ taps):

- **Floating-point Operations (FLOPs)** per pixel:
  - 25 multiplications + 25 additions = 25 FMAs = **50 FLOPs** per pixel.
- **Ideal DRAM Traffic** (with perfect on-chip caching):
  - Read input $X$: $4$ bytes per pixel (each pixel loaded exactly once into cache/SRAM).
  - Write output $Y$: $4$ bytes per pixel.
  - Read filter mask $M$: $25 \times 4 = 100$ bytes total (negligible; cached in constant cache).
  - Ideal DRAM Traffic: **8 bytes per pixel**.
- **Ideal Arithmetic Intensity (AI)**:
  $$\text{AI}_{\text{ideal}} = \frac{50 \text{ FLOPs}}{8 \text{ Bytes}} = 6.25 \text{ FLOPs/Byte}$$
- **Naive DRAM Traffic** (no caching, 25 global reads per pixel):
  - Read input $X$: $25 \times 4 = 100$ bytes.
  - Write output $Y$: $4$ bytes.
  - Naive DRAM Traffic: **104 bytes per pixel**.
  $$\text{AI}_{\text{naive}} = \frac{50 \text{ FLOPs}}{104 \text{ Bytes}} \approx 0.48 \text{ FLOPs/Byte}$$

### Leonardo A100 Roofline Limits:
On Leonardo Booster's NVIDIA A100-SXM4-64GB:
- **Peak FP32 Compute**: $19.49 \text{ TFLOPS}$
- **Peak HBM2e Bandwidth**: $1,555 \text{ GB/s}$
- **Ridge Point (Knee)**:
  $$\text{Knee Point} = \frac{19.49 \times 10^{12} \text{ FLOPs/s}}{1555 \times 10^9 \text{ Bytes/s}} \approx 12.53 \text{ FLOPs/Byte}$$

Since $\text{AI}_{\text{ideal}} = 6.25 < 12.53$, **2D Stencil & Convolution is strictly memory-bandwidth bound**.
The maximum achievable compute performance is capped by memory bandwidth:
$$\text{Max Theoretical Compute} = 6.25 \text{ FLOPs/Byte} \times 1555 \text{ GB/s} \approx 9.72 \text{ TFLOPS}$$

Under naive global memory access ($\text{AI} = 0.48$), throughput drops by over an order of magnitude!

---

## 🧠 The 4-Stage Optimization Progression

```text
[Milestone 1: Naive Global]
         │ (Mask passed via global pointer, redundant DRAM reads)
         ▼
[Milestone 2: Constant Memory]
         │ (Mask stored in __constant__, 1-cycle warp broadcast)
         ▼
[Milestone 3: Shared Memory Apron Tiling]
         │ (Cooperative halo loading, zero DRAM redundancy in compute loop)
         ▼
[Milestone 4: Read-Only Streaming Cache]
           (Ampere unified L1/L2 cache via const __restrict__)
```

### Milestone 1: Naive 2D Convolution (Global Memory)
- Threads independently read $X[r+i, c+j]$ and $M[i, j]$ from global memory pointers `d_in` and `d_mask`.
- Boundary conditions are evaluated inside the innermost loops.
- Serves as the un-optimized baseline.

### Milestone 2: Constant Memory Filter Mask (`__constant__`)
- Filter weights (100 bytes) are placed into CUDA's `__constant__` memory using `cudaMemcpyToSymbol`.
- Constant memory features a dedicated 64 KB cache per SM.
- **Warp Broadcast**: When all 32 threads in a warp access the same filter tap `c_mask[k]` simultaneously, the hardware serves all 32 threads in a single clock cycle with zero bank conflicts, offloading L1/L2 data cache bandwidth!

### Milestone 3: Shared Memory Apron Tiling (Cooperative Halo Loading)
- Output tile size: $16 \times 16$ pixels (256 threads per block).
- With filter radius $R = 2$, computing a $16 \times 16$ output tile requires an input tile of size $(16 + 2 \times 2) \times (16 + 2 \times 2) = 20 \times 20 = 400$ elements.
- **Cooperative Apron Loading**:
  Instead of complicated nested branching for corners and halos, the 256 threads cooperatively load the 400 elements into `__shared__ float s_in[20][20]` in 2 clean strided iterations (`stride = 256`):
  ```cuda
  for (int i = tid; i < 400; i += 256) {
      int s_r = i / 20;
      int s_c = i % 20;
      int g_r = top_left_r + s_r;
      int g_c = top_left_c + s_c;
      s_in[s_r][s_c] = (in_bounds) ? in[g_r * width + g_c] : 0.0f;
  }
  __syncthreads();
  ```
- **Zero-Branch Compute Loop**: Once the apron is staged in shared memory with halo zero-padding already applied, the convolution inner loop runs directly out of `s_in` and `c_mask` with **zero global memory traffic** and **zero boundary checks**!

### Milestone 4: Modern Ampere Read-Only Cache Streaming (`const __restrict__`)
- Ampere GPUs combine L1 data cache and shared memory into a single high-bandwidth 192 KB SRAM structure.
- By tagging pointers `const float* __restrict__ in`, the compiler emits non-coherent cache read instructions (`ld.global.nc`), streaming spatial neighborhoods through the unified L1 cache.
- Explores how modern GPU hardware cache lines compare against manual software-managed shared memory staging.


---

## 🎯 Milestones & A100 Achieved Results

Benchmark size: $H = 8,192 \times W = 8,192$ matrix ($64\text{M}$ pixels, $512.0\text{ MB}$ DRAM traffic, $3.355\text{ GFLOPs}$) on Leonardo Booster NVIDIA A100-SXM4-64GB:

| Milestone | Memory Strategy | Mask Storage | Latency | Achieved Bandwidth | Compute Throughput | Peak HBM2e % |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **1. Naive Global** (Baseline) | Direct DRAM reads | Global pointer | 1.193 ms | 450.0 GB/s | 2.813 TFLOPS | 28.9% |
| **2. Constant Mask** | Direct DRAM reads | `__constant__` cache | 0.890 ms | 603.2 GB/s | 3.770 TFLOPS | 38.8% |
| **3. Shared Apron** | Cooperative `__shared__` tile | `__constant__` cache | 1.097 ms | 489.2 GB/s | 3.058 TFLOPS | 31.5% |
| **4. Read-Only Streaming** | `const __restrict__` (L1 cache) | `__constant__` cache | 0.804 ms | 667.8 GB/s | 4.174 TFLOPS | 42.9% |

### Key Architectural Takeaways:
1. **Constant Memory Warp Broadcast Cuts Latency by 25%**:
   Moving the 25 filter weights from global pointers to `__constant__` memory reduces latency from **1.193 ms down to 0.890 ms** (+34% speedup). Because all 32 threads in each warp access the exact same filter weight simultaneously, the constant cache serves all 32 threads in a single clock cycle broadcast, completely freeing L1/L2 and register ports.
2. **Modern Ampere Unified L1 Cache vs Software Shared Memory**:
   On Ampere `sm_80`, the 192 KB unified L1 data cache is wide enough to stream spatial 2D windows without the synchronization overhead (`__syncthreads()`) of manual shared memory staging. However, manual shared memory staging (Milestone 3) remains the essential technique for older architectures, multi-pass stencils, or custom boundary handling.

---

## 🚀 How to Run

```bash
# 1. Activate Leonardo environment
source .env

# 2. Build target
leetgpu-build

# 3. Run test suite & A100 roofline benchmark
leetgpu-run ./build/problems/05_stencil_conv/stencil_conv
```
