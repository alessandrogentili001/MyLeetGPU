# Problem 03: Parallel Reduction ($\sum A[i]$)

> **Difficulty**: Medium / Hard  
> **Type**: Memory-Bound / Architecture Mastery  
> **Key Concepts**: Warp Divergence, Shared Memory Bank Conflicts, Sequential vs Interleaved Addressing, Warp-Level Primitives (`__shfl_down_sync`), Grid-Stride Aggregation.

---

## 📖 Problem Description

Given a 1D vector $A$ of length $N$ containing single-precision floating point numbers (`float`), compute the sum of all elements:

$$S = \sum_{i=0}^{N-1} A[i]$$

This operation is called a **reduction**. While simple on a CPU, performing a tree-based reduction on a massive GPU with thousands of threads requires careful management of parallel thread execution, branching, and memory hierarchies.

---

## 🔬 Hardware & Roofline Analysis (NVIDIA A100-SXM4-64GB)

Parallel reduction performs **1 FLOP** (addition) per element and reads the element once from memory.
- **Arithmetic Intensity**: $1 \text{ FLOP} / 4 \text{ Bytes} = 0.25 \text{ FLOPs/Byte}$.
- $0.25 \ll 12.5$ (A100 Knee Point) $\implies$ **Strictly Memory-Bandwidth Bound**.

For a benchmark size of $N = 67,108,864$ elements ($64\text{M}$ floats, $256\text{ MB}$):
- Total data read: $256\text{ MB}$ (exceeds 32 MB L2 cache, measuring pure HBM2e bandwidth).
- Total data written: 4 bytes (the single scalar result).
- Theoretical minimum latency on A100 ($1,555\text{ GB/s}$):
  $$t_{\min} = \frac{0.256 \text{ GB}}{1,555 \text{ GB/s}} \approx 0.165 \text{ ms} \; (165 \; \mu\text{s})$$

---

## 🧠 The Mark Harris Reduction Progression

This problem guides you through the legendary CUDA reduction optimization sequence originally developed by NVIDIA's Mark Harris.

### 1. Interleaved Addressing with Warp Divergence
We start by loading elements into shared memory (`sdata`), and pairing elements.
```cuda
for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```
**The Problem:** `tid % (2 * s) == 0` evaluates differently for adjacent threads. Within a 32-thread warp, some threads take the branch (active) and others do not (idle). This causes **warp divergence**, forcing the GPU hardware to serialize execution of the branches and slashing performance.

### 2. Interleaved Addressing without Divergence
We fix warp divergence by clustering active threads together using strided indexing:
```cuda
int index = 2 * s * tid;
if (index < blockDim.x) {
    sdata[index] += sdata[index + s];
}
```
**The Problem:** While warps are no longer divergent, look at the memory addresses! At $s=16$, `index` jumps by 32. Threads in the warp access `sdata[0], sdata[32], sdata[64]...` which all map to **Bank 0** in shared memory. This causes a massive **32-way bank conflict**!

### 3. Sequential Addressing (Conflict-Free)
We fix the bank conflicts by reversing the reduction tree structure. Instead of active threads jumping across the array, we cut the array in half and adjacent threads access adjacent elements:
```cuda
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
}
```
**The Fix:** Thread $tid$ accesses `sdata[tid]` and `sdata[tid + s]`. Since adjacent threads access adjacent indices, there are **zero bank conflicts**!

### 4. Warp Shuffle Reduction (`__shfl_down_sync`)
Once the active elements in the block drop to 32, the entire remaining reduction fits inside a **single warp**.
Instead of synchronizing the whole block with `__syncthreads()` and reading/writing shared memory, threads can exchange data directly through registers using `__shfl_down_sync`!
```cuda
val += __shfl_down_sync(0xffffffff, val, offset);
```
This reduces latency to virtually zero for the final 5 steps of the tree.

---

## 🎯 Milestones & A100 Achieved Results

Benchmark size: $N = 67,108,864$ floats ($256\text{ MB}$ total memory traffic) on Leonardo Booster NVIDIA A100-SXM4-64GB (using an optimized `grid_size = 1024`):

| Milestone | Optimization Strategy | Theoretical Bottleneck | Achieved Bandwidth | Latency |
| :--- | :--- | :--- | :--- | :--- |
| **1. Divergent** | Interleaved `if (tid % (2*s) == 0)` | Severe Warp Divergence | **796.43 GB/s** (51.2% Peak) | 0.337 ms |
| **2. Interleaved** | Strided `index = 2 * s * tid` | 32-way Shared Memory Bank Conflicts | **746.42 GB/s** (48.0% Peak) | 0.360 ms |
| **3. Sequential** | Halving `for (s = blockDim.x/2; s > 0; s >>= 1)` | `__syncthreads()` overhead at warp level | **749.73 GB/s** (48.2% Peak) | 0.358 ms |
| **4. Warp Shuffle**| `__shfl_down_sync` for last 32 threads | None (Peak HBM2e Bandwidth Saturation) | **750.16 GB/s** (48.2% Peak) | 0.358 ms |

### Key Takeaways from the Numbers:
1. **The Memory Wall**: The parallel reduction is profoundly memory bound ($0.25$ FLOPs/Byte). Fetching 256 MB of data from HBM2e entirely dominates the execution time.
2. **Atomic Bottleneck Mitigation**: The naive launch configuration (`grid_size = N / block_size = 250,000` blocks) caused 250,000 threads to collide on a single `atomicAdd` write to `out`, artificially capping bandwidth to ~400 GB/s. By aggressively capping the grid size to 1,024 blocks, each thread processed 250 elements in a fast grid-stride loop, eliminating the atomic bottleneck and driving bandwidth to ~796 GB/s!
3. **Micro-Optimizations vs Global Bottlenecks**: Warp divergence (M1), bank conflicts (M2), and `__syncthreads()` overhead (M3) are extremely important for compute-bound kernels. However, in a kernel that only does a single addition per float fetched from VRAM, these SM-level inefficiencies are almost entirely hidden behind the massive latency of global memory reads. This is why all milestones performed very similarly once the atomic contention was resolved!

---

## 🚀 How to Run

```bash
# 1. Activate environment
source .env

# 2. Build target
leetgpu-build

# 3. Run correctness test suite & A100 roofline benchmark
leetgpu-run ./build/problems/03_parallel_reduction/parallel_reduction
```
