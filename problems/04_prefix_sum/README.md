# Problem 04: Prefix Sum (Exclusive Scan)

> **Difficulty**: Hard  
> **Type**: Algorithmic & Memory Architecture Mastery  
> **Key Concepts**: Work-Efficiency, Step-Efficiency, Up-Sweep / Down-Sweep (Blelloch Tree), Shared Memory Bank Conflicts.

---

## 📖 Problem Description

Given a 1D vector $A$ of length $N$ containing single-precision floating point numbers (`float`), compute its **exclusive prefix sum** (or exclusive scan) and store the result in vector $C$:

$$C[i] = \sum_{j=0}^{i-1} A[j]$$

Where $C[0] = 0$.

Prefix sum is a fundamental parallel primitive used extensively in stream compaction, radix sort, string matching, and sparse matrix operations. 
Because the $i$-th element depends on all $i-1$ previous elements, parallelizing it is not trivial!

---

## 🔬 Hardware & Roofline Analysis

- **Arithmetic Intensity**: Extremely low. $\approx 1 \text{ FLOP} / 8 \text{ Bytes} = 0.125 \text{ FLOPs/Byte}$.
- Like Parallel Reduction, Prefix Sum is **strictly memory-bandwidth bound**.
- However, doing it efficiently in parallel requires significantly more internal memory manipulation than a simple reduction.

---

## 🧠 The Scan Progression

### 1. Hillis-Steele (Step-Efficient, Work-Inefficient)
The Hillis-Steele algorithm uses double-buffered shared memory. In each step $d$, a thread adds the element $2^d$ positions to its left.
- **Steps**: $\log_2(N)$
- **Work (Additions)**: $O(N \log N)$
- This algorithm is extremely fast on small arrays because the hardware is massively parallel, but it performs exponentially more additions than the sequential $O(N)$ CPU algorithm!

### 2. Blelloch (Work-Efficient)
The Blelloch algorithm matches the CPU's $O(N)$ work complexity by performing two phases over a binary tree mapped to the array:
1. **Up-Sweep (Reduce)**: Build a sum tree from leaves to root.
2. **Down-Sweep**: Push sums back down the tree to compute the prefix sum.
- **Work**: $O(N)$
- This is the standard for large-scale GPU scans.

### 3. Bank Conflict Avoidance
Just like the interleaved reduction, the power-of-2 striding in Blelloch causes massive 32-way shared memory bank conflicts!
We use a padding macro to shift addresses and eliminate these conflicts.

---

## 🎯 Milestones & A100 Achieved Results

Benchmark size: $N = 67,108,864$ floats ($64\text{M}$ elements, $512\text{ MB}$ total memory traffic: $256\text{ MB}$ read + $256\text{ MB}$ write) on Leonardo Booster NVIDIA A100-SXM4-64GB:

| Milestone | Optimization Strategy | Algorithmic Work | Achieved Bandwidth | Avg Latency | % of A100 Peak (1,555 GB/s) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Hillis-Steele** | Step-Efficient (2 el/thread) | $O(N \log N)$ | **473.53 GB/s** | 1.134 ms | 30.45% |
| **2. Blelloch (Naive)** | Work-Efficient Up/Down-Sweep Tree | $O(N)$ | **335.92 GB/s** | 1.598 ms | 21.60% |
| **3. Blelloch (Padded)**| Bank Conflict Avoidance via `>> 5` padding | $O(N)$ | **496.34 GB/s** | **1.082 ms** | **31.92%** |

### 🔍 Key Takeaways from the Numbers:

1. **The Shared Memory Bank Conflict Penalty**:
   - In Milestone 2, the power-of-2 strided accesses (`offset * (2 * tid + 1) - 1`) cause threads in the same 32-thread warp to access shared memory addresses mapping to identical banks.
   - The hardware serializes these conflicting requests (up to 16-way and 32-way bank conflicts).
   - In Milestone 3, inserting 1 dummy float pad per 32 elements (`CONFLICT_FREE_OFFSET(n) = n >> 5`) shifts the addresses so that each thread accesses a distinct bank.
   - **Result**: A massive **~48% speedup** (`1.598 ms` $\to$ `1.082 ms`), raising bandwidth from **335.9 GB/s** to **496.3 GB/s**!

2. **Work-Efficiency vs Hardware Concurrency**:
   - Theoretically, Hillis-Steele performs $O(N \log N)$ additions while Blelloch does only $O(N)$ additions.
   - However, naive Blelloch was slower than Hillis-Steele due to bank conflicts and tree traversal step dependencies.
   - Once bank conflicts were eliminated, Blelloch Padded surpassed Hillis-Steele, delivering the fastest execution time on the A100 GPU.

3. **Multi-Block Recursive Scanning**:
   - For arrays spanning multiple thread blocks, each block writes its total sum to `d_block_sums`.
   - `launch_scan_*` recursively scans the block sums on the GPU and then launches `add_block_sums_kernel` to add the scanned block prefixes back to the respective blocks, handling arbitrary array sizes up to tens of millions of elements seamlessly.

---

## 🚀 How to Run

```bash
# 1. Activate environment
source .env

# 2. Build target
leetgpu-build

# 3. Run correctness test suite & benchmark
leetgpu-run ./build/problems/04_prefix_sum/prefix_sum
```

