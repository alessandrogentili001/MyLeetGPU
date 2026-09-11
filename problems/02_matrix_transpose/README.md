# Problem 02: Matrix Transpose ($B = A^T$)

> **Difficulty**: Medium  
> **Type**: Memory-Bound / Memory Architecture Mastery  
> **Key Concepts**: Global Memory Coalescing, 2D Grid/Block Indexing, Shared Memory Tiling, Shared Memory Bank Conflicts & Padding (+1 Stride), Rectangular Matrix Boundaries.

---

## 📖 Problem Description

Given a 2D matrix $A$ of dimensions $M \times N$ ($M$ rows, $N$ columns) stored in row-major order with single-precision floating point numbers (`float`), compute its transpose matrix $B = A^T$ of dimensions $N \times M$ ($N$ rows, $M$ columns):

$$B[c, r] = A[r, c], \quad \forall r \in [0, M-1], \; c \in [0, N-1]$$

In 1D flattened row-major memory:
- Input address: `in[r * cols + c]`
- Output address: `out[c * rows + r]`

---

## 🔬 Hardware & Roofline Analysis (NVIDIA A100-SXM4-64GB)

Matrix transpose performs **0 arithmetic operations (FLOPs)** and transfers **2 memory words** per element:
- 1 read from global memory (`in`): 4 bytes
- 1 write to global memory (`out`): 4 bytes
- **Total memory traffic**: $8 \times M \times N\text{ bytes}$

### Roofline Numbers:
- **Arithmetic Intensity**: $0\text{ FLOPs/Byte}$ (Strictly memory-bandwidth bound)
- **A100 Peak Memory Bandwidth**: $\sim 1,555\text{ GB/s}$ ($1.55\text{ TB/s}$ HBM2e)
- **A100 Peak FP32 Compute**: $19.49\text{ TFLOPS}$

For an $8192 \times 8192$ matrix ($67,108,864$ floats, $268.4\text{ MB}$ input, $268.4\text{ MB}$ output):
- Total memory traffic = $536.87\text{ MB}$ (far exceeds the 32 MB L2 cache, measuring pure HBM2e memory bus throughput).
- Theoretical minimum latency on A100:
  $$t_{\min} = \frac{0.53687 \text{ GB}}{1,555 \text{ GB/s}} \approx 0.345 \text{ ms} \; (345 \; \mu\text{s})$$

---

## 🧠 Why is Matrix Transpose Hard on GPUs?

### 1. The Coalescing Dilemma
NVIDIA GPUs read and write global memory in aligned **32-byte sectors** (or 128-byte cache lines).
When all 32 threads in a warp access 32 consecutive 4-byte floats (e.g. `c, c+1, ..., c+31`), the memory controller issues a **single coalesced transaction**.

In a naive transpose:
- **Read**: Thread $(tx, ty)$ reads `in[r * cols + c]`. Consecutive $tx$ threads read consecutive columns $c$ in the same row $\to$ **100% coalesced read**!
- **Write**: Thread $(tx, ty)$ writes to `out[c * rows + r]`. Consecutive $tx$ threads write to rows separated by `rows * 4` bytes!
  Unless `rows == 1`, each thread in the warp writes to a completely different 32-byte sector. The memory controller must issue **32 separate memory transactions** for a single warp!
- **Penalty**: 87.5% of requested bandwidth is wasted. On an A100 SXM4 capable of $>1,500\text{ GB/s}$, naive transpose typically crawls at only $\approx 200 - 350\text{ GB/s}$!

---

### 2. The Solution: Shared Memory Tiling
We use fast on-chip SRAM (`__shared__` memory) as an intermediate staging buffer:
1. Load a $32 \times 32$ tile from global memory into shared memory using **coalesced reads**.
2. Synchronize threads with `__syncthreads()`.
3. Read from shared memory in transposed order and write to global memory using **coalesced writes**.

Now, **both** global memory reads and writes are fully coalesced!

---

### 3. The Trap: 32-Way Shared Memory Bank Conflicts
Shared memory on Ampere (`sm_80`) is split into **32 memory banks** of 4 bytes (32 bits) each:
$$\text{Bank Index} = (\text{byte\_address} / 4) \pmod{32}$$

If multiple threads in a warp access different words in the *same* bank simultaneously, the hardware serializes the accesses (**bank conflict**).

Consider a tile `__shared__ float tile[32][32]`:
- During write to output, thread $tx \in [0, 31]$ reads `tile[tx][ty]`.
- For thread $tx$, the element index is $32 \times tx + ty$.
- The bank index is:
  $$\text{Bank} = (32 \times tx + ty) \pmod{32} = ty \pmod{32}$$
- Every single thread $tx \in [0, 31]$ in the warp accesses the exact same bank $ty$!
- This causes a **catastrophic 32-way bank conflict**, forcing the hardware to execute the read across **32 sequential clock cycles**!

---

### 4. The Fix: Bank Conflict Padding (`tile[32][33]`)
By adding **1 float of padding** to each row:
```cuda
__shared__ float tile[TILE_DIM][TILE_DIM + 1]; // [32][33]
```
- The row stride becomes 33 floats instead of 32.
- The address of `tile[tx][ty]` is now:
  $$\text{Bank} = (33 \times tx + ty) \pmod{32} = (32 \times tx + tx + ty) \pmod{32} = (tx + ty) \pmod{32}$$
- As $tx$ varies from $0$ to $31$, $(tx + ty) \pmod{32}$ cycles uniquely through banks $0, 1, 2, \dots, 31$!
- **Result: 0 bank conflicts! All 32 reads complete concurrently in 1 clock cycle.**

---

## 🎯 Milestones & A100 Achieved Results

Benchmark size: $8192 \times 8192$ matrix ($64\text{M}$ floats, $512\text{ MB}$ total memory traffic) on Leonardo Booster NVIDIA A100-SXM4-64GB:

| Milestone | Strategy | Read Pattern | Write Pattern | Shared Memory Banks | Achieved Bandwidth | Latency | Speedup vs Naive |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Naive** | Direct global memory | Coalesced | Strided (Uncoalesced) | N/A | **172.55 GB/s** (11.1% Peak) | 3.111 ms | 1.00x *(Baseline)* |
| **2. Shared Memory** | $32 \times 32$ tile buffer | Coalesced | Coalesced | 32-way Bank Conflicts | **702.39 GB/s** (45.2% Peak) | 0.764 ms | **4.07x** |
| **3. Padded Shared** | $32 \times 33$ tile buffer | Coalesced | Coalesced | **0 Bank Conflicts** | **1,090.99 GB/s** (70.2% Peak) | 0.492 ms | **6.32x** |
| **4. Coarse-Grained** | $32 \times 8$ block (4 rows/th) | Coalesced + ILP | Coalesced + ILP | **0 Bank Conflicts** | **1,187.14 GB/s** (76.3% Peak) | 0.452 ms | **6.88x** 🚀 |

### Key Takeaways from the Numbers:
1. **Coalescing gives the first massive leap**: Moving from Naive (uncoalesced global writes) to Shared Memory Tiling jumps bandwidth from **$172.55\text{ GB/s} \to 702.39\text{ GB/s}$ ($4.07\times$ speedup)**.
2. **Bank conflicts are expensive**: Removing the 32-way bank conflicts via `[32][33]` padding gains another **$+388.6\text{ GB/s}$** ($702.39 \to 1,090.99\text{ GB/s}$), an additional $1.55\times$ boost!
3. **Thread coarsening increases ILP**: Using $32 \times 8$ threads to process a $32 \times 32$ tile amortizes block launch overhead, increases instruction-level parallelism and register reuse, reaching **$1,187.14\text{ GB/s}$ ($76.3\%$ of theoretical peak)**!

---

## 🚀 How to Run

```bash
# 1. Activate environment
source .env

# 2. Build target
leetgpu-build

# 3. Run correctness test suite & A100 roofline benchmark
leetgpu-run ./build/problems/02_matrix_transpose/matrix_transpose
```
