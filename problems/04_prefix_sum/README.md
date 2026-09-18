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

## 🚀 How to Run

```bash
# 1. Activate environment
source .env

# 2. Build target
leetgpu-build

# 3. Run correctness test suite & benchmark
leetgpu-run ./build/problems/04_prefix_sum/prefix_sum
```
