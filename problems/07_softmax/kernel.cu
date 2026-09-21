#include "kernel.cuh"
#include <cfloat>
#include <cmath>
#include <cstdint>

// ==============================================================================
// MILESTONE 1: Two-Pass Safe Softmax (Shared Memory Block Reduction)
// ==============================================================================
// Each thread block processes one row of length n.
//
// Pass 1: Cooperative reduction to find row maximum m = max(x_j)
// Pass 2: Cooperative reduction to compute sum of exponentials d = sum(exp(x_j - m))
// Pass 3: Normalize and write output y_j = exp(x_j - m) / d
// ==============================================================================
__global__ void softmax_block_twopass_kernel(const float* in, float* out, int m, int n) {
    // TODO: Implement Milestone 1:
    // 1. Identify current row: int row = blockIdx.x; (exit if row >= m)
    // 2. Allocate shared memory for reduction:
    //    __shared__ float s_max[SOFTMAX_BLOCK_SIZE];
    //    __shared__ float s_sum[SOFTMAX_BLOCK_SIZE];
    // 3. Grid-stride loop over columns to find thread-local max, then reduce in s_max:
    //    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) ...
    // 4. Grid-stride loop over columns to compute thread-local sum of exp(val - max_val),
    //    then reduce in s_sum.
    // 5. Broadcast row_max and row_sum, compute inv_sum = 1.0f / row_sum.
    // 6. Write out[row * n + col] = expf(in[row * n + col] - row_max) * inv_sum.

    int row = blockIdx.x;
    if (row>=m) return;

    __shared__ float s_max[SOFTMAX_BLOCK_SIZE];
    __shared__ float s_sum[SOFTMAX_BLOCK_SIZE];

    int col_stride = blockDim.x;
    int tid = threadIdx.x;
    //int lane = tid%32;
    //int warp = tid/32;

    // First Pass: MAX reduction
    float thread_max = -1e20;
    for (int i = tid; i < n; i += col_stride) {
        thread_max = fmaxf(thread_max, in[row*n+i]);
    }
    s_max[tid] = thread_max;
    __syncthreads();

    for (int stride = col_stride/2; stride > 0; stride >>= 1) {
        s_max[tid] = fmaxf(s_max[tid], s_max[tid+stride]);
        __syncthreads();
    }
    float block_max = s_max[0];

    // Second Pass: SUM reduction
    float thread_sum = 0.0f;
    for (int i = tid; i<n; i+= col_stride) {
        thread_sum += expf(in[row*n+i]-block_max);
    }
    s_sum[tid] = thread_sum;
    __syncthreads();

    for (int stride = col_stride/2; stride > 0; stride>>=1) {
        s_sum[tid] += s_sum[tid+stride];
        __syncthreads();
    }
    float block_sum = s_sum[0];
    float inv_sum = 1.0f/block_sum;

    // Third Pass: WRITE normalized output
    for (int i = tid; i<n; i+= col_stride) {
        out[row*n+i] = expf(in[row*n+i]-block_max)*inv_sum;
    }
}

void launch_softmax_block_twopass(const float* d_in, float* d_out, int m, int n) {
    dim3 grid(m);
    dim3 block(SOFTMAX_BLOCK_SIZE);
    softmax_block_twopass_kernel<<<grid, block>>>(d_in, d_out, m, n);
}

bool is_block_twopass_implemented() {
    return true; // Change to true once implemented!
}


// ==============================================================================
// MILESTONE 2: Warp-Accelerated Safe Softmax (__shfl_down_sync)
// ==============================================================================
// Warp shuffles allow intra-warp communication without touching shared memory
// or requiring __syncthreads(), drastically cutting latency.
//
// Strategy:
// 1. Reduce within each 32-thread warp using __shfl_down_sync.
// 2. Warp leaders (lane == 0) store partial warp results into small shared memory
//    (only blockDim.x / 32 elements).
// 3. Warp 0 performs the final reduction across warp leaders.
// 4. Broadcast result to all threads in the block.
// ==============================================================================

// Helper: Warp-level reduction for max
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// Helper: Warp-level reduction for sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_warp_shuffle_kernel(const float* in, float* out, int m, int n) {
    // TODO: Implement Milestone 2:
    // 1. Each thread computes local max over its strided column elements.
    // 2. Perform warp reduction with warp_reduce_max().
    // 3. Lane 0 of each warp writes to __shared__ float s_warp_max[32].
    // 4. Synchronize, then threadIdx.x < num_warps performs final reduction with Warp 0.
    // 5. Repeat the reduction for the sum of exponentials using warp_reduce_sum().
    // 6. Write normalized outputs to out[row * n + col].

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int col_stride = blockDim.x;
    int lane = tid % 32;
    int warp = tid / 32;
    int num_warps = blockDim.x/32;

    __shared__ float s_max[32];
    __shared__ float s_sum[32];

    float thread_max = -1e20;
    for (int i = tid; i < n; i+= col_stride) {
        thread_max = fmaxf(thread_max, in[row*n+i]);
    }
    thread_max = warp_reduce_max(thread_max);

    if (lane == 0) s_max[warp] = thread_max;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < num_warps) ? s_max[lane] : -INFINITY;
        val = warp_reduce_max(val);
        if (lane == 0) s_max[0] = val;
    }
    __syncthreads();
    float block_max = s_max[0]; 

    float thread_sum = 0;
    for (int i = tid; i < n; i+= col_stride) {
        thread_sum += expf(in[row*n+i]-block_max);
    }
    thread_sum = warp_reduce_sum(thread_sum);

    if (lane == 0) s_sum[warp] = thread_sum;
    __syncthreads();

    if (warp == 0) {
        float val = (lane < num_warps) ? s_sum[lane] : 0;
        val = warp_reduce_sum(val);
        if (lane == 0) s_sum[0] = val;
    }
    __syncthreads();
    float block_sum = s_sum[0]; 

    float inv_sum = 1.0f/block_sum;
    for (int i = tid; i < n; i+= col_stride) {
        out[row*n+i] = expf(in[row*n+i]-block_max)*inv_sum;
    }
}

void launch_softmax_warp_shuffle(const float* d_in, float* d_out, int m, int n) {
    dim3 grid(m);
    dim3 block(SOFTMAX_BLOCK_SIZE);
    softmax_warp_shuffle_kernel<<<grid, block>>>(d_in, d_out, m, n);
}

bool is_warp_shuffle_implemented() {
    return true; // Change to true once implemented!
}


// ==============================================================================
// MILESTONE 3: Online Safe Softmax (FlashSoftmax / Single-Pass)
// ==============================================================================
// In traditional softmax, data must be read multiple times from DRAM/cache.
// The Online Softmax algorithm (Milakov & Gimelshein 2018 / Dao et al. FlashAttention)
// maintains running max (m) and running sum (d) simultaneously in a single pass:
//
// Given current state (m_A, d_A) and new values (m_B, d_B):
//   m_new = max(m_A, m_B)
//   d_new = d_A * exp(m_A - m_new) + d_B * exp(m_B - m_new)
//
// Combine this with float4 vectorized loads for maximum DRAM bus saturation!
// ==============================================================================

// Helper: Combine two online softmax states (max, sum)
__device__ __forceinline__ void combine_online_stats(float& m1, float& d1, float m2, float d2) {
    if (m1 > m2) {
        d1 = d1 + d2 * __expf(m2 - m1);
    } else {
        d1 = d1 * __expf(m1 - m2) + d2;
        m1 = m2;
    }
}

// Helper: Update running online stats with a single scalar element
__device__ __forceinline__ void update_online_val(float& m, float& d, float x) {
    if (x > m) {
        d = d * __expf(m - x) + 1.0f;
        m = x;
    } else {
        d += __expf(x - m);
    }
}

__global__ void softmax_online_kernel(const float* in, float* out, int m, int n) {
    // TODO: Implement Milestone 3:
    // 1. Maintain running (thread_m, thread_d) initialized to (-INFINITY, 0.0f).
    // 2. Loop over columns using vectorized float4 loads when aligned and within bounds.
    // 3. For each element x:
    //      float m_prev = thread_m;
    //      thread_m = fmaxf(thread_m, x);
    //      thread_d = thread_d * __expf(m_prev - thread_m) + __expf(x - thread_m);
    // 4. Reduce (thread_m, thread_d) across the block using warp shuffles and shared memory:
    //      Use combine_online_stats() inside warp reductions!
    // 5. Broadcast global (row_m, row_d) to all threads in block.
    // 6. Loop over columns once more to write out the normalized values y = exp(x - row_m) / row_d.

    __shared__ float s_m[32];
    __shared__ float s_d[32];

    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int col_stride = blockDim.x;
    int lane = tid % 32;
    int warp = tid / 32;
    int num_warps = blockDim.x / 32;

    const float* row_in = in + static_cast<size_t>(row) * n;
    float* row_out = out + static_cast<size_t>(row) * n;

    float thread_m = -1e20f;
    float thread_d = 0.0f;

    // Check if row addresses are 16-byte aligned for float4 vectorization
    bool is_aligned = (reinterpret_cast<uintptr_t>(row_in) % sizeof(float4) == 0) &&
                      (reinterpret_cast<uintptr_t>(row_out) % sizeof(float4) == 0);

    // Pass 1: Online stats reduction (using float4 when aligned)
    if (is_aligned) {
        int n4 = n / 4;
        const float4* in4 = reinterpret_cast<const float4*>(row_in);
        for (int i = tid; i < n4; i += col_stride) {
            float4 v = in4[i];
            float local_max = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
            float local_d = __expf(v.x - local_max) + __expf(v.y - local_max) +
                            __expf(v.z - local_max) + __expf(v.w - local_max);
            combine_online_stats(thread_m, thread_d, local_max, local_d);
        }
        for (int i = n4 * 4 + tid; i < n; i += col_stride) {
            update_online_val(thread_m, thread_d, row_in[i]);
        }
    } else {
        for (int i = tid; i < n; i += col_stride) {
            update_online_val(thread_m, thread_d, row_in[i]);
        }
    }

    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float m2 = __shfl_down_sync(0xffffffff, thread_m, offset);
        float d2 = __shfl_down_sync(0xffffffff, thread_d, offset);
        combine_online_stats(thread_m, thread_d, m2, d2);
    }

    if (lane == 0) s_m[warp] = thread_m;
    if (lane == 0) s_d[warp] = thread_d;
    __syncthreads();

    if (tid == 0) {
        for (int i = 1; i < num_warps; i++) {
            combine_online_stats(s_m[0], s_d[0], s_m[i], s_d[i]);
        }
    }
    __syncthreads();
    
    float row_m = s_m[0];
    float row_d = s_d[0];
    float inv_d = 1.0f / row_d;

    // Pass 2: Write normalized outputs (using float4 when aligned)
    if (is_aligned) {
        int n4 = n / 4;
        const float4* in4 = reinterpret_cast<const float4*>(row_in);
        float4* out4 = reinterpret_cast<float4*>(row_out);
        for (int i = tid; i < n4; i += col_stride) {
            float4 v = in4[i];
            float4 res;
            res.x = __expf(v.x - row_m) * inv_d;
            res.y = __expf(v.y - row_m) * inv_d;
            res.z = __expf(v.z - row_m) * inv_d;
            res.w = __expf(v.w - row_m) * inv_d;
            out4[i] = res;
        }
        for (int i = n4 * 4 + tid; i < n; i += col_stride) {
            row_out[i] = __expf(row_in[i] - row_m) * inv_d;
        }
    } else {
        for (int i = tid; i < n; i += col_stride) {
            row_out[i] = __expf(row_in[i] - row_m) * inv_d;
        }
    }
}

void launch_softmax_online(const float* d_in, float* d_out, int m, int n) {
    dim3 grid(m);
    dim3 block(SOFTMAX_BLOCK_SIZE);
    softmax_online_kernel<<<grid, block>>>(d_in, d_out, m, n);
}

bool is_online_implemented() {
    return true; // Change to true once implemented!
}
