#include "kernel.cuh"
#include <cmath>
#include <cstdint>

// ==============================================================================
// MILESTONE 1: Multi-Pass LayerNorm (Shared Memory Block Reduction)
// ==============================================================================
// Each thread block processes one row of the M x N matrix.
//
// Pass 1: Cooperative reduction to compute row mean:
//         mean = (1 / N) * sum(X[i, j])
// Pass 2: Cooperative reduction to compute row variance:
//         var  = (1 / N) * sum((X[i, j] - mean)^2)
// Pass 3: Normalize and write output:
//         Y[i, j] = ((X[i, j] - mean) / sqrt(var + eps)) * gamma[j] + beta[j]
// ==============================================================================
__global__ void layernorm_twopass_kernel(const float* __restrict__ x,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ y,
                                         int m, int n, float eps) {
    // TODO: Implement Milestone 1
    // 1. Identify row: int row = blockIdx.x; if (row >= m) return;
    // 2. Allocate shared memory for reduction:
    //    __shared__ float s_data[LAYERNORM_BLOCK_SIZE];
    // 3. Thread-stride loop to compute sum of x elements, then block reduce in s_data.
    // 4. Compute and broadcast mean = s_data[0] / n.
    // 5. Thread-stride loop to compute sum of squared differences (x - mean)^2,
    //    then block reduce in s_data.
    // 6. Compute inv_std = rsqrtf(s_data[0] / n + eps).
    // 7. Thread-stride loop to normalize:
    //    val = (x[idx] - mean) * inv_std * gamma[col] + beta[col];
    //    y[idx] = val;

    int row = blockIdx.x;
    if (row >= m) return;
    __shared__ float s_data[LAYERNORM_BLOCK_SIZE];
    int tid = threadIdx.x;
    const float* row_x = x + row * n;
    float* row_y = y + row * n;
    // Pass 1: Compute row mean
    float sum = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        sum += row_x[j];
    }
    s_data[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    float mean = s_data[0] / n;
    __syncthreads();
    // Pass 2: Compute row variance
    float var_sum = 0.0f;
    for (int j = tid; j < n; j += blockDim.x) {
        float diff = row_x[j] - mean;
        var_sum += diff * diff;
    }
    s_data[tid] = var_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }
    float var = s_data[0] / n;
    float inv_std = rsqrtf(var + eps);
    __syncthreads();
    // Pass 3: Normalize and write output
    for (int j = tid; j < n; j += blockDim.x) {
        float val = (row_x[j] - mean) * inv_std;
        if (gamma != nullptr) val *= gamma[j];
        if (beta != nullptr) val += beta[j];
        row_y[j] = val;
    }
}

void launch_layernorm_twopass(const float* d_x, const float* d_gamma, const float* d_beta,
                              float* d_y, int m, int n, float eps) {
    dim3 block(LAYERNORM_BLOCK_SIZE);
    dim3 grid(m);
    layernorm_twopass_kernel<<<grid, block>>>(d_x, d_gamma, d_beta, d_y, m, n, eps);
}

bool is_layernorm_twopass_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 2: One-Pass Welford LayerNorm (Warp Shuffle Reduction)
// ==============================================================================
// Welford's algorithm computes the mean and variance online in a single pass:
//
// Combining two states A = (count_A, mean_A, M2_A) and B = (count_B, mean_B, M2_B):
//   count = count_A + count_B
//   delta = mean_B - mean_A
//   mean  = mean_A + delta * (count_B / count)
//   M2    = M2_A + M2_B + delta^2 * (count_A * count_B / count)
//
// Here M2 is sum of squared differences from the mean: var = M2 / N.
//
// We combine states across warps using `__shfl_down_sync` in register space,
// completely eliminating shared memory tree reductions for within-warp work!
// ==============================================================================

struct WelfordState {
    float mean;
    float m2;
    float count;
};

__device__ __forceinline__ WelfordState merge_welford(WelfordState a, WelfordState b) {
    if (a.count == 0.0f) return b;
    if (b.count == 0.0f) return a;

    float new_count = a.count + b.count;
    float delta = b.mean - a.mean;
    float new_mean = a.mean + delta * (b.count / new_count);
    float new_m2 = a.m2 + b.m2 + delta * delta * (a.count * b.count / new_count);

    return {new_mean, new_m2, new_count};
}

__device__ __forceinline__ WelfordState warp_reduce_welford(WelfordState val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        WelfordState other;
        other.mean  = __shfl_down_sync(0xffffffff, val.mean, offset);
        other.m2    = __shfl_down_sync(0xffffffff, val.m2, offset);
        other.count = __shfl_down_sync(0xffffffff, val.count, offset);
        val = merge_welford(val, other);
    }
    return val;
}

__global__ void layernorm_welford_kernel(const float* __restrict__ x,
                                         const float* __restrict__ gamma,
                                         const float* __restrict__ beta,
                                         float* __restrict__ y,
                                         int m, int n, float eps) {
    // TODO: Implement Milestone 2
    // 1. Each thread accumulates local Welford state across its column slice.
    // 2. Reduce Welford state within warp using warp_reduce_welford.
    // 3. Store warp leaders to shared memory (size = blockDim.x / 32).
    // 4. Warp 0 performs final reduction of warp leaders.
    // 5. Broadcast final mean and inv_std = rsqrtf(M2 / n + eps) to all threads.
    // 6. Write normalized elements to y.

    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    constexpr int num_warps = LAYERNORM_BLOCK_SIZE / WARP_SIZE;

    const float* row_x = x + row * n;
    float* row_y = y + row * n;

    // Step 1: Thread-local Welford accumulation across column slice
    WelfordState thread_state = {0.0f, 0.0f, 0.0f};
    for (int j = tid; j < n; j += blockDim.x) {
        WelfordState x_state = {row_x[j], 0.0f, 1.0f};
        thread_state = merge_welford(thread_state, x_state);
    }

    // Step 2: Intra-warp reduction via register shuffle
    thread_state = warp_reduce_welford(thread_state);

    // Step 3: Inter-warp block reduction via tiny shared memory buffer
    __shared__ WelfordState s_warp[num_warps];
    if (lane == 0) {
        s_warp[warp_id] = thread_state;
    }
    __syncthreads();

    // Warp 0 reduces the warp leaders
    if (warp_id == 0) {
        WelfordState warp_leader = (lane < num_warps) ? s_warp[lane] : WelfordState{0.0f, 0.0f, 0.0f};
        warp_leader = warp_reduce_welford(warp_leader);
        if (lane == 0) {
            s_warp[0] = warp_leader;
        }
    }
    __syncthreads();

    // Step 4: Broadcast reduction results and normalize
    float mean = s_warp[0].mean;
    float var = s_warp[0].m2 / n;
    float inv_std = rsqrtf(var + eps);

    for (int j = tid; j < n; j += blockDim.x) {
        float val = (row_x[j] - mean) * inv_std;
        if (gamma != nullptr) val *= gamma[j];
        if (beta != nullptr) val += beta[j];
        row_y[j] = val;
    }
}

void launch_layernorm_welford(const float* d_x, const float* d_gamma, const float* d_beta,
                              float* d_y, int m, int n, float eps) {
    dim3 block(LAYERNORM_BLOCK_SIZE);
    dim3 grid(m);
    layernorm_welford_kernel<<<grid, block>>>(d_x, d_gamma, d_beta, d_y, m, n, eps);
}

bool is_layernorm_welford_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 3: Fused Vectorized RMSNorm (float4 + Warp Shuffle)
// ==============================================================================
// RMSNorm (Root Mean Square Normalization, used in LLaMA, Mistral, Gemma):
//   RMS = sqrt( (1 / N) * sum(X[i, j]^2) + eps )
//   Y[i, j] = (X[i, j] / RMS) * gamma[j]
//
// Key Advantages over LayerNorm:
// 1. Eliminates mean centering (no subtraction needed).
// 2. Requires only a single reduction (sum of squares x^2).
// 3. Fused with `float4` (128-bit) loads/stores to saturate A100 memory bandwidth.
// ==============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void rmsnorm_vectorized_kernel(const float* __restrict__ x,
                                          const float* __restrict__ gamma,
                                          float* __restrict__ y,
                                          int m, int n, float eps) {
    // TODO: Implement Milestone 3
    // 1. Identify row: int row = blockIdx.x; if (row >= m) return;
    // 2. Thread-stride loop using float4 to compute thread-local sum of squares (val.x^2 + val.y^2 + ...)
    // 3. Handle leftover non-multiple of 4 elements if n % 4 != 0.
    // 4. Block-level sum reduction using warp_reduce_sum + small shared memory buffer.
    // 5. Compute inv_rms = rsqrtf(total_sq_sum / n + eps).
    // 6. Thread-stride loop using float4 to load x, scale by inv_rms and gamma, and write to y.

    int row = blockIdx.x;
    if (row >= m) return;

    int tid = threadIdx.x;
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    constexpr int num_warps = LAYERNORM_BLOCK_SIZE / WARP_SIZE;

    const float* row_x = x + row * n;
    float* row_y = y + row * n;

    // Check 16-byte alignment for 128-bit float4 loads/stores
    bool is_aligned = (reinterpret_cast<uintptr_t>(row_x) % sizeof(float4) == 0) &&
                      (gamma == nullptr || reinterpret_cast<uintptr_t>(gamma) % sizeof(float4) == 0) &&
                      (reinterpret_cast<uintptr_t>(row_y) % sizeof(float4) == 0);

    float thread_sq_sum = 0.0f;

    // Step 1: Compute thread-local sum of squares
    if (is_aligned) {
        int num_vec = n / 4;
        const float4* row_x_vec = reinterpret_cast<const float4*>(row_x);

        // Vectorized 128-bit loads
        for (int i = tid; i < num_vec; i += blockDim.x) {
            float4 v = row_x_vec[i];
            thread_sq_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }

        // Remainder scalar elements (when n % 4 != 0)
        for (int j = num_vec * 4 + tid; j < n; j += blockDim.x) {
            float v = row_x[j];
            thread_sq_sum += v * v;
        }
    } else {
        // Fallback scalar loop for unaligned edge-case dimensions
        for (int j = tid; j < n; j += blockDim.x) {
            float v = row_x[j];
            thread_sq_sum += v * v;
        }
    }

    // Step 2: Intra-warp sum reduction
    thread_sq_sum = warp_reduce_sum(thread_sq_sum);

    // Step 3: Inter-warp block reduction via shared memory
    __shared__ float s_warp[num_warps];
    if (lane == 0) {
        s_warp[warp_id] = thread_sq_sum;
    }
    __syncthreads();

    if (warp_id == 0) {
        float warp_leader = (lane < num_warps) ? s_warp[lane] : 0.0f;
        warp_leader = warp_reduce_sum(warp_leader);
        if (lane == 0) {
            s_warp[0] = warp_leader;
        }
    }
    __syncthreads();

    // Step 4: Compute inv_rms normalizer
    float mean_sq = s_warp[0] / n;
    float inv_rms = rsqrtf(mean_sq + eps);

    // Step 5: Normalize and write output with vectorized stores
    if (is_aligned) {
        int num_vec = n / 4;
        const float4* row_x_vec = reinterpret_cast<const float4*>(row_x);
        const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
        float4* row_y_vec = reinterpret_cast<float4*>(row_y);

        // Vectorized normalization and write with gamma handling
        for (int i = tid; i < num_vec; i += blockDim.x) {
            float4 v = row_x_vec[i];
            float4 g = (gamma_vec != nullptr) ? gamma_vec[i] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 out;
            out.x = v.x * inv_rms * g.x;
            out.y = v.y * inv_rms * g.y;
            out.z = v.z * inv_rms * g.z;
            out.w = v.w * inv_rms * g.w;
            row_y_vec[i] = out;
        }

        // Remainder scalar elements
        for (int j = num_vec * 4 + tid; j < n; j += blockDim.x) {
            float val = row_x[j] * inv_rms;
            if (gamma != nullptr) val *= gamma[j];
            row_y[j] = val;
        }
    } else {
        // Fallback scalar loop for unaligned edge-case dimensions
        for (int j = tid; j < n; j += blockDim.x) {
            float val = row_x[j] * inv_rms;
            if (gamma != nullptr) val *= gamma[j];
            row_y[j] = val;
        }
    }
}

void launch_rmsnorm_vectorized(const float* d_x, const float* d_gamma,
                               float* d_y, int m, int n, float eps) {
    dim3 block(LAYERNORM_BLOCK_SIZE);
    dim3 grid(m);
    rmsnorm_vectorized_kernel<<<grid, block>>>(d_x, d_gamma, d_y, m, n, eps);
}

bool is_rmsnorm_vectorized_implemented() {
    return true;
}
