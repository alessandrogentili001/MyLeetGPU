#include "kernel.cuh"

// ==============================================================================
// Helper Kernel: Add block sums to the final output (Provided)
// ==============================================================================
__global__ void add_block_sums_kernel(float* d_out, const float* d_block_sums, int n) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    if (blockIdx.x == 0) return; // First block doesn't need addition

    float block_sum = d_block_sums[blockIdx.x];

    if (idx < n) d_out[idx] += block_sum;
    if (idx + blockDim.x < n) d_out[idx + blockDim.x] += block_sum;
}


// ==============================================================================
// MILESTONE 1: Hillis-Steele Scan (Step-Efficient, Work-Inefficient)
// ==============================================================================
__global__ void hillis_steele_kernel(const float* in, float* out, float* block_sums, int n) {
    // TODO: Implement the Hillis-Steele exclusive scan.
    // 1. Load elements into shared memory (remember to handle out-of-bounds!)
    // 2. Perform the Hillis-Steele scan: for (int offset = 1; offset < blockDim.x; offset *= 2)
    // 3. Keep double buffering or synchronization in mind.
    // 4. Write the total block sum to `block_sums[blockIdx.x]` if block_sums is not null.
    // 5. Write the exclusive scan results to `out`.

    __shared__ float s_val[SCAN_BLOCK_SIZE * 2]; // one thread manage two inputs 
    int tid1 = threadIdx.x;
    int tid2 = threadIdx.x + SCAN_BLOCK_SIZE;
    int global_idx1 = blockIdx.x * blockDim.x * 2 + tid1;
    int global_idx2 = blockIdx.x * blockDim.x * 2 + tid2;

    if (global_idx1 < n) s_val[tid1] = in[global_idx1];
    else s_val[tid1] = 0;
    if (global_idx2 < n) s_val[tid2] = in[global_idx2];
    else s_val[tid2] = 0;
    __syncthreads();

    for (int offset = 1; offset < SCAN_BLOCK_SIZE * 2; offset *= 2) {
        float temp1 = (tid1 >= offset) ? s_val[tid1 - offset] : 0.0f;
        float temp2 = (tid2 >= offset) ? s_val[tid2 - offset] : 0.0f;
        __syncthreads();
        s_val[tid1] += temp1;
        s_val[tid2] += temp2;
        __syncthreads();    
    }

    if (block_sums != nullptr && tid2 == SCAN_BLOCK_SIZE * 2 - 1) {
        block_sums[blockIdx.x] = s_val[tid2];
    }
    __syncthreads();

    if (global_idx1 < n) {
        out[global_idx1] = (tid1 == 0) ? 0.0f : s_val[tid1 - 1]; // exclusive scan 
    }
    if (global_idx2 < n) {
        out[global_idx2] = (tid2 == 0) ? 0.0f : s_val[tid2 - 1]; // exclusive scan 
    }
}

void launch_scan_hillis_steele(const float* d_in, float* d_out, int n) {
    // Boilerplate for multi-block scan (provided)
    int num_blocks = (n + SCAN_BLOCK_SIZE * 2 - 1) / (SCAN_BLOCK_SIZE * 2);
    
    float* d_block_sums = nullptr;
    if (num_blocks > 1) {
        cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
    }

    hillis_steele_kernel<<<num_blocks, SCAN_BLOCK_SIZE>>>(d_in, d_out, d_block_sums, n);

    if (num_blocks > 1) {
        // Recursively scan the block sums
        launch_scan_hillis_steele(d_block_sums, d_block_sums, num_blocks);
        // Add the scanned block sums back to the main array
        add_block_sums_kernel<<<num_blocks, SCAN_BLOCK_SIZE>>>(d_out, d_block_sums, n);
        cudaFree(d_block_sums);
    }
}

bool is_hillis_steele_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 2: Blelloch Scan (Work-Efficient)
// ==============================================================================
__global__ void blelloch_kernel(const float* in, float* out, float* block_sums, int n) {
    // TODO: Implement the Blelloch exclusive scan.
    // 1. Load elements into shared memory.
    // 2. Up-Sweep (Reduce) phase.
    // 3. Save the total block sum, then clear the root to 0.
    // 4. Down-Sweep phase.
    // 5. Write results to `out` and `block_sums`.

    __shared__ float s_val[SCAN_BLOCK_SIZE * 2]; // one thread manage two inputs 

    int tid1 = threadIdx.x; 
    int tid2 = tid1 + SCAN_BLOCK_SIZE;
    int global_idx1 = blockIdx.x * blockDim.x * 2 + tid1;
    int global_idx2 = blockIdx.x * blockDim.x * 2 + tid2;

    if (global_idx1 < n) s_val[tid1] = in[global_idx1];
    else s_val[tid1] = 0.0f;
    if (global_idx2 < n) s_val[tid2] = in[global_idx2];
    else s_val[tid2] = 0.0f;
    __syncthreads();

    int offset = 1;

    // Up-Sweep (Reduction) phase
    for (int d = SCAN_BLOCK_SIZE; d > 0; d >>= 1) {
        if (tid1 < d) {
            int ai = offset * (2 * tid1 + 1) - 1;
            int bi = offset * (2 * tid1 + 2) - 1;
            s_val[bi] += s_val[ai];
        }
        offset *= 2;
        __syncthreads();
    }

    // Save block sum and clear the root to 0
    if (tid1 == 0) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = s_val[SCAN_BLOCK_SIZE * 2 - 1];
        }
        s_val[SCAN_BLOCK_SIZE * 2 - 1] = 0.0f;
    }

    // Down-Sweep phase
    for (int d = 1; d <= SCAN_BLOCK_SIZE; d *= 2) {
        offset >>= 1;
        if (tid1 < d) {
            int ai = offset * (2 * tid1 + 1) - 1;
            int bi = offset * (2 * tid1 + 2) - 1;
            float t = s_val[ai];
            s_val[ai] = s_val[bi];
            s_val[bi] += t;
        }
        __syncthreads();
    }

    // Write results to out
    if (global_idx1 < n) out[global_idx1] = s_val[tid1];
    if (global_idx2 < n) out[global_idx2] = s_val[tid2];
}

void launch_scan_blelloch(const float* d_in, float* d_out, int n) {
    int num_blocks = (n + SCAN_BLOCK_SIZE * 2 - 1) / (SCAN_BLOCK_SIZE * 2);
    
    float* d_block_sums = nullptr;
    if (num_blocks > 1) {
        cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
    }

    blelloch_kernel<<<num_blocks, SCAN_BLOCK_SIZE>>>(d_in, d_out, d_block_sums, n);

    if (num_blocks > 1) {
        launch_scan_blelloch(d_block_sums, d_block_sums, num_blocks);
        add_block_sums_kernel<<<num_blocks, SCAN_BLOCK_SIZE>>>(d_out, d_block_sums, n);
        cudaFree(d_block_sums);
    }
}

bool is_blelloch_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 3: Blelloch Scan with Bank Conflict Avoidance
// ==============================================================================
// Use this macro to compute padded indices in shared memory!
#define CONFLICT_FREE_OFFSET(n) ((n) >> 5)

__global__ void blelloch_padded_kernel(const float* in, float* out, float* block_sums, int n) {
    // Shared memory with padding to prevent 32-bank conflicts
    __shared__ float s_val[SCAN_BLOCK_SIZE * 2 + ((SCAN_BLOCK_SIZE * 2) >> 5)];

    int tid1 = threadIdx.x; 
    int tid2 = tid1 + SCAN_BLOCK_SIZE;
    int global_idx1 = blockIdx.x * blockDim.x * 2 + tid1;
    int global_idx2 = blockIdx.x * blockDim.x * 2 + tid2;

    int tid1_padded = tid1 + CONFLICT_FREE_OFFSET(tid1);
    int tid2_padded = tid2 + CONFLICT_FREE_OFFSET(tid2);

    if (global_idx1 < n) s_val[tid1_padded] = in[global_idx1];
    else s_val[tid1_padded] = 0.0f;
    if (global_idx2 < n) s_val[tid2_padded] = in[global_idx2];
    else s_val[tid2_padded] = 0.0f;
    __syncthreads();

    int offset = 1;

    // Up-Sweep (Reduction) phase
    for (int d = SCAN_BLOCK_SIZE; d > 0; d >>= 1) {
        if (tid1 < d) {
            int ai = offset * (2 * tid1 + 1) - 1;
            int bi = offset * (2 * tid1 + 2) - 1;
            int ai_padded = ai + CONFLICT_FREE_OFFSET(ai);
            int bi_padded = bi + CONFLICT_FREE_OFFSET(bi);
            s_val[bi_padded] += s_val[ai_padded];
        }
        offset *= 2;
        __syncthreads();
    }

    // Save block sum and clear the root to 0
    int root = SCAN_BLOCK_SIZE * 2 - 1;
    int root_padded = root + CONFLICT_FREE_OFFSET(root);
    if (tid1 == 0) {
        if (block_sums != nullptr) {
            block_sums[blockIdx.x] = s_val[root_padded];
        }
        s_val[root_padded] = 0.0f;
    }
    __syncthreads();

    // Down-Sweep phase
    for (int d = 1; d <= SCAN_BLOCK_SIZE; d *= 2) {
        offset >>= 1;
        if (tid1 < d) {
            int ai = offset * (2 * tid1 + 1) - 1;
            int bi = offset * (2 * tid1 + 2) - 1;
            int ai_padded = ai + CONFLICT_FREE_OFFSET(ai);
            int bi_padded = bi + CONFLICT_FREE_OFFSET(bi);
            float t = s_val[ai_padded];
            s_val[ai_padded] = s_val[bi_padded];
            s_val[bi_padded] += t;
        }
        __syncthreads();
    }

    // Write results to out
    if (global_idx1 < n) out[global_idx1] = s_val[tid1_padded];
    if (global_idx2 < n) out[global_idx2] = s_val[tid2_padded];
}

void launch_scan_blelloch_padded(const float* d_in, float* d_out, int n) {
    int num_blocks = (n + SCAN_BLOCK_SIZE * 2 - 1) / (SCAN_BLOCK_SIZE * 2);
    
    float* d_block_sums = nullptr;
    if (num_blocks > 1) {
        cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
    }

    blelloch_padded_kernel<<<num_blocks, SCAN_BLOCK_SIZE>>>(d_in, d_out, d_block_sums, n);

    if (num_blocks > 1) {
        launch_scan_blelloch_padded(d_block_sums, d_block_sums, num_blocks);
        add_block_sums_kernel<<<num_blocks, SCAN_BLOCK_SIZE>>>(d_out, d_block_sums, n);
        cudaFree(d_block_sums);
    }
}

bool is_blelloch_padded_implemented() {
    return true;
}
