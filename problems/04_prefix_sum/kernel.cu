#include "kernel.cuh"

// ==============================================================================
// Helper Kernel: Add block sums to the final output (Provided)
// ==============================================================================
__global__ void add_block_sums_kernel(float* d_out, const float* d_block_sums, int n) {
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    
    if (blockIdx.x == 0) return; // First block doesn't need addition

    float block_sum = d_block_sums[blockIdx.x - 1]; // Exclusive scan of block sums

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
    return false;
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
    return false;
}

// ==============================================================================
// MILESTONE 3: Blelloch Scan with Bank Conflict Avoidance
// ==============================================================================
// Use this macro to compute padded indices in shared memory!
#define CONFLICT_FREE_OFFSET(n) ((n) >> 5)

__global__ void blelloch_padded_kernel(const float* in, float* out, float* block_sums, int n) {
    // TODO: Implement the padded Blelloch exclusive scan.
    // Use `int ai_padded = ai + CONFLICT_FREE_OFFSET(ai);` when accessing shared memory.
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
    return false;
}
