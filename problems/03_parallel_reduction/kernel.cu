#include "kernel.cuh"
#include "cuda_utils.cuh"

// ==============================================================================
// MILESTONE 1: Interleaved Addressing with Warp Divergence
// ==============================================================================
// In this baseline kernel, stride doubles each iteration: s = 1, 2, 4, 8, ...
// The condition `if (tid % (2 * s) == 0)` causes severe WARP DIVERGENCE because
// threads within the same 32-thread warp evaluate different branch paths!
__global__ void reduction_divergent_kernel(const float* in, float* out, int n) {
    // Each thread within thesame block can access this shared memory
    __shared__ float sdata[REDUCTION_BLOCK_SIZE];
    // Each thread has a unique ID within the block
    int tid = threadIdx.x;

    // Step 1: Grid-stride loop to load and accumulate all elements into shared memory
    float sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) {
        sum += in[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    // Step 2: Interleaved reduction tree in shared memory (DIVERGENT)
    for (int s = 1; s < blockDim.x; s *= 2) {
        // Highly divergent branch:
        // When s = 1: threads 0, 2, 4, 6... active (50% warp divergence)
        // When s = 2: threads 0, 4, 8, 12... active (75% warp divergence)
        // When s = 4: threads 0, 8, 16, 24... active (87.5% warp divergence)
        if ((tid % (2 * s)) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Step 3: Thread 0 writes block partial sum to global accumulator
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

void launch_reduction_divergent(const float* d_in, float* d_out, int n) {
    // Allocate REDUCTION_BLOCK_SIZE threads per block
    int block_size = REDUCTION_BLOCK_SIZE;
    // Number of blocks to cover all elements
    int grid_size = 1024;
    // Launch kernel
    reduction_divergent_kernel<<<grid_size, block_size>>>(d_in, d_out, n);
}

// ==============================================================================
// MILESTONE 2: Interleaved Addressing without Divergence (Strided Indexing)
// ==============================================================================
// We eliminate warp divergence by restructuring the thread index mapping:
// Instead of `if (tid % (2 * s) == 0)`, active threads are clustered into
// consecutive thread IDs: `int index = 2 * s * tid;`
// However, this causes severe SHARED MEMORY BANK CONFLICTS as the stride increases!
__global__ void reduction_interleaved_kernel(const float* in, float* out, int n) {
    // TODO: 1. Allocate shared memory: __shared__ float sdata[REDUCTION_BLOCK_SIZE];
    //
    // TODO: 2. Grid-stride loop to load from global memory into sdata[tid]
    //          __syncthreads();
    //
    // TODO: 3. Interleaved reduction without branch divergence:
    //          for (unsigned int s = 1; s < blockDim.x; s *= 2) {
    //              int index = 2 * s * tid;
    //              if (index < blockDim.x) {
    //                  sdata[index] += sdata[index + s];
    //              }
    //              __syncthreads();
    //          }
    //          Notice: Adjacent threads tid = 0, 1, 2... are ALL active together!
    //          No warp divergence for active warps!
    //          BUT: at s = 16, index = 32 * tid -> all 32 threads access bank 0!
    //          (32-way shared memory bank conflict!).
    //
    // TODO: 4. if (tid == 0) atomicAdd(out, sdata[0]);

    __shared__ float sdata[REDUCTION_BLOCK_SIZE];
    int tid = threadIdx.x;
    
    float sum = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i<n; i += gridDim.x * blockDim.x) {
        sum += in[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        int index = 2 * s * tid; // slide to the next contiguous thread avoiding holes in the same warp
        if (index < blockDim.x) {
            sdata[index] += sdata[index + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

void launch_reduction_interleaved(const float* d_in, float* d_out, int n) {
    // TODO: Configure grid and block dimensions, then launch reduction_interleaved_kernel
    (void)d_in; (void)d_out; (void)n;
    // Allocate REDUCTION_BLOCK_SIZE threads per block
    int block_size = REDUCTION_BLOCK_SIZE;
    // Number of blocks to cover all elements
    int grid_size = 1024;
    // Launch kernel
    reduction_interleaved_kernel<<<grid_size, block_size>>>(d_in, d_out, n);
}

// Set to true once you implement Milestone 2!
bool is_interleaved_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 3: Sequential Addressing (Conflict-Free Shared Memory Reduction)
// ==============================================================================
// Reverse the reduction loop direction: start stride at blockDim.x / 2 and halve!
// for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
//     if (tid < s) sdata[tid] += sdata[tid + s];
//     __syncthreads();
// }
// Adjacent active threads `tid = 0..s-1` access adjacent shared memory locations.
// ZERO bank conflicts!
__global__ void reduction_sequential_kernel(const float* in, float* out, int n) {
    // TODO: 1. Allocate shared memory
    // TODO: 2. Grid-stride loop to load from global memory into sdata[tid]
    //          __syncthreads();
    //
    // TODO: 3. Sequential reduction loop:
    //          for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    //              if (tid < s) {
    //                  sdata[tid] += sdata[tid + s];
    //              }
    //              __syncthreads();
    //          }
    //
    // TODO: 4. if (tid == 0) atomicAdd(out, sdata[0]);

    __shared__ float sdata[REDUCTION_BLOCK_SIZE];
    int tid = threadIdx.x;

    float sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<n; i += gridDim.x * blockDim.x) {
        sum += in[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; // Access contiguous indices in shared memory avoiding bank conflicts
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

void launch_reduction_sequential(const float* d_in, float* d_out, int n) {
    // TODO: Configure grid and block dimensions, then launch reduction_interleaved_kernel
    (void)d_in; (void)d_out; (void)n;
    // Allocate REDUCTION_BLOCK_SIZE threads per block
    int block_size = REDUCTION_BLOCK_SIZE;
    // Number of blocks to cover all elements
    int grid_size = 1024;
    // Launch kernel
    reduction_sequential_kernel<<<grid_size, block_size>>>(d_in, d_out, n);
}

// Set to true once you implement Milestone 3!
bool is_sequential_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 4: Warp Shuffle Reduction (__shfl_down_sync)
// ==============================================================================
// When reducing the final 32 elements (1 warp), bypass shared memory and
// barriers completely by exchanging values across registers with `__shfl_down_sync`!
// Hardware register exchange is virtually instantaneous with zero SRAM traffic.
__device__ inline float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void reduction_warp_shuffle_kernel(const float* in, float* out, int n) {
    // TODO: 1. Grid-stride loop into sdata[tid]
    //
    // TODO: 2. Reduce shared memory down to 32 elements (only 1 warp remains!):
    //          for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    //              if (tid < s) sdata[tid] += sdata[tid + s];
    //              __syncthreads();
    //          }
    //
    // TODO: 3. Warp shuffle for the final 32 elements:
    //          if (tid < 32) {
    //              float val = sdata[tid];
    //              val = warp_reduce_sum(val);
    //              if (tid == 0) atomicAdd(out, val);
    //          }

    __shared__ float sdata[REDUCTION_BLOCK_SIZE];
    int tid = threadIdx.x;

    float sum = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i<n; i += gridDim.x * blockDim.x) {
        sum += in[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) { // stop before 32 to let the independent warp handle the final reduction
        if (tid < s) {
            sdata[tid] += sdata[tid + s]; // Access contiguous indices in shared memory avoiding bank conflicts
        }
        __syncthreads();
    }

    // Warp shuffle for the final 32 elements (no need for __syncthreads() between warps)
    if (tid < 32) {
        // Load and merge the last 32 elements from the previous warp
        float val = sdata[tid] + sdata[tid + 32];
        // Warp shuffle to reduce the sum to a single value
        val = warp_reduce_sum(val);
        // Store final result 
        if (tid == 0) atomicAdd(out, val);
    }
}

void launch_reduction_warp_shuffle(const float* d_in, float* d_out, int n) {
    // TODO: Configure grid and block dimensions, then launch reduction_warp_shuffle_kernel
    // Allocate REDUCTION_BLOCK_SIZE threads per block
    int block_size = REDUCTION_BLOCK_SIZE;
    // Number of blocks to cover all elements
    int grid_size = 1024;
    // Launch kernel
    reduction_warp_shuffle_kernel<<<grid_size, block_size>>>(d_in, d_out, n);
}

// Set to true once you implement Milestone 4!
bool is_warp_shuffle_implemented() {
    return true;
}
