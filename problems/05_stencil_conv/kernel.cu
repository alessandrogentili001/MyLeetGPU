#include "kernel.cuh"
#include "cuda_utils.cuh"

// Constant memory allocation for the 2D filter mask (5x5 = 25 floats = 100 bytes)
// Constant memory has a dedicated 64 KB cache per SM. When all threads in a warp
// read the same mask element simultaneously, hardware performs a single-cycle broadcast!
__constant__ float c_mask[KERNEL_SIZE];

// ==============================================================================
// MILESTONE 1: Naive 2D Convolution (Global Memory for Input & Mask) - PROVIDED
// ==============================================================================
// Every thread computes one output pixel (r, c).
// For a 5x5 filter, each thread reads 25 pixels directly from global DRAM.
// Redundant global memory loads: each input pixel is read up to 25 times!
// ==============================================================================
__global__ void conv2d_naive_kernel(const float* in, const float* mask, float* out, 
                                   int height, int width) {
    int c = blockIdx.x * blockDim.x + threadIdx.x; // column
    int r = blockIdx.y * blockDim.y + threadIdx.y; // row

    if (r >= height || c >= width) return;

    float sum = 0.0f;
    for (int kr = -KERNEL_RADIUS; kr <= KERNEL_RADIUS; ++kr) {
        for (int kc = -KERNEL_RADIUS; kc <= KERNEL_RADIUS; ++kc) {
            int in_r = r + kr;
            int in_c = c + kc;
            // Zero-padding boundary check
            if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
                float pixel = in[in_r * width + in_c];
                float weight = mask[(kr + KERNEL_RADIUS) * KERNEL_DIAMETER + (kc + KERNEL_RADIUS)];
                sum += pixel * weight;
            }
        }
    }
    out[r * width + c] = sum;
}

void launch_conv2d_naive(const float* d_in, const float* d_mask, float* d_out, int height, int width) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);

    conv2d_naive_kernel<<<grid, block>>>(d_in, d_mask, d_out, height, width);
}

// ==============================================================================
// MILESTONE 2: Constant Memory Filter Mask
// ==============================================================================
// 🎯 YOUR TASK:
// 1. Copy `h_mask` to device constant memory symbol `c_mask` using `cudaMemcpyToSymbol`.
// 2. In `conv2d_constant_mask_kernel`, read the filter weights from `c_mask[...]`
//    instead of global memory.
// 3. Keep input pixel loads from `in` (global memory) with zero-padding check.
// 4. Return `true` in `is_constant_mask_implemented()`.
// ==============================================================================
__global__ void conv2d_constant_mask_kernel(const float* in, float* out, int height, int width) {
    // TODO: Implement constant memory stencil kernel
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r >= height || c >= width) return;

    // TODO: Loop through neighborhood [-KERNEL_RADIUS, KERNEL_RADIUS],
    // fetch weights from `c_mask`, and accumulate sum.
    float sum = 0.0f;
    for (int kr = -KERNEL_RADIUS; kr <= KERNEL_RADIUS; ++kr) {
        for (int kc = -KERNEL_RADIUS; kc <= KERNEL_RADIUS; ++kc) {
            int in_r = r + kr;
            int in_c = c + kc;
            // Zero-padding boundary check
            if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
                float pixel = in[in_r * width + in_c];
                float weight = c_mask[(kr + KERNEL_RADIUS) * KERNEL_DIAMETER + (kc + KERNEL_RADIUS)];
                sum += pixel * weight;
            }
        }
    }
    out[r * width + c] = sum;
}

void launch_conv2d_constant_mask(const float* d_in, const float* h_mask, float* d_out, int height, int width) {
    (void)d_in; (void)h_mask; (void)d_out; (void)height; (void)width;

    // TODO:
    // 1. CUDA_CHECK(cudaMemcpyToSymbol(c_mask, h_mask, KERNEL_SIZE * sizeof(float)));
    // 2. Configure grid and block dimensions (e.g., dim3 block(16, 16))
    // 3. Launch conv2d_constant_mask_kernel<<<grid, block>>>(d_in, d_out, height, width);
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, h_mask, KERNEL_SIZE * sizeof(float)));
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    conv2d_constant_mask_kernel<<<grid, block>>>(d_in, d_out, height, width);
}

// Set to true once you implement Milestone 2!
bool is_constant_mask_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 3: Shared Memory Apron Tiling (Cooperative Halo Loading)
// ==============================================================================
// 🎯 YOUR TASK:
// For a 16x16 output tile and radius R = 2, load an apron of size 20x20 into shared memory.
//
// 1. Allocate shared memory: `__shared__ float s_in[APRON_DIM][APRON_DIM];` (20x20 = 400 floats)
// 2. Cooperative Loading:
//    Have the 256 threads in the block cooperatively load the 400 elements:
//    - Map 1D index `i` (from tid to 400, step 256) to 2D shared coords (s_r, s_c).
//    - Compute global coords: `g_r = top_left_r + s_r`, `g_c = top_left_c + s_c`.
//    - Check bounds; store `in[g_r * width + g_c]` if in bounds, else `0.0f`.
// 3. Synchronize: `__syncthreads();`
// 4. Compute convolution entirely from `s_in` and `c_mask` (zero boundary checks needed!).
// 5. Write to `out[out_r * width + out_c]`.
// 6. Return `true` in `is_shared_tiled_implemented()`.
// ==============================================================================
__global__ void conv2d_shared_tiled_kernel(const float* in, float* out, int height, int width) {
    // TODO: Allocate 2D shared memory apron
    // __shared__ float s_in[APRON_DIM][APRON_DIM];
    // TODO: Step 1: Cooperative Apron Loading with zero-padding
    // TODO: Step 2: __syncthreads();
    // TODO: Step 3: Compute convolution from shared memory and write output
    
    // Allocate shared memory
    __shared__ float s_in[APRON_DIM][APRON_DIM];

    // Global row and column
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    // Same reference corner for all threads in a block
    int top_left_r = blockIdx.y * blockDim.y - KERNEL_RADIUS; 
    int top_left_c = blockIdx.x * blockDim.x - KERNEL_RADIUS;
    // Cooperative Apron loading
    int tid = threadIdx.y * blockDim.x + threadIdx.x; // 0 .. 255
    for (int idx = tid; idx < APRON_DIM * APRON_DIM; idx += blockDim.x * blockDim.y) {
        int s_r = idx / APRON_DIM;
        int s_c = idx % APRON_DIM;
        int g_r = top_left_r + s_r;
        int g_c = top_left_c + s_c;
        s_in[s_r][s_c] = (g_r >= 0 && g_r < height && g_c >= 0 && g_c < width) ? in[g_r * width + g_c] : 0.0f;
    }
    __syncthreads();

    if (r >= height || c >= width) return;

    // Compute convolution from shared memory and write output
    float sum = 0.0f;
    for (int kr = -KERNEL_RADIUS; kr <= KERNEL_RADIUS; ++kr) {
        for (int kc = -KERNEL_RADIUS; kc <= KERNEL_RADIUS; ++kc) {
            float pixel = s_in[threadIdx.y + kr + KERNEL_RADIUS][threadIdx.x + kc + KERNEL_RADIUS];
            float weight = c_mask[(kr + KERNEL_RADIUS) * KERNEL_DIAMETER + (kc + KERNEL_RADIUS)];
            sum += pixel * weight;
        }
    }
    out[r * width + c] = sum;
}

void launch_conv2d_shared_tiled(const float* d_in, const float* h_mask, float* d_out, int height, int width) {
    (void)d_in; (void)h_mask; (void)d_out; (void)height; (void)width;

    // TODO:
    // 1. Copy mask to c_mask
    // 2. dim3 block(TILE_DIM, TILE_DIM);
    // 3. dim3 grid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    // 4. conv2d_shared_tiled_kernel<<<grid, block>>>(d_in, d_out, height, width);
    
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, h_mask, KERNEL_SIZE * sizeof(float)));
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    conv2d_shared_tiled_kernel<<<grid, block>>>(d_in, d_out, height, width);
}

// Set to true once you implement Milestone 3!
bool is_shared_tiled_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 4: Modern Ampere Read-Only Cache Streaming (const __restrict__)
// ==============================================================================
// 🎯 YOUR TASK:
// On Ampere (sm_80), investigate modern hardware caching:
// 1. Tag pointer with `const float* __restrict__ in`.
// 2. Loop over [-KERNEL_RADIUS, KERNEL_RADIUS] with `#pragma unroll`.
// 3. Observe whether Ampere's unified 192 KB L1/L2 cache matches or beats manual
//    shared memory apron loading!
// 4. Return `true` in `is_readonly_cached_implemented()`.
// ==============================================================================
__global__ void conv2d_readonly_cached_kernel(const float* __restrict__ in, 
                                             float* __restrict__ out, 
                                             int height, int width) {
    // TODO: Implement read-only cached convolution kernel
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    const float* __restrict__ in_ptr = in; // Tag pointer as read-only, allowing hardware prefetching
    if (r >= height || c >= width) return;
    float sum = 0.0f;
    for (int kr = -KERNEL_RADIUS; kr <= KERNEL_RADIUS; ++kr) {
        #pragma unroll
        for (int kc = -KERNEL_RADIUS; kc <= KERNEL_RADIUS; ++kc) {
            int in_r = r + kr;
            int in_c = c + kc;
            // Zero-padding boundary check
            if (in_r >= 0 && in_r < height && in_c >= 0 && in_c < width) {
                float pixel = in_ptr[in_r * width + in_c];
                float weight = c_mask[(kr + KERNEL_RADIUS) * KERNEL_DIAMETER + (kc + KERNEL_RADIUS)];
                sum += pixel * weight;
            }
        }
    }
    out[r * width + c] = sum;
}

void launch_conv2d_readonly_cached(const float* d_in, const float* h_mask, float* d_out, int height, int width) {
    (void)d_in; (void)h_mask; (void)d_out; (void)height; (void)width;

    // TODO: Copy mask and launch conv2d_readonly_cached_kernel
    CUDA_CHECK(cudaMemcpyToSymbol(c_mask, h_mask, KERNEL_SIZE * sizeof(float)));
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    conv2d_readonly_cached_kernel<<<grid, block>>>(d_in, d_out, height, width);
}

// Set to true once you implement Milestone 4!
bool is_readonly_cached_implemented() {
    return true;
}
