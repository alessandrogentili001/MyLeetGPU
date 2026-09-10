#include "kernel.cuh"
#include "cuda_utils.cuh"

// ==============================================================================
// MILESTONE 1: Naive (1 Thread per Element)
// ==============================================================================
__global__ void vector_add_naive_kernel(const float* a, const float* b, float* c, int n) {
    // TODO: 1. Calculate the global 1D thread index (idx)
    // TODO: 2. Boundary check: ensure idx < n
    // TODO: 3. Compute element-wise addition: c[idx] = a[idx] + b[idx]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_vector_add_naive(const float* d_a, const float* d_b, float* d_c, int n) {
    // TODO: Configure block and grid dimensions, then launch the kernel
    int block_size = 1024;
    int grid_size = (n + block_size - 1) / block_size;
    vector_add_naive_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
}

// ==============================================================================
// MILESTONE 2: Grid-Stride Loop
// ==============================================================================
__global__ void vector_add_grid_stride_kernel(const float* a, const float* b, float* c, int n) {
    // TODO: Write a grid-stride loop where threads stride by: gridDim.x * blockDim.x
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n; 
         idx += gridDim.x * blockDim.x) {
        c[idx] = a[idx] + b[idx];
    }
}

void launch_vector_add_grid_stride(const float* d_a, const float* d_b, float* d_c, int n) {
    // TODO: Choose block_size and a fixed grid_size (e.g. multiples of SM count = 124 for Leonardo Booster A100 64G)
    // and launch vector_add_grid_stride_kernel
    int block_size = 1024;
    int grid_size = 124 * 4;
    vector_add_grid_stride_kernel<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
}

// Set to true once you have implemented Milestone 2!
bool is_grid_stride_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 3: Vectorized Memory Access (float4 / 128-bit loads)
// ==============================================================================
__global__ void vector_add_vectorized_kernel(const float4* a, const float4* b, float4* c, int n4) {
    // TODO: Load float4 vectors, add component-wise (x, y, z, w), and store to c
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
        idx < n4; 
        idx += gridDim.x * blockDim.x) {
        // Load float4 vectors
        float4 va = a[idx];
        float4 vb = b[idx];
        // Add component wise
        float4 vc = make_float4(va.x + vb.x, va.y + vb.y, va.z + vb.z, va.w + vb.w);
        // Store result
        c[idx] = vc;
    }
}

void launch_vector_add_vectorized(const float* d_a, const float* d_b, float* d_c, int n) {
    // TODO: Launch vector_add_vectorized_kernel for n / 4 elements
    // TODO: Handle any remaining elements (n % 4) with a scalar tail
    int block_size = 1024;
    int grid_size = 124 * 4;
    int n4 = n/4;
    const float4* d_a4 = reinterpret_cast<const float4*>(d_a);
    const float4* d_b4 = reinterpret_cast<const float4*>(d_b);
    float4* d_c4 = reinterpret_cast<float4*>(d_c);
    
    vector_add_vectorized_kernel<<<grid_size, block_size>>>(d_a4, d_b4, d_c4, n4);

    // Scalar tail
    int remainder = n - n4 * 4;
    if (remainder > 0) {
        int offset = n4 * 4;
        // Launch 1 block with 'remainder' threads for the remaining 1 to 3 elements
        vector_add_naive_kernel<<<1, remainder>>>(d_a + offset, d_b + offset, d_c + offset, remainder);
    }
}

// Set to true once you have implemented Milestone 3!
bool is_vectorized_implemented() {
    return true;
}
