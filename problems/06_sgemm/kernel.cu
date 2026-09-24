#include "kernel.cuh"
#include "cuda_utils.cuh"

// ==============================================================================
// MILESTONE 1: Naive SGEMM (Global Memory)
// ==============================================================================
// Every thread computes one element of C.
// C[i, j] = dot(A[i, :], B[:, j])
// Reads from global memory are not fully coalesced and heavily redundant.
// ==============================================================================
__global__ void sgemm_naive_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: Implement naive SGEMM
}

void launch_sgemm_naive(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    // TODO: Configure grid/block and launch naive kernel
}

bool is_naive_implemented() {
    return false;
}

// ==============================================================================
// MILESTONE 2: Shared Memory Tiling (Cache Blocking)
// ==============================================================================
// Use shared memory to cache tiles of A and B.
// This reduces global memory bandwidth by a factor of TILE_SIZE.
// ==============================================================================
__global__ void sgemm_shared_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: Allocate shared memory tiles
    // __shared__ float sA[TILE_SIZE][TILE_SIZE];
    // __shared__ float sB[TILE_SIZE][TILE_SIZE];
    
    // TODO: Loop over tiles along the K dimension
        // TODO: Load data into shared memory cooperatively
        // __syncthreads();
        // TODO: Compute dot product for the current tile
        // __syncthreads();
    
    // TODO: Write result to C
}

void launch_sgemm_shared_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    // TODO: Configure grid/block and launch shared memory tiled kernel
}

bool is_shared_tiled_implemented() {
    return false;
}

// ==============================================================================
// MILESTONE 3: 2D Register Tiling (Thread Coarsening)
// ==============================================================================
// Each thread computes a TM x TN tile of C.
// This dramatically increases the arithmetic intensity (FLOPs per byte) 
// by keeping accumulators and fetched elements in extremely fast registers.
// ==============================================================================
__global__ void sgemm_2d_register_tiled_kernel(const float* A, const float* B, float* C, int M, int N, int K) {
    // TODO: Implement 2D Register Tiling
    // BM = 64, BN = 64, BK = 8, TM = 8, TN = 8
    // Thread block: (BM/TM) x (BN/TN) = 8 x 8 = 64 threads.
    
    // TODO: Allocate shared memory for A and B
    // __shared__ float sA[BM][BK];
    // __shared__ float sB[BK][BN];
    
    // TODO: Allocate thread-local registers for accumulators and fetched data
    // float accum[TM][TN] = {0.0f};
    // float regA[TM];
    // float regB[TN];
    
    // TODO: Loop over the K dimension in blocks of BK
        // TODO: Load into shared memory (with bounds checking and padding)
        // __syncthreads();
        // TODO: Compute outer products into accumulators (loop over BK)
        // __syncthreads();
    
    // TODO: Write accumulators to C (with bounds checking)
}

void launch_sgemm_2d_register_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    // TODO: Configure grid/block and launch 2D register tiled kernel
}

bool is_2d_register_tiled_implemented() {
    return false;
}

// ==============================================================================
// MILESTONE 4: cuBLAS Baseline (Provided)
// ==============================================================================
// Uses NVIDIA's highly optimized cuBLAS library.
// Since cuBLAS uses column-major by default and C/C++ uses row-major,
// we compute C^T = B^T * A^T to get the row-major C = A * B.
// ==============================================================================
void launch_sgemm_cublas(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    // cuBLAS is column-major. To perform row-major C = A * B:
    // C^T = B^T * A^T
    // lda, ldb, ldc are leading dimensions. For row-major:
    // lda = K, ldb = N, ldc = N
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                d_B, N, // B^T
                d_A, K, // A^T
                &beta,
                d_C, N); // C^T
}
