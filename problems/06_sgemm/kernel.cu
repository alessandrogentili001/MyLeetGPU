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
    int col = blockIdx.x * blockDim.x + threadIdx.x; // j in [0, N)
    int row = blockIdx.y * blockDim.y + threadIdx.y; // i in [0, M)
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void launch_sgemm_naive(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    // TODO: Configure grid/block and launch naive kernel
    // 16x16 (256 threads) or 32x32 (1024 threads, max block size)
    dim3 blockDim(32, 32);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);
    sgemm_naive_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

bool is_naive_implemented() {
    return true;
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

    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    float sum = 0.0f;

    for (int tile = 0; tile < numTiles; ++tile) {
        int start = tile * TILE_SIZE;

        // Load A into sA with boundary check (pad with 0.0f if out of bounds)
        if (row < M && (start + threadIdx.x) < K) {
            sA[threadIdx.y][threadIdx.x] = A[row * K + start + threadIdx.x];
        } else {
            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load B into sB with boundary check (pad with 0.0f if out of bounds)
        if ((start + threadIdx.y) < K && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(start + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }
        __syncthreads();

        // Compute dot product for the current tile
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Write result to C with boundary check
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

void launch_sgemm_shared_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE,
                 (M + TILE_SIZE - 1) / TILE_SIZE);
    sgemm_shared_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

bool is_shared_tiled_implemented() {
    return true;
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

    __shared__ float sA[BM][BK];
    __shared__ float sB[BK][BN];

    int tid = threadIdx.y * blockDim.x + threadIdx.x; // linear thread ID [0, 63]

    // Accumulators for TM x TN sub-tile stored in registers
    float accum[TM][TN] = {0.0f};
    float regA[TM];
    float regB[TN];

    int numTiles = (K + BK - 1) / BK;

    constexpr int threads_per_block = (BM / TM) * (BN / TN); // 64
    constexpr int loads_a = (BM * BK) / threads_per_block;   // 8
    constexpr int loads_b = (BK * BN) / threads_per_block;   // 8

    for (int tile = 0; tile < numTiles; ++tile) {
        // Cooperatively load sA [BM][BK]
        #pragma unroll
        for (int l = 0; l < loads_a; ++l) {
            int idx_a = tid + l * threads_per_block;
            int r_a = idx_a / BK;
            int c_a = idx_a % BK;
            int global_r_a = blockIdx.y * BM + r_a;
            int global_c_a = tile * BK + c_a;

            if (global_r_a < M && global_c_a < K) {
                sA[r_a][c_a] = A[global_r_a * K + global_c_a];
            } else {
                sA[r_a][c_a] = 0.0f;
            }
        }

        // Cooperatively load sB [BK][BN]
        #pragma unroll
        for (int l = 0; l < loads_b; ++l) {
            int idx_b = tid + l * threads_per_block;
            int r_b = idx_b / BN;
            int c_b = idx_b % BN;
            int global_r_b = tile * BK + r_b;
            int global_c_b = blockIdx.x * BN + c_b;

            if (global_r_b < K && global_c_b < N) {
                sB[r_b][c_b] = B[global_r_b * N + global_c_b];
            } else {
                sB[r_b][c_b] = 0.0f;
            }
        }

        __syncthreads();

        // Compute outer products in registers along BK
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                regA[i] = sA[threadIdx.y * TM + i][k];
            }

            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                regB[j] = sB[k][threadIdx.x * TN + j];
            }

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    accum[i][j] += regA[i] * regB[j];
                }
            }
        }

        __syncthreads();
    }

    // Write accumulators to C with boundary checking
    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            int r = blockIdx.y * BM + threadIdx.y * TM + i;
            int c = blockIdx.x * BN + threadIdx.x * TN + j;
            if (r < M && c < N) {
                C[r * N + c] = accum[i][j];
            }
        }
    }
}

void launch_sgemm_2d_register_tiled(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    dim3 blockDim(BN / TN, BM / TM);
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    sgemm_2d_register_tiled_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, N, K);
}

bool is_2d_register_tiled_implemented() {
    return true;
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
