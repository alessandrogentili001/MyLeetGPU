#include "kernel.cuh"
#include "cuda_utils.cuh"

// ==============================================================================
// MILESTONE 1: Naive Matrix Transpose (Coalesced Read, Strided Write)
// ==============================================================================
// Input matrix:  rows x cols  (M x N)
// Output matrix: cols x rows  (N x M)
__global__ void matrix_transpose_naive_kernel(const float* in, float* out, int rows, int cols) {
    // Global 2D thread coordinates
    int c = blockIdx.x * blockDim.x + threadIdx.x; // column in input [0, cols)
    int r = blockIdx.y * blockDim.y + threadIdx.y; // row in input    [0, rows)

    // Boundary check for non-multiple matrix dimensions
    if (r < rows && c < cols) {
        // Coalesced Read: Adjacent threads in a warp (tx = 0..31) access consecutive 'c'
        // -> in[r * cols + c] accesses consecutive 4-byte words -> 100% coalesced read.
        //
        // Strided Write: Adjacent threads write to out[c * rows + r]
        // -> thread tx writes to (c + tx) * rows + r. Address stride is 'rows'!
        // -> High global memory transaction overhead and severe bandwidth penalty.
        out[c * rows + r] = in[r * cols + c];
    }
}

void launch_matrix_transpose_naive(const float* d_in, float* d_out, int rows, int cols) {
    dim3 block(32, 32);
    dim3 grid((cols + block.x - 1) / block.x, 
              (rows + block.y - 1) / block.y);

    matrix_transpose_naive_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
}

// ==============================================================================
// MILESTONE 2: Shared Memory Tiling (Coalesced Access, but Bank Conflicts)
// ==============================================================================
// Uses a 2D tile in shared memory to stage data so BOTH global reads and writes
// are 100% coalesced. However, reading columns from tile[tx][ty] causes a
// 32-way shared memory bank conflict!
__global__ void matrix_transpose_shared_conflict_kernel(const float* in, float* out, int rows, int cols) {
    // TODO: 1. Allocate 2D shared memory tile of size [TILE_DIM][TILE_DIM]
    //          __shared__ float tile[TILE_DIM][TILE_DIM];
    //
    // TODO: 2. Calculate input tile coordinates and load data into shared memory coalesced:
    //          int in_c = blockIdx.x * TILE_DIM + threadIdx.x;
    //          int in_r = blockIdx.y * TILE_DIM + threadIdx.y;
    //          if (in_r < rows && in_c < cols) {
    //              tile[threadIdx.y][threadIdx.x] = in[in_r * cols + in_c];
    //          }
    //
    // TODO: 3. Synchronize all threads in the block:
    //          __syncthreads();
    //
    // TODO: 4. Calculate output tile coordinates (transposed block positions):
    //          Note: blockIdx.x corresponds to input columns, which map to output rows!
    //          Note: blockIdx.y corresponds to input rows, which map to output columns!
    //          int out_c = blockIdx.y * TILE_DIM + threadIdx.x; // column in output matrix
    //          int out_r = blockIdx.x * TILE_DIM + threadIdx.y; // row in output matrix
    //
    // TODO: 5. Write coalesced from shared memory to global memory:
    //          Notice we read tile[threadIdx.x][threadIdx.y]!
    //          if (out_r < cols && out_c < rows) {
    //              out[out_r * rows + out_c] = tile[threadIdx.x][threadIdx.y];
    //          }

    __shared__ float tile[TILE_DIM][TILE_DIM];

    int in_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int in_col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (in_row < rows && in_col < cols) {
        tile[threadIdx.y][threadIdx.x] = in[in_row * cols + in_col];
    }
    __syncthreads();

    int out_row = blockIdx.x * TILE_DIM + threadIdx.y; // Swapping blockIdx.x and
    int out_col = blockIdx.y * TILE_DIM + threadIdx.x; // Swapping blockIdx.y

    if (out_row < cols && out_col < rows) {
        out[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
    }
}

void launch_matrix_transpose_shared_conflict(const float* d_in, float* d_out, int rows, int cols) {
    // TODO: Configure block (TILE_DIM, TILE_DIM) and grid dimensions, then launch kernel
    dim3 block (TILE_DIM, TILE_DIM);
    dim3 grid ( (cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM); // Swapping rows and cols
    // Each block is managing a tile of 32x32 elements
    matrix_transpose_shared_conflict_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
}

// Set to true once you implement Milestone 2!
bool is_shared_conflict_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 3: Bank-Conflict-Free Shared Memory Tiling (Padded [32][33])
// ==============================================================================
// By padding the shared memory row dimension by +1 float:
// __shared__ float tile[TILE_DIM][TILE_DIM + 1];
// Consecutive elements in a column now map to distinct banks (stride 33 instead of 32)!
// (address % 32) cycles cleanly through all 32 banks with zero bank conflicts.
__global__ void matrix_transpose_shared_padded_kernel(const float* in, float* out, int rows, int cols) {
    // TODO: 1. Allocate padded 2D shared memory tile:
    //          __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    //
    // TODO: 2. Read input coalesced into padded tile:
    //          tile[threadIdx.y][threadIdx.x] = in[...];
    //
    // TODO: 3. Synchronize threads:
    //          __syncthreads();
    //
    // TODO: 4. Write output coalesced from padded tile without bank conflicts:
    //          out[...] = tile[threadIdx.x][threadIdx.y];

    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Padding ensures bank conflict free SRAM access 
    
    int in_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int in_col = blockIdx.x * TILE_DIM + threadIdx.x;

    if (in_row < rows && in_col < cols) {
        tile[threadIdx.y][threadIdx.x] = in[in_row * cols + in_col];
    }
    __syncthreads();

    int out_row = blockIdx.x * TILE_DIM + threadIdx.y; // Swapping blockIdx.x and
    int out_col = blockIdx.y * TILE_DIM + threadIdx.x; // Swapping blockIdx.y

    if (out_row < cols && out_col < rows) {
        out[out_row * rows + out_col] = tile[threadIdx.x][threadIdx.y];
    }

}

void launch_matrix_transpose_shared_padded(const float* d_in, float* d_out, int rows, int cols) {
    // TODO: Configure block and grid dimensions, then launch kernel
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid( (cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM); // Swapping rows and cols
    matrix_transpose_shared_padded_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
}

// Set to true once you implement Milestone 3!
bool is_shared_padded_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 4 (BONUS): Coarse-Grained Tiling (32x8 threads per 32x32 tile)
// ==============================================================================
// Instead of 1024 threads per block, use 256 threads (blockDim = 32 x 8).
// Each thread processes 4 elements (TILE_DIM / BLOCK_ROWS) using a loop,
// reducing block scheduling overhead and increasing register reuse & ILP.
__global__ void matrix_transpose_coarse_kernel(const float* in, float* out, int rows, int cols) {
    // TODO: Implement coarse-grained transpose with padded shared memory

    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // Padding ensures bank conflict free SRAM access 
    
    int in_row = blockIdx.y * TILE_DIM + threadIdx.y;
    int in_col = blockIdx.x * TILE_DIM + threadIdx.x;

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS){
        int r = in_row + i; // Ensure coarse grained access 
        if (r < rows && in_col < cols) {
            tile[threadIdx.y + i][threadIdx.x] = in[r * cols + in_col];
        }
    }
    __syncthreads();

    int out_row = blockIdx.x * TILE_DIM + threadIdx.y; // Swapping blockIdx.x and
    int out_col = blockIdx.y * TILE_DIM + threadIdx.x; // Swapping blockIdx.y

    for (int i = 0; i < TILE_DIM; i+=BLOCK_ROWS) {
        int r = out_row + i; // Ensure coarse grained access 
        if (r < cols && out_col < rows) {
            out[r * rows + out_col] = tile[threadIdx.x][threadIdx.y + i];
        }
    }

}

void launch_matrix_transpose_coarse(const float* d_in, float* d_out, int rows, int cols) {
    (void)d_in; (void)d_out; (void)rows; (void)cols;

    // TODO: Configure block and grid dimensions, then launch kernel
    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((cols + TILE_DIM - 1) / TILE_DIM, (rows + TILE_DIM - 1) / TILE_DIM);
    matrix_transpose_coarse_kernel<<<grid, block>>>(d_in, d_out, rows, cols);
}

// Set to true once you implement Milestone 4!
bool is_coarse_implemented() {
    return true;
}
