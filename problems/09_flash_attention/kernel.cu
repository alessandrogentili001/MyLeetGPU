#include "kernel.cuh"
#include <cfloat>
#include <cmath>

// ==============================================================================
// MILESTONE 1: Standard Attention (Global Memory Baseline)
// ==============================================================================
// Materializes the full N x N attention matrix S in global memory:
// 1. Compute logits: S = scale * (Q @ K^T)  [N x N]
// 2. Row-wise Softmax: P = softmax(S)       [N x N]
// 3. Output projection: O = P @ V           [N x d]
//
// Memory Complexity: O(N^2) DRAM traffic. For large N, memory traffic
// quickly overwhelms the GPU memory bus.
// ==============================================================================

__global__ void attention_naive_kernel(const float* __restrict__ Q,
                                       const float* __restrict__ K,
                                       const float* __restrict__ V,
                                       float* __restrict__ O,
                                       float* __restrict__ S,
                                       int seq_len, int d,
                                       float scale, bool is_causal) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= seq_len) return;

    float* S_row = S + static_cast<size_t>(i) * seq_len;
    const float* q_vec = Q + i * d;

    // Cache Q[i] if head dimension is reasonably sized
    float q_reg[128];
    if (d <= 128) {
        for (int k = 0; k < d; ++k) {
            q_reg[k] = q_vec[k];
        }
    }

    int max_j = is_causal ? (i + 1) : seq_len;

    // 1. Compute logits against all K rows: S[j] = scale * dot(Q[i], K[j])
    // 2. Find row max m_i = max_j S[j] (apply causal mask: j > i -> -inf)
    float m_i = -INFINITY;
    for (int j = 0; j < seq_len; ++j) {
        if (is_causal && j > i) {
            S_row[j] = -INFINITY;
        } else {
            const float* k_vec = K + j * d;
            float dot = 0.0f;
            if (d <= 128) {
                for (int k = 0; k < d; ++k) {
                    dot += q_reg[k] * k_vec[k];
                }
            } else {
                for (int k = 0; k < d; ++k) {
                    dot += q_vec[k] * k_vec[k];
                }
            }
            float val = dot * scale;
            S_row[j] = val;
            if (val > m_i) {
                m_i = val;
            }
        }
    }

    // 3. Compute sum of exp: l_i = sum_j exp(S[j] - m_i)
    float l_i = 0.0f;
    for (int j = 0; j < max_j; ++j) {
        l_i += expf(S_row[j] - m_i);
    }
    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;

    // 4. Compute P[j] = exp(S[j] - m_i) / l_i and store in S_row
    for (int j = 0; j < max_j; ++j) {
        S_row[j] = expf(S_row[j] - m_i) * inv_l;
    }
    if (is_causal) {
        for (int j = max_j; j < seq_len; ++j) {
            S_row[j] = 0.0f;
        }
    }

    // 5. Accumulate O[i, k] = sum_j (P[j] * V[j, k])
    float* o_vec = O + i * d;
    for (int k = 0; k < d; ++k) {
        float sum = 0.0f;
        for (int j = 0; j < max_j; ++j) {
            sum += S_row[j] * V[j * d + k];
        }
        o_vec[k] = sum;
    }
}

void launch_attention_naive(const float* d_Q, const float* d_K, const float* d_V,
                            float* d_O, int seq_len, int d, float scale, bool is_causal) {
    static float* d_S = nullptr;
    static size_t d_S_capacity = 0;

    size_t required = static_cast<size_t>(seq_len) * seq_len;
    if (required > d_S_capacity) {
        if (d_S) {
            cudaFree(d_S);
        }
        cudaMalloc(&d_S, required * sizeof(float));
        d_S_capacity = required;
    }

    dim3 block(128);
    dim3 grid((seq_len + block.x - 1) / block.x);
    attention_naive_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, d_S, seq_len, d, scale, is_causal);
}

bool is_attention_naive_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 2: Tiled FlashAttention (SRAM Tiling + Online Softmax)
// ==============================================================================
// Key Idea (Dao et al., 2022):
// Never write the N x N attention matrix S to DRAM!
// Load tiles of Q (Br x d) and tiles of K, V (Bc x d) into fast __shared__ memory (SRAM).
//
// Use the Online Softmax recurrence to maintain running row max m_i and running sum l_i:
//   m_new = max(m_old, m_tile)
//   l_new = l_old * exp(m_old - m_new) + l_tile * exp(m_tile - m_new)
//   O_new = O_old * exp(m_old - m_new) + P_tile @ V_tile
//
// Memory Complexity: O(N) DRAM traffic (only reads Q, K, V and writes O).
// ==============================================================================

__global__ void flash_attention_tiled_kernel(const float* __restrict__ Q,
                                             const float* __restrict__ K,
                                             const float* __restrict__ V,
                                             float* __restrict__ O,
                                             int seq_len, int d,
                                             float scale, bool is_causal) {
    // Allocate shared memory for tiles
    __shared__ float s_Q[TILE_BR][HEAD_DIM];
    __shared__ float s_K[TILE_BC][HEAD_DIM];
    __shared__ float s_V[TILE_BC][HEAD_DIM];

    int q_tile_idx = blockIdx.x;
    int q_row_start = q_tile_idx * TILE_BR;
    int tx = threadIdx.x;
    int row_i = q_row_start + tx;

    // 1. Cooperatively load Q tile into s_Q
    if (row_i < seq_len) {
        for (int k = 0; k < d; ++k) {
            s_Q[tx][k] = Q[row_i * d + k];
        }
    } else {
        for (int k = 0; k < d; ++k) {
            s_Q[tx][k] = 0.0f;
        }
    }
    __syncthreads();

    // Running statistics for online softmax
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_accum[HEAD_DIM];
    for (int k = 0; k < HEAD_DIM; ++k) {
        o_accum[k] = 0.0f;
    }

    int num_kv = (seq_len + TILE_BC - 1) / TILE_BC;

    // 2. Loop over K, V blocks
    for (int kv_tile = 0; kv_tile < num_kv; ++kv_tile) {
        int kv_row_start = kv_tile * TILE_BC;

        // Causal skipping: if entire K/V tile is in the future for this Q tile
        if (is_causal && kv_row_start > q_row_start + TILE_BR - 1) {
            break;
        }

        // Cooperatively load K and V tiles into SRAM
        int kv_row = kv_row_start + tx;
        if (kv_row < seq_len) {
            for (int k = 0; k < d; ++k) {
                s_K[tx][k] = K[kv_row * d + k];
                s_V[tx][k] = V[kv_row * d + k];
            }
        } else {
            for (int k = 0; k < d; ++k) {
                s_K[tx][k] = 0.0f;
                s_V[tx][k] = 0.0f;
            }
        }
        __syncthreads();

        if (row_i < seq_len) {
            int valid_keys = (kv_row_start + TILE_BC <= seq_len) ? TILE_BC : (seq_len - kv_row_start);

            // Compute local dot products in SRAM
            float S_tile[TILE_BC];
            float m_tile = -INFINITY;

            for (int j = 0; j < valid_keys; ++j) {
                int col_j = kv_row_start + j;
                if (is_causal && col_j > row_i) {
                    S_tile[j] = -INFINITY;
                } else {
                    float dot = 0.0f;
                    for (int k = 0; k < d; ++k) {
                        dot += s_Q[tx][k] * s_K[j][k];
                    }
                    float val = dot * scale;
                    S_tile[j] = val;
                    if (val > m_tile) {
                        m_tile = val;
                    }
                }
            }

            // Online Softmax Update
            if (m_tile > -INFINITY) {
                float m_new = fmaxf(m_i, m_tile);
                float alpha = (m_i > -INFINITY) ? expf(m_i - m_new) : 0.0f;

                float l_tile = 0.0f;
                float P_tile[TILE_BC];
                for (int j = 0; j < valid_keys; ++j) {
                    if (S_tile[j] > -INFINITY) {
                        float p = expf(S_tile[j] - m_new);
                        P_tile[j] = p;
                        l_tile += p;
                    } else {
                        P_tile[j] = 0.0f;
                    }
                }

                l_i = l_i * alpha + l_tile;

                for (int k = 0; k < d; ++k) {
                    float pv = 0.0f;
                    for (int j = 0; j < valid_keys; ++j) {
                        pv += P_tile[j] * s_V[j][k];
                    }
                    o_accum[k] = o_accum[k] * alpha + pv;
                }

                m_i = m_new;
            }
        }
        __syncthreads();
    }

    // 3. Write normalized output to DRAM
    if (row_i < seq_len) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        for (int k = 0; k < d; ++k) {
            O[row_i * d + k] = o_accum[k] * inv_l;
        }
    }
}

void launch_flash_attention_tiled(const float* d_Q, const float* d_K, const float* d_V,
                                  float* d_O, int seq_len, int d, float scale, bool is_causal) {
    dim3 block(TILE_BR);
    dim3 grid((seq_len + TILE_BR - 1) / TILE_BR);
    flash_attention_tiled_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d, scale, is_causal);
}

bool is_flash_attention_tiled_implemented() {
    return true;
}

// ==============================================================================
// MILESTONE 3: FlashAttention-2 (Outer-Loop over Q, Causal Masking, Register Rescaling)
// ==============================================================================
// FlashAttention-2 Innovations (Dao, 2023):
// 1. Invert loop order: Outer loop over Q tiles (parallelized across thread blocks).
//    Inner loop over K, V tiles. This keeps output accumulators O entirely in
//    fast thread registers until the block finishes!
// 2. Rescale O only by exp(m_old - m_new) without dividing by l_new until the very end:
//    O_final = O_accum / l_final.
// 3. Efficient Causal Masking: Skip entire K, V tiles where j_start > i_end!
// ==============================================================================

__global__ void flash_attention_2_kernel(const float* __restrict__ Q,
                                         const float* __restrict__ K,
                                         const float* __restrict__ V,
                                         float* __restrict__ O,
                                         int seq_len, int d,
                                         float scale, bool is_causal) {
    // Shared memory tiles only for K and V (Q is kept purely in thread registers!)
    __shared__ float s_K[TILE_BC][HEAD_DIM];
    __shared__ float s_V[TILE_BC][HEAD_DIM];

    int q_tile_idx = blockIdx.x;
    int q_row_start = q_tile_idx * TILE_BR;
    int tx = threadIdx.x;
    int row_i = q_row_start + tx;

    // Load Q row entirely into thread registers (persistent across all K, V iterations)
    float q_reg[HEAD_DIM];
    if (row_i < seq_len) {
        if (d == HEAD_DIM) {
            const float4* q_ptr = reinterpret_cast<const float4*>(Q + row_i * d);
            float4* q_reg4 = reinterpret_cast<float4*>(q_reg);
            #pragma unroll
            for (int v = 0; v < HEAD_DIM / 4; ++v) {
                q_reg4[v] = q_ptr[v];
            }
        } else {
            for (int k = 0; k < d; ++k) {
                q_reg[k] = Q[row_i * d + k];
            }
        }
    } else {
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) {
            q_reg[k] = 0.0f;
        }
    }

    // FlashAttention-2 Register Accumulators
    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_accum[HEAD_DIM];
    #pragma unroll
    for (int k = 0; k < HEAD_DIM; ++k) {
        o_accum[k] = 0.0f;
    }

    int num_kv = (seq_len + TILE_BC - 1) / TILE_BC;

    // Inner loop over K, V tiles
    for (int kv_tile = 0; kv_tile < num_kv; ++kv_tile) {
        int kv_row_start = kv_tile * TILE_BC;

        // Efficient causal masking: skip entire tile if all keys are in the causal future
        if (is_causal && kv_row_start > q_row_start + TILE_BR - 1) {
            break;
        }

        // Vectorized cooperative load into shared memory
        int kv_row = kv_row_start + tx;
        if (kv_row < seq_len) {
            if (d == HEAD_DIM) {
                const float4* k_ptr = reinterpret_cast<const float4*>(K + kv_row * d);
                const float4* v_ptr = reinterpret_cast<const float4*>(V + kv_row * d);
                float4* sk_ptr = reinterpret_cast<float4*>(&s_K[tx][0]);
                float4* sv_ptr = reinterpret_cast<float4*>(&s_V[tx][0]);
                #pragma unroll
                for (int v = 0; v < HEAD_DIM / 4; ++v) {
                    sk_ptr[v] = k_ptr[v];
                    sv_ptr[v] = v_ptr[v];
                }
            } else {
                for (int k = 0; k < d; ++k) {
                    s_K[tx][k] = K[kv_row * d + k];
                    s_V[tx][k] = V[kv_row * d + k];
                }
            }
        } else {
            if (d == HEAD_DIM) {
                float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4* sk_ptr = reinterpret_cast<float4*>(&s_K[tx][0]);
                float4* sv_ptr = reinterpret_cast<float4*>(&s_V[tx][0]);
                #pragma unroll
                for (int v = 0; v < HEAD_DIM / 4; ++v) {
                    sk_ptr[v] = zero4;
                    sv_ptr[v] = zero4;
                }
            } else {
                for (int k = 0; k < d; ++k) {
                    s_K[tx][k] = 0.0f;
                    s_V[tx][k] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (row_i < seq_len) {
            int valid_keys = (kv_row_start + TILE_BC <= seq_len) ? TILE_BC : (seq_len - kv_row_start);

            float S_tile[TILE_BC];
            float m_tile = -INFINITY;

            // Fast path: if not causal or this tile is strictly before the diagonal, avoid branches
            bool is_full_tile = !is_causal || (kv_row_start + valid_keys - 1 <= row_i);

            if (is_full_tile) {
                #pragma unroll 4
                for (int j = 0; j < valid_keys; ++j) {
                    float dot = 0.0f;
                    #pragma unroll
                    for (int k = 0; k < HEAD_DIM; ++k) {
                        dot += q_reg[k] * s_K[j][k];
                    }
                    float val = dot * scale;
                    S_tile[j] = val;
                    m_tile = fmaxf(m_tile, val);
                }
            } else {
                for (int j = 0; j < valid_keys; ++j) {
                    int col_j = kv_row_start + j;
                    if (col_j > row_i) {
                        S_tile[j] = -INFINITY;
                    } else {
                        float dot = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < HEAD_DIM; ++k) {
                            dot += q_reg[k] * s_K[j][k];
                        }
                        float val = dot * scale;
                        S_tile[j] = val;
                        m_tile = fmaxf(m_tile, val);
                    }
                }
            }

            // FlashAttention-2 Online Softmax & Register Rescaling
            if (m_tile > -INFINITY) {
                float m_new = fmaxf(m_i, m_tile);
                float alpha = (m_i > -INFINITY) ? expf(m_i - m_new) : 0.0f;

                float l_tile = 0.0f;
                float P_tile[TILE_BC];
                for (int j = 0; j < valid_keys; ++j) {
                    if (S_tile[j] > -INFINITY) {
                        float p = expf(S_tile[j] - m_new);
                        P_tile[j] = p;
                        l_tile += p;
                    } else {
                        P_tile[j] = 0.0f;
                    }
                }

                l_i = l_i * alpha + l_tile;

                // Accumulate P @ V into register accumulators O
                #pragma unroll
                for (int k = 0; k < HEAD_DIM; ++k) {
                    float pv = 0.0f;
                    #pragma unroll 4
                    for (int j = 0; j < valid_keys; ++j) {
                        pv += P_tile[j] * s_V[j][k];
                    }
                    o_accum[k] = o_accum[k] * alpha + pv;
                }

                m_i = m_new;
            }
        }
        __syncthreads();
    }

    // FlashAttention-2: Single division by l_final at the very end
    if (row_i < seq_len) {
        float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
        #pragma unroll
        for (int k = 0; k < HEAD_DIM; ++k) {
            o_accum[k] *= inv_l;
        }

        if (d == HEAD_DIM) {
            float4* o_ptr = reinterpret_cast<float4*>(O + row_i * d);
            const float4* o_accum4 = reinterpret_cast<const float4*>(o_accum);
            #pragma unroll
            for (int v = 0; v < HEAD_DIM / 4; ++v) {
                o_ptr[v] = o_accum4[v];
            }
        } else {
            for (int k = 0; k < d; ++k) {
                O[row_i * d + k] = o_accum[k];
            }
        }
    }
}

void launch_flash_attention_2(const float* d_Q, const float* d_K, const float* d_V,
                              float* d_O, int seq_len, int d, float scale, bool is_causal) {
    dim3 block(TILE_BR);
    dim3 grid((seq_len + TILE_BR - 1) / TILE_BR);
    flash_attention_2_kernel<<<grid, block>>>(d_Q, d_K, d_V, d_O, seq_len, d, scale, is_causal);
}

bool is_flash_attention_2_implemented() {
    return true;
}
