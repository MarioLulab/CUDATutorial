#include "impl_sgemm.h"


__global__ void sgemm_1dtile_thread(
    float* A, float* B, float* out,
    int M, int N, int K
)
{
    __shared__ float bufA[kBDIMY * kbK];
    __shared__ float bufB[kBDIMX * kbK];
    
    unsigned int block_offset_x = kBDIMX * blockIdx.x;
    unsigned int block_offset_y = kBDIMY * blockIdx.y;
    unsigned int tid = threadIdx.x;

    unsigned int inner_A_row = tid / kbK;
    unsigned int inner_A_col = tid % kbK;
    unsigned int inner_B_row = tid / kbN;
    unsigned int inner_B_col = tid % kbN;

    unsigned int ASize = M * K;
    unsigned int BSize = N * K;

    unsigned int thread_row = tid / kbN;
    unsigned int thread_col = tid % kbN;


    unsigned int global_A_pos = block_offset_y * K;
    unsigned int global_B_pos = block_offset_x;
    unsigned int out_offset = block_offset_y * kBDIMY * N + block_offset_x * kBDIMX;

    A += global_A_pos;
    B += global_B_pos;
    out += out_offset;


    float local_thread_bufA[krM] = {0.f};

    for (unsigned int offset = 0; offset < K; offset += kbK){
        bufA[inner_A_row * kbK + inner_A_col] = global_A_pos + inner_A_row * K + inner_A_col < ASize ? A[inner_A_row * K + inner_A_col] : 0.f;
        bufB[inner_B_row * kbN + inner_B_col] = global_B_pos + inner_B_row * N + inner_B_col < BSize ? B[inner_B_row * N + inner_B_col] : 0.f;
        __syncthreads();

        global_A_pos += kbK;
        global_B_pos += kbK * N;
        A += kbK;
        B += kbK * N;

        for (unsigned int dot_idx = 0; dot_idx < kbK; dot_idx++){
            float tmpB = bufB[dot_idx * kbN + thread_col];
            for (unsigned int res_idx = 0; res_idx < krM; res_idx++){
                local_thread_bufA[res_idx] += tmpB * bufA[(thread_row * krM + res_idx) * kbK + dot_idx];
            }
        }

        __syncthreads();
    }

    for (unsigned int res_idx = 0; res_idx < krM; res_idx++){

        if (out_offset + (thread_row * krM + res_idx) * N + thread_col < M * N){
            out[(thread_row * krM + res_idx) * N + thread_col] = local_thread_bufA[res_idx];
        }
    }
    
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void sgemm_blocktiling_1d_kernel(float *A, float *B, float *C, int M, int N, int K)
{
    // the output block that we want to compute in this threadblock
    const uint c_row = blockIdx.y;
    const uint c_col = blockIdx.x;

    // allocate shared memory for the input and output submatrices
    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    // the inner row & col that we're accessing in this thread
    const uint thread_row = threadIdx.x / BN;
    const uint thread_col = threadIdx.x % BN;

    // advance pointers to the starting positions
    A += c_row * BM * K;
    B += c_col * BN;
    C += c_row * BM * N + c_col * BN;

    // use to avoid out-of-bounds accesses
    int global_m_pos = c_row * BM * K;
    int global_n_pos = c_col * BN;
    const uint m_size = M * K;
    const uint n_size = N * K;

    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);

    const uint A_inner_row = threadIdx.x / BK; // warp-level GMEM coalescing
    const uint A_inner_col = threadIdx.x % BK;
    const uint B_inner_row = threadIdx.x / BN; // warp-level GMEM coalescing
    const uint B_inner_col = threadIdx.x % BN;

    // allocate thread-local cache for results in registerfile
    float thread_results[TM] = {0.0};

    // outer loop over block tiles
    for (uint bk_idx = 0; bk_idx < K; bk_idx += BK)
    {
        // load the next block of the input matrices into shared memory
        A_shared[A_inner_row * BK + A_inner_col] = (global_m_pos + A_inner_row * K + A_inner_col < m_size) ? A[A_inner_row * K + A_inner_col] : 0.0f;
        B_shared[B_inner_row * BN + B_inner_col] = (global_n_pos + B_inner_row * N + B_inner_col < n_size) ? B[B_inner_row * N + B_inner_col] : 0.0f;

        // wait for all threads to finish loading
        __syncthreads();

        // advance the pointers
        A += BK;
        B += BK * N;
        global_m_pos += BK;
        global_n_pos += BK * N;

        // compute the partial sum
        for (uint dot_idx = 0; dot_idx < BK; dot_idx++)
        {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            float tmp_b = B_shared[dot_idx * BN + thread_col];
            for (uint res_idx = 0; res_idx < TM; res_idx++)
            {
                thread_results[res_idx] += A_shared[(thread_row * TM + res_idx) * BK + dot_idx] * tmp_b;
            }
        }

        // wait for all threads to finish computing
        __syncthreads();
    }

    for (uint res_idx = 0; res_idx < TM; res_idx++)
    {
        if (c_row * BM + thread_row * TM + res_idx < M && c_col * BN + thread_col < N)
        {
            C[(thread_row * TM + res_idx) * N + thread_col] = thread_results[res_idx];
        }
    }
}