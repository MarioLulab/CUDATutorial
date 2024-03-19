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
    unsigned int out_offset = block_offset_y * N + block_offset_x;

    // NOTE: The reason why assert here is that : to make threads load gm to sm fully
    assert(kBDIMY * kbK == blockDim.x);
    assert(kBDIMX * kbK == blockDim.x);


    A += global_A_pos;
    B += global_B_pos;
    out += out_offset;


    float local_thread_buf[krM] = {0.f};

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
                local_thread_buf[res_idx] += tmpB * bufA[(thread_row * krM + res_idx) * kbK + dot_idx];
            }
        }

        __syncthreads();
    }

    for (unsigned int res_idx = 0; res_idx < krM; res_idx++){

        if (out_offset + (thread_row * krM + res_idx) * N + thread_col < M * N){
            out[(thread_row * krM + res_idx) * N + thread_col] = local_thread_buf[res_idx];
        }
    }
    
}



__global__ void sgemm_2dtile_thread(
    float* A, float* B, float* out,
    int M, int N, int K
)
{
    __shared__ float bufA[kBDIMY * kbK];
    __shared__ float bufB[kBDIMX * kbK];

    unsigned int block_offset_y = kBDIMY * blockIdx.y;
    unsigned int block_offset_x = kBDIMX * blockIdx.x;
    unsigned int out_offset = block_offset_y * N + block_offset_x;
    unsigned int tid = threadIdx.x;

    unsigned int thread_row = threadIdx.x / (kbN / krN);
    unsigned int thread_col = threadIdx.x % (kbN / krN);

    // unsigned int inner_A_row = tid / kbK;
    // unsigned int inner_A_col = tid % kbK;
    // unsigned int inner_B_row = tid / kbN;
    // unsigned int inner_B_col = tid % kbN;

    unsigned int global_A_pos = block_offset_y * K;
    unsigned int global_B_pos = block_offset_x;
    unsigned int ASize = M * K;
    unsigned int BSize = K * N;
    
    A += block_offset_y * K;
    B += block_offset_x;
    out += out_offset;


    float thread_local_buf[krM * krN] = {0.f};
    float thread_local_bufA[krM] = {0.f};
    float thread_local_bufB[krN] = {0.f};

    for (unsigned int offset = 0; offset < K; offset += kbK){

        for (unsigned int tid_offset = tid; tid_offset < kBDIMY * kbK; tid_offset += blockDim.x){
            unsigned int inner_A_row = tid_offset / kbK;
            unsigned int inner_A_col = tid_offset % kbK;
            bufA[inner_A_row * kbK + inner_A_col] = global_A_pos + inner_A_row * K + inner_A_col < ASize ? A[inner_A_row * K + inner_A_col] : 0.f;            
        }
        for (unsigned int tid_offset = tid; tid_offset < kBDIMX * kbK; tid_offset += blockDim.x){
            unsigned int inner_B_row = tid_offset / kbN;
            unsigned int inner_B_col = tid_offset % kbN;
            bufB[inner_B_row * kbN + inner_B_col] = global_B_pos + inner_B_row * N + inner_B_col < BSize ? B[inner_B_row * N + inner_B_col] : 0.f;
        }
        __syncthreads();

        A += kbK;
        B += kbK * N;
        global_A_pos += kbK;
        global_B_pos += kbK * N;

        for (unsigned int dot_idx = 0; dot_idx < kbK; dot_idx++){
            for (unsigned int i = 0; i < krM; i++){
                thread_local_bufA[i] = bufA[(thread_row * krM + i) * kbK + dot_idx];
            }

            for (unsigned int i = 0; i < krN; i++){
                thread_local_bufB[i] = bufB[dot_idx * kbN + (thread_col * krN + i)];
            }

            for (unsigned int i = 0; i < krM; i++){
                for (unsigned int j = 0; j < krN; j++){
                    thread_local_buf[i * krN + j] += thread_local_bufA[i] * thread_local_bufB[j];
                }
            }


        }

        __syncthreads();
    }

    for (unsigned int res_y = 0; res_y < krM; res_y++){
        for (unsigned int res_x = 0; res_x < krN; res_x++){
            out[(thread_row * krM + res_y) * N + thread_col * krN + res_x] = thread_local_buf[res_y * krN + res_x];
        }
    }
}