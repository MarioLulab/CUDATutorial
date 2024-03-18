#include "impl_sgemm.h"

__global__ void naive_sgemm(
    const float* A, const float* B, float* out,
    int M, int N, int K
)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (y < M && x < N){
        out[y*N + x] = 0;
        for (int k = 0; k < K; ++k){
            out[y * N + x] += A[y * K + k] * B[k * N + x];
        }
    }
}