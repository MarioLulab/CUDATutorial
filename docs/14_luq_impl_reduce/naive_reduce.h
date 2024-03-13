#include "impl_reduce.h"

__global__ void warmUp(float *in, float *out, int n) {
    __shared__ float buf[kBDIMX];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        buf[threadIdx.x] = in[tid];
    }
    __syncthreads();

    for (int s = 1; s < kBDIMX; s *= 2){
        if (threadIdx.x % (2 * s) == 0 && threadIdx.x + s < n){
            buf[threadIdx.x] += buf[threadIdx.x + s];
        }
        __syncthreads();
    }

}

__global__ void naive_reduce(float *in, float *out, int n) {
    __shared__ float buf[kBDIMX];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        buf[threadIdx.x] = in[tid];
    }
    __syncthreads();

    for (int s = 1; s < kBDIMX; s *= 2){
        if (threadIdx.x % (2 * s) == 0 && threadIdx.x + s < n){
            buf[threadIdx.x] += buf[threadIdx.x + s];
        }
        __syncthreads();
    }


    if (threadIdx.x == 0){
        out[blockIdx.x] = buf[0];
    }
}

__global__ void reduce_interleaved(float *in, float *out, int n) {
    __shared__ float buf[kBDIMX];

    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n){
        buf[threadIdx.x] = in[tid];
    }
    __syncthreads();

    for (int s = 1; s < kBDIMX; s *= 2){
        unsigned int index = 2 * s * threadIdx.x;
        if (index < kBDIMX && index + s < kBDIMX){
            buf[index] += buf[index + s];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0){
        out[blockIdx.x] = buf[0];
    }

}

// alias: bank-free version
__global__ void reduce_sequential_addressing(float *in, float *out, int n) {
    __shared__ float buf[kBDIMX];
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int id = threadIdx.x;
    unsigned int bid = blockIdx.x;
    if (tid < n){
        buf[id] = in[tid];
    }
    __syncthreads();

    for (unsigned int s = kBDIMX >> 1; s > 0; s >>= 1){
        if(id < s){
            buf[id] += buf[id + s];
        }
        __syncthreads();
    }

    if (id == 0){
        out[bid] = buf[0];
    }

}


template<int UnrollNum>
__global__ void reduce_unroll_first(float *in, float *out, int n) {
    __shared__ float buf[kBDIMX];
    unsigned int tid = blockIdx.x * blockDim.x * UnrollNum + threadIdx.x;
    unsigned int id = threadIdx.x;
    unsigned int bid = blockIdx.x;
    if (tid < n){
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < UnrollNum; ++i){
            sum += in[tid + i * kBDIMX];
        }
        buf[id] = sum;
    }
    __syncthreads();

    for (unsigned int s = kBDIMX >> 1; s > 0; s >>= 1){
        if(id < s){
            buf[id] += buf[id + s];
        }
        __syncthreads();
    }

    if (id == 0){
        out[bid] = buf[0];
    }

}

__device__ void warpReduce(volatile float *buf, unsigned int id) {
    buf[id] += buf[id + 32];
    buf[id] += buf[id + 16];
    buf[id] += buf[id + 8];
    buf[id] += buf[id + 4];
    buf[id] += buf[id + 2];
    buf[id] += buf[id + 1];
}

__device__ float warpReduceUsingShuffle(float warpSum) {
    warpSum = warpSum + __shfl_xor_sync(FULL_MASK, warpSum, 16, kWarpSize);
    warpSum = warpSum + __shfl_xor_sync(FULL_MASK, warpSum, 8, kWarpSize);
    warpSum = warpSum + __shfl_xor_sync(FULL_MASK, warpSum, 4, kWarpSize);
    warpSum = warpSum + __shfl_xor_sync(FULL_MASK, warpSum, 2, kWarpSize);
    warpSum = warpSum + __shfl_xor_sync(FULL_MASK, warpSum, 1, kWarpSize);
    return warpSum;
}

template<int UnrollNum>
__global__ void reduce_unroll_first_and_unroll_last_warp(float *in, float *out, int n) {
    __shared__ float buf[kBDIMX];
    unsigned int tid = blockIdx.x * blockDim.x * UnrollNum + threadIdx.x;
    unsigned int id = threadIdx.x;
    unsigned int bid = blockIdx.x;
    if (tid < n){
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < UnrollNum; ++i){
            sum += in[tid + i * kBDIMX];
        }
        buf[id] = sum;
    }
    __syncthreads();

    for (unsigned int s = kBDIMX >> 1; s > 32; s >>= 1){
        if(id < s){
            buf[id] += buf[id + s];
        }
        __syncthreads();
    }

    if (id < 32){
        warpReduce(buf, id);
    }

    if (id == 0){
        out[bid] = buf[0];
    }
}

template<int UnrollNum>
__global__ void reduce_unroll_first_and_warp_sum_unroll_last_warp(float *in, float *out, int n) {
    __shared__ float buf[kBDIMX];
    unsigned int tid = blockIdx.x * blockDim.x * UnrollNum + threadIdx.x;
    unsigned int id = threadIdx.x;
    unsigned int bid = blockIdx.x;
    if (tid < n){
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < UnrollNum; ++i){
            sum += in[tid + i * kBDIMX];
        }
        buf[id] = sum;
    }
    __syncthreads();

    for (unsigned int s = kBDIMX >> 1; s >= kWarpSize; s >>= 1){
        if(id < s){
            buf[id] += buf[id + s];
        }
        __syncthreads();
    }

    float warpSum = 0.f;
    if (id < kWarpSize){
        warpSum = warpReduceUsingShuffle(buf[id]);
    }

    if (id == 0){
        out[bid] = warpSum;
    }

}

__device__ void warpReduceUnrollAll(volatile float *buf, unsigned int id) {
    if (kBDIMX >= 64) buf[id] += buf[id + 32];
    if (kBDIMX >= 32) buf[id] += buf[id + 16];
    if (kBDIMX >= 16) buf[id] += buf[id + 8];
    if (kBDIMX >= 8) buf[id] += buf[id + 4];
    if (kBDIMX >= 4) buf[id] += buf[id + 2];
    if (kBDIMX >= 2) buf[id] += buf[id + 1];
}

template<int UnrollNum>
__global__ void reduce_unroll_first_and_unroll_all(float *in, float *out, int n) {
    __shared__ float buf[kBDIMX];
    unsigned int tid = blockIdx.x * blockDim.x * UnrollNum + threadIdx.x;
    unsigned int id = threadIdx.x;
    unsigned int bid = blockIdx.x;
    if (tid < n){
        float sum = 0.f;
        #pragma unroll
        for (int i = 0; i < UnrollNum; ++i){
            sum += in[tid + i * kBDIMX];
        }
        buf[id] = sum;
    }
    __syncthreads();

    if (kBDIMX >= 512 && id < 256){
        buf[id] += buf[id + 256];
        __syncthreads();
    }

    if (kBDIMX >= 256 && id < 128){
        buf[id] += buf[id + 128];
        __syncthreads();
    }

    if (kBDIMX >= 128 && id < 64){
        buf[id] += buf[id + 64];
        __syncthreads();
    }

    if (id < 32){
        warpReduceUnrollAll(buf, id);
    }

    if (id == 0){
        out[bid] = buf[0];
    }

}