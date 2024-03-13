#pragma once
#include <cuda_runtime.h>

#include "impl_transpose.h"

const int kIPAD = 2;

__global__ void transposeMat2D_sm(const float* in, float* out, int width, int height) {
    __shared__ float buf[kBDIMY][kBDIMX];

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = tid / blockDim.y;
    unsigned int icol = tid % blockDim.y;
    unsigned int o_row = blockIdx.x * blockDim.x + irow;
    unsigned int o_col = blockIdx.y * blockDim.y + icol;

    if (col < width && row < height) {
        buf[threadIdx.y][threadIdx.x] = in[row * width + col];
        __syncthreads();
        out[o_row * height + o_col] = buf[icol][irow];
    }
}


__global__ void transposeMat2D_sm_pad(const float* in, float* out, int width, int height) {
    __shared__ float buf[kBDIMY][kBDIMX + kIPAD];

    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = tid / blockDim.y;
    unsigned int icol = tid % blockDim.y;
    unsigned int o_row = blockIdx.x * blockDim.x + irow;
    unsigned int o_col = blockIdx.y * blockDim.y + icol;

    if (col < width && row < height) {
        buf[threadIdx.y][threadIdx.x] = in[row * width + col];
        __syncthreads();
        out[o_row * height + o_col] = buf[icol][irow];
    }
}


__global__ void transposeMat2D_sm_pad_unroll(const float* in, float* out, int width, int height) {
    // __shared__ float buf[kBDIMY][kBDIMX*2 + kIPAD];
    __shared__ float buf[kBDIMY * (kBDIMX*2 + kIPAD)];  // better-performance in V100-32GB

    unsigned int col = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

    unsigned int tid = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = tid / blockDim.y;
    unsigned int icol = tid % blockDim.y;
    unsigned int o_row = blockIdx.x * blockDim.x * 2 + irow;
    unsigned int o_col = blockIdx.y * blockDim.y + icol;

    if (col + kBDIMX < width && row < height) {
        // buf[threadIdx.y][threadIdx.x] = in[row * width + col];
        // buf[threadIdx.y][threadIdx.x + kBDIMX] = in[row * width + col + kBDIMX];
        buf[threadIdx.y * (kBDIMX*2 + kIPAD) + threadIdx.x] = in[row * width + col];
        buf[threadIdx.y * (kBDIMX*2 + kIPAD) + threadIdx.x + kBDIMX] = in[row * width + col + kBDIMX];
        __syncthreads();
        // out[o_row * height + o_col] = buf[icol][irow];
        // out[(o_row + kBDIMX) * height + o_col] = buf[icol][irow + kBDIMX];
        out[o_row * height + o_col] = buf[icol * (kBDIMX*2 + kIPAD) + irow];
        out[(o_row + kBDIMX) * height + o_col] = buf[icol * (kBDIMX*2 + kIPAD) + irow + kBDIMX];
    }
}