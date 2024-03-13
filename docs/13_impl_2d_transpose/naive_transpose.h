#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

__global__ void transposeMat2D_naive(const float* in, float* out, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        out[col*height + row] = in[row*width + col];
    }
}