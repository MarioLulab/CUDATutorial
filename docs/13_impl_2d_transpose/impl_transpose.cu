#include "impl_transpose.h"

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include "utils/prng.h"
#include "naive_transpose.h"
#include "shared_mem_transpose.h"

#define ALIGN_UP(x, div) (unsigned int)((x + div - 1) / div)

void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,dev);
  printf("Using device %d: %s\n",dev,deviceProp.name);
  cudaSetDevice(dev);
}

bool checkResult(const float* out_cpu, const float* out_gpu, size_t n, float atol = 1e-5f) {
    bool pass = true;
    for (int i = 0; i < n; ++i) {
        if (abs(out_cpu[i] - out_gpu[i]) > atol) {
            pass = false;
            std::cout << "Verification failed at " << i << "!" << std::endl;
            std::cout << "CPU: " << out_cpu[i] << " GPU: " << out_gpu[i] << std::endl;
            break;
        }
    }

    return pass;
}

void transposeMat2D_CPU(const float* in, float* out, int width, int height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            out[j*height + i] = in[i*width + j];
        }
    }
}

__global__ void warmUp(const float* in, float* out, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int index = row*width + col;
        out[index] = in[index];
    }
}

__global__ void copyRow(const float* in, float* out, int width, int height) {
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    if (col < width && row < height) {
        int index = row*width + col;
        out[index] = in[index];
    }
}

int main(int argc, char** argv) {

    initDevice(6);

    int transform_type = 0;
    if(argc==2) {
        transform_type=atoi(argv[1]);
    }


    const int kWidth = 1 << 12;
    const int kHeight = 1 << 12;

    // 分配内存并随机生成输入数据
    // ===== Host Memory =====
    float *in, *out_cpu;
    in = (float *)malloc(kWidth * kHeight * sizeof(float));
    out_cpu = (float *)malloc(kWidth * kHeight * sizeof(float));
    Randomize(in, kWidth * kHeight, -10.0, 10.0);

    // ===== Device Memory =====
    float *d_in, *d_out;
    cudaMalloc(&d_in, kWidth * kHeight * sizeof(float));
    cudaMalloc(&d_out, kWidth * kHeight * sizeof(float));

    dim3 blockDim(kBDIMX, kBDIMY, 1);
    dim3 gridDim(ALIGN_UP(kWidth, kBDIMX), ALIGN_UP(kHeight, kBDIMY), 1);

    // ===== CUDA Warm Up =====
    // ===== Host -> Device =====
    cudaMemcpy(d_in, in, kWidth * kHeight * sizeof(float), cudaMemcpyHostToDevice);

    warmUp<<<gridDim, blockDim>>>(d_in, d_out, kWidth, kHeight);
    cudaDeviceSynchronize();

    // ===== CPU transpose =====
    transposeMat2D_CPU(in, out_cpu, kWidth, kHeight);

    // ===== GPU transpose =====
#define RUN_TRANSPOSE_IMPL(TYPE)    \
    switch (TYPE) { \
        case 0: \
        {   \
            copyRow<<<gridDim, blockDim>>>(d_in, d_out, kWidth, kHeight);   \
            std::cout << "====== copyRow finished ======" << std::endl; \
            break;  \
        }   \
        case 1: \
        {   \
            transposeMat2D_naive<<<gridDim, blockDim>>>(d_in, d_out, kWidth, kHeight);  \
            std::cout << "====== transposeMat2D_naive finished ======" << std::endl;    \
            break;  \
        }   \
        case 2: \
        {   \
            transposeMat2D_sm<<<gridDim, blockDim>>>(d_in, d_out, kWidth, kHeight); \
            std::cout << "====== transposeMat2D_sm finished ======" << std::endl;   \
            break;  \
        }   \
        case 3: \
        {   \
            transposeMat2D_sm_pad<<<gridDim, blockDim>>>(d_in, d_out, kWidth, kHeight); \
            std::cout << "====== transposeMat2D_sm_pad finished ======" << std::endl;   \
            break;  \
        }   \
        case 4: \
        {   \
            transposeMat2D_sm_pad_unroll<<<gridDim, blockDim>>>(d_in, d_out, kWidth, kHeight); \
            std::cout << "====== transposeMat2D_sm_pad finished ======" << std::endl;   \
            break;  \
        }   \
        default:    \
            break;  \
    }   


    RUN_TRANSPOSE_IMPL(transform_type);
    // ===== Device -> Host =====
    float* out_gpu = (float *)malloc(kWidth * kHeight * sizeof(float));
    cudaMemcpy(out_gpu, d_out, kWidth * kHeight * sizeof(float), cudaMemcpyDeviceToHost);

    if (checkResult(out_cpu, out_gpu, kWidth * kHeight)) {
        std::cout << "Test Passed!" << std::endl;

        for (int i = 0; i < 20; i ++){
            RUN_TRANSPOSE_IMPL(transform_type);
        }

    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(in);
    free(out_cpu);
    free(out_gpu);

    return 0;
}