#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include "utils/prng.h"

#include "impl_softmax.h"
#include "block_softmax.h"
#include "warp_softmax.h"


#define ALIGN_UP(x, div) (unsigned int)((x + div - 1) / div)
void initDevice(int devNum)
{
  int dev = devNum;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp,dev);
  printf("Using device %d: %s\n",dev,deviceProp.name);
  cudaSetDevice(dev);

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, devNum);
  std::cout << "max thread num: " << prop.maxThreadsPerBlock << std::endl;
  std::cout << "max grad dimensions: " << prop.maxGridSize[0] << ", " << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << std::endl;
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


void softmax_CPU(const float* in, float* out, size_t rows, size_t cols) {
    for (int i = 0; i < rows; ++i) {
        float row_data[cols];
        float row_max = *std::max_element(in + i * cols, in + (i+1) * cols);
        float row_sum = 0.;
        for (int j = 0; j < cols; ++j) {
            row_data[j] = std::exp(in[i * cols + j] - row_max);
            row_sum += row_data[j];
        }

        for (int j = 0; j < cols; ++j) {
            out[i * cols + j] = row_data[j] / row_sum;
        }
    }
}


int main(int argc, char** argv) {
    initDevice(6);

    int kernel_type = 0;
    if(argc==2) {
        kernel_type=atoi(argv[1]);
    }

    dim3 gridDim(kROWS, 1, 1);
    dim3 blockDim(kBDIMX, 1, 1);

    // 分配内存并随机生成输入数据
    // ===== Host Memory =====
    constexpr unsigned int kTotalNum = kROWS * kCOLS;
    float *in, *out_cpu;
    in = (float *)malloc(kTotalNum * sizeof(float));
    out_cpu = (float *)malloc(kTotalNum * sizeof(float));
    Randomize((in), kTotalNum, -10.0, 10.0);

    // ===== Device Memory =====
    float *d_in, *d_out;
    cudaMalloc(&d_in, kTotalNum * sizeof(float));
    cudaMalloc(&d_out, kTotalNum * sizeof(float));

    // ===== CUDA Warm Up =====
    // ===== Host -> Device =====
    cudaMemcpy(d_in, in, kTotalNum * sizeof(float), cudaMemcpyHostToDevice);

    SoftmaxNoSM<<<gridDim, blockDim>>>(d_in, d_out, kROWS, kCOLS);
    cudaDeviceSynchronize();

    // ===== CPU softmax =====
    softmax_CPU(in, out_cpu, kROWS, kCOLS);

    // ===== GPU softmax =====
#define RUN_SOFTMAX_IMPL(TYPE)    \
    switch (TYPE) { \
        case 0: \
        {   \
            SoftmaxNoSM<<<gridDim, blockDim>>>(d_in, d_out, kROWS, kCOLS);   \
            std::cout << "====== SoftmaxNoSM finished ======" << std::endl; \
            break;  \
        }   \
        case 1: \
        {   \
            size_t smem_size = kCOLS * sizeof(float);  \
            SoftmaxSharedMemory<<<gridDim, blockDim, smem_size>>>(d_in, d_out, kROWS, kCOLS);   \
            std::cout << "====== SoftmaxSharedMemory finished ======" << std::endl; \
            break;  \
        }   \
        case 2: \
        {   \
            size_t smem_size = kCOLS * sizeof(float);  \
            blockDim.x = 32; \
            gridDim.x = kCOLS;   \
            SoftmaxWarpReduce<<<gridDim, blockDim, smem_size>>>(d_in, d_out, kROWS, kCOLS);   \
            std::cout << "====== SoftmaxWarpReduce finished ======" << std::endl; \
            break;  \
        }   \
        default:    \
            break;  \
    }   


    for (int i = 0; i < 20; ++i) {
        RUN_SOFTMAX_IMPL(kernel_type);
    }

    // ===== Device -> Host =====
    float* out_gpu = (float *)malloc(kTotalNum * sizeof(float));
    cudaMemcpy(out_gpu, d_out, kTotalNum * sizeof(float), cudaMemcpyDeviceToHost);

    if (checkResult(out_cpu, out_gpu, kTotalNum)) {
        std::cout << "Test Passed!" << std::endl;

        for (int i = 0; i < 20; i ++){
            RUN_SOFTMAX_IMPL(kernel_type);
        }

    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(in);
    free(out_cpu);
    free(out_gpu);
}