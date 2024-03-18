#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include "utils/prng.h"

#include "impl_sgemm.h"
#include "naive_sgemm.h"


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


inline void call_naive_sgemm(const float* A, const float* B, float* out, int M, int N, int K){
    dim3 gridDim(ALIGN_UP(N, kBDIMX), ALIGN_UP(M, kBDIMY), 1);
    dim3 blockDim(kBDIMX, kBDIMY, 1);

    naive_sgemm<<<gridDim, blockDim>>>(A, B, out, M, N, K);    
}

void sgemm_CPU(const float* A, const float* B, float* out, int M, int N, int K){
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.;
            for (int k = 0 ; k < K; ++k) {
                sum += A[i*K + k] * B[k*N + j];
            }
            out[i * N + j] = sum;
        }
    }
}


int main(int argc, char** argv) {
    initDevice(6);

    int kernel_type = 0;
    if(argc==2) {
        kernel_type=atoi(argv[1]);
    }

    // 分配内存并随机生成输入数据
    // ===== Host Memory =====
    constexpr unsigned int kATotalNum = kM * kK;
    constexpr unsigned int kBTotalNum = kK * kN;
    constexpr unsigned int kOutTotalNum = kM * kN;
    float *inA, *inB, *out_cpu;

    inA = (float *)malloc(kATotalNum * sizeof(float));
    inB = (float *)malloc(kBTotalNum * sizeof(float));
    out_cpu = (float *)malloc(kOutTotalNum * sizeof(float));
    Randomize(inA, kATotalNum, -10.0, 10.0);
    Randomize(inB, kBTotalNum, -10.0, 10.0);

    // ===== Device Memory =====
    float *d_inA, *d_inB, *d_out;
    cudaMalloc(&d_inA, kATotalNum * sizeof(float));
    cudaMalloc(&d_inB, kBTotalNum * sizeof(float));
    cudaMalloc(&d_out, kOutTotalNum * sizeof(float));

    // ===== CUDA Warm Up =====
    // ===== Host -> Device =====
    cudaMemcpy(d_inA, inA, kATotalNum * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_inB, inB, kBTotalNum * sizeof(float), cudaMemcpyHostToDevice);

    call_naive_sgemm(d_inA, d_inB, d_out, kM, kN, kK);
    cudaDeviceSynchronize();

    // ===== CPU sgemm =====
    sgemm_CPU(inA, inB, out_cpu, kM, kN, kK);

    // ===== GPU sgemm =====
#define RUN_SGEMM_IMPL(TYPE)    \
    switch (TYPE) { \
        case 0: \
        {   \
            call_naive_sgemm(d_inA, d_inB, d_out, kM, kN, kK);   \
            std::cout << "====== call_naive_sgemm finished ======" << std::endl; \
            break;  \
        }   \
        default:    \
            break;  \
    }   


    for (int i = 0; i < 20; ++i) {
        RUN_SGEMM_IMPL(kernel_type);
    }

    // ===== Device -> Host =====
    float* out_gpu = (float *)malloc(kOutTotalNum * sizeof(float));
    cudaMemcpy(out_gpu, d_out, kOutTotalNum * sizeof(float), cudaMemcpyDeviceToHost);

    if (checkResult(out_cpu, out_gpu, kOutTotalNum)) {
        std::cout << "Test Passed!" << std::endl;

        for (int i = 0; i < 20; i ++){
            RUN_SGEMM_IMPL(kernel_type);
        }

    } else {
        std::cout << "Test Failed!" << std::endl;
    }

    cudaFree(d_inA);
    cudaFree(d_inB);
    cudaFree(d_out);
    free(inA);
    free(inB);
    free(out_cpu);
    free(out_gpu);
}