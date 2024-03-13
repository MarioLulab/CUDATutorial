#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include "utils/prng.h"

#include "impl_reduce.h"
#include "naive_reduce.h"


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

void reduce_CPU(const float* in, float* out, size_t n) {
    float sum = 0.f;
    for (int i = 0; i < n; i++){
        sum += in[i];
    }
    
    out[0] = sum;
}

int main(int argc, char** argv) {

    initDevice(6);

    int kernel_type = 0;
    if(argc==2) {
        kernel_type=atoi(argv[1]);
    }

    size_t kTotalNum = 1 << 20;
    constexpr int kUnrollNum = 2;

    // 分配内存并随机生成输入数据
    // ===== Host Memory =====
    float *in, *out_cpu;
    in = (float *)malloc(kTotalNum * sizeof(float));
    out_cpu = (float *)malloc(kTotalNum * sizeof(float));
    Randomize((in), kTotalNum, -10.0, 10.0);

    // ===== Device Memory =====
    float *d_in, *d_out;
    cudaMalloc(&d_in, kTotalNum * sizeof(float));
    cudaMalloc(&d_out, kTotalNum * sizeof(float));

    dim3 blockDim(kBDIMX, 1, 1);
    dim3 gridDim(ALIGN_UP(kTotalNum, kBDIMX), 1, 1);

    if (kernel_type >= 3) {
        // using unroll first strategy
        gridDim.x /= kUnrollNum;
    }

    std::cout << "gridDim: " << gridDim.x << ", blockDim: " << blockDim.x << std::endl;

    // ===== CUDA Warm Up =====
    // ===== Host -> Device =====
    cudaMemcpy(d_in, in, kTotalNum * sizeof(float), cudaMemcpyHostToDevice);

    warmUp<<<gridDim, blockDim>>>(d_in, d_out, kTotalNum);
    cudaDeviceSynchronize();

    // ===== CPU Reduce =====
    reduce_CPU(in, out_cpu, kTotalNum);

    // ===== GPU Reduce =====
#define RUN_REDUCE_IMPL(TYPE)    \
    switch (TYPE) { \
        case 0: \
        {   \
            naive_reduce<<<gridDim, blockDim>>>(d_in, d_out, kTotalNum);   \
            std::cout << "====== naive_reduce finished-part1 ======" << std::endl; \
            break;  \
        }   \
        case 1: \
        {   \
            reduce_interleaved<<<gridDim, blockDim>>>(d_in, d_out, kTotalNum);   \
            std::cout << "====== reduce_interleaved finished-part1 ======" << std::endl; \
            break;  \
        }   \
        case 2: \
        {   \
            reduce_sequential_addressing<<<gridDim, blockDim>>>(d_in, d_out, kTotalNum);   \
            std::cout << "====== reduce_sequential_addressing finished-part1 ======" << std::endl; \
            break;  \
        }   \
        case 3: \
        {   \
            reduce_unroll_first<kUnrollNum><<<gridDim, blockDim>>>(d_in, d_out, kTotalNum);   \
            std::cout << "====== reduce_unroll_first finished-part1 ======" << std::endl; \
            break;  \
        }   \
        case 4: \
        {   \
            reduce_unroll_first_and_unroll_last_warp<kUnrollNum><<<gridDim, blockDim>>>(d_in, d_out, kTotalNum);   \
            std::cout << "====== reduce_unroll_first_and_unroll_last_warp finished-part1 ======" << std::endl; \
            break;  \
        }   \
        case 5: \
        {   \
            reduce_unroll_first_and_unroll_all<kUnrollNum><<<gridDim, blockDim>>>(d_in, d_out, kTotalNum);   \
            std::cout << "====== reduce_unroll_first_and_unroll_all finished-part1 ======" << std::endl; \
            break;  \
        }   \
        case 6: \
        {   \
            reduce_unroll_first_and_warp_sum_unroll_last_warp<kUnrollNum><<<gridDim, blockDim>>>(d_in, d_out, kTotalNum);   \
            std::cout << "====== reduce_unroll_first_and_warp_sum_unroll_last_warp finished-part1 ======" << std::endl; \
            break;  \
        }   \
        default:    \
            break;  \
    }   


    RUN_REDUCE_IMPL(kernel_type);

    // ===== Device -> Host =====
    float* out_gpu = (float *)malloc(kTotalNum * sizeof(float));
    cudaMemcpy(out_gpu, d_out, kTotalNum * sizeof(float), cudaMemcpyDeviceToHost);


    float sum = 0;
    switch (kernel_type) {
        case 0:
        {
            for (int i = 0; i < gridDim.x; i++){
                sum += out_gpu[i];
            }
            std::cout << "====== naive_reduce finished-part2 ======" << std::endl; \
        };
        break;
        case 1:
        {
            for (int i = 0; i < gridDim.x; i++){
                sum += out_gpu[i];
            }
            std::cout << "====== reduce_interleaved finished-part2 ======" << std::endl; \
        };
        break;
        case 2:
        {
            for (int i = 0; i < gridDim.x; i++){
                sum += out_gpu[i];
            }
            std::cout << "====== reduce_sequential_addressing finished-part2 ======" << std::endl; \
        };
        case 3:
        {
            for (int i = 0; i < gridDim.x; i++){
                sum += out_gpu[i];
            }
            std::cout << "====== reduce_unroll_first finished-part2 ======" << std::endl; \
        };
        break;
        case 4:
        {
            for (int i = 0; i < gridDim.x; i++){
                sum += out_gpu[i];
            }
            std::cout << "====== reduce_unroll_first_and_unroll_last_warp finished-part2 ======" << std::endl; \
        };
        break;
        case 5:
        {
            for (int i = 0; i < gridDim.x; i++){
                sum += out_gpu[i];
            }
            std::cout << "====== reduce_unroll_first_and_unroll_all finished-part2 ======" << std::endl; \
        };
        break;
        case 6:
        {
            for (int i = 0; i < gridDim.x; i++){
                sum += out_gpu[i];
            }
            std::cout << "====== reduce_unroll_first_and_warp_sum_unroll_last_warp finished-part2 ======" << std::endl; \
        };
        break;
        default: break;
    }

    if (checkResult(&(out_cpu[0]), &sum, 1)) {
        std::cout << "Test Passed!" << std::endl;

        for (int i = 0; i < 20; i ++){
            RUN_REDUCE_IMPL(kernel_type);
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
