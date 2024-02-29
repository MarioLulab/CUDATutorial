#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include "utils/prng.h"

#include "impl_fused_rope.h"

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

int main(int argc, char** argv) {

    initDevice(6);

    const int64_t batch_size = 4;
    const int64_t seq_len = 4096;
    const int64_t num_heads = 32;
    const int64_t head_dim = 128;
    const int64_t numel = batch_size * seq_len * num_heads * head_dim;

    const int64_t qkv_shape[4] = {batch_size, seq_len, num_heads, head_dim}; // bs, seq_len, num_heads, head_dim
    const int64_t sin_cos_shape[4] = {1, seq_len, 1, head_dim};

    dim3 gridDim(kGDIMX, 1, 1);
    dim3 blockDim(kBDIMX, 1, 1);

    constexpr const int vec_size = 2;

    // allocate outs_data
    float* outs_data[3];
    float* outs_data_gpu[3];
    for (int i = 0; i < 3; i++) {
        outs_data[i] = (float*)malloc(numel * sizeof(float));
        cudaMalloc(&outs_data_gpu[i], numel * sizeof(float));
    }

    // allocate qkv_data
    float* ins_data[3];
    float* ins_data_gpu[3];
    for (int i = 0; i < 3; i++) {
        ins_data[i] = (float*)malloc(numel * sizeof(float));
        Randomize(ins_data[i], numel, -10, 10);
        cudaMalloc(&ins_data_gpu[i], numel * sizeof(float));
        cudaMemcpy(ins_data_gpu[i], ins_data[i], numel * sizeof(float), cudaMemcpyHostToDevice);
    }

    // allocate sin_cos_data
    float* sin_cos_data[2];
    float* sin_cos_data_gpu[2];
    for (int i = 0; i < 2; i++) {
        sin_cos_data[i] = (float*)malloc(seq_len * head_dim * sizeof(float));
        Randomize(sin_cos_data[i], seq_len * head_dim, -1, 1);
        cudaMalloc(&sin_cos_data_gpu[i], seq_len * head_dim * sizeof(float));
        cudaMemcpy(sin_cos_data_gpu[i], sin_cos_data[i], seq_len * head_dim * sizeof(float), cudaMemcpyHostToDevice);
    }


    int num_inputs = 2;
    int64_t inputs_num_heads[num_inputs] = {num_heads, num_heads};
    bool flag_sin_cos = true;
    int sign = 1;

    const int64_t batch_stride = seq_len * num_heads * head_dim;
    const int64_t seq_stride = num_heads * head_dim;

    Array<float*, 3> out_array {outs_data_gpu[0], outs_data_gpu[1], outs_data_gpu[2]};
    Array<const float*, 3> ins_data_array {ins_data_gpu[0], ins_data_gpu[1], ins_data_gpu[2]};
    Array<const float*, 2> sin_cos_data_array {sin_cos_data_gpu[0], sin_cos_data_gpu[1]};

    for (int i = 0; i < 100; i++){
        // default stream
        VectorizedFusedRopeWithRotateEveryTwoKernel<float, float, vec_size><<<gridDim, blockDim, 0>>>(
         ins_data_array,
         sin_cos_data_array,
         static_cast<const int64_t*>(nullptr),   
         10000,
         flag_sin_cos,
         sign,
         batch_size,
         seq_len,
         num_heads,
         head_dim,
         batch_stride,
         seq_stride,
         out_array,
         num_inputs);
    }  

    // free
    for (int i = 0; i < 3; i++) {
        free(outs_data[i]);
        free(ins_data[i]);
        cudaFree(outs_data_gpu[i]);
        cudaFree(ins_data_gpu[i]);
    }

    for (int i = 0; i < 2; i++){
        free(sin_cos_data[i]);
        cudaFree(sin_cos_data_gpu[i]);
    }

    return 0;
}
