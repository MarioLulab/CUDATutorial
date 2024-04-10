#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <algorithm>
#include "utils/prng.h"

#include "impl_paddle.h"


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

int main(int argc, char** argv) {
    initDevice(6);

    int kernel_type = 0;
    if(argc==2) {
        kernel_type=atoi(argv[1]);
    }

    dim3 gridDim(65536, 1, 1);
    dim3 blockDim(512, 1, 1);


    // paddle::Array<int64_t, 3> inputs_num_heads;

    // q.shape: [seq_len, batch_size, num_heads, head_dim] if time_major else
    // [batch_size, seq_len, num_heads, head_dim]
    constexpr int64_t batch_size = 4;
    constexpr int64_t seq_len = 4096;
    constexpr int64_t num_heads = 32;
    constexpr int64_t head_dim = 128;

    constexpr unsigned int vec_size = 2;

    paddle::Array<float*, 3> outs_data;
    paddle::Array<const float*, 3> ins_data;
    paddle::Array<const float*, 2> sin_cos_data;
    const int64_t* position_ids_data = NULL;

    // 分配内存并随机生成输入数据
    // ===== Host Memory =====
    constexpr unsigned int kTotalNum = batch_size * seq_len * num_heads * head_dim;
    float *in;
    in = (float *)malloc(kTotalNum * sizeof(float));
    Randomize((in), kTotalNum, -10.0, 10.0);

    // ===== Device Memory =====
    float *q_d_in, *k_d_in, *q_d_out, *k_d_out;
    cudaMalloc(&q_d_in, kTotalNum * sizeof(float));
    cudaMalloc(&k_d_in, kTotalNum * sizeof(float));
    cudaMalloc(&q_d_out, kTotalNum * sizeof(float));
    cudaMalloc(&k_d_out, kTotalNum * sizeof(float));

    // ===== CUDA Warm Up =====
    // ===== Host -> Device =====
    cudaMemcpy(q_d_in, in, kTotalNum * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(k_d_in, in, kTotalNum * sizeof(float), cudaMemcpyHostToDevice);


    ins_data[0] = q_d_in;
    outs_data[0] = q_d_out;
    inputs_num_heads[0] = num_heads;

    ins_data[1] = k_d_in;
    outs_data[1] = k_d_out;
    inputs_num_heads[1] = num_heads;

    int num_inputs = 2;
    float div_c = 1.0f / head_dim;

    int64_t batch_stride = seq_len * num_heads * head_dim;
    int64_t seq_stride = num_heads * head_dim;
    
    int sign = 1;
    float rotary_emb_base = 10000.0;
    paddle::VectorizedFusedRopeWithRotateEveryTwoKernel<float, float, vec_size><<<gridDim, blockDim>>>(
        ins_data,
        sin_cos_data,
        position_ids_data,
        /*flag_sin_cos*/ false,
        sign,
        batch_size,
        seq_len,
        inputs_num_heads[0],
        head_dim,
        batch_stride,
        seq_stride,
        outs_data,
        num_inputs,
        div_c,
        rotary_emb_base
    );

    cudaDeviceSynchronize();



    // ===== GPU fused rope =====
#define RUN_FUSDE_ROPE()    \
    paddle::VectorizedFusedRopeWithRotateEveryTwoKernel<float, float, vec_size><<<gridDim, blockDim>>>(   \
        ins_data,   \
        sin_cos_data,   \
        position_ids_data,  \
        /*flag_sin_cos*/ false, \
        sign,   \
        batch_size, \
        seq_len,    \
        inputs_num_heads[0],    \
        head_dim,   \
        batch_stride,   \
        seq_stride, \
        outs_data,  \
        num_inputs, \
        div_c,  \
        rotary_emb_base \
    );


    for (int i = 0; i < 20; ++i) {
        RUN_FUSDE_ROPE();
        std::cout << "====== RUN_FUSDE_ROPE finished ======" << std::endl; \

    }

    cudaFree(q_d_in);
    cudaFree(k_d_in);
    cudaFree(q_d_out);
    cudaFree(k_d_out);
    free(in);
}