#pragma once
#include "impl_softmax.h"
#include <cub/cub.cuh>

template<class ReduceOp, int BlockSize = kBDIMX>
__forceinline__ __device__ float BlockAllReduce(float val) {
    typedef cub::BlockReduce<float, BlockSize> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ float shared_result;
    val = BlockReduce(temp_storage).Reduce(val, ReduceOp());

    if (threadIdx.x == 0) {
        shared_result = val;
    }
    __syncthreads();
    return shared_result;
}

// assume nopadding : `cols % kBIMX == 0` and `(cols / kBIMX) % VecSize == 0`
template<int VecSize = 2>
__global__ void SoftmaxNoSM(const float* in, float* out, const unsigned int rows, const unsigned int cols){
    unsigned int global_row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int value_per_thread = cols / kBDIMX;
    unsigned int packs_per_thread = value_per_thread / VecSize;
    using VecType = AlignedVector<float, VecSize>;

    float thread_max = -CUDART_INF_F;
    VecType load;
    for (int pack_idx = 0; pack_idx < packs_per_thread; pack_idx++) {
        Load<float, VecSize>(&in[global_row * cols + (pack_idx * kBDIMX + tid) * VecSize], &load);
        #pragma unroll
        for (int i = 0; i < VecSize; i++) {
            thread_max = max(load[i], thread_max);
        }
    }

    float row_max = BlockAllReduce<ReduceMax>(thread_max);

    VecType store;
    float thread_sum = 0.f;
    for (int pack_idx = 0; pack_idx < packs_per_thread; pack_idx++) {
        Load<float, VecSize>(&in[global_row * cols + (pack_idx * kBDIMX + tid) * VecSize], &load);
        #pragma unroll
        for (int i = 0; i < VecSize; i++) {
            // thread_max = max(load[i], thread_max);
            store[i] = exp(load[i] - row_max);
            thread_sum += store[i];
        }
        Store<float, VecSize>(store, &out[global_row * cols + (pack_idx * kBDIMX + tid) * VecSize]);
    }

    float row_sum = BlockAllReduce<ReduceSum>(thread_sum);

    for (int pack_idx = 0; pack_idx < packs_per_thread; pack_idx++) {
        Load<float, VecSize>(&out[global_row * cols + (pack_idx * kBDIMX + tid) * VecSize], &load);
        #pragma unroll
        for (int i = 0; i < VecSize; i++) {
            // thread_max = max(load[i], thread_max);
            load[i] = load[i] / row_sum;
        }
        Store<float, VecSize>(load, &out[global_row * cols + (pack_idx * kBDIMX + tid) * VecSize]);
    }
}


// assume no padding : `cols % kBIMX == 0` and `(cols / kBIMX) % VecSize == 0`
template<int VecSize = 2>
__global__ void SoftmaxSharedMemory(const float* in, float* out, const unsigned int rows, const unsigned int cols){
    extern __shared__ __align__(sizeof(float)) unsigned char shared_buf[];
    
    float* buf = reinterpret_cast<float*>(shared_buf);
    unsigned int global_row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int value_per_thread = cols / kBDIMX;
    unsigned int packs_per_thread = value_per_thread / VecSize;
    using VecType = AlignedVector<float, VecSize>;

    float thread_max = - CUDART_INF_F;
    VecType load;
    for (int pack_idx = 0; pack_idx < packs_per_thread; pack_idx++) {
        unsigned int block_offset = (pack_idx * kBDIMX + tid) * VecSize;
        Load<float, VecSize>(&in[global_row * cols + block_offset], &load);
        #pragma unroll
        for (int i = 0; i < VecSize; i++) {
            thread_max = max(load[i], thread_max);
        }
        Store<float, VecSize>(load, &buf[block_offset]);
    }

    float row_max = BlockAllReduce<ReduceMax>(thread_max);

    float thread_sum = 0.f;
    for (int pack_idx = 0; pack_idx < packs_per_thread; pack_idx++) {
        unsigned int block_offset = (pack_idx * kBDIMX + tid) * VecSize;
        #pragma unroll
        for (int i = 0; i < VecSize; i++) {
            float x_minus_max_exp = exp(buf[block_offset + i] - row_max);         
            buf[block_offset + i] = x_minus_max_exp;
            thread_sum += x_minus_max_exp;
        }
    }

    float row_sum = BlockAllReduce<ReduceSum>(thread_sum);
    VecType store;
    for (int pack_idx = 0; pack_idx < packs_per_thread; pack_idx++) {
        unsigned int block_offset = (pack_idx * kBDIMX + tid) * VecSize;
        #pragma unroll
        for (int i = 0; i < VecSize; i++) {
            store[i] = buf[block_offset] / row_sum;
        }

        Store<float, VecSize>(store, &out[global_row * cols + block_offset]);
    }
}
