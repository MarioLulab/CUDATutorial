#pragma once
#include "impl_softmax.h"


template<class ReduceOp, int WarpSize = kWARPSIZE>
__forceinline__ __device__ float WarpAllReduce(float val){
    for (unsigned int mask = WarpSize >> 1; mask > 0; mask >>= 1){
        val = ReduceOp()(val, __shfl_xor_sync(0xffffffff, val, mask, WarpSize));
    }
    
    val = __shfl_sync(0xffffffff, val, 0, WarpSize);
    return val;
}

// assume no padding : `cols % kBIMX == 0` and `(cols / kBIMX) % VecSize == 0`
template<int VecSize = 2>
__global__ void SoftmaxWarpReduce(const float* in, float* out, unsigned int rows, unsigned int cols){
    extern __shared__ __align__(sizeof(float)) unsigned char shared_buf[];
    
    float* buf = reinterpret_cast<float*>(shared_buf);
    unsigned int global_row = blockIdx.x;
    unsigned int tid = threadIdx.x;
    unsigned int value_per_thread = cols / kWARPSIZE;
    unsigned int packs_per_thread = value_per_thread / VecSize;
    using VecType = AlignedVector<float, VecSize>;

    float thread_max = - CUDART_INF_F;
    VecType load;
    for (int pack_idx = 0; pack_idx < packs_per_thread; pack_idx++) {
        unsigned int block_offset = (pack_idx * kWARPSIZE + tid) * VecSize;
        Load<float, VecSize>(&in[global_row * cols + block_offset], &load);
        #pragma unroll
        for (int i = 0; i < VecSize; i++) {
            thread_max = max(load[i], thread_max);
        }
        Store<float, VecSize>(load, &buf[block_offset]);
    }
    float row_max = WarpAllReduce<ReduceMax>(thread_max);

    float thread_sum = 0.f;
    for (int pack_idx = 0; pack_idx < packs_per_thread; pack_idx++) {
        unsigned int block_offset = (pack_idx * kWARPSIZE + tid) * VecSize;
        #pragma unroll
        for (int i = 0; i < VecSize; i++) {
            float x_minus_max_exp = exp(buf[block_offset + i] - row_max);         
            buf[block_offset + i] = x_minus_max_exp;
            thread_sum += x_minus_max_exp;
        }
    }


    float row_sum = WarpAllReduce<ReduceSum>(thread_sum);
    VecType store;
    for (int pack_idx = 0; pack_idx < packs_per_thread; pack_idx++) {
        unsigned int block_offset = (pack_idx * kWARPSIZE + tid) * VecSize;
        #pragma unroll
        for (int i = 0; i < VecSize; i++) {
            store[i] = buf[block_offset] / row_sum;
        }

        Store<float, VecSize>(store, &out[global_row * cols + block_offset]);
    }
}