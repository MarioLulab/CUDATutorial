#pragma once


#include "./impl_fused_rope.h"


namespace paddle{

#define HOSTDEVICE __host__ __device__
#define UNUSED __attribute__((unused))


template <typename T, size_t kStart, size_t kEnd, bool kStop>
struct UnrollVarArgsAssignImpl {
  template <typename... Args>
  HOSTDEVICE inline static void Run(T *d, T val, Args... args) {
    static_assert(sizeof...(args) + 1 == kEnd - kStart, "Wrong argument");
    d[kStart] = val;
    UnrollVarArgsAssignImpl<T, kStart + 1, kEnd, kStart + 1 == kEnd>::Run(
        d, args...);
  }
};

template <typename T, size_t kStart, size_t kEnd>
struct UnrollVarArgsAssignImpl<T, kStart, kEnd, true> {
  HOSTDEVICE inline static void Run(T *d) {}
};


template <typename T>
struct UnrollVarArgsAssign {
  template <typename... Args>
  HOSTDEVICE inline static void Run(T *d, Args... args) {
    UnrollVarArgsAssignImpl<T, 0, sizeof...(Args), sizeof...(Args) == 0>::Run(
        d, args...);
  }
};


namespace detail{
template <size_t kStart, size_t kEnd, bool kStop>
struct UnrollFillConstant {
  template <typename T>
  HOSTDEVICE inline static void Run(T *data, T val) {
    data[kStart] = val;
    UnrollFillConstant<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(data, val);
  }
};

template <size_t kStart, size_t kEnd>
struct UnrollFillConstant<kStart, kEnd, true> {
  template <typename T>
  HOSTDEVICE inline static void Run(T *data UNUSED, T val UNUSED) {}
};


template <size_t kStart, size_t kEnd, bool kStop>
struct UnrollCompare {
  template <typename T>
  HOSTDEVICE inline static bool Run(const T *d1, const T *d2) {
    return d1[kStart] == d2[kStart] &&
           UnrollCompare<kStart + 1, kEnd, kStart + 1 == kEnd>::Run(d1, d2);
  }
};

template <size_t kStart, size_t kEnd>
struct UnrollCompare<kStart, kEnd, true> {
  template <typename T>
  HOSTDEVICE inline constexpr static bool Run(const T *d1 UNUSED,
                                              const T *d2 UNUSED) {
    return true;
  }
};
} // namespace detail

template <size_t N>
using UnrollFillConstant = detail::UnrollFillConstant<0, N, N == 0>;

template <size_t N>
using UnrollCompare = detail::UnrollCompare<0, N, N == 0>;

template <typename T, size_t N>
class Array {
 public:
  static constexpr size_t kSize = N;

  HOSTDEVICE inline Array() {}

  template <typename... Args>
  HOSTDEVICE inline explicit Array(const T &val, Args... args) {
    static_assert(N == sizeof...(Args) + 1, "Invalid argument");
    UnrollVarArgsAssign<T>::Run(data_, val, args...);
  }

  HOSTDEVICE inline void Fill(const T &val) {
    UnrollFillConstant<N>::Run(data_, val);
  }

  HOSTDEVICE inline const T *Get() const { return data_; }

  HOSTDEVICE inline T *GetMutable() { return data_; }

  HOSTDEVICE inline T &operator[](size_t i) { return *advance(data_, i); }

  // Writing "return data_[i]" would cause compilation warning/error:
  // "array subscript is above array bound" in Python 35 CI.
  // It seems that it is a false warning of GCC if we do not check the bounds
  // of array index. But for better performance, we do not check in operator[]
  // like what is in STL. If users want to check the bounds, use at() instead
  HOSTDEVICE inline const T &operator[](size_t i) const {
    return *advance(data_, i);
  }

  HOSTDEVICE inline T &at(size_t i) {
    return (*this)[i];
  }

  HOSTDEVICE inline const T &at(size_t i) const {
    return (*this)[i];
  }

  HOSTDEVICE constexpr size_t size() const { return N; }

  HOSTDEVICE inline bool operator==(const Array<T, N> &other) const {
    return UnrollCompare<N>::Run(data_, other.data_);
  }

  HOSTDEVICE inline bool operator!=(const Array<T, N> &other) const {
    return !(*this == other);
  }

 private:
  template <typename U>
  HOSTDEVICE static inline U *advance(U *ptr, size_t i) {
    return ptr + i;
  }

  T data_[N] = {};
};

// Aligned vector generates vectorized load/store on CUDA.
template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];

  HOSTDEVICE inline const T& operator[](int i) const { return val[i]; }
  HOSTDEVICE inline T& operator[](int i) { return val[i]; }
};

template <typename T, typename MPType, int VecSize>
using VectorizedFusedRopeCudaKernelFunc =
    void (*)(Array<const T*, 3> ins_data,
             Array<const T*, 2> sin_cos_data,
             const int64_t* position_ids_data,
             bool flag_sin_cos,
             int sign,
             int64_t batch_size,
             int64_t seq_len,
             int64_t num_heads,
             int64_t head_dim,
             int64_t batch_stride,
             int64_t seq_stride,
             Array<T*, 3> outs_data,
             int num_inputs,
             MPType div_c);

template <typename T, typename MPType, int VecSize = 2>
__device__ void VectorizedGetSinCos(Array<const T*, 2> sin_cos_data,
                                    const int64_t* position_ids_data,
                                    bool flag_sin_cos,
                                    int64_t index,
                                    int64_t batch_size,
                                    int64_t seq_len,
                                    int64_t num_heads,
                                    int64_t head_dim,
                                    int64_t batch_stride,
                                    int64_t seq_stride,
                                    MPType* out_sin,
                                    MPType* out_cos,
                                    MPType div_c,
                                    float rotary_emb_base) {
  MPType* sin_value = out_sin;
  MPType* cos_value = out_cos;

  if (flag_sin_cos) {
#pragma unroll
    for (int64_t nx = 0; nx < VecSize; ++nx) {
      int64_t pos_seq_ori = (index + nx) / seq_stride % seq_len;
      int64_t pos_seq;
      if (position_ids_data) {
        int64_t pos_bs = (index + nx) / batch_stride % batch_size;
        int64_t index_ids = pos_bs * seq_len + pos_seq_ori;
        pos_seq = position_ids_data[index_ids];
      } else {
        pos_seq = pos_seq_ori;
      }
      int64_t pos_head = (index + nx) % head_dim;
      int64_t index_sc = pos_seq * head_dim + pos_head;
      const T* sin_input = sin_cos_data[0] + index_sc;
      const T* cos_input = sin_cos_data[1] + index_sc;

      sin_value[nx] = static_cast<MPType>(sin_input[0]);
      cos_value[nx] = static_cast<MPType>(cos_input[0]);
    }
  } else {
#pragma unroll
    for (int nx = 0; nx < VecSize; ++nx) {
      // get sin_index and cos_index
      int64_t pos_seq = (index + nx) / seq_stride % seq_len;

      MPType idx = static_cast<MPType>(((index + nx) % head_dim) / 2 * 2.0);
      MPType indicses =
          static_cast<MPType>(1) /
          pow(static_cast<MPType>(rotary_emb_base), idx * static_cast<MPType>(div_c));
      MPType value = pos_seq * indicses;
      sin_value[nx] = sin(value);
      cos_value[nx] = cos(value);
    }
  }
}

template <typename T, typename MPType, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateEveryTwoKernel(
    Array<const T*, 3> ins_data,
    Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t batch_stride,
    int64_t seq_stride,
    Array<T*, 3> outs_data,
    int num_inputs,
    MPType div_c,
    float rotary_emb_base) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;

  for (; index < size; index += stride) {
    VectorizedGetSinCos(sin_cos_data,
                        position_ids_data,
                        flag_sin_cos,
                        index,
                        batch_size,
                        seq_len,
                        num_heads,
                        head_dim,
                        batch_stride,
                        seq_stride,
                        sin_value,
                        cos_value,
                        div_c,
                        rotary_emb_base);

#pragma unroll
    for (int iter = 0; iter < 3; iter++) {
      if (iter >= num_inputs) break;
      const T* input = ins_data[iter] + index;
      VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

#pragma unroll
      for (int nx = 0; nx < kVectorsPerThread; ++nx) {
        int pr_index = nx * 2;
        int ls_index = pr_index + 1;

        MPType p0 = static_cast<MPType>(input[pr_index]);
        MPType p1 = static_cast<MPType>(input[ls_index]);

        if (sign == 1) {
          result[pr_index] = cos_value[pr_index] * p0;
          result[pr_index] -= sin_value[pr_index] * p1;

          result[ls_index] = sin_value[ls_index] * p0;
          result[ls_index] += cos_value[ls_index] * p1;
        } else if (sign == -1) {
          result[pr_index] =
              cos_value[pr_index] * p0 + sin_value[ls_index] * p1;
          result[ls_index] =
              cos_value[ls_index] * p1 - sin_value[pr_index] * p0;
        }

        store[pr_index] = static_cast<T>(result[pr_index]);
        store[ls_index] = static_cast<T>(result[ls_index]);
      }
      out[0] = *(reinterpret_cast<VecType*>(store));
    }
  }
}

template <typename T, typename MPType, int VecSize = 2>
__global__ void VectorizedFusedRopeWithRotateHalfKernel(
    Array<const T*, 3> ins_data,
    Array<const T*, 2> sin_cos_data,
    const int64_t* position_ids_data,
    bool flag_sin_cos,
    int sign,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim,
    int64_t batch_stride,
    int64_t seq_stride,
    Array<T*, 3> outs_data,
    int num_inputs,
    MPType div_c,
    float rotary_emb_base) {
  int64_t index =
      (static_cast<int64_t>(blockIdx.x) * static_cast<int64_t>(blockDim.x) +
       threadIdx.x) *
      VecSize;
  int64_t stride = static_cast<int64_t>(gridDim.x) *
                   static_cast<int64_t>(blockDim.x) * VecSize;
  int64_t size = batch_size * seq_len * num_heads * head_dim;
  MPType sin_value[VecSize];
  MPType cos_value[VecSize];
  MPType result[VecSize];
  T store[VecSize];
  using VecType = AlignedVector<T, VecSize>;
  constexpr int kVectorsPerThread = VecSize / 2;

  for (; index < size; index += stride) {
    VectorizedGetSinCos(sin_cos_data,
                        position_ids_data,
                        flag_sin_cos,
                        index,
                        batch_size,
                        seq_len,
                        num_heads,
                        head_dim,
                        batch_stride,
                        seq_stride,
                        sin_value,
                        cos_value,
                        div_c,
                        rotary_emb_base);

    // use rotate_half mode
    int stride_r = head_dim / 2;
#pragma unroll
    for (int iter = 0; iter < 3; iter++) {
      if (iter >= num_inputs) break;
      // get value_index and rotate_half_index
      int index_v = index;
      int index_r = (index % head_dim) < stride_r ? (index + stride_r)
                                                  : (index - stride_r);
      MPType sign_r = (index % head_dim) < stride_r ? static_cast<MPType>(-1)
                                                    : static_cast<MPType>(1);
      const T* input_v = ins_data[iter] + index_v;
      const T* input_r = ins_data[iter] + index_r;
      VecType* out = reinterpret_cast<VecType*>(outs_data[iter] + index);

#pragma unroll
      for (int nx = 0; nx < VecSize; ++nx) {
        MPType p0 = static_cast<MPType>(input_v[nx]);
        MPType p1 = static_cast<MPType>(input_r[nx]);

        result[nx] = cos_value[nx] * p0 + sign * sign_r * sin_value[nx] * p1;

        store[nx] = static_cast<T>(result[nx]);
      }
      out[0] = *(reinterpret_cast<VecType*>(store));
    }
  }
}

} // namespace paddle