#pragma once
#include <math_constants.h>

constexpr unsigned int kROWS = 1024;
constexpr unsigned int kCOLS = 1024;
constexpr unsigned int kBDIMX = 256;
constexpr unsigned int kWARPSIZE = 32;

#define HOSTDEVICE __host__ __device__


template <typename T, int Size>
struct alignas(sizeof(T) * Size) AlignedVector {
  T val[Size];

  HOSTDEVICE inline const T& operator[](int i) const { return val[i]; }
  HOSTDEVICE inline T& operator[](int i) { return val[i]; }
};

template <typename T, int Size>
HOSTDEVICE inline void Load(const T* addr, AlignedVector<T, Size>* vec) {
  const AlignedVector<T, Size>* addr_vec =
      reinterpret_cast<const AlignedVector<T, Size>*>(addr);
  *vec = *addr_vec;
}

template <typename T, int Size>
HOSTDEVICE inline void Store(const AlignedVector<T, Size>& vec, T* addr) {
  AlignedVector<T, Size>* addr_vec =
      reinterpret_cast<AlignedVector<T, Size>*>(addr);
  *addr_vec = vec;
}

struct ReduceSum{
    __forceinline__ __device__ float operator()(float a, float b) const {
        return a + b;
    }
};

struct ReduceMax{
    __forceinline__ __device__ float operator()(float a, float b) const {
        return max(a, b);
    }
};
