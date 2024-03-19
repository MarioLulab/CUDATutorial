#pragma once
#include <math_constants.h>
#include <cassert>

constexpr unsigned int kM = 1024;
constexpr unsigned int kN = 1024;
constexpr unsigned int kK = 512;
constexpr unsigned int kBDIMX = 64;
constexpr unsigned int kBDIMY = 64;
constexpr unsigned int kbM = kBDIMY;
constexpr unsigned int kbN = kBDIMX;
constexpr unsigned int kbK = 8;
constexpr unsigned int krM = 8;
constexpr unsigned int krN = 8;

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
