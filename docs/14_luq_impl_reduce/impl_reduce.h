#pragma once
#include <cuda_runtime.h>

constexpr unsigned int kBDIMX = 256;
constexpr unsigned int kWarpSize = 32;
constexpr unsigned int FULL_MASK = 0xffffffff;
