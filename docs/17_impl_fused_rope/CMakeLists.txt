Project(CUDATurorial CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-g -G -O3")
include_directories("../")

add_executable(impl_fused_rope impl_fused_rope.cu)