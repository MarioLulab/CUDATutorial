Project(CUDATurorial CXX C CUDA)
set(CMAKE_CUDA_FLAGS "-g -G -O3")
include_directories("../")
include_directories("/luqi/thrust/dependencies/cub/cub/")

add_executable(impl_softmax impl_softmax.cu)