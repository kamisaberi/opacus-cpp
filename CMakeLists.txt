cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(opacus_cpp LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directory
include_directories(include ${TORCH_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/privacy_engine.cpp
    src/dp_optimizer.cpp
    src/per_sample_grad.cpp
    src/accountants/rdp_accountant.cpp
    src/kernels/dp_kernels.cu
)

# Build static library
add_library(opacus_lib STATIC ${SOURCES})
target_link_libraries(opacus_lib PUBLIC ${TORCH_LIBRARIES})

# Set CUDA compilation flags
set_target_properties(opacus_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89;90"
)

# Build benchmark binary
add_executable(benchmark examples/benchmark.cpp)
target_link_libraries(benchmark PRIVATE opacus_lib ${TORCH_LIBRARIES})