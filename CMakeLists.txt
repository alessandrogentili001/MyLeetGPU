cmake_minimum_required(VERSION 3.20)
project(MyLeetGPU LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Default to NVIDIA A100 (sm_80) if not explicitly set
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Target CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Compiler flags
# -lineinfo: adds source line mapping for Nsight Systems and Nsight Compute profiling
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -lineinfo -Xcompiler -Wall")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Device Inspection Tool
add_executable(device_info src/device_info.cu)

# Add problems directory if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/problems/CMakeLists.txt")
    add_subdirectory(problems)
endif()
