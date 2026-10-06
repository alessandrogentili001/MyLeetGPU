add_executable(sgemm main.cu kernel.cu)


# Link against cuBLAS for the baseline comparison
find_package(CUDAToolkit REQUIRED)
target_link_libraries(sgemm PRIVATE CUDA::cublas)
