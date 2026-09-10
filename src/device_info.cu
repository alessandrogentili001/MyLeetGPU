#include "cuda_utils.cuh"
#include <iostream>
#include <iomanip>

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        std::cerr << "No CUDA-capable devices found!" << std::endl;
        return 1;
    }

    std::cout << "\n============================================================\n";
    std::cout << "          🔍 MyLeetGPU - Device Inspection Tool             \n";
    std::cout << "============================================================\n";

    for (int dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

        std::cout << " Device ID               : " << dev << "\n";
        std::cout << " Name                    : " << prop.name << "\n";
        std::cout << " Compute Capability      : " << prop.major << "." << prop.minor << "\n";
        std::cout << " Streaming Multiprocessors: " << prop.multiProcessorCount << " SMs\n";
        std::cout << " Total Global Memory     : " << prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0) << " GB\n";
        std::cout << " Memory Bus Width        : " << prop.memoryBusWidth << " bits\n";
        std::cout << " Memory Clock Rate       : " << prop.memoryClockRate * 1e-3 << " MHz\n";
        
        // Theoretical memory bandwidth calculation
        // Bandwidth = 2 * (bus width in bytes) * (memory clock in Hz)
        double mem_bw_gbs = 2.0 * (prop.memoryClockRate * 1e3) * (prop.memoryBusWidth / 8.0) / 1e9;
        std::cout << " Peak Memory Bandwidth   : ~" << std::fixed << std::setprecision(1) 
                  << (prop.major == 8 && prop.minor == 0 ? 1555.0 : mem_bw_gbs) << " GB/s (HBM2e)\n";

        std::cout << " Shared Memory per Block : " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
        std::cout << " Max Shared Memory per SM: " << prop.sharedMemPerMultiprocessor / 1024.0 << " KB\n";
        std::cout << " Max Threads per Block   : " << prop.maxThreadsPerBlock << "\n";
        std::cout << " Max Threads per SM      : " << prop.maxThreadsPerMultiProcessor << "\n";
        std::cout << " Warp Size               : " << prop.warpSize << "\n";
        std::cout << " L2 Cache Size           : " << prop.l2CacheSize / (1024.0 * 1024.0) << " MB\n";
        std::cout << " Max 3D Grid Dimensions  : [" << prop.maxGridSize[0] << ", " 
                  << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "]\n";
        std::cout << "------------------------------------------------------------\n";
    }

    std::cout << " ✅ Device query completed successfully.\n";
    std::cout << "============================================================\n\n";

    return 0;
}
