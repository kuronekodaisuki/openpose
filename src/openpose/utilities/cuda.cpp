#ifndef CPU_ONLY
    #include <cuda.h>
    #include <cuda_runtime.h>
#endif
#include <openpose/utilities/fastMath.hpp>
#include <openpose/utilities/cuda.hpp>

namespace op
{
#ifndef CPU_ONLY
    const dim3 THREADS_PER_BLOCK_TINY{32, 32, 1};
    const dim3 THREADS_PER_BLOCK_SMALL{64, 64, 1};
    const dim3 THREADS_PER_BLOCK_MEDIUM{128, 128, 1};
    const dim3 THREADS_PER_BLOCK_BIG{256, 256, 1};
#endif

    void cudaCheck(const int line, const std::string& function, const std::string& file)
    {
        #ifndef CPU_ONLY
            const auto errorCode = cudaPeekAtLastError();
            if(errorCode != cudaSuccess)
                error("Cuda check failed (" + std::to_string(errorCode) + " vs. " + std::to_string(cudaSuccess) + "): " + cudaGetErrorString(errorCode), line, function, file);
        #endif
    }

    int getGpuNumber()
    {
        int gpuNumber = 0;
        #ifndef CPU_ONLY
            cudaGetDeviceCount(&gpuNumber);
            cudaCheck(__LINE__, __FUNCTION__, __FILE__);
        #endif
        return gpuNumber;
    }

#ifndef CPU_ONLY
    dim3 getNumberCudaBlocks(const Point<int>& frameSize, const dim3 numberCudaThreads)
    {
        try
        {
            return dim3{getNumberCudaBlocks((unsigned int)frameSize.x, numberCudaThreads.x),
                        getNumberCudaBlocks((unsigned int)frameSize.y, numberCudaThreads.y),
                        numberCudaThreads.z};
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return dim3{};
        }
    }

    std::pair<dim3, dim3> getNumberCudaThreadsAndBlocks(const Point<int>& frameSize)
    {
        try
        {
            // Image <= 480p    --> THREADS_PER_BLOCK_TINY
            // Image <= 720p    --> THREADS_PER_BLOCK_SMALL
            // Image <= 16K     --> THREADS_PER_BLOCK_MEDIUM
            // Image > 16K      --> THREADS_PER_BLOCK_BIG
            const auto maxValue = fastMax(frameSize.x, frameSize.y);
            auto threadsPerBlock = THREADS_PER_BLOCK_TINY;
            if (maxValue >= 16384)
                threadsPerBlock = THREADS_PER_BLOCK_BIG;
            else if (maxValue > 1280)
                threadsPerBlock = THREADS_PER_BLOCK_MEDIUM;
            else if (maxValue > 640)
                threadsPerBlock = THREADS_PER_BLOCK_SMALL;
            return std::make_pair(threadsPerBlock, getNumberCudaBlocks(frameSize, threadsPerBlock));
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
            return std::make_pair(dim3{1,1,1}, dim3{1,1,1});
        }
    }
#endif
}
