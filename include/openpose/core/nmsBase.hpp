#ifndef OPENPOSE_CORE_NMS_BASE_HPP
#define OPENPOSE_CORE_NMS_BASE_HPP

#include <openpose/core/common.hpp>

namespace op
{
    template <typename T>
    OP_API void nmsCpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);

#ifndef CPU_ONLY
    template <typename T>
    OP_API void nmsGpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
#endif
}

#endif // OPENPOSE_CORE_NMS_BASE_HPP
