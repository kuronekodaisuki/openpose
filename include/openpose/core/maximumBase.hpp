#ifndef OPENPOSE_CORE_MAXIMUM_BASE_HPP
#define OPENPOSE_CORE_MAXIMUM_BASE_HPP

#include <openpose/core/common.hpp>

namespace op
{
    template <typename T>
    OP_API void maximumCpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);

#ifdef CPU_ONLY
    template <typename T>
    OP_API void maximumGpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
#endif
}

#endif // OPENPOSE_CORE_MAXIMUM_BASE_HPP
