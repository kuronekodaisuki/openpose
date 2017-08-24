#include <openpose/core/nmsBase.hpp>

namespace op
{
    template <typename T>
    void nmsCpu(T* targetPtr, int* kernelPtr, const T* const sourcePtr, const T threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize)
    {
        try
        {
            UNUSED(kernelPtr);

            const auto num = sourceSize[0];
            const auto height = sourceSize[2];
            const auto width = sourceSize[3];
            const auto channels = targetSize[1];
            const auto maxPeaks = targetSize[2]-1;
            const auto imageOffset = height * width;
            const auto offsetTarget = (maxPeaks+1)*targetSize[3];

            // stupid method
            for (int n = 0; n < num; n++)
            {
                for (auto c = 0; c < channels; c++)
                {
                    // log("channel: " + std::to_string(c));
                    const auto offsetChannel = (n * channels + c);
                    const auto* const sourcePtrOffsetted = sourcePtr + offsetChannel * imageOffset;
                    auto* targetPtrOffsetted = targetPtr + offsetChannel * offsetTarget;

                    int peakCount = 0;
                    for (int y = 1; y < height-1; y++)
                    {
                        for (int x = 1; x < width-1; x++)
                        {
                            const T value = sourcePtrOffsetted[y*width + x];
                            if (value >= threshold)
                            {
                                const auto top_left        = sourcePtrOffsetted[(y-1)*width + x-1];
                                const auto top             = sourcePtrOffsetted[(y-1)*width + x];
                                const auto top_right       = sourcePtrOffsetted[(y-1)*width + x+1];
                                const auto left            = sourcePtrOffsetted[y*width + (x-1)];
                                const auto right           = sourcePtrOffsetted[y*width + (x+1)];
                                const auto bottom_left     = sourcePtrOffsetted[(y+1)*width + x-1];
                                const auto bottom          = sourcePtrOffsetted[(y+1)*width + x];
                                const auto bottom_right    = sourcePtrOffsetted[(y+1)*width + x+1];

                                if (value > top_left && 
                                    value > top && 
                                    value > top_right && 
                                    value > left && 
                                    value > right && 
                                    value > bottom_left && 
                                    value > bottom && 
                                    value > bottom_right)
                                {
                                    //means A[globalIdx] == A[globalIdx + 1] as the kernelPtr[globalIdx]-th repeat
                                    const auto peakIndex = peakCount; //0-index
                                    const auto peakLocX = x; 
                                    const auto peakLocY = y;

                                    if (peakIndex < maxPeaks) // limitation
                                    {
                                        T xAcc = 0.f;
                                        T yAcc = 0.f;
                                        T scoreAcc = 0.f;
                                        for (auto dy = -3 ; dy < 4 ; dy++)
                                        {
                                            const auto py = peakLocY + dy;
                                            if (0 <= py && py < height) // height = 368
                                            {
                                                for (auto dx = -3 ; dx < 4 ; dx++)
                                                {
                                                    const auto px = peakLocX + dx;
                                                    if (0 <= px && px < width) // width = 656
                                                    {
                                                        const auto score = sourcePtrOffsetted[py * width + px];
                                                        if (score > 0)
                                                        {
                                                            xAcc += px*score;
                                                            yAcc += py*score;
                                                            scoreAcc += score;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                
                                        const auto outputIndex = (peakIndex + 1) * 3;
                                        targetPtrOffsetted[outputIndex] = xAcc / scoreAcc;
                                        targetPtrOffsetted[outputIndex + 1] = yAcc / scoreAcc;
                                        targetPtrOffsetted[outputIndex + 2] = sourcePtrOffsetted[peakLocY*width + peakLocX];
                                        peakCount++;
                                    }
                                }
                            }
                        }
                    }
                    targetPtrOffsetted[0] = peakCount;
                }
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void nmsCpu(float* targetPtr, int* kernelPtr, const float* const sourcePtr, const float threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
    template void nmsCpu(double* targetPtr, int* kernelPtr, const double* const sourcePtr, const double threshold, const std::array<int, 4>& targetSize, const std::array<int, 4>& sourceSize);
}
