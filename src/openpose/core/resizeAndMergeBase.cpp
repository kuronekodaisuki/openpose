#include <opencv2/imgproc/imgproc.hpp>
#include <openpose/core/resizeAndMergeBase.hpp>

namespace op
{

    template <typename T>
    void resizeKernel(T* targetPtr, const T* const sourcePtr, const int sourceWidth, const int sourceHeight, const int targetWidth,
                      const int targetHeight) 
    {
        //fill source
        cv::Mat source(sourceWidth, sourceHeight, CV_32FC1);
        float *_source = (float*)source.data;
        for (int y = 0; y < sourceHeight; y++) {
            for (int x = 0; x < sourceWidth; x++) {
                _source[y*sourceWidth + x] = (float)sourcePtr[y*sourceWidth + x];
            }
        }
        
        // spatial resize
        cv::Mat target;
        cv::resize(source, target, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);
    
        //fill top
        float *_target = (float*)target.data;
        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                targetPtr[y*targetWidth + x] = _target[y*targetWidth + x];
            }
        }
    }

    template <typename T>
    void resizeKernelAndMerge(T* targetPtr, const T* const sourcePtr, const int sourceNumOffset, const int num, const T* scaleRatios,
                              const int sourceWidth, const int sourceHeight, const int targetWidth, const int targetHeight)
    {
        cv::Mat target(targetWidth, targetHeight, CV_32FC1, cv::Scalar(0));
        for (auto n = 0; n < num; n++)
        {
            const auto currentWidth = (int)(sourceWidth * scaleRatios[n]);
            const auto currentHeight = (int)(sourceHeight * scaleRatios[n]);
            const T* const sourcePtrN = sourcePtr + n * sourceNumOffset;

            //fill source
            cv::Mat source(currentWidth, currentHeight, CV_32FC1);
            float *_source = (float*)source.data;
            for (int y = 0; y < currentHeight; y++) {
                for (int x = 0; x < currentWidth; x++) {
                    _source[y*currentWidth + x] = (float)sourcePtrN[y*currentWidth + x];
                }
            }
            
            // spatial resize
            cv::Mat targetN;
            cv::resize(source, targetN, {targetWidth, targetHeight}, 0, 0, CV_INTER_CUBIC);
            target += targetN;
        }
        target /= num;

        //fill top
        float *_target = (float*)target.data;
        for (int y = 0; y < targetHeight; y++) {
            for (int x = 0; x < targetWidth; x++) {
                targetPtr[y*targetWidth + x] = _target[y*targetWidth + x];
            }
        }
    }

    template <typename T>
    void resizeAndMergeCpu(T* targetPtr, const T* const sourcePtr, const std::array<int, 4>& targetSize,
                           const std::array<int, 4>& sourceSize, const std::vector<T>& scaleRatios)
    {
        try
        {
            //UNUSED(targetPtr);
            //UNUSED(sourcePtr);
            //UNUSED(scaleRatios);
            //UNUSED(targetSize);
            //UNUSED(sourceSize);
            //error("CPU version not completely implemented.", __LINE__, __FUNCTION__, __FILE__);

            // TODO: THIS CODE IS WORKING, BUT IT DOES NOT CONSIDER THE SCALES (I.E. SCALE NUMBER, START AND GAP) 
            const int num = sourceSize[0];
            const int channels = sourceSize[1];
            const int sourceHeight = sourceSize[2];
            const int sourceWidth = sourceSize[3];
            const int targetHeight = targetSize[2];
            const int targetWidth = targetSize[3];

            const auto sourceChannelOffset = sourceHeight * sourceWidth;
            const auto targetChannelOffset = targetWidth * targetHeight;

            // No multi-scale merging
            if (targetSize[0] > 1)
            {
                //stupid method
                for (int n = 0; n < num; n++)
                {
                    const auto offsetBase = n*channels;
                    for (int c = 0; c < channels; c++)
                    {
                        const auto offset = offsetBase + c;
                        resizeKernel(targetPtr + offset * targetChannelOffset,
                                     sourcePtr + offset * sourceChannelOffset,
                                     sourceWidth, sourceHeight, targetWidth, targetHeight);
                    }
                }
            }
            // Multi-scale merging
            else
            {
                // If scale_number > 1 --> scaleRatios must be set
                if (scaleRatios.size() != num)
                    error("The scale ratios size must be equal than the number of scales.", __LINE__, __FUNCTION__, __FILE__);
                const auto maxScales = 10;
                if (scaleRatios.size() > maxScales)
                    error("The maximum number of scales is " + std::to_string(maxScales) + ".", __LINE__, __FUNCTION__, __FILE__);

                // Perform resize + merging
                const auto sourceNumOffset = channels * sourceChannelOffset;
                for (auto c = 0 ; c < channels ; c++)
                    resizeKernelAndMerge(targetPtr + c * targetChannelOffset,
                                         sourcePtr + c * sourceChannelOffset, sourceNumOffset,
                                         num, scaleRatios.data(), sourceWidth, sourceHeight, targetWidth, targetHeight);
            }
        }
        catch (const std::exception& e)
        {
            error(e.what(), __LINE__, __FUNCTION__, __FILE__);
        }
    }

    template void resizeAndMergeCpu(float* targetPtr, const float* const sourcePtr, const std::array<int, 4>& targetSize,
                                    const std::array<int, 4>& sourceSize, const std::vector<float>& scaleRatios);
    template void resizeAndMergeCpu(double* targetPtr, const double* const sourcePtr, const std::array<int, 4>& targetSize,
                                    const std::array<int, 4>& sourceSize, const std::vector<double>& scaleRatios);
}
