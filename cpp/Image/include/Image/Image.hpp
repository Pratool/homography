#pragma once

#include <Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>

namespace pcv
{


template<class BitDepth, std::size_t ChannelNumber>
class Image
{
public:
    /**
     * Delete default constructor because that makes this type difficult to work
     * with.
     */
    Image() = delete;


    /**
     * Construct with a path to an image. Will read in data, using appropriate
     * codec, into a matrix that can later be accessed directly, if necessary.
     */
    explicit Image(const std::filesystem::path &imagePath)
    {
        cv::Mat cvBuffer = cv::imread(imagePath.native(), 1);

        if (! cvBuffer.data)
        {
            throw std::runtime_error("OpenCV failed to load data from file.");
        }

        std::array<cv::Mat, ChannelNumber> channels;
        cv::split(cvBuffer, channels.data());
        std::transform(
            std::cbegin(channels), std::cend(channels), std::begin(matrices),
            [](const cv::Mat &channel)
            {
                Eigen::Matrix<BitDepth, Eigen::Dynamic, Eigen::Dynamic> eigenBuffer;
                cv::cv2eigen<BitDepth>(channel, eigenBuffer);
                return eigenBuffer;
            });
    }


    /**
     * Prevent a deep copy default constructor to prevent excessive copying.
     * There should exist helper methods to do deep copies in order to keep the
     * library computationally efficent.
     */
    Image(const Image& other) = delete;
    Image &operator=(const Image& other) = delete;

    /**
     * Default move constructor is exactly how one would implment a custom move
     * constructor. Don't be fooled by the lack of a default constructor!
     */
    Image(Image &&other) = default;
    Image &operator=(Image&& other) = default;

private:
    std::array<
        Eigen::Matrix<BitDepth, Eigen::Dynamic, Eigen::Dynamic>,
        ChannelNumber> matrices;
};


/*
template<class BitDepth>
Image::Image(const std::filesystem::path &imagePath)
{
    cv::Mat cvBuffer = cv::imread(imagePath.native(), 1);

    if (! cvBuffer.data)
    {
        throw ImageLoadError("OpenCV failed to load data from file.");
    }

    matrix = cv::cv2eigen<BitDepth>(cvBuffer);
}
*/


}
