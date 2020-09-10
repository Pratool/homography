#pragma once

#include <Geometry/ConvexPolygon.hpp>

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

#include <boost/filesystem.hpp>

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
    explicit Image(const boost::filesystem::path &imagePath);

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

    /**
     * Writes the image to a file using a specified codec. Right now the output
     * file type is specified by the file name extension.
     */
    void write(const boost::filesystem::path &savePath);

    /**
     * Crop the image to any polygon specified while retaining the original
     * image size.
     */
    template<class NumericType>
    void crop(const ConvexPolygon<NumericType> &polygon);

    /**
     * Draw a polygon onto the image. Will overwrite the image held in memory.
     * This is a very slow function and meant for debugging purposes only.
     */
    template<class NumericType>
    void drawPolygon(const ConvexPolygon<NumericType> &polygon);

    /**
     * Will keep the transformed image "within frame" -- meaning it will
     * increase (but never decrease) the rasterization frame size to include
     * the full transformed image. This may require using a transformation
     * different from the one intended by the caller, often by a translation to
     * keep the transformed image's bounding box only in positive pixel
     * coordinates. This function will return the actual transformation applied,
     * leaving the caller with complete information to deduce what happened.
     */
    Eigen::Matrix3d transform(const Eigen::Matrix3d &transform);

private:
    std::array<
        Eigen::Matrix<BitDepth, Eigen::Dynamic, Eigen::Dynamic>,
        ChannelNumber> matrices;
};

template<class BitDepth, std::size_t ChannelNumber>
inline Image<BitDepth, ChannelNumber>::Image(
    const boost::filesystem::path &imagePath)
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

template<class BitDepth, std::size_t ChannelNumber>
inline void Image<BitDepth, ChannelNumber>::write(
    const boost::filesystem::path &savePath)
{
    std::array<cv::Mat, ChannelNumber> channels;
    std::transform(
        std::cbegin(matrices), std::cend(matrices), std::begin(channels),
        [](const Eigen::Matrix<BitDepth, Eigen::Dynamic, Eigen::Dynamic> &channel)
        {
            cv::Mat cvBuffer;
            cv::eigen2cv<BitDepth>(channel, cvBuffer);
            return cvBuffer;
        });

    cv::Mat output;
    cv::merge(channels.data(), 3, output);
    cv::imwrite(savePath.native(), output);
}

template<class BitDepth, std::size_t ChannelNumber>
template<class NumericType>
inline void Image<BitDepth, ChannelNumber>::crop(
    const ConvexPolygon<NumericType> &polygon)
{
    decltype(matrices) matrixBuffers{};

    std::transform(
        std::cbegin(matrices), std::cend(matrices), std::begin(matrixBuffers),
        [&polygon](const Eigen::Matrix<BitDepth, Eigen::Dynamic, Eigen::Dynamic> &tmp)
        {
            auto channel = tmp;
            for (auto r = 0; r < channel.rows(); ++r)
            {
                for (auto c = 0; c < channel.cols(); ++c)
                {
                    if (! polygon.isPointContained(Eigen::Vector2d({r, c})))
                    {
                        channel(r, c) = 0;
                    }
                }
            }
            return channel;
        });

    matrices = std::move(matrixBuffers);
}

template<class BitDepth, std::size_t ChannelNumber>
template<class NumericType>
inline void Image<BitDepth, ChannelNumber>::drawPolygon(
    const ConvexPolygon<NumericType> &polygon)
{
    decltype(matrices) matrixBuffers{};

    std::transform(
        std::cbegin(matrices), std::cend(matrices), std::begin(matrixBuffers),
        [&polygon](const Eigen::Matrix<BitDepth, Eigen::Dynamic, Eigen::Dynamic> &tmp)
        {
            auto channel = tmp;
            for (auto r = 0; r < channel.rows(); ++r)
            {
                for (auto c = 0; c < channel.cols(); ++c)
                {
                    if (polygon.isPointContained(Eigen::Vector2d({r, c})))
                    {
                        channel(r, c) = 0;
                    }
                }
            }
            return channel;
        });

    matrices = std::move(matrixBuffers);
}

template<class BitDepth, std::size_t ChannelNumber>
inline Eigen::Matrix3d transform(const Eigen::Matrix3d &transform)
{
    // Transform the bounding box (represented as a convex poly.) to get new
    // transformed poly. Get new transformed poly's bounding box size and
    // top-left corner to potentially add a translation on top of the transform
    // or allocate more memory to this->matrices due to increased bounding box
    // size. Don't need to worry about decreasing memory usage if the image
    // shrinks -- for now.
}

}
