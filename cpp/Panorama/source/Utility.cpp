#include <iostream>
#include <vector>
#include <random>

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

#include <Geometry/ConvexPolygon.hpp>
#include <Solvers/Image.hpp>
#include <Solvers/Ransac.hpp>

#include "Panorama/Utility.hpp"

struct WarpBounds
{
    cv::Size size{0, 0};
    cv::Mat targetFrameHomography;
};

cv::Mat eigenToCv(const Eigen::Matrix3d &rhs)
{
    cv::Mat m = (cv::Mat_<double>(3,3) <<
        rhs(0, 0), rhs(0, 1), rhs(0, 2),
        rhs(1, 0), rhs(1, 1), rhs(1, 2),
        rhs(2, 0), rhs(2, 1), rhs(2, 2) );

    return m;
}

WarpBounds getWarpedBounds(const cv::Size &sourceImageSize,
                           const cv::Size &targetImageSize,
                           const cv::Mat &homography)
{
    // Set the output of sourceWarped to be the max height and max width of the
    // warped image.  Due to the homography property that straight lines will
    // stay straight after being transformed, this can be determined by getting
    // the coordinates of the four corners of the image after being transformed.
    // The maximum x and y-coordinates of the four remapped points will be used.

    // Each of these points will be represented in homogenous coordinates.
    // Simply appending 1 assumes the imager plane is of unit distance from the
    // viewer.
    std::array<cv::Vec3d, 4> corners({
        cv::Vec3d(0, 0, 1),
        cv::Vec3d(sourceImageSize.width, 0, 1),
        cv::Vec3d(0, sourceImageSize.height, 1),
        cv::Vec3d(sourceImageSize.width, sourceImageSize.height, 1)});

    double minWidth = std::numeric_limits<double>::infinity();
    double minHeight = std::numeric_limits<double>::infinity();
    double maxWidth = -std::numeric_limits<double>::infinity();
    double maxHeight = -std::numeric_limits<double>::infinity();
    for (auto &corner : corners)
    {
        // Transform corner. Note that values in the output matrix can be
        // negative!
        const auto tmpCorner = cv::Mat(homography * corner);

        // Convert homogenous coordinates back to the imager plane coordinates.
        const auto &z = tmpCorner.at<double>(0, 2);
        cv::Point2d tmpPoint(tmpCorner.at<double>(0, 0)/z,
                             tmpCorner.at<double>(0, 1)/z);

        minWidth = std::min(tmpPoint.x, minWidth);
        minHeight = std::min(tmpPoint.y, minHeight);

        maxWidth = std::max(tmpPoint.x, maxWidth);
        maxHeight = std::max(tmpPoint.y, maxHeight);
    }

    auto width = static_cast<int>(maxWidth-minWidth);
    auto height = static_cast<int>(maxHeight-minHeight);

    // In case the target image would be outside the frame after applying the
    // minWidth and minHeight translation: adjust the emitted size to entirely
    // bound the target image.
    auto newSize = cv::Size(
        width + minWidth > targetImageSize.width
            ? width
            : targetImageSize.width - minWidth,
        height + minHeight > targetImageSize.height
            ? height
            : targetImageSize.height - minHeight);

    // Translate the warped source by an amount to get its edges
    // to line up with the bounds of the image frame, which must be 0.
    cv::Mat targetFrameHomography = cv::Mat::eye(3, 3, CV_64F);
    targetFrameHomography.at<double>(0, 2) -= minWidth;
    targetFrameHomography.at<double>(1, 2) -= minHeight;

    return WarpBounds{newSize, targetFrameHomography};
}

cv::Mat stitchImages(const cv::Mat &sourceImage, const cv::Mat &targetImage)
{
    if (! sourceImage.data || ! targetImage.data)
    {
        throw std::runtime_error("Null image data.");
    }

    std::vector<cv::KeyPoint> sourceKeyPoints;
    cv::Mat sourceDescriptors;

    std::vector<cv::KeyPoint> targetKeyPoints;
    cv::Mat targetDescriptors;

    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    akaze->detectAndCompute(targetImage,
                            cv::noArray(),
                            targetKeyPoints,
                            targetDescriptors);
    akaze->detectAndCompute(sourceImage,
                            cv::noArray(),
                            sourceKeyPoints,
                            sourceDescriptors);


    cv::BFMatcher bruteForceMatcher{cv::NORM_HAMMING};

    std::vector< std::vector<cv::DMatch> > nearestNeighborMatches;
    // Last parameter indicates the "k" in knn i.e. get only the 2 best nearest-
    // neighbor matches. Useful for Lowe's ratio.
    bruteForceMatcher.knnMatch(
            targetDescriptors,
            sourceDescriptors,
            nearestNeighborMatches,
            2);


    // Get all nearest neighbor matches where the best and second-best matches
    // are not too far apart.
    std::vector<cv::DMatch> matches;
    for (const auto &matchVec : nearestNeighborMatches)
    {
        constexpr const double nearestNeighborRatio = 0.8;

        // Use Lowe's ratio here to filter matches.
        if (matchVec[0].distance < nearestNeighborRatio * matchVec[1].distance)
        {
            matches.push_back(matchVec[0]);
        }
    }

    // Create a vector of correspondences between the source and target points.
    std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences;
    for (const auto &match : matches)
    {
        // Get the pixel coordinates of each of the matches.
        correspondences.emplace_back(
            std::make_pair(
                sourceKeyPoints[match.trainIdx].pt,
                targetKeyPoints[match.queryIdx].pt));
    }

    std::clog << "inliers to total: "
              << matches.size() << "/" << sourceKeyPoints.size()
              << " = " << matches.size()/double(sourceKeyPoints.size())
              << std::endl;

    auto eigenHomography = pcv::Ransac<std::pair<cv::Point2f, cv::Point2f>, Eigen::Matrix3d>(
        correspondences,
        pcv::findHomographyWithDirectLinearTransform<float>,
        4,
        pcv::getReprojectionError<float>,
        3.0,
        52980);

    cv::Mat homography = eigenToCv(eigenHomography);

    // Find the bounds and translation vector necessary to align the sourceImage
    // into the targetImage frame.
    WarpBounds warpBounds = getWarpedBounds(
            sourceImage.size(), targetImage.size(), homography);

    // Apply translation transform (captured by targetFrameHomography) AFTER
    // applying projective transformation i.e. order of operations here matter.
    homography = cv::Mat(warpBounds.targetFrameHomography*homography);

    cv::Mat sourceWarped;
    cv::warpPerspective(sourceImage, sourceWarped, homography, warpBounds.size);
    cv::Mat targetWarped;
    cv::warpPerspective(targetImage, targetWarped, warpBounds.targetFrameHomography, warpBounds.size);

    cv::Mat dst;
    cv::addWeighted(sourceWarped, 0.5, targetWarped, 0.5, 0.0, dst);
    cv::imwrite("blended.png", dst);
    cv::imwrite("sourceWarped.png", sourceWarped);
    cv::imwrite("targetWarped.png", targetWarped);

    pcv::ConvexPolygon<int> poly0;
    pcv::ConvexPolygon<int> poly1;
    std::array<cv::Vec3d, 4> corners0({
        cv::Vec3d(0, 0, 1),
        cv::Vec3d(0, sourceImage.size().height, 1),
        cv::Vec3d(sourceImage.size().width, sourceImage.size().height, 1),
        cv::Vec3d(sourceImage.size().width, 0, 1)});
    std::array<cv::Vec3d, 4> corners1({
        cv::Vec3d(0, 0, 1),
        cv::Vec3d(0, targetImage.size().height, 1),
        cv::Vec3d(targetImage.size().width, targetImage.size().height, 1),
        cv::Vec3d(targetImage.size().width, 0, 1)});

    for (const auto &corner : corners0)
    {
        const auto tmpCorner = cv::Mat(homography * corner);
        const auto &z = tmpCorner.at<double>(0, 2);
        poly0.addVertex({static_cast<int>(tmpCorner.at<double>(0, 0)/z),
                         static_cast<int>(tmpCorner.at<double>(0, 1)/z)});
    }

    for (const auto &corner : corners1)
    {
        const auto tmpCorner = cv::Mat(warpBounds.targetFrameHomography * corner);
        const auto &z = tmpCorner.at<double>(0, 2);
        poly1.addVertex({static_cast<int>(tmpCorner.at<double>(0, 0)/z),
                         static_cast<int>(tmpCorner.at<double>(0, 1)/z)});
    }

    std::vector<pcv::ConvexPolygon<int>> polys({poly0, poly1});

    std::array<cv::Mat, 3> channels;
    cv::split(dst, channels.data());
    cv::Mat_<uint8_t> tmpcvmat;
    pcv::makePolygonIntersectionOpencvGrid(
        polys, channels[0].rows, channels[0].cols, tmpcvmat);

    std::array<cv::Mat, 3> outputChannels;
    std::array<cv::Mat, 3> outputChannels2;
    for (std::size_t i = 0; i < 3; ++i)
    {
        outputChannels[i] = cv::Mat(channels[i].mul(tmpcvmat));
    }
    tmpcvmat = (tmpcvmat * -1) + 1;
    for (std::size_t i = 0; i < 3; ++i)
    {
        outputChannels2[i] = cv::Mat(channels[i].mul(tmpcvmat));
        outputChannels[i] = outputChannels[i]+2*outputChannels2[i];
    }

    cv::Mat output;
    cv::merge(outputChannels.data(), 3, output);

    cv::imwrite("intersection.png", output);
    std::clog << "wrote intersection.png" << std::endl;

    return sourceWarped;
}
