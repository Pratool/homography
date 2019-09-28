#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "Panorama/Utility.hpp"

struct WarpBounds
{
    cv::Size size{0, 0};
    double minWidth;
    double minHeight;
};

WarpBounds getWarpedBounds(const cv::Size &sourceImageSize,
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

    return WarpBounds{
        cv::Size{static_cast<int>(maxWidth-minWidth),
                 static_cast<int>(maxHeight-minHeight)},
        minWidth,
        minHeight};
}

cv::Mat distortSourceToMatchTarget(
    const cv::Mat &sourceImage, const cv::Mat &targetImage)
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
    std::clog << "finished running matcher" << std::endl;


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
    std::clog << "filtered matches" << std::endl;


    // Get the pixel coordinates of each of the matches.
    std::vector<cv::Point2f> sourceRansacInliers;
    std::vector<cv::Point2f> targetRansacInliers;
    for (const auto &match : matches)
    {
        targetRansacInliers.push_back(targetKeyPoints[match.queryIdx].pt);
        sourceRansacInliers.push_back(sourceKeyPoints[match.trainIdx].pt);
    }

    std::clog << "inliers to total: "
              << matches.size() << "/" << sourceKeyPoints.size()
              << " = " << matches.size()/double(sourceKeyPoints.size())
              << std::endl;

    cv::Mat homography = cv::findHomography(sourceRansacInliers,
                                            targetRansacInliers,
                                            cv::RANSAC,
                                            3);
    std::clog << "found homography matrix:" << std::endl;
    std::clog << homography << std::endl;


    // Find the bounds and translation vector necessary to align the sourceImage
    // into the targetImage frame.
    WarpBounds warpBounds = getWarpedBounds(sourceImage.size(), homography);
    std::clog << "warped image size " << warpBounds.size << std::endl;
    std::clog << "warped image min. bounds: "
              << warpBounds.minWidth << " x " << warpBounds.minHeight << std::endl;

    // Translate the warped source by an amount to get its edges
    // to line up with the bounds of the image frame, which must be 0.
    cv::Mat targetFrameHomography = cv::Mat::eye(3, 3, CV_64F);
    targetFrameHomography.at<double>(0, 2) -= warpBounds.minWidth;
    targetFrameHomography.at<double>(1, 2) -= warpBounds.minHeight;

    std::clog << "translation matrix to keep source and target images in bound:" << std::endl;
    std::clog << targetFrameHomography << std::endl;

    // Apply translation transform (captured by targetFrameHomography) AFTER
    // applying projective transformation i.e. order of operations here matter.
    homography = cv::Mat(targetFrameHomography*homography);
    std::clog << homography << std::endl;

    const auto newSize = cv::Size(
        warpBounds.size.width + warpBounds.minWidth > targetImage.size().width ?
            warpBounds.size.width
                :
            targetImage.size().width-warpBounds.minWidth,

        warpBounds.size.height + warpBounds.minHeight > targetImage.size().height ?
            warpBounds.size.height
                :
            targetImage.size().height-warpBounds.minHeight);

    cv::Mat sourceWarped;
    cv::warpPerspective(sourceImage, sourceWarped, homography, newSize);
    cv::Mat targetWarped;
    cv::warpPerspective(targetImage, targetWarped, targetFrameHomography, newSize);

    cv::Mat dst;
    cv::addWeighted(sourceWarped, 0.5, targetWarped, 0.5, 0.0, dst);
    cv::imwrite("blended.png", dst);
    std::clog << "wrote blended.png" << std::endl;

    return sourceWarped;
}

namespace utility
{

Eigen::Matrix3d
findHomographyWithDirectLinearTransform(
    std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences)
{
    Eigen::Matrix<double, 8, 9> joinedPoints;

    std::array<Eigen::Matrix<double, 2, 9>, 4> points;

    std::transform(
        std::cbegin(correspondences),
        std::cend(correspondences),
        std::begin(points),
        [](std::pair<cv::Point2f, cv::Point2f> pointPair)
        {
            Eigen::Matrix<double, 2, 9> tmp;
            tmp <<
                -pointPair.first.x,
                -pointPair.first.y, -1, 0,
                0,
                0,
                pointPair.first.x*pointPair.second.x,
                pointPair.first.y*pointPair.second.x,
                pointPair.second.x,
                0,
                0,
                0,
                -pointPair.first.x,
                -pointPair.first.y,
                -1,
                pointPair.first.x*pointPair.second.y,
                pointPair.first.y*pointPair.second.y,
                pointPair.second.y;

            return tmp;
        });

    joinedPoints << points[0], points[1], points[2], points[3];

    Eigen::JacobiSVD<decltype(joinedPoints)> svd(joinedPoints, Eigen::ComputeFullV);

    return Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(svd.matrixV().col(8).data());
}

Eigen::Matrix3d
findHomographyWithLeastSquares(
    std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences)
{
    // Allocate appropriate amount of memory for Matrix.
    Eigen::MatrixXd joinedPoints(2*correspondences.size(), 9);

    for (std::size_t correspondenceIndex = 0; correspondenceIndex < correspondences.size(); ++correspondenceIndex)
    {
        const auto &pointPair = correspondences[correspondenceIndex];

        joinedPoints.row(2*correspondenceIndex) <<
            -pointPair.first.x,
            -pointPair.first.y,
            -1,
            0,
            0,
            0,
            pointPair.first.x*pointPair.second.x,
            pointPair.first.y*pointPair.second.x,
            pointPair.second.x;

        joinedPoints.row(2*correspondenceIndex+1) <<
            0,
            0,
            0,
            -pointPair.first.x,
            -pointPair.first.y,
            -1,
            pointPair.first.x*pointPair.second.y,
            pointPair.first.y*pointPair.second.y,
            pointPair.second.y;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(
        joinedPoints.transpose()*joinedPoints, Eigen::ComputeFullV);

    return Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(svd.matrixV().col(8).data());
}

} // end namespace utility
