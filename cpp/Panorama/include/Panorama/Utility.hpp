#pragma once

#include <numeric>
#include <vector>
#include <utility>

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

cv::Mat distortSourceToMatchTarget(
    const cv::Mat &sourceImage, const cv::Mat &targetImage);

namespace utility
{

cv::Mat eigenToCv(const Eigen::Matrix3d &rhs);

Eigen::Matrix3d
findHomographyWithRansac(
    std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences,
    double reprojectionErrorThreshold,
    std::size_t iterations);

template<class PointType>
double getReprojectionError(
    const Eigen::Matrix3d &model,
    const std::pair<PointType, PointType> &correspondence)
{
    // Make the first correspondence a homogenous-coordinate column vector.
    Eigen::Vector3d homogenousFirst;
    homogenousFirst <<
        correspondence.first.x,
        correspondence.first.y,
        1.0;

    Eigen::Vector3d estimate = model * homogenousFirst;
    estimate /= estimate(2);

    return std::sqrt(
            std::pow(estimate(0)-correspondence.second.x, 2)
            + std::pow(estimate(1)-correspondence.second.y, 2));
}

Eigen::Matrix3d
findHomographyWithDirectLinearTransform(
    std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences);

Eigen::Matrix3d
findHomographyWithLeastSquares(
    std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences);

} // end namespace utility
