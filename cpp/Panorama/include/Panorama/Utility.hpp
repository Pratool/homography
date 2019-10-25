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

} // end namespace utility
