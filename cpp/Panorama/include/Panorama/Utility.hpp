#pragma once

#include <opencv2/core/core.hpp>

cv::Mat distortSourceToMatchTarget(
    const cv::Mat &sourceImage, const cv::Mat &targetImage);
