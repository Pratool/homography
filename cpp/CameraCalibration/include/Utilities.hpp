/**
 * \file Utilities.hpp
 * Utility functions useful for improving readability, testability, DRY-ness of
 * other code.
 **/

#pragma once

#include <vector>

#include <opencv2/core/core.hpp>

std::vector<cv::Vec3f>
createObjectPoint(const unsigned int innerPointsLength,
                  const unsigned int innerPointsWidth,
                  const double squareSize_mm = 1.0);
