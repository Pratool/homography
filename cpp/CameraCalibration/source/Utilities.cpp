#include "Utilities.hpp"

std::vector<cv::Vec3f>
createObjectPoint(const unsigned int innerPointsLength,
                  const unsigned int innerPointsWidth,
                  const double squareSize_mm)
{
    std::vector<cv::Vec3f> outputPoint;
    for (unsigned int i = 0; i < innerPointsLength; ++i)
    {
        for (unsigned int ii = 0; ii < innerPointsWidth; ++ii)
        {
            outputPoint.emplace_back(
                    cv::Vec3f(i*squareSize_mm, ii*squareSize_mm, 0));
        }
    }

    return outputPoint;
}
