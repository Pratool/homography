#pragma once

#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Dense>


namespace pcv
{

template<class NumericType>
double getLineModelError(
    const Eigen::Vector2d &model,
    const Eigen::Matrix<NumericType, 2, 1> point)
{
    static_assert(std::is_arithmetic<NumericType>::value,
                  "Must have a numerical point type.");

    double yEstimate = (1.0 - model[0]*point.x())/model[1];

    return std::pow(yEstimate-point.y(), 2);
}

template<class NumericType>
Eigen::Vector2d
findLineModelFromPoints(
    std::vector<Eigen::Matrix<NumericType, 2, 1>> points)
{
    static_assert(std::is_arithmetic<NumericType>::value,
                  "Must have a numerical point type.");

    if (points.size() < 2)
    {
        throw std::runtime_error("There must be at least the minimum number of "
                                 "data points required by modeling function.");
    }

    Eigen::Matrix<double, 2, 2> linearEquationsLhs;
    linearEquationsLhs << points[0].x(), points[0].y(),
                          points[1].x(), points[1].y();

    Eigen::Vector2d linearEquationsRhs;
    linearEquationsRhs << 1, 1;

    Eigen::Vector2d model =
        linearEquationsLhs.colPivHouseholderQr().solve(linearEquationsRhs);

    return model;
}

} // end namespace pcv
