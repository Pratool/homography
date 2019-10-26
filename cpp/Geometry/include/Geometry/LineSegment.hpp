#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <iostream>
#include <type_traits>

namespace pcv
{

template<class NumericType0, class NumericType1>
double getEuclideanDistance(
    const Eigen::Matrix<NumericType0, 2, 1> &point0,
    const Eigen::Matrix<NumericType1, 2, 1> &point1)
{
    return std::sqrt(
            std::pow(point0.x()-point1.x(), 2)
          + std::pow(point0.y()-point1.y(), 2));
}

template<class NumericType>
std::optional<Eigen::Vector2d> getSegmentIntersectionPoint2(
    const Eigen::Matrix<NumericType, 2, 1> &segmentOnePoint0,
    const Eigen::Matrix<NumericType, 2, 1> &segmentOnePoint1,
    const Eigen::Matrix<NumericType, 2, 1> &segmentTwoPoint0,
    const Eigen::Matrix<NumericType, 2, 1> &segmentTwoPoint1,
    double error = 1e-10)
{
    static_assert(std::is_arithmetic<NumericType>::value,
                  "Must have a numerical point type.");

    Eigen::Matrix<double, 2, 2> linearEquationsLhs;
    linearEquationsLhs <<
        segmentOnePoint0.y() - segmentOnePoint1.y(),
        segmentOnePoint1.x() - segmentOnePoint0.x(),
        segmentTwoPoint0.y() - segmentTwoPoint1.y(),
        segmentTwoPoint1.x() - segmentTwoPoint0.x();

    Eigen::Vector2d linearEquationsRhs;
    linearEquationsRhs <<
          segmentOnePoint0.y() * (segmentOnePoint1.x() - segmentOnePoint0.x())
        - segmentOnePoint0.x() * (segmentOnePoint1.y() - segmentOnePoint0.y()),
          segmentTwoPoint0.y() * (segmentTwoPoint1.x() - segmentTwoPoint0.x())
        - segmentTwoPoint0.x() * (segmentTwoPoint1.y() - segmentTwoPoint0.y());

    Eigen::Vector2d intersection =
        linearEquationsLhs.colPivHouseholderQr().solve(linearEquationsRhs);

    if (intersection.x() >= std::min(segmentOnePoint0.x(), segmentOnePoint1.x())-error
        && intersection.x() <= std::max(segmentOnePoint0.x(), segmentOnePoint1.x())+error
        && intersection.y() >= std::min(segmentOnePoint0.y(), segmentOnePoint1.y())-error
        && intersection.y() <= std::max(segmentOnePoint0.y(), segmentOnePoint1.y())+error
        && intersection.x() >= std::min(segmentTwoPoint0.x(), segmentTwoPoint1.x())-error
        && intersection.x() <= std::max(segmentTwoPoint0.x(), segmentTwoPoint1.x())+error
        && intersection.y() >= std::min(segmentTwoPoint0.y(), segmentTwoPoint1.y())-error
        && intersection.y() <= std::max(segmentTwoPoint0.y(), segmentTwoPoint1.y())+error)
    {
        return intersection;
    }

    return {};
}

}
