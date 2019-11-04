#pragma once

#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>


namespace pcv
{

template<class NumericType>
double getReprojectionError(
    const Eigen::Matrix3d &model,
    const std::pair<cv::Point_<NumericType>, cv::Point_<NumericType>>
        &correspondence)
{
    static_assert(std::is_arithmetic<NumericType>::value,
                  "Must have a numerical point type.");

    // Make the first correspondence a homogenous-coordinate column vector.
    Eigen::Vector3d homogenousFirst;
    homogenousFirst <<
        correspondence.first.x,
        correspondence.first.y,
        1.0;

    Eigen::Vector3d estimate = model * homogenousFirst;
    estimate /= estimate(2);

    return std::sqrt(  std::pow(estimate(0)-correspondence.second.x, 2)
                     + std::pow(estimate(1)-correspondence.second.y, 2));
}

template<class NumericType>
Eigen::Matrix3d
findHomographyWithLeastSquares(
    std::vector<std::pair<
        cv::Point_<NumericType>, cv::Point_<NumericType> >>
        correspondences)
{
    static_assert(std::is_arithmetic<NumericType>::value,
                  "Must have a numerical point type.");

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

    auto output =
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(svd.matrixV().col(8).data());

    return output / output(2, 2);
}

template<class NumericType>
Eigen::Matrix3d
findHomographyWithDirectLinearTransform(
    std::vector<std::pair<
        cv::Point_<NumericType>, cv::Point_<NumericType> >>
        correspondences)
{
    static_assert(std::is_arithmetic<NumericType>::value,
                  "Must have a numerical point type.");

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

    auto output =
        Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(svd.matrixV().col(8).data());

    return output / output(2, 2);
}

} // end namespace pcv
