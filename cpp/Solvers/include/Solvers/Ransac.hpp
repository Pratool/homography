#pragma once

#include <numeric>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

cv::Mat distortSourceToMatchTarget(
    const cv::Mat &sourceImage, const cv::Mat &targetImage);

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

    return Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(svd.matrixV().col(8).data());
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

    return Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(svd.matrixV().col(8).data());
}

template<class DataType, class ModelType>
ModelType
Ransac(
    const std::vector<DataType> &data,
    const std::function<ModelType(std::vector<DataType>)> &modelingFunction,
    std::size_t dataPointsRequiredByModel,
    const std::function<double(ModelType, DataType)> &costFunction,
    double errorThreshold,
    std::size_t iterations)
{
    if (data.size() < dataPointsRequiredByModel)
    {
        throw std::runtime_error("There must be at least the minimum number of "
                                 "data points required by modeling function.");
    }

    // There will at minimum be 4 inliers as 4 randomly selected points are
    // inliers.
    std::size_t bestInlierCount = 0;

    // The default value of bestCost shall never be used. It should always
    // get overwritten otherwise discarded.
    double bestCost = 0;

    ModelType bestModel{};

    // Always run at least one iteration of RANSAC.
    for (std::size_t iteration = 0; iteration < iterations+1; ++iteration)
    {
        std::vector<DataType> tmpInliers;
        for (std::size_t minDataIter = 0; minDataIter < 4; ++minDataIter)
        {
            std::uniform_int_distribution<std::size_t> indexRandomizer(0, data.size());
            std::random_device randomDevice;
            tmpInliers.push_back(data[indexRandomizer(randomDevice)]);
        }

        auto tmpModel = modelingFunction(tmpInliers);
        std::size_t tmpInlierCount = 0;

        const auto tmpTotalCost = std::accumulate(
            std::cbegin(data),
            std::cend(data),
            0.0,
            [tmpModel, errorThreshold, &tmpInlierCount, &costFunction](double accumulator, DataType dataIter)
            {
                const auto &error = costFunction(tmpModel, dataIter);

                // Do not count this point's reprojection error toward the total
                // cost because it is an outlier.
                if (error > errorThreshold)
                {
                    return accumulator;
                }

                ++tmpInlierCount;
                return accumulator + error;
            });


        // If the current model has most inliers insofar, or if the the number
        // of inliers matches the most inliers insofar and the cost has been
        // reduced, then update the best model with the current model.
        if (tmpInlierCount > bestInlierCount
           || ((tmpInlierCount == bestInlierCount) && tmpTotalCost < bestCost))
        {
            bestCost = tmpTotalCost;
            bestInlierCount = tmpInlierCount;

            // Ensure that the bottom-right corner of the 3x3 matrix is 1 and
            // divide element-wise through the matrix.
            bestModel = tmpModel / tmpModel(2,2);
        }
    }
    return bestModel;
}

} // end namespace utility
