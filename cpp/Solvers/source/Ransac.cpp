#include <iostream>
#include <vector>
#include <random>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>

#include "Solvers/Ransac.hpp"

namespace pcv
{

Eigen::Matrix3d
findHomographyWithRansac(
    std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences,
    double reprojectionErrorThreshold,
    std::size_t iterations)
{
    using DataType = std::pair<cv::Point2f, cv::Point2f>;

    if (correspondences.size() < 4)
    {
        throw std::runtime_error("There must be at least 4 point correspondences"
                                 "to calculate a homography.");
    }

    // There will at minimum be 4 inliers as 4 randomly selected points are
    // inliers.
    std::size_t bestInlierCount = 0;

    // The default value of bestCost shall never be used. It should always
    // get overwritten otherwise discarded.
    double bestCost = 0;

    Eigen::Matrix3d bestModel{};

    // Always run at least one iteration of RANSAC.
    for (std::size_t iteration = 0; iteration < iterations+1; ++iteration)
    {
        std::vector<DataType> tmpInliers;
        for (std::size_t minDataIter = 0; minDataIter < 4; ++minDataIter)
        {
            std::uniform_int_distribution<std::size_t> indexRandomizer(0, correspondences.size());
            std::random_device randomDevice;
            tmpInliers.push_back(correspondences[indexRandomizer(randomDevice)]);
        }

        auto tmpModel = pcv::findHomographyWithDirectLinearTransform(tmpInliers);
        std::size_t tmpInlierCount = 0;

        const auto tmpTotalCost = std::accumulate(
            std::cbegin(correspondences),
            std::cend(correspondences),
            0.0,
            [tmpModel, reprojectionErrorThreshold, &tmpInlierCount](double accumulator, DataType dataIter)
            {
                const auto &reprojectionError = getReprojectionError(tmpModel, dataIter);

                // Do not count this point's reprojection error toward the total
                // cost because it is an outlier.
                if (reprojectionError > reprojectionErrorThreshold)
                {
                    return accumulator;
                }

                ++tmpInlierCount;
                return accumulator + reprojectionError;
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

} // end namespace pcv
