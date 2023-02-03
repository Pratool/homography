#include <Solvers/Image.hpp>
#include <Solvers/Line.hpp>
#include <Solvers/Ransac.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>


TEST(HomogenousCoordinates, Distance)
{
    Eigen::Matrix3d identity;
    identity <<
        1, 0, 0,
        0, 1, 0,
        0, 0, 1;

    const double distance = pcv::getReprojectionError(
        identity,
        std::pair(cv::Point2f{1,1}, cv::Point2f{0,0}) );

    ASSERT_EQ(distance, std::sqrt(static_cast<double>(2.0)));
}


TEST(HomogenousCoordinates, TranslationFloats)
{
    Eigen::Matrix3d translate;
    translate <<
        1, 0, 2,
        0, 1, 0,
        0, 0, 1;

    using namespace pcv;

    const double result0 = getReprojectionError(
        translate,
        std::pair(cv::Point2f{1,1}, cv::Point2f{0,0}) );

    ASSERT_EQ(result0, std::sqrt(static_cast<double>(10.0)));

    const double result1 = getReprojectionError(
        translate,
        std::pair(cv::Point2f{1,1}, cv::Point2f{3,1}) );

    ASSERT_EQ(result1, std::sqrt(static_cast<double>(0.0)));

    const double result2 = getReprojectionError(
        translate,
        std::pair(cv::Point2f{-2,-1}, cv::Point2f{0,-1}));

    ASSERT_EQ(result2, std::sqrt(static_cast<double>(0.0)));
}


TEST(HomogenousCoordinates, TranslationDoubles)
{
    Eigen::Matrix3d translate;
    translate <<
        1, 0, 2,
        0, 1, 0,
        0, 0, 1;

    using namespace pcv;

    const double result0 = getReprojectionError(
        translate,
        std::pair(cv::Point2d{1,1}, cv::Point2d{0,0}) );

    ASSERT_EQ(result0, std::sqrt(static_cast<double>(10.0)));

    const double result1 = getReprojectionError(
        translate,
        std::pair(cv::Point2d{1,1}, cv::Point2d{3,1}) );

    ASSERT_EQ(result1, std::sqrt(static_cast<double>(0.0)));

    const double result2 = getReprojectionError(
        translate,
        std::pair(cv::Point2d{-2,-1}, cv::Point2d{0,-1}));

    ASSERT_EQ(result2, std::sqrt(static_cast<double>(0.0)));
}


TEST(FindHomography, LeastSquares)
{
    Eigen::Matrix3d translate;
    translate <<
        1, 0, 2,
        0, 1, 0,
        0, 0, 1;

    using namespace pcv;

    std::vector<std::pair<cv::Point2f, cv::Point2f>> correspondences{
        std::pair(cv::Point2f{0,0}, cv::Point2f{1,1}),
        std::pair(cv::Point2f{1,0}, cv::Point2f{2,1}),
        std::pair(cv::Point2f{1,1}, cv::Point2f{2,2}),
        std::pair(cv::Point2f{0,1}, cv::Point2f{1,2}) };

    Eigen::Matrix3d dltResult =
        findHomographyWithDirectLinearTransform(correspondences);

    Eigen::Matrix3d leastSquaresResult =
        findHomographyWithLeastSquares(correspondences);

    for (Eigen::Index i = 0; i < dltResult.rows(); ++i)
    {
        for (Eigen::Index ii = 0; ii < dltResult.cols(); ++ii)
        {
            ASSERT_NEAR(dltResult(i,ii), leastSquaresResult(i,ii), 1e-12);
        }
    }
}


TEST(Ransac, BestFitLine)
{
    std::vector<Eigen::Vector2d> points;
    points.emplace_back(Eigen::Vector2d({0.0, 2.0}));
    points.emplace_back(Eigen::Vector2d({2.0, 8.0}));
    points.emplace_back(Eigen::Vector2d({4.0, 5.5}));
    points.emplace_back(Eigen::Vector2d({6.0, 6.2}));

    auto eigenModel = pcv::Ransac<Eigen::Vector2d, Eigen::Vector2d>(
        points,
        pcv::findLineModelFromPoints<double>,
        2,
        pcv::getLineModelError<double>,
        1e-4,
        52980);

    // Assumption here is that after 52981 iterations, an optimal solution will
    // be found with Ransac. The optimal line goes through (0, 2) and (4, 5.5).
    ASSERT_NEAR(eigenModel[0], -0.4375, 1e-12);
    ASSERT_NEAR(eigenModel[1], 0.5, 1e-12);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
