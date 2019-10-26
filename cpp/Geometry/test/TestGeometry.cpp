#include <Geometry/LineSegment.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <iostream>

TEST(LineSegment, Intersection)
{
    using namespace pcv;
    using namespace Eigen;

    const auto intersection = getSegmentIntersectionPoint2(
        Vector2i{0, 0},
        Vector2i{2, 2},
        Vector2i{0, 2},
        Vector2i{2, 0});

    ASSERT_TRUE(intersection);
    ASSERT_DOUBLE_EQ(intersection->x(), 1.0);
    ASSERT_DOUBLE_EQ(intersection->y(), 1.0);
}


int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
