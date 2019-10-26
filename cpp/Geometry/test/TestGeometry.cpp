#include <Geometry/ConvexPolygon.hpp>
#include <Geometry/LineSegment.hpp>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include <iostream>
#include <cmath>

TEST(ConvexPolygon, AddVerticesConstruction)
{
    using namespace pcv;

    ConvexPolygon<int> polygon{};

    polygon.addVertex(Eigen::Vector2i{2, 4});
    polygon.addVertex(Eigen::Vector2i{3, 5});
    polygon.addVertex(Eigen::Vector2i{4, 0});
    polygon.addVertex(Eigen::Vector2i{0, 0});

    ConvexPolygon<int> polygonReplica;
    polygonReplica.addVertex(Eigen::Vector2i{2, 4});
    polygonReplica.addVertex(Eigen::Vector2i{3, 5});
    polygonReplica.addVertex(Eigen::Vector2i{4, 0});
    polygonReplica.addVertex(Eigen::Vector2i{0, 0});

    ASSERT_EQ(polygon, polygonReplica);
}


TEST(Triagle, SignedArea)
{
    using namespace pcv;
    using namespace Eigen;

    const double negativeArea = getSignedArea2D(Vector2d{2.0, 4.0}, Vector2d{3.0, 5.0}, Vector2d{4.0, 0.0});
    ASSERT_EQ(negativeArea, -3.0);

    const double positiveArea = getSignedArea2D(Vector2d{4.0, 0.0}, Vector2d{3.0, 5.0}, Vector2d{2.0, 4.0});
    ASSERT_EQ(positiveArea, 3.0);
}

TEST(ConvexPolygon, PointContained)
{
    using namespace pcv;
    using namespace Eigen;

    ConvexPolygon<int> polygon{};

    polygon.addVertex(Vector2i{2, 4});
    polygon.addVertex(Vector2i{3, 5});
    polygon.addVertex(Vector2i{4, 0});
    polygon.addVertex(Vector2i{0, 0});

    // Check that all the vertices are contained in the polygon.
    ASSERT_TRUE(polygon.isPointContained(Vector2i{2, 4}.cast<double>()));
    ASSERT_TRUE(polygon.isPointContained(Vector2d{3, 5}));
    ASSERT_TRUE(polygon.isPointContained(Vector2d{4, 0}));
    ASSERT_TRUE(polygon.isPointContained(Vector2d{0, 0}));

    ASSERT_TRUE(polygon.isPointContained(Vector2d{2, 2}));
    ASSERT_TRUE(polygon.isPointContained(Vector2d{1, 1}));
    ASSERT_FALSE(polygon.isPointContained(Vector2d{5, 5}));
}


TEST(ConvexPolygon, Rasterize)
{
    using namespace pcv;
    using namespace Eigen;

    ConvexPolygon<int> polygon{};

    polygon.addVertex(Vector2i{2, 4});
    polygon.addVertex(Vector2i{3, 5});
    polygon.addVertex(Vector2i{4, 0});
    polygon.addVertex(Vector2i{0, 0});

    MatrixXi autoMask = makePolygonIntersectionGrid<int>({polygon}, 7, 7);

    MatrixXi mask(7, 7);
    mask <<
        0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 1, 0, 0;

    ASSERT_TRUE(mask == autoMask);
}

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
