#pragma once

#include <Geometry/LineSegment.hpp>

#include <opencv2/core/core.hpp>
#include <Eigen/Dense>

#include <numeric>
#include <vector>

namespace pcv
{


/**
 * Utility function to help determine "leftness" of a point compared to the line
 * segment drawn between the first two points. Leftness is indicated by a
 * negative return value, which is why the tempalate parameter must be signed.
 * Idea from Computational Geometry in C by Joseph O'Rourke.
 */
template<class SignedNumericType>
SignedNumericType getTwiceSignedArea2D(
    const Eigen::Matrix<SignedNumericType, 2, 1> &endpoint0,
    const Eigen::Matrix<SignedNumericType, 2, 1> &endpoint1,
    const Eigen::Matrix<SignedNumericType, 2, 1> &testpoint)
{
    static_assert(std::is_signed<SignedNumericType>::value);

    // Aliases for more readable arithmetic.
    const auto &a = endpoint0;
    const auto &b = endpoint1;
    const auto &c = testpoint;

    return ((b.x() - a.x()) * (c.y() - a.y()))
         - ((c.x() - a.x()) * (b.y() - a.y()));
}


/**
 * \class ConvexPolygon
 * Create a polygon by specifying its vertices in a clockwise direction. Points
 * are 2D Eigen::Vectors.
 */
template<class NumericType>
class ConvexPolygon
{
public:
    ConvexPolygon() = default;


    ConvexPolygon(const ConvexPolygon &polygon) = default;


    /**
     * Move construction leverages std::vector's move constructor.
     */
    ConvexPolygon(ConvexPolygon &&polygon) = default;


    ~ConvexPolygon() = default;


    ConvexPolygon(const std::vector<NumericType> &vertices) : vertices(vertices)
    {}


    ConvexPolygon(std::vector<NumericType> &&vertices) : vertices(vertices)
    {}


    void addVertex(Eigen::Matrix<NumericType, 2, 1> &&vertex)
    {
        vertices.emplace_back(vertex);
    }


    void addVertex(const Eigen::Matrix<NumericType, 2, 1> &vertex)
    {
        vertices.push_back(vertex);
    }


    std::vector<Eigen::Matrix<NumericType, 2, 1>> getVertices() const
    {
        return vertices;
    }


    /**
     * Determine if a given point is inside or outside the polygon in O(n) where
     * n is the number of vertices of the polygon. This function requires a
     * point to be represented as two doubles, even if the points can be
     * represented with integers or the polygon has a non-floating point
     * templated NumericType. The current implementation does not account for
     * floating-point error.
     */
    bool isPointContained(Eigen::Vector2d testPoint) const
    {
        auto cachedVertex = vertices.back();

        return std::accumulate(
            std::cbegin(vertices),
            std::cend(vertices),
            true,
            [&cachedVertex, testPoint](bool isContained, decltype(cachedVertex) vertexIter)
            {
                isContained &=
                    getTwiceSignedArea2D(
                        static_cast<Eigen::Vector2d>(cachedVertex.template cast<double>()),
                        static_cast<Eigen::Vector2d>(vertexIter.template cast<double>()),
                        testPoint) <= 0;

                cachedVertex = vertexIter;
                return isContained;
            });
    }

private:
    std::vector<Eigen::Matrix<NumericType, 2, 1>> vertices;
};


/**
 * An O(n) operation where n is the max of the two polygon's number of vertices.
 */
template<class NumericType>
bool operator==(
    const ConvexPolygon<NumericType> &lhs,
    const ConvexPolygon<NumericType> &rhs)
{
    return lhs.getVertices() == rhs.getVertices();
}


template<class NumericType>
Eigen::Matrix<NumericType, Eigen::Dynamic, Eigen::Dynamic>
makePolygonIntersectionEigenGrid(
    const std::vector<ConvexPolygon<NumericType>> &polygons,
    std::size_t gridRows,
    std::size_t gridCols)
{
    static_assert(std::is_arithmetic<NumericType>::value,
                  "Must have numerical polygon type.");

    Eigen::Matrix<NumericType, Eigen::Dynamic, Eigen::Dynamic> mask =
        Eigen::Matrix<NumericType, Eigen::Dynamic, Eigen::Dynamic>::Zero(gridRows, gridCols);

    for (std::size_t r = 0; r < gridRows; ++r)
    {
        for (std::size_t c = 0; c < gridCols; ++c)
        {
            bool containment = true;
            for (const auto &polygon : polygons)
            {
                containment &= polygon.isPointContained(Eigen::Vector2d{c, gridRows-r-1});
            }
            mask(r, c) = containment;
        }
    }

    return mask;
}


template<class PolygonNumericType, class MatNumericType>
void makePolygonIntersectionOpencvGrid(
    const std::vector<ConvexPolygon<PolygonNumericType>> &polygons,
    std::size_t gridRows,
    std::size_t gridCols,
    cv::Mat_<MatNumericType> &output)
{
    cv::Mat_<MatNumericType> mask(gridRows, gridCols);

    for (std::size_t r = 0; r < gridRows; ++r)
    {
        for (std::size_t c = 0; c < gridCols; ++c)
        {
            bool containment = true;
            for (const auto &polygon : polygons)
            {
                containment &= polygon.isPointContained(Eigen::Vector2d{c, gridRows-r-1});
            }
            mask(r, c) = static_cast<MatNumericType>(containment);
        }
    }

    output = mask;
}


} // end namespace pcv
