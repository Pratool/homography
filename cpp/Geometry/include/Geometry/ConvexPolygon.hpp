#pragma once

#include <Geometry/LineSegment.hpp>

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


template<class NumericType>
class ConvexPolygon
{
public:
    ConvexPolygon() = default;


    ~ConvexPolygon() = default;


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


template<class NumericType>
bool operator==(
    const ConvexPolygon<NumericType> &lhs,
    const ConvexPolygon<NumericType> &rhs)
{
    return lhs.getVertices() == rhs.getVertices();
}


template<class NumericType>
Eigen::Matrix<NumericType, Eigen::Dynamic, Eigen::Dynamic>
makePolygonIntersectionGrid(
    const std::vector<ConvexPolygon<NumericType>> &polygons,
    std::size_t gridRows,
    std::size_t gridCols)
{
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

} // end namespace pcv
