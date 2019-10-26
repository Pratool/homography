#pragma once

#include <Geometry/LineSegment.hpp>

#include <Eigen/Dense>

#include <numeric>
#include <vector>

namespace pcv
{


template<class NumericType>
double getSignedArea2D(
    const Eigen::Matrix<NumericType, 2, 1> &a,
    const Eigen::Matrix<NumericType, 2, 1> &b,
    const Eigen::Vector2d &c)
{
    double ax = static_cast<double>(a.x());
    double ay = static_cast<double>(a.y());
    double bx = static_cast<double>(b.x());
    double by = static_cast<double>(b.y());
    double cx = static_cast<double>(c.x());
    double cy = static_cast<double>(c.y());

    return 0.5
           * (((bx - ax) * (cy - ay))
            - ((cx - ax) * (by - ay))); 
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
                isContained &= getSignedArea2D(cachedVertex, vertexIter, testPoint) < 1e-15;
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
