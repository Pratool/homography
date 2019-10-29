#pragma once

#include <Geometry/ConvexPolygon.hpp>

#include <Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <memory>

namespace pcv
{

class InternalImageFrame
{
public:
    InternalImageFrame() = default;

    InternalImageFrame(const Eigen::Index topLeft,
                       const Eigen::Index bottomRight);

    InternalImageFrame(const Eigen::Index topLeft,
                       const Eigen::Index topRight,
                       const Eigen::Index bottomLeft,
                       const Eigen::Index bottomRight);

    InternalImageFrame(const Eigen::Matrix<Eigen::Index, 4, 1> &inputBounds)
        : bounds(inputBounds) {};


    ~InternalImageFrame() = default;


    InternalImageFrame(const InternalImageFrame &rhs) = default;

    InternalImageFrame(InternalImageFrame &&rhs) = default;


    InternalImageFrame &operator=(const InternalImageFrame &rhs) = default;

    InternalImageFrame &operator=(InternalImageFrame &&rhs) = default;


    Eigen::Index topLeft() const;

    Eigen::Index topRight() const;

    Eigen::Index bottomLeft() const;

    Eigen::Index bottomRight() const;


    Eigen::Matrix<Eigen::Index, 4, 1> getBounds() const;

private:

    Eigen::Matrix<Eigen::Index, 4, 1> bounds{};

};

cv::Mat eigenToCv(const Eigen::Matrix3d &rhs)
{
    cv::Mat m = (cv::Mat_<double>(3,3) <<
        rhs(0, 0), rhs(0, 1), rhs(0, 2),
        rhs(1, 0), rhs(1, 1), rhs(1, 2),
        rhs(2, 0), rhs(2, 1), rhs(2, 2) );

    return m;
}

cv::Mat eigenToCv(const Eigen::MatrixXd &rhs)
{
    cv::Mat_<double> m(rhs.rows(), rhs.cols());
    for (auto i = 0; i < rhs.rows(); ++i)
    {
        for (auto ii = 0; ii < rhs.cols(); ++ii)
        {
            m(i, ii) = rhs(i, ii);
        }
    }

    return m;
}

Eigen::MatrixXd cvToEigen(const cv::Mat &rhs);

//std::unique_ptr<>
//Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic>
template<class ElementType>
void stepBlendImages(
    const Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic> &image0,
    const Eigen::Matrix<ElementType, Eigen::Dynamic, Eigen::Dynamic> &image1,
    const Eigen::Matrix3d &transform_image0_image1)
{
    using namespace Eigen;

    // Create a matrix where each column is a homogenous coordinate of the image
    // plane at each of the four corners of both images.
    Matrix<ElementType, 3, 4> image0Corners;
    image0Corners <<
        0, image0.rows(), 0,             image0.cols(),
        0, 0,             image0.cols(), image0.cols(),
        1, 1,             1,             1;

    Matrix<ElementType, 3, 4> image1Corners;
    image1Corners <<
        0, image1.rows(), 0,             image1.cols(),
        0, 0,             image1.cols(), image1.cols(),
        1, 1,             1,             1;

    // Matrix representation of corners allows for a concise transformation of
    // each of image0's corner coordinates.
    Matrix<ElementType, 3, 4> transformedImage0Corners =
        transform_image0_image1 * image0Corners;

    // Divide through all transformed coordinates with the z-dimension to keep
    // all image coordinates on the same plane a unit distance away from the
    // viewer.
    for (Index col = 0; col < transformedImage0Corners.cols(); ++col)
    {
        transformedImage0Corners.col(col) /= transformedImage0Corners.col(col)(2);
    }

    Matrix<ElementType, Dynamic, Dynamic> allCorners(3, 8);
    allCorners << transformedImage0Corners, image1Corners;

    Index width =
        allCorners.row(0).maxCoeff() - allCorners.row(0).minCoeff();

    Index height =
        allCorners.row(1).maxCoeff() - allCorners.row(1).minCoeff();

    Matrix<ElementType, 3, 3> transform_image1_outputCanvas =
        Matrix<ElementType, 3, 3>::Identity(3, 3);
    transform_image1_outputCanvas(0, 2) -= allCorners.row(0).minCoeff();
    transform_image1_outputCanvas(1, 2) -= allCorners.row(1).minCoeff();

    Matrix3d transform_image0_outputCanvas =
        transform_image1_outputCanvas*transform_image0_image1;

    if (width + allCorners.row(0).minCoeff() < image1.cols())
    {
        width = image1.cols() - allCorners.row(0).minCoeff();
    }

    if (height + allCorners.row(0).minCoeff() < image1.rows())
    {
        height = image1.rows() - allCorners.row(0).minCoeff();
    }

    // At this point, the height and width now indicate the bounding box of the
    // the union of the images.
    //
    // Find the coordinates of the corners of the two images in the final
    // canvas's image frame. This is not simply the transformed coordinates of
    // image1's corners nor image0's corners. Find the minimum x-coordinate and
    // note the image its contained in. Find the minimum y-coordinate and note
    // the image its contained in. Figure out how much each image needs to be
    // translated to fit the blended image frame and add the values to the
    // original image corner coordinates.
    //
    // Use these coordinates to generate two polygons (convex quadrilaterals).
    // Rasterize the intersection of these two polygons to a masking matrix.

    pcv::ConvexPolygon<double> transformedImage0Poly{};
    for (Index col = 0; col < transformedImage0Corners.cols(); ++col)
    {
        Eigen::Vector3d tmp = transform_image0_outputCanvas*transformedImage0Corners.col(col);
        tmp = tmp/tmp(2, 0);
        std::clog << "tform img0 corner:\n" << Eigen::Vector2d{tmp.x(), tmp.y()} << std::endl;
        transformedImage0Poly.addVertex(Eigen::Vector2d{tmp.x(), tmp.y()});
    }

    pcv::ConvexPolygon<double> image1Poly{};
    for (Index col = 0; col < image1Corners.cols(); ++col)
    {
        Eigen::Vector3d tmp = transform_image1_outputCanvas*image1Corners.col(col);
        tmp = tmp/tmp(2, 0);
        std::clog << "img1 corner:\n" << Eigen::Vector2d{tmp.x(), tmp.y()} << std::endl;
        transformedImage0Poly.addVertex(Eigen::Vector2d{tmp.x(), tmp.y()});
    }

    std::clog << "got this far0" << std::endl;

    auto grid = 0.5*eigenToCv(
        pcv::makePolygonIntersectionGrid(
            std::vector<decltype(image1Poly)>{image1Poly, transformedImage0Poly},
            height,
            width));

    std::clog << "got this far1" << std::endl;

    cv::Mat targetFrameHomography = cv::Mat::eye(3, 3, CV_64F);
    targetFrameHomography.at<double>(0, 2) -= allCorners.row(0).minCoeff();
    targetFrameHomography.at<double>(1, 2) -= allCorners.row(1).minCoeff();

    auto homography = eigenToCv(transform_image0_image1);
    homography = cv::Mat(targetFrameHomography*homography);

    cv::Mat image0Warped;
    //cv::warpPerspective(eigenToCv(image0), image0Warped, homography, {width, height});
    //image0Warped = image0Warped.mul(grid);

    cv::imwrite("image0warped.png", eigenToCv(image0));
    std::clog << "wrote image0warped.png" << std::endl;

    cv::Mat image1Warped;
    cv::warpPerspective(eigenToCv(image1), image1Warped, targetFrameHomography, {width, height});
    image1Warped = image1Warped.mul(grid);

    cv::imwrite("image1warped.png", image1Warped);
    std::clog << "wrote image1warped.png" << std::endl;

    cv::Mat dst;
    //cv::addWeighted(image0Warped, 1.0, image1Warped, 1.0, 0.0, dst);

    //cv::imwrite("blended-from-blender.png", dst);
    std::clog << "wrote blended-from-blender.png" << std::endl;
}

} // end namespace pcv

/*
    Matrix<Index, 2, 1> outerBoundTopLeft, outerBoundTopRight,
          outerBoundBottomLeft, outerBoundBottomRight;
    Matrix<Index, 2, 1> innerBoundTopLeft, innerBoundTopRight,
          innerBoundBottomLeft, innerBoundBottomRight;

    for (Index col = 0; col < image0Corners.cols(); ++cols)
    {
        const auto &image0X = image0Corners(0, col);
        const auto &image0Y = image0Corners(1, col);

        const auto &image1X = image1Corners(0, col);
        const auto &image1Y = image1Corners(1, col);

        switch(col)
        {
            case 0:
                outerBoundTopLeft(0) = std::min(image0X, image1X);
                innerBoundTopLeft(0) = std::max(image0X, image1X);
                outerBoundTopLeft(1) = std::min(image0Y, image1Y);
                innerBoundTopLeft(1) = std::max(image0Y, image1Y);
                break;
            case 1:
                outerBoundTopRight(0) = std::max(image0X, image1X);
                innerBoundTopRight(0) = std::min(image0X, image1X);
                outerBoundTopRight(1) = std::min(image0Y, image1Y);
                innerBoundTopRight(1) = std::max(image0Y, image1Y);
                break;
            case 2:
                outerBoundBottomLeft(0) = std::min(image0X, image1X);
                innerBoundBottomLeft(0) = std::max(image0X, image1X);
                outerBoundBottomLeft(1) = std::max(image0Y, image1Y);
                innerBoundBottomLeft(1) = std::min(image0Y, image1Y);
                break;
            case 3:
                outerBoundBottomRight(0) = std::max(image0X, image1X);
                innerBoundBottomRight(0) = std::min(image0X, image1X);
                outerBoundBottomRight(1) = std::max(image0Y, image1Y);
                innerBoundBottomRight(1) = std::min(image0Y, image1Y);
                break;
            default:
                break;
        }
    }
*/
