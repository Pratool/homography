/**
 * \file CameraCalibration.cpp
 * \brief Functions useful for calibrating cameras.
 **/

/* Must provide the following features:
 * Open JPG images
 * Convert JPG images to matrix of numerical data.
 * Save images, ideally in same input format.
 */

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "CameraCalibration.hpp"

void findChessBoard(const std::string &imagePath)
{
    cv::Mat image;
    image = cv::imread(imagePath, 1);

    if (! image.data)
    {
        throw std::runtime_error("Could not open or find the image.");
    }

    cv::Size patternSize(10, 7);
    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
    std::vector<cv::Point2f> corners;

    bool successFindingChessboard =
        cv::findChessboardCorners(grayImage, patternSize, corners);

    if (successFindingChessboard)
    {
        for (auto i : corners)
        {
            std::clog << i << std::endl;
        }
    }

    cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
    cv::imshow("Display window", image);

    cv::waitKey(0);
}
