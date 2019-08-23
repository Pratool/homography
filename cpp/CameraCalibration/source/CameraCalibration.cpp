/**
 * \file CameraCalibration.cpp
 * \brief Functions useful for calibrating cameras.
 **/

#include <iostream>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "CameraCalibration.hpp"

void computeCameraCalibration(const std::string &imagePath)
{
    cv::Mat image;
    image = cv::imread(imagePath, 1);

    if (! image.data)
    {
        throw std::runtime_error("Could not open or find the image.");
    }

    constexpr int innerPointsLength = 6;
    constexpr int innerPointsWidth = 4;
    constexpr double squareSize_mm = 30.0;

    std::vector<cv::Vec3f> objectPoint;
    for (int i = 0; i < innerPointsLength; ++i)
    {
        for (int ii = 0; ii < innerPointsWidth; ++ii)
        {
            objectPoint.emplace_back(
                    cv::Vec3f(i*squareSize_mm, ii*squareSize_mm, 0));
        }
    }

    cv::Size patternSize(innerPointsLength, innerPointsWidth);

    cv::Mat grayImage;
    cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

    std::vector<std::vector<cv::Vec3f>> objectPoints;
    std::vector<std::vector<cv::Vec2f>> imagePoints;
    std::vector<cv::Vec2f> corners;

    bool successFindingChessboard =
        cv::findChessboardCorners(grayImage, patternSize, corners);

    if (successFindingChessboard)
    {
        cv::cornerSubPix(grayImage, corners, cv::Size(5,5), cv::Size(-1,-1),
                         cv::TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));

        imagePoints.emplace_back(corners);
        objectPoints.push_back(objectPoint);

        cv::drawChessboardCorners(image, patternSize, corners, successFindingChessboard);
        cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
        cv::imshow("Display window", image);
        cv::waitKey(0);
    }

    cv::Mat K;
    cv::Mat D;
    std::vector<cv::Mat> rvecs, tvecs;
    int flag = 0;
    flag |= cv::CALIB_FIX_K4;
    flag |= cv::CALIB_FIX_K5;

    cv::calibrateCamera(objectPoints, imagePoints, image.size(), K, D, rvecs, tvecs, flag);

    std::clog << K << std::endl;
    std::clog << D << std::endl;
}
