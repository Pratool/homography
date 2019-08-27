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
#include "Utilities.hpp"

std::ostream& operator<< (std::ostream &stream, const CameraCalibrationData &data)
{
    stream << "Camera Matrix:" << std::endl
           << data.cameraMatrix << std::endl
           << "Distortion Coefficients:" << std::endl
           << data.distortionCoefficients << std::endl;

    stream << "Rotation Vectors:\n";
    for (const auto &vec : data.rotationVectors)
    {
           stream << vec << std::endl;
    }

    stream << "Translation Vectors:\n";
    for (const auto &vec : data.translationVectors)
    {
           stream << vec << std::endl;
    }

    return stream;
}


std::optional<CameraCalibrationData>
computeCameraCalibrationFromChessboard(
    cv::Mat&& image,
    unsigned int innerPointsLength,
    unsigned int innerPointsWidth,
    double squareSize_mm)
{
    std::vector<cv::Vec3f> objectPoint =
        createObjectPoint(innerPointsLength, innerPointsWidth, squareSize_mm);

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

        CameraCalibrationData calibrationData;
        int flag = 0;
        flag |= cv::CALIB_FIX_K4;
        flag |= cv::CALIB_FIX_K5;

        cv::calibrateCamera(
            objectPoints, imagePoints, image.size(),
            calibrationData.cameraMatrix,
            calibrationData.distortionCoefficients,
            calibrationData.rotationVectors,
            calibrationData.translationVectors,
            flag);

        return calibrationData;
    }

    return {};
}
