#pragma once

#include <string>
#include <vector>
#include <optional>

#include <opencv2/core/core.hpp>

struct CameraCalibrationData
{
    cv::Mat cameraMatrix;

    cv::Mat distortionCoefficients;

    std::vector<cv::Mat> rotationVectors;

    std::vector<cv::Mat> translationVectors;
};

std::ostream& operator<< (
        std::ostream &stream, const CameraCalibrationData &data);

std::optional<CameraCalibrationData>
computeCameraCalibrationFromChessboard(
    cv::Mat&& image,
    unsigned int innerPointsLength,
    unsigned int innerPointsWidth,
    double squareSize_mm = 1.0);
