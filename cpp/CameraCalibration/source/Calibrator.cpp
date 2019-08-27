#include <iostream>
#include <string>

#include <opencv2/highgui/highgui.hpp>

#include "CameraCalibration.hpp"

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        return -1;
    }

    cv::Mat image;
    image = cv::imread(argv[1], 1);

    if (! image.data)
    {
        throw std::runtime_error("Could not open or find the image.");
    }

    // Using images with 30 mm square side lengths.
    auto calibrationData = computeCameraCalibrationFromChessboard(
        std::move(image), std::atoi(argv[2]), std::atoi(argv[3]), 30.0);

    if (calibrationData)
    {
        std::cout << *calibrationData;
    }

    return 0;
}
