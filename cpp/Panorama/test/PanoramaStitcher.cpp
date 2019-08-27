#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Panorama/Utility.hpp>

int main(int argc, char** argv)
{
    if (argc != 3)
    {
        return -1;
    }

    cv::Mat sourceImage = cv::imread(argv[1], 1);

    if (! sourceImage.data)
    {
        throw std::runtime_error("Could not open or find the image.");
    }

    cv::Mat targetImage = cv::imread(argv[2], 1);

    if (! targetImage.data)
    {
        throw std::runtime_error("Could not open or find the image.");
    }

    distortSourceToMatchTarget(sourceImage, targetImage);

    return 0;
}
