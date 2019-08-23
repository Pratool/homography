#include "CameraCalibration.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        return -1;
    }

    computeCameraCalibration(argv[1]);

    return 0;
}
