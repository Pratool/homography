#include "CameraCalibration.hpp"

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        return -1;
    }

    findChessBoard(argv[1]);

    return 0;
}
