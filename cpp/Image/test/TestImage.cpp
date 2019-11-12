#include <iostream>
#include <filesystem>

#include <Image/Image.hpp>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        return -1;
    }

    std::clog << "Reading image into memory." << std::endl;

    pcv::Image<uint8_t, 3>(std::filesystem::path(argv[1]));

    std::clog << "Successfully read image into memory." << std::endl;

    return 0;
}
