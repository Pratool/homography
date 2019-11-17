#include <iostream>
#include <filesystem>

#include <Geometry/ConvexPolygon.hpp>
#include <Image/Image.hpp>

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        return -1;
    }

    std::clog << "Reading image into memory." << std::endl;

    pcv::Image<uint8_t, 3> testImage{std::filesystem::path(argv[1])};

    std::clog << "Successfully read image into memory." << std::endl;

    testImage.write("testimage.png");

    pcv::ConvexPolygon<unsigned int> polygon{};
    polygon.addVertex(Eigen::Matrix<unsigned int, 2, 1>({100, 400}));
    polygon.addVertex(Eigen::Matrix<unsigned int, 2, 1>({200, 600}));
    polygon.addVertex(Eigen::Matrix<unsigned int, 2, 1>({300, 250}));
    polygon.addVertex(Eigen::Matrix<unsigned int, 2, 1>({200, 200}));
    testImage.crop(polygon);

    testImage.write("testimage_cropped.png");

    return 0;
}
