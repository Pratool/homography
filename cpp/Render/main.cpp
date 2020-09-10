#include <boost/gil/io/write_view.hpp>
#include <boost/gil/extension/io/jpeg/tags.hpp>
#include <boost/gil/image.hpp>
#include <iostream>

int main()
{
    using rgb8_pixel_t = boost::gil::rgb8_pixel_t;

    const rgb8_pixel_t red{255, 0, 0};
    const rgb8_pixel_t green{0, 255, 0};
    const rgb8_pixel_t blue{0, 0, 255};
    const rgb8_pixel_t purple{0, 255, 255};
    const rgb8_pixel_t white{255, 255, 255};
    const rgb8_pixel_t black{0, 0, 0};

    boost::gil::rgb8_image_t img{3, 3};
    auto &view = boost::gil::view(img);

    view(0, 0) = red;
    view(0, 1) = green;
    view(0, 2) = blue;
    view(1, 0) = purple;
    view(1, 1) = white;
    view(1, 2) = black;
    view(2, 0) = blue;
    view(2, 1) = red;
    view(2, 2) = green;

    boost::gil::write_view("image.jpg", img, boost::gil::jpeg_tag{});

    std::cout << "success." << std::endl;
    return 0;
}
