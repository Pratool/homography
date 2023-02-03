#include <boost/gil.hpp>
#include <boost/gil/io/io.hpp>
#include <boost/gil/image.hpp>
#include <Eigen/Dense>

//#include <boost/gil/extension/io/jpeg/write.hpp>
//#include <boost/gil/extension/io/jpeg/old.hpp>
#include <boost/gil/extension/io/png/old.hpp>
//extension/io/jpeg/detail/writer_backend.hpp>

#include <iostream>
#include <vector>
#include <random>


int main()
{
    std::string filename{"output.jpg"};
    boost::gil::rgb8_image_t img(512, 512);
    auto img_view = boost::gil::view(img);
    //Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic> mat{};

    for (auto rowIdx = 0; rowIdx < 512; ++rowIdx)
    {
        for (auto colIdx = 0; colIdx < 512; ++colIdx)
        {
            //mat(rowIdx, colIdx) = static_cast<uint8_t>(std::min(rowIdx * colIdx, 255));
            auto val = static_cast<unsigned char>(std::min(rowIdx * colIdx, 255));
            img_view(rowIdx, colIdx) =
                boost::gil::pixel<unsigned char, boost::gil::rgb_layout_t>(
                    val, val, val);
        }
    }
    boost::gil::png_write_view(filename, img_view);

    return 0;
}
