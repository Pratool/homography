#include "FeatureExtractors/ImageGradient.hpp"
#include "boost/gil/extension/io/png.hpp"
#include "boost/gil/extension/io/tiff.hpp"

#include <iostream>
#include <string>

namespace gil = boost::gil;

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    return -1;
  }

  std::cout << "opening file: " << argv[1] << std::endl;

  gil::gray8_image_t img;
  gil::read_image(argv[1], img, gil::png_tag());

  std::cout << "creating const_view" << std::endl;
  gil::gray8c_view_t src = gil::const_view(img);

  std::cout << "creating destination image buffer" << std::endl;
  gil::gray8s_image_t dst_img{img.dimensions()};

  std::cout << "creating non-const view of destination buffer" << std::endl;
  gil::gray8s_view_t dst = gil::view(dst_img);

  std::cout << "running x_gradient" << std::endl;
  x_gradient(src, dst);

  gil::gray8_image_t interim_image = dst_img;
  gil::gray8s_view_t interim_view = gil::const_view(interim_image);

  std::cout << "running y_gradient" << std::endl;
  y_gradient(interim_view, dst);

  std::string output_file("output.tiff");
  gil::write_view(output_file, gil::view(dst_img), gil::tiff_tag());

  return 0;
}
