#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/extension/numeric/convolve.hpp>
#include <boost/gil/extension/numeric/kernel.hpp>
#include <boost/gil/extension/io/png/read.hpp>
#include <boost/gil/extension/io/png/write.hpp>

int main()
{
  using namespace boost::gil;

  rgb8_image_t img;
  read_image("test.png", img, png_tag{});

  rgb8_image_t convolved(img);

  std::array<float, 3> sobel_x_r = {1.0f, 2.0f, 1.0f};
  kernel_1d_fixed<float, sobel_x_r.size()> sobel_x_r_kernel(sobel_x_r.begin(), 1);
  convolve_rows_fixed<rgb32f_pixel_t>(const_view(convolved), sobel_x_r_kernel, view(convolved));

  std::array<float, 3> sobel_x_c = {1.0f, 0.0f, -1.0f};
  kernel_1d_fixed<float, sobel_x_c.size()> sobel_x_c_kernel(sobel_x_c.begin(), 1);
  convolve_rows_fixed<rgb32f_pixel_t>(const_view(convolved), sobel_x_c_kernel, view(convolved));

  kernel_1d_fixed<float, sobel_x_c.size()> sobel_y_r_kernel(sobel_x_c.begin(), 1);
  convolve_rows_fixed<rgb32f_pixel_t>(const_view(convolved), sobel_y_r_kernel, view(convolved));

  kernel_1d_fixed<float, sobel_x_r.size()> sobel_y_c_kernel(sobel_x_r.begin(), 1);
  convolve_rows_fixed<rgb32f_pixel_t>(const_view(convolved), sobel_x_c_kernel, view(convolved));

  write_view("test-sobel-filter.png", view(convolved), boost::gil::png_tag{});

  return 0;
}
