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

  std::array<float, 9> gaussian = {
    0.075, 0.075, 0.075,
    0.075, 0.4, 0.075,
    0.075, 0.075, 0.075 };

  kernel_2d_fixed<float, 9> guassian_kernel(gaussian.begin(), 1, 1);
  convolve_rows_fixed<rgb32f_pixel_t>(const_view(convolved), guassian_kernel, view(convolved));

  /*
  float sobel_x[] = { 1.0f, 0.0f, -1.0f, 2.0f, 0.0f, -2.0f, 1.0f, 0.0f, -1.0f };
  kernel_1d_fixed<float, 9> sobel_x_kernel(sobel_x, 4);
  float sobel_y[] = { 1.0f, 2.0f, 1.0f, 0.0f, 0.0f, 0.0f, -1.0f, -2.0f, -1.0f };
  kernel_1d_fixed<float, 9> sobel_y_kernel(sobel_y, 4);
  convolve_cols_fixed<rgb32f_pixel_t>(const_view(convolved), guassian_kernel, view(convolved));
  convolve_rows_fixed<rgb32f_pixel_t>(const_view(convolved), sobel_x_kernel, view(convolved));
  convolve_cols_fixed<rgb32f_pixel_t>(const_view(convolved), sobel_y_kernel, view(convolved));
  */

  write_view("test-blur.png", view(convolved), boost::gil::png_tag{});

  return 0;
}
