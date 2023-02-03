/// SPDX-License-Identifier: MIT

#include <boost/gil.hpp>

#include <type_traits>

namespace gil = boost::gil;

void x_gradient(const gil::gray8c_view_t& source, const gil::gray8s_view_t& destination)
{
  assert(source.dimensions() == destination.dimensions());

  for (int y = 0; y < source.height(); ++y)
  {
    auto src_it = source.row_begin(y);
    auto dst_it = destination.row_begin(y);

    for (int x = 1; x < source.width()-1; ++x)
    {
      dst_it[x] = (src_it[x-1] - src_it[x+1]) / 2;
    }
  }
}

void y_gradient(const gil::gray8c_view_t& source, const gil::gray8s_view_t& destination)
{
  assert(source.dimensions() == destination.dimensions());

  for (int y = 1; y < source.height()-1; ++y)
  {
    auto src1_it = source.row_begin(y-1);
    auto src2_it = source.row_begin(y+1);
    auto dst_it = destination.row_begin(y);

    for (int x = 0; x < source.width(); ++x)
    {
      *dst_it = ((*src1_it) - (*src2_it)) / 2;
      ++dst_it;
      ++src1_it;
      ++src2_it;
    }
  }
}
