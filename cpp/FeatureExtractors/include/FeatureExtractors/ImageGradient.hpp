/// SPDX-License-Identifier: MIT

#pragma once

#include <boost/gil.hpp>

void x_gradient(const boost::gil::gray8c_view_t& source, const boost::gil::gray8s_view_t& destination);

void y_gradient(const boost::gil::gray8c_view_t& source, const boost::gil::gray8s_view_t& destination);
