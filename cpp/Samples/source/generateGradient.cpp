#include <boost/gil/image.hpp>
#include <boost/gil/typedefs.hpp>
#include <boost/gil/extension/io/png/write.hpp>

/// Models a unary function
/// @tparam P Models PixelValueConcept
template <typename P>
struct gradient_function
{
  using point_t = boost::gil::point_t;
  using const_t = gradient_function;
  using value_type = P;
  using reference = value_type;
  using const_reference = value_type;
  using argument_type = point_t;
  using result_type = reference;
  static constexpr bool is_mutable = false;

private:
  point_t image_size_;

public:
  gradient_function() = default;
  gradient_function(const point_t& image_size) : image_size_(image_size)
  {
  }

  result_type operator()(const point_t& p) const
  {
    using namespace boost::gil;

    const auto x_intensity = static_cast<double>(p.x)/image_size_.x;
    const auto y_intensity = static_cast<double>(p.y)/image_size_.y;
    const double intensity = (x_intensity + y_intensity) / 2.0;
    const auto pixel_value = channel_traits<typename channel_type<P>::type>::max_value() * intensity;

    value_type return_value;

    for (std::size_t channel_index = 0; channel_index < num_channels<P>::value; ++channel_index)
    {
      return_value[channel_index] = pixel_value;
    }

    return return_value;
  }
};

int main()
{
    using deref_t = gradient_function<boost::gil::rgb8_pixel_t>;
    using point_t = deref_t::point_t;
    using locator_t = boost::gil::virtual_2d_locator<deref_t,false>;
    using virtual_view_t = boost::gil::image_view<locator_t>;

    boost::function_requires<boost::gil::PixelLocatorConcept<locator_t>>();
    boost::gil::gil_function_requires<boost::gil::StepIteratorConcept<locator_t::x_iterator>>();

    point_t dimensions(1028, 1028);
    virtual_view_t mandel(dimensions, locator_t(point_t(0,0), point_t(1,1), deref_t(dimensions)));
    write_view("out-gradient.png", mandel, boost::gil::png_tag{});

    return 0;
}
