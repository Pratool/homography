#pragma once

#include <stdexcept>

namespace pcv
{

class ImageLoadError : public std::runtime_error
{
public:
    const char *what() const throw()
    {
        return "pcv::Image could not be constructed properly.";
    }

};

}
