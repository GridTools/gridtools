#include <gridtools.hpp>



extern "C"
{
    int get_backend_float_size ( ) {
        return sizeof(gridtools::float_type) * 8;
    };
}
