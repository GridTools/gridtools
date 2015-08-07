#pragma once
#include <common/string_c.h>
namespace gridtools {
    template <int I, ushort_t NColors>
    struct location_type {
        static const int value = I;
        static const uint_t n_colors = NColors; //! <- is the number of locations of this type
    };

    template <int I, ushort_t NColors>
    std::ostream& operator<<(std::ostream& s, location_type<I, NColors>) {
        return s << "location_type<" << I << "> with "<< NColors<< " colors";
    }
} //namespace gridtools
