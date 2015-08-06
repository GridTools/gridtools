#pragma once
#include <common/string_c.h>
namespace gridtools {
    template <int I, ushort_t NColors, char const * Name=NULL>
    struct location_type {
        typedef string_c<print, Name> print_name;
        static const int value = I;
        static const uint_t n_colors = NColors; //! <- is the number of locations of this type
    };

    template <int I, ushort_t NColors, char const * Name=NULL>
    std::ostream& operator<<(std::ostream& s, location_type<I, NColors, Name>) {
        return s << "location_type<" << I << "> with "<< NColors<< " colors";
    }
} //namespace gridtools
