#pragma once

namespace gridtools {
    template <int I>
    struct location_type {
        static const int value = I;
    };

    template <int I>
    std::ostream& operator<<(std::ostream& s, location_type<I>) {
        return s << "location_type<" << I << ">";
    }
} //namespace gridtools
