#pragma once
#include <common/string_c.hpp>
namespace gridtools {
    template < int I, ushort_t NColors >
    struct location_type {
        static const int value = I;
        using n_colors = static_ushort< NColors >; //! <- is the number of locations of this type
    };

    template < typename T >
    struct is_location_type : boost::mpl::false_ {};

    template < int I, ushort_t NColors >
    struct is_location_type< location_type< I, NColors > > : boost::mpl::true_ {};

    template < int I, ushort_t NColors >
    std::ostream &operator<<(std::ostream &s, location_type< I, NColors >) {
        return s << "location_type<" << I << "> with " << NColors << " colors";
    }
} // namespace gridtools
