#pragma once
#include "../../common/defs.hpp"

namespace gridtools {
    enum color { downard = 0, upward };

    template < uint_t c >
    struct color_type {
        static_uint< c > color_t;
    };

    template < typename T >
    struct is_color_type : boost::mpl::false_ {};

    template < uint_t c >
    struct is_color_type< color_type< c > > : boost::mpl::true_ {};

    template <>
    struct is_color_type< notype > : boost::mpl::true_ {};

}
