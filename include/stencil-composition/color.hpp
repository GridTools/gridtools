#pragma once

namespace gridtools {
    template < uint_t c >
    struct color_type {
        typedef static_uint< c > color_t;
    };

    template < typename T >
    struct is_color_type : boost::mpl::false_ {};

    template < uint_t c >
    struct is_color_type< color_type< c > > : boost::mpl::true_ {};

    template <>
    struct is_color_type< notype > : boost::mpl::true_ {};

    typedef notype nocolor;
};
