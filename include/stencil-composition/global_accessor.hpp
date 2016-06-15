#pragma once

#include "../common/defs.hpp"
#include "./empty_extent.hpp"

namespace gridtools {

    template < uint_t I, enumtype::intend Intend >
    struct global_accessor {

        typedef global_accessor< I, Intend > type;

        typedef static_uint< I > index_type;

        typedef empty_extent extent_t;
    };

    template < typename Type >
    struct is_global_accessor : boost::false_type {};

    template < uint_t I, enumtype::intend Intend >
    struct is_global_accessor< global_accessor< I, Intend > > : boost::true_type {};

} // namespace gridtools
