#pragma once

#include "../common/defs.hpp"

namespace gridtools {

    template < uint_t I, enumtype::intend Intend >
    struct global_accessor {

        typedef global_accessor< I, Intend > type;
        // static const ushort_t n_dim=Dim;
        typedef static_uint< I > index_type;
        // typedef Range range_type;
    };

} // namespace gridtools
