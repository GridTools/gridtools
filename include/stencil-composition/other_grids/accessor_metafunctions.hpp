#pragma once

#include "accessor.hpp"

namespace gridtools{

    template <typename Accessor>
    struct accessor_index
    {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Internal Error: wrong type");
        typedef typename Accessor::index_type type;
    };

} //namespace gridtools
