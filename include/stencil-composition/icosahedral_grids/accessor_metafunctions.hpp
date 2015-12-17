#pragma once

#include "accessor.hpp"

namespace gridtools{

    template <typename Accessor>
    struct accessor_index
    {
        GRIDTOOLS_STATIC_ASSERT((is_accessor<Accessor>::value), "Internal Error: wrong type");
        typedef typename Accessor::index_type type;
    };

    template<typename Accessor> struct is_accessor_readonly : boost::mpl::false_{};

    template <int ID, typename LocationType, typename Radius>
    struct is_accessor_readonly<ro_accessor<ID, LocationType, Radius> > : boost::mpl::true_{};

} //namespace gridtools
