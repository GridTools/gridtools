#pragma once

#include "./grid_traits.hpp"

namespace gridtools {
    template <typename T>
    struct is_grid_traits_from_id : boost::false_type {};

    template <enumtype::grid_type G>
    struct is_grid_traits_from_id<grid_traits_from_id<G> > : boost::true_type {};
} // namespace gridtools
