#pragma once

#include <boost/mpl/quote.hpp>
#include "compute_extents_metafunctions.hpp"

namespace gridtools {

    template < enumtype::grid_type GridId >
    struct grid_traits_from_id {
        struct select_mss_compute_extent_sizes {
            typedef boost::mpl::quote1< strgrid::mss_compute_extent_sizes > type;
        };
    };

    typedef extent<0,0,0,0> null_extent_t;
}
