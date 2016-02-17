#pragma once
#include <boost/mpl/quote.hpp>

#include "compute_extents_metafunctions.hpp"
#include "icosahedral_grid_traits.hpp"

namespace gridtools {

    template <>
    struct grid_traits_from_id< icosahedral > {
        struct select_mss_compute_extent_sizes {
            typedef boost::mpl::quote1< icgrid::mss_compute_extent_sizes > type;
        };

        typedef extent< 0 > null_extent_t;

        template < enumtype::platform BackendId >
        struct with_arch {
            typedef icgrid::grid_traits< BackendId > type;
        };
    };
}
